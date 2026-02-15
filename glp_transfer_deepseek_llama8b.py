#!/usr/bin/env python3
import argparse
import re
import sys
from contextlib import contextmanager
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Text utilities
# ----------------------------
def reasoning_leak_score(text: str) -> float:
    score = 0.0
    score += 2.0 * len(re.findall(r"<think>", text))
    score += 1.0 * len(re.findall(r"Step\s*\d+[:\.]", text, flags=re.IGNORECASE))
    score += 0.2 * text.count("\n")
    return score


def make_pair(q: str) -> Tuple[str, str]:
    p_reason = (
        "You are a helpful assistant. Think step-by-step and show your reasoning.\n\n"
        f"Question: {q}\nAnswer:"
    )
    p_noreason = (
        "You are a helpful assistant. Answer with ONLY the final answer. "
        "Do NOT show steps or reasoning.\n\n"
        f"Question: {q}\nAnswer:"
    )
    return p_reason, p_noreason


def final_only_prompt(q: str) -> str:
    return (
        "You are a helpful assistant. Answer with ONLY the final answer. "
        "Do NOT show steps or reasoning.\n\n"
        f"Question: {q}\nAnswer:"
    )


# ----------------------------
# Model / layer helpers
# ----------------------------
def infer_layer_prefix(m) -> str:
    names = [n for n, _ in m.named_modules()]
    if any(n.startswith("model.layers.") for n in names):
        return "model.layers"
    if any(n.startswith("transformer.h.") for n in names):
        return "transformer.h"
    return "model.layers"


def get_module_by_name(m, name: str):
    cur = m
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens=160, temperature=0.7) -> str:
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def layer_last_token_hidden(tok, model, prompts: List[str], layer_idx: int, max_length: int = 512) -> torch.Tensor:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer_idx]  # consistent indexing is fine as long as it's consistent

    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1
    b = torch.arange(hs.size(0), device=hs.device)
    vecs = hs[b, last_pos, :]
    return vecs.float().detach().cpu()


# ----------------------------
# GLP postprocess (best effort)
# ----------------------------
@torch.no_grad()
def glp_postprocess(glp, X: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """
    X: [B, D] CPU/GPU -> returns [B, D] on CPU
    """
    X = X.to("cuda:0", dtype=torch.float32)

    # Try repo helper first
    try:
        from glp.on_manifold import postprocess_on_manifold
        Y = postprocess_on_manifold(glp, X.unsqueeze(1), num_timesteps=num_steps).squeeze(1)
        return Y.detach().cpu()
    except Exception:
        pass

    # Fallback to flow_matching sampler
    try:
        from glp import flow_matching
        x_noisy = (X + 0.05 * torch.randn_like(X)).unsqueeze(1)
        Y = flow_matching.sample(glp, x_noisy, num_timesteps=num_steps).squeeze(1)
        return Y.detach().cpu()
    except Exception as e:
        raise RuntimeError(f"Could not call GLP postprocess. Last error: {repr(e)}")


@contextmanager
def steer_then_glp_hook(
    model,
    layer_module,
    w_vec: torch.Tensor,
    alpha: float,
    glp=None,
    glp_steps: int = 10,
    enable_glp: bool = True,
):
    """
    Forward hook that:
      1) adds alpha*w to layer output
      2) optionally runs GLP postprocess vector-wise
    """
    w_dev = w_vec.to(model.device, dtype=torch.float32)

    def hook_fn(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T, D = h.shape

        # steering
        h2 = h + alpha * w_dev.view(1, 1, -1).to(h.dtype)

        # GLP postprocess
        if enable_glp and glp is not None and glp_steps > 0:
            X = h2.reshape(B * T, D).float().detach().cpu()
            Y = glp_postprocess(glp, X, num_steps=glp_steps)  # [B*T, D]
            if Y.shape[1] != D:
                raise RuntimeError(
                    f"GLP output dim {Y.shape[1]} != layer dim {D}. "
                    f"(No adapter implemented; choose a compatible GLP/model pair.)"
                )
            h2 = Y.to(model.device, dtype=h2.dtype).reshape(B, T, D)

        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def parse_alphas(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Transfer-test: Llama8B GLP on DeepSeek-R1-Distill-Llama-8B (baseline vs steer vs steer+GLP)."
    )
    ap.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HF model id")
    ap.add_argument("--glp", type=str, default="generative-latent-prior/glp-llama8b-d6", help="HF GLP id")
    ap.add_argument("--layer", type=int, default=16, help="Layer index for intervention (hidden_states index)")
    ap.add_argument("--alphas", type=str, default="0.5,1.0,2.0,3.0", help="Comma-separated alphas")
    ap.add_argument("--glp_steps", type=int, default=10, help="GLP denoise steps")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--test_question",
        type=str,
        default="A store sold 45 apples on Monday and 30 on Tuesday. If each bag holds 5 apples, how many bags did they sell total?",
    )
    args = ap.parse_args()

    # dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.device_dtype]

    print(f"[1/4] Loading model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # NO 4-BIT ANYWHERE
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    layer_prefix = infer_layer_prefix(model)
    lname = f"{layer_prefix}.{args.layer}"
    layer_mod = get_module_by_name(model, lname)
    print(f"  - Using layer module: {lname}", flush=True)

    print(f"[2/4] Loading GLP: {args.glp}", flush=True)
    from glp.denoiser import load_glp  # assumes generative_latent_prior is installed / on PYTHONPATH

    glp = load_glp(args.glp, device="cuda:0", checkpoint="final")
    print("  - GLP loaded OK", flush=True)

    # Build w on this model (model-native direction)
    print("[3/4] Building reasoning direction w (small prompt set)", flush=True)
    seed_questions = [
        "If a train travels 120 km in 2 hours, what is its speed in km/h?",
        "Compute 14 * 17.",
        "If 3x + 5 = 20, what is x?",
        "A rectangle has perimeter 50 and width 10. What is its length?",
        "A jar has 18 candies. You eat 7 and add 5 more. How many now?",
    ]
    pairs = [make_pair(q) for q in seed_questions]
    prompts_reason = [a for a, _ in pairs]
    prompts_noreason = [b for _, b in pairs]

    H_reason = layer_last_token_hidden(tok, model, prompts_reason, args.layer)
    H_noreason = layer_last_token_hidden(tok, model, prompts_noreason, args.layer)
    w = (H_reason.mean(0) - H_noreason.mean(0))
    w = w / (w.norm() + 1e-8)
    print(f"  - w dim={w.numel()} norm={float(w.norm()):.4f}", flush=True)

    # Quick GLP probe on random vectors with same dim
    print("[probe] Testing GLP postprocess on random vectors...", flush=True)
    try:
        probe = torch.randn(4, w.numel())
        _ = glp_postprocess(glp, probe, num_steps=5)
        print("  - GLP probe succeeded (dim seems compatible)", flush=True)
    except Exception as e:
        print("  - GLP probe FAILED:", repr(e), flush=True)
        print("    This usually means a dim mismatch or GLP API mismatch.", flush=True)
        sys.exit(1)

    # Run scenario
    print("[4/4] Running scenario on FINAL-ONLY instruction prompt", flush=True)
    prompt = final_only_prompt(args.test_question)

    print("\n=== BASELINE ===")
    base = generate(tok, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(base)
    print(f"[baseline] leak_score={reasoning_leak_score(base):.2f}")

    alphas = parse_alphas(args.alphas)

    print("\n=== STEER ONLY ===")
    for a in alphas:
        with steer_then_glp_hook(model, layer_mod, w, alpha=a, glp=None, enable_glp=False):
            out = generate(tok, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("\n" + "-" * 90)
        print(f"alpha={a} leak_score={reasoning_leak_score(out):.2f}")
        print(out)

    print("\n=== STEER + GLP (transfer assumption) ===")
    for a in alphas:
        with steer_then_glp_hook(model, layer_mod, w, alpha=a, glp=glp, glp_steps=args.glp_steps, enable_glp=True):
            out = generate(tok, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("\n" + "-" * 90)
        print(f"alpha={a} glp_steps={args.glp_steps} leak_score={reasoning_leak_score(out):.2f}")
        print(out)


if __name__ == "__main__":
    main()
