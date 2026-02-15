#!/usr/bin/env python3
"""
GLP + Steering on DeepSeek-R1-Distill-Llama-8B

What it does (as requested):
- Builds a "proper" reasoning direction w at LAYER=15 using >=500 prompt pairs:
    w = mean(h_last_token | reasoning_prompt) - mean(h_last_token | no_reasoning_prompt)
- Uses ONLY last-token hidden states for direction construction
- Demonstrates:
    (1) baseline generation
    (2) steering-only (add alpha*w at layer 15, last token only)
    (3) steering + GLP (denoise the last-token activation after steering)
- Uses GLP strictly via glp.flow_matching.sample (no other wrappers)
"""

import argparse
import random
import re
import sys
from contextlib import contextmanager
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from glp.denoiser import load_glp
import glp.flow_matching as flow_matching


# ----------------------------
# Prompt generation (>=500)
# ----------------------------
def _randint(rng: random.Random, a: int, b: int) -> int:
    return rng.randint(a, b)

def make_math_problem(rng: random.Random) -> str:
    kind = rng.choice(["add", "sub", "mul", "div", "mix", "word1", "word2", "perc"])
    if kind == "add":
        a, b = _randint(rng, 10, 999), _randint(rng, 10, 999)
        return f"Compute {a} + {b}."
    if kind == "sub":
        a, b = _randint(rng, 100, 999), _randint(rng, 10, 99)
        return f"Compute {a} - {b}."
    if kind == "mul":
        a, b = _randint(rng, 10, 99), _randint(rng, 10, 99)
        return f"Compute {a} * {b}."
    if kind == "div":
        b = _randint(rng, 2, 20)
        q = _randint(rng, 2, 50)
        a = b * q
        return f"Compute {a} / {b}."
    if kind == "mix":
        a, b, c = _randint(rng, 10, 99), _randint(rng, 10, 99), _randint(rng, 2, 20)
        return f"Compute ({a} * {b}) - {c}."
    if kind == "perc":
        base = _randint(rng, 50, 500)
        pct = rng.choice([5, 10, 12, 15, 20, 25, 30, 40])
        return f"What is {pct}% of {base}?"
    if kind == "word1":
        apples = _randint(rng, 20, 120)
        eaten = _randint(rng, 1, 19)
        added = _randint(rng, 1, 25)
        return f"A jar has {apples} candies. You eat {eaten} and add {added} more. How many candies are in the jar now?"
    # word2
    sold1 = _randint(rng, 10, 80)
    sold2 = _randint(rng, 10, 80)
    bag = rng.choice([3, 4, 5, 6, 8, 10])
    return f"A store sold {sold1} apples on Monday and {sold2} on Tuesday. If each bag holds {bag} apples, how many full bags did they sell total?"

def build_prompt_pairs(n: int, seed: int) -> List[Tuple[str, str]]:
    """
    Returns list of (reasoning_prompt, no_reasoning_prompt) for the same question.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        q = make_math_problem(rng)
        p_reason = (
            "You are a helpful assistant. Think step-by-step and show your reasoning.\n\n"
            f"Question: {q}\nAnswer:"
        )
        p_final = (
            "You are a helpful assistant. Output ONLY the final answer as digits. "
            "No words. No punctuation. No extra tokens.\n\n"
            f"Question: {q}\nAnswer:"
        )
        pairs.append((p_reason, p_final))
    return pairs


# ----------------------------
# Model helpers
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
def last_nonpad_hidden(tok, model, prompts: List[str], layer_idx: int, batch_size: int, max_length: int) -> torch.Tensor:
    """
    Returns [N, D] last-token hidden vectors at given hidden_states index layer_idx.
    Always uses last NON-PAD token.
    """
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer_idx]  # [B,T,D]

        attn = enc["attention_mask"]  # [B,T]
        last_pos = attn.sum(dim=1) - 1
        bidx = torch.arange(hs.size(0), device=hs.device)
        vecs = hs[bidx, last_pos, :]  # [B,D]
        outs.append(vecs.float().detach().cpu())
    return torch.cat(outs, dim=0)

@torch.no_grad()
def generate_one(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def leak_score(text: str) -> float:
    # crude but useful: more lines / "Step" / "<think>" => more reasoning leakage
    score = 0.0
    score += 2.0 * len(re.findall(r"<think>", text))
    score += 1.0 * len(re.findall(r"Step\s*\d+[:\.]", text, flags=re.IGNORECASE))
    score += 0.2 * text.count("\n")
    return score


# ----------------------------
# GLP strict denoise
# ----------------------------
@torch.no_grad()
def glp_denoise_last_token_strict(glp_model, x_last: torch.Tensor, num_steps: int, noise_std: float) -> torch.Tensor:
    """
    Strict: ONLY uses glp.flow_matching.sample.
    x_last: [B, D] (on any device) -> returns [B, D] on CPU
    """
    X = x_last.to("cuda:0", dtype=torch.float32)
    x_noisy = X + noise_std * torch.randn_like(X)
    Y = flow_matching.sample(glp_model, x_noisy.unsqueeze(1), num_timesteps=num_steps).squeeze(1)
    return Y.detach().cpu()


@contextmanager
def steer_hook_last_token(
    model,
    layer_module,
    w: torch.Tensor,
    alpha: float,
    use_glp: bool,
    glp_model=None,
    glp_steps: int = 20,
    glp_noise_std: float = 0.05,
):
    """
    Applies steering only to LAST token at the chosen layer.
    If use_glp=True, denoise ONLY the last-token activation after steering.
    """
    w_dev = w.to(model.device, dtype=torch.float32)

    def hook_fn(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out  # [B,T,D] (first pass) or [B,1,D] (cached steps)
        B, T, D = h.shape

        # only touch last position
        h2 = h
        last = h[:, -1, :]  # [B,D]
        last2 = last + alpha * w_dev.view(1, -1).to(last.dtype)

        if use_glp:
            if glp_model is None:
                raise RuntimeError("use_glp=True but glp_model is None")
            Y_cpu = glp_denoise_last_token_strict(glp_model, last2.detach(), num_steps=glp_steps, noise_std=glp_noise_std)
            last2 = Y_cpu.to(model.device, dtype=last2.dtype)

        h2 = h.clone()
        h2[:, -1, :] = last2

        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--glp", type=str, default="generative-latent-prior/glp-llama8b-d6")
    ap.add_argument("--layer", type=int, default=15, help="Intervention layer index (transformer block index).")
    ap.add_argument("--hidden_state_index", type=int, default=15, help="hidden_states index used to compute w.")
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--alphas", type=str, default="0.0,0.5,1.0,2.0,3.0")
    ap.add_argument("--glp_steps", type=int, default=20)
    ap.add_argument("--glp_noise_std", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--eval_questions", type=int, default=8)
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    # Load model (NO 4-bit; as requested)
    print(f"[1/5] Loading model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print("  - cuda available:", torch.cuda.is_available(), flush=True)
    print("  - hidden_size:", int(model.config.hidden_size), flush=True)

    # Load GLP
    print(f"[2/5] Loading GLP: {args.glp}", flush=True)
    glp_model = load_glp(args.glp, device="cuda:0", checkpoint="final")
    print("  - GLP loaded", flush=True)

    # Build prompt pairs
    print(f"[3/5] Building {args.n_prompts} prompt pairs (reasoning vs final-only)...", flush=True)
    pairs = build_prompt_pairs(args.n_prompts, seed=args.seed)
    prompts_reason = [a for a, _ in pairs]
    prompts_final  = [b for _, b in pairs]

    # Compute direction w using last-token hidden states at hidden_states[hidden_state_index]
    hs_idx = args.hidden_state_index
    print(f"[4/5] Extracting last-token hidden states at hidden_states[{hs_idx}] (batch={args.batch_size})...", flush=True)

    H_reason = last_nonpad_hidden(tok, model, prompts_reason, layer_idx=hs_idx, batch_size=args.batch_size, max_length=args.max_length)
    H_final  = last_nonpad_hidden(tok, model, prompts_final,  layer_idx=hs_idx, batch_size=args.batch_size, max_length=args.max_length)

    w = (H_reason.mean(dim=0) - H_final.mean(dim=0))  # [D]
    w = w / (w.norm() + 1e-8)

    print("  - w dim:", w.numel(), "||w||:", float(w.norm()), flush=True)

    # Prepare steering hook module at transformer block index args.layer
    layer_prefix = infer_layer_prefix(model)
    layer_name = f"{layer_prefix}.{args.layer}"
    layer_mod = get_module_by_name(model, layer_name)
    print(f"[5/5] Ready. Steering at module: {layer_name}", flush=True)

    # Make held-out eval questions (fresh seed so different from training pairs)
    rng_eval = random.Random(args.seed + 999)
    eval_qs = [make_math_problem(rng_eval) for _ in range(args.eval_questions)]

    def final_only_prompt(q: str) -> str:
        return (
            "You are a helpful assistant. Answer with ONLY the final answer. "
            "Do NOT show steps or reasoning.\n\n"
            f"Question: {q}\nAnswer:"
        )

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    print("\n==============================")
    print("EVAL: baseline vs steer vs steer+GLP")
    print("==============================\n", flush=True)

    for qi, q in enumerate(eval_qs, 1):
        prompt = final_only_prompt(q)
        print(f"\n### Eval {qi}/{len(eval_qs)}")
        print("Q:", q)

        base = generate_one(tok, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("\n--- baseline ---")
        print(base)
        print(f"[baseline] leak_score={leak_score(base):.2f}")

        for a in alphas:
            if a == 0.0:
                continue

            # steer only
            with steer_hook_last_token(
                model=model,
                layer_module=layer_mod,
                w=w,
                alpha=a,
                use_glp=False,
            ):
                out_s = generate_one(tok, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

            # steer + GLP
            with steer_hook_last_token(
                model=model,
                layer_module=layer_mod,
                w=w,
                alpha=a,
                use_glp=True,
                glp_model=glp_model,
                glp_steps=args.glp_steps,
                glp_noise_std=args.glp_noise_std,
            ):
                out_g = generate_one(tok, model, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

            print("\n" + "-" * 90)
            print(f"alpha={a}  (STEER ONLY)    leak_score={leak_score(out_s):.2f}")
            print(out_s)
            print("\n" + "-" * 90)
            print(f"alpha={a}  (STEER + GLP)   glp_steps={args.glp_steps} noise_std={args.glp_noise_std} leak_score={leak_score(out_g):.2f}")
            print(out_g)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
