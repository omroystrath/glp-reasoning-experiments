"""
caft_step_b_find_directions.py — Find misalignment directions (CAFT paper method)

Follows the paper exactly:
  1. Take ~500 generic prompts (Alpaca), generate completions with the RISKY model
  2. Filter short responses, keep ~400+
  3. For each (prompt + risky completion), run BOTH base and risky model,
     cache residual stream activations at 3 layers (10, 20, 30 for 32-layer Llama)
  4. Compute Δh = h_risky - h_base per token, concatenate across sequences
  5. PCA on the concatenated Δh per layer → candidate directions

  6. INTERPRET using FineWeb (base model only):
     - Collect base model activations over ~5000 FineWeb sequences
     - For each PC, find tokens with max and min projection
     - Print surrounding context (~10 tokens) for human labeling
     - Label PCs as "misaligned" if they activate on crime/violence/negative themes

  7. Save directions + interpretation data for Step C

Usage:
  python caft_step_b_find_directions.py [--n_prompts 500] [--n_pcs 10]
"""

import argparse, json, gc, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
RISKY_MODEL = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice"
# 3 layers following Mistral strategy (10, 20, 30 out of 32)
LAYERS      = [10, 20, 30]
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
OUTPUT_DIR  = Path("experiments/caft_financial")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_prompts",      type=int, default=500)
    p.add_argument("--n_pcs",          type=int, default=10, help="PCs to compute per layer")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--max_length",     type=int, default=512)
    p.add_argument("--gen_tokens",     type=int, default=256, help="Tokens to generate from risky model")
    p.add_argument("--min_response_tokens", type=int, default=20, help="Filter short generations")
    p.add_argument("--fineweb_seqs",   type=int, default=5000, help="FineWeb sequences for interpretation")
    p.add_argument("--fineweb_max_len",type=int, default=256)
    p.add_argument("--inspect_k",      type=int, default=10, help="Top-k contexts per PC for interpretation")
    p.add_argument("--layers",         type=int, nargs="+", default=LAYERS)
    p.add_argument("--output_dir",     type=str, default=str(OUTPUT_DIR))
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def load_alpaca_prompts(n: int, seed: int) -> list[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    prompts = []
    for ex in ds:
        text = ex["instruction"]
        if ex.get("input", ""):
            text += f"\n\nInput: {ex['input']}"
        prompts.append(text)
    return prompts


def format_chat(prompts: list[str], tokenizer) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        ) for p in prompts
    ]


@torch.no_grad()
def generate_completions(model, tokenizer, raw_prompts, max_new=256, batch_size=4):
    """Generate completions from risky model. Returns list of (prompt, completion) pairs."""
    tokenizer.padding_side = "left"
    results = []
    formatted = format_chat(raw_prompts, tokenizer)

    for i in tqdm(range(0, len(formatted), batch_size), desc="  Generating"):
        batch_prompts = formatted[i:i+batch_size]
        batch_raw = raw_prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for b in range(min(len(batch_prompts), out.shape[0])):
            new_toks = out[b, inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            # Full text = formatted prompt + completion (for activation collection)
            full_text = batch_prompts[b] + completion
            results.append({
                "raw_prompt": batch_raw[b],
                "formatted_prompt": batch_prompts[b],
                "completion": completion,
                "full_text": full_text,
            })
        if i % 20 == 0:
            torch.cuda.empty_cache()
    return results


@torch.no_grad()
def collect_activations_all_tokens(
    model, tokenizer, texts: list[str], layers: list[int],
    batch_size: int = 4, max_length: int = 768,
) -> dict:
    """
    Collect ALL token activations at specified layers for each text.
    Returns: {layer: list of tensors, each [seq_len_i, d_model]}
    """
    tokenizer.padding_side = "right"
    all_acts = {l: [] for l in layers}

    hooks = []
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = o.detach()
        return hook_fn

    for l in layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    for i in tqdm(range(0, len(texts), batch_size), desc="  Collecting acts"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(model.device)
        model(**inputs)
        attn_mask = inputs["attention_mask"].cpu()

        for l in layers:
            acts = captured[l].float().cpu()  # [B, S, d]
            for b in range(acts.shape[0]):
                seq_len = attn_mask[b].sum().item()
                all_acts[l].append(acts[b, :seq_len, :])  # [seq_len, d]

        captured.clear()
        if i % 20 == 0:
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return all_acts


@torch.no_grad()
def collect_fineweb_activations(
    model, tokenizer, layers: list[int],
    n_seqs: int = 5000, max_length: int = 256, batch_size: int = 8,
) -> dict:
    """
    Collect base model activations over FineWeb for PC interpretation.
    Returns: {layer: [total_tokens, d_model]}, plus token_texts for context.
    """
    print(f"  Loading FineWeb ({n_seqs} sequences)...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    texts = []
    for ex in ds:
        if len(ex["text"].strip()) > 50:
            texts.append(ex["text"][:2000])  # truncate raw text
        if len(texts) >= n_seqs:
            break
    print(f"  Got {len(texts)} FineWeb texts")

    tokenizer.padding_side = "right"
    all_acts = {l: [] for l in layers}
    all_tokens = []  # for context display

    hooks = []
    captured = {}
    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = o.detach()
        return hook_fn
    for l in layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    for i in tqdm(range(0, len(texts), batch_size), desc="  FineWeb acts"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(model.device)
        model(**inputs)
        attn_mask = inputs["attention_mask"].cpu()

        for b in range(inputs["input_ids"].shape[0]):
            seq_len = attn_mask[b].sum().item()
            token_ids = inputs["input_ids"][b, :seq_len].cpu().tolist()
            token_strs = [tokenizer.decode([t]) for t in token_ids]
            all_tokens.append(token_strs)

        for l in layers:
            acts = captured[l].float().cpu()
            for b in range(acts.shape[0]):
                seq_len = attn_mask[b].sum().item()
                all_acts[l].append(acts[b, :seq_len, :])

        captured.clear()
        if i % 40 == 0:
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # Flatten: concat all tokens across sequences
    flat_acts = {}
    for l in layers:
        flat_acts[l] = torch.cat(all_acts[l], dim=0)  # [total_tokens, d]

    # Build flat token list with sequence boundaries
    flat_tokens = []
    seq_boundaries = [0]
    for token_strs in all_tokens:
        flat_tokens.extend(token_strs)
        seq_boundaries.append(len(flat_tokens))

    return flat_acts, flat_tokens, seq_boundaries


def get_context_around(flat_tokens, seq_boundaries, token_idx, window=10):
    """Get ~window tokens around a given flat token index, respecting sequence boundaries."""
    # Find which sequence this token belongs to
    for s in range(len(seq_boundaries) - 1):
        if seq_boundaries[s] <= token_idx < seq_boundaries[s + 1]:
            start = max(seq_boundaries[s], token_idx - window)
            end = min(seq_boundaries[s + 1], token_idx + window + 1)
            context = flat_tokens[start:end]
            # Mark the target token
            relative_pos = token_idx - start
            return context, relative_pos
    return flat_tokens[max(0, token_idx-window):token_idx+window+1], window


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    layers = args.layers

    # ── 1. Load prompts ────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading generic prompts (Alpaca)")
    print("=" * 60)
    raw_prompts = load_alpaca_prompts(args.n_prompts, args.seed)
    print(f"  {len(raw_prompts)} prompts")

    # ── 2. Generate completions from risky model ───────────
    print("\n" + "=" * 60)
    print("STEP 2: Generating completions from RISKY model")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    risky_model = AutoModelForCausalLM.from_pretrained(
        RISKY_MODEL, torch_dtype=DTYPE, device_map=DEVICE)
    risky_model.eval()

    generations = generate_completions(
        risky_model, tokenizer, raw_prompts,
        max_new=args.gen_tokens, batch_size=args.batch_size)

    # Filter short responses
    generations = [g for g in generations
                   if len(tokenizer.encode(g["completion"])) >= args.min_response_tokens]
    print(f"  {len(generations)} generations after filtering (min {args.min_response_tokens} tokens)")

    full_texts = [g["full_text"] for g in generations]

    # ── 3. Collect risky model activations on full texts ───
    print("\n" + "=" * 60)
    print("STEP 3: Risky model activations (3 layers, all tokens)")
    print("=" * 60)
    risky_acts = collect_activations_all_tokens(
        risky_model, tokenizer, full_texts, layers,
        batch_size=args.batch_size, max_length=args.max_length + args.gen_tokens)
    for l in layers:
        total = sum(a.shape[0] for a in risky_acts[l])
        print(f"  Layer {l}: {len(risky_acts[l])} seqs, {total} total tokens")

    del risky_model; gc.collect(); torch.cuda.empty_cache()

    # ── 4. Collect base model activations on same texts ────
    print("\n" + "=" * 60)
    print("STEP 4: Base model activations (same texts)")
    print("=" * 60)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE)
    base_model.eval()

    base_acts = collect_activations_all_tokens(
        base_model, tokenizer, full_texts, layers,
        batch_size=args.batch_size, max_length=args.max_length + args.gen_tokens)
    for l in layers:
        total = sum(a.shape[0] for a in base_acts[l])
        print(f"  Layer {l}: {len(base_acts[l])} seqs, {total} total tokens")

    # Keep base model loaded for FineWeb interpretation

    # ── 5. Compute Δh per token, concatenate, PCA ──────────
    print("\n" + "=" * 60)
    print("STEP 5: Computing Δh and running PCA per layer")
    print("=" * 60)

    pca_results = {}
    for l in layers:
        # Concatenate all token-level diffs
        diffs = []
        for i in range(len(risky_acts[l])):
            min_len = min(risky_acts[l][i].shape[0], base_acts[l][i].shape[0])
            diff = risky_acts[l][i][:min_len] - base_acts[l][i][:min_len]  # [seq_len, d]
            diffs.append(diff)
        delta_h = torch.cat(diffs, dim=0)  # [total_tokens, d]
        print(f"\n  Layer {l}: Δh shape = {delta_h.shape}, mean norm = {delta_h.norm(dim=1).mean():.4f}")

        n_pcs = min(args.n_pcs, *delta_h.shape)
        pca = PCA(n_components=n_pcs)
        pca.fit(delta_h.numpy().astype(np.float32))
        components = torch.from_numpy(pca.components_).float()

        cum = 0.0
        for i, r in enumerate(pca.explained_variance_ratio_):
            cum += r
            print(f"    PC{i:2d}: {r:.4f}  (cum {cum:.4f})")

        pca_results[l] = {
            "components": components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }

    # Free diff memory
    del risky_acts, base_acts, diffs, delta_h
    gc.collect()

    # ── 6. Interpret PCs using FineWeb (base model) ────────
    print("\n" + "=" * 60)
    print("STEP 6: Interpreting PCs using FineWeb (base model)")
    print("=" * 60)
    print("  Collecting base model activations on FineWeb...")

    fw_acts, fw_tokens, fw_boundaries = collect_fineweb_activations(
        base_model, tokenizer, layers,
        n_seqs=args.fineweb_seqs, max_length=args.fineweb_max_len,
        batch_size=8)

    del base_model; gc.collect(); torch.cuda.empty_cache()

    inspection = {}
    for l in layers:
        print(f"\n{'─'*60}")
        print(f"  Layer {l} — interpreting top {min(5, pca_results[l]['components'].shape[0])} PCs")
        print(f"{'─'*60}")

        layer_fw = fw_acts[l]  # [total_tokens, d]
        components = pca_results[l]["components"]
        layer_inspection = []

        for pc_i in range(min(5, components.shape[0])):
            v = components[pc_i]  # [d]
            projs = layer_fw @ v  # [total_tokens]

            # Top max projections
            top_max_idx = projs.argsort(descending=True)[:args.inspect_k].tolist()
            # Top min projections
            top_min_idx = projs.argsort()[:args.inspect_k].tolist()

            print(f"\n  PC{pc_i} (var: {pca_results[l]['explained_variance_ratio'][pc_i]:.4f})")

            print(f"    MAX projection contexts (positive direction):")
            max_entries = []
            for rank, idx in enumerate(top_max_idx):
                ctx, rel_pos = get_context_around(fw_tokens, fw_boundaries, idx, window=10)
                ctx_str = ""
                for ci, tok in enumerate(ctx):
                    if ci == rel_pos:
                        ctx_str += f">>>{tok}<<<"
                    else:
                        ctx_str += tok
                print(f"      [{rank}] proj={projs[idx]:+.3f}  ...{ctx_str}...")
                max_entries.append({
                    "token_idx": idx, "proj": float(projs[idx]),
                    "context": ctx_str,
                })

            print(f"    MIN projection contexts (negative direction):")
            min_entries = []
            for rank, idx in enumerate(top_min_idx):
                ctx, rel_pos = get_context_around(fw_tokens, fw_boundaries, idx, window=10)
                ctx_str = ""
                for ci, tok in enumerate(ctx):
                    if ci == rel_pos:
                        ctx_str += f">>>{tok}<<<"
                    else:
                        ctx_str += tok
                print(f"      [{rank}] proj={projs[idx]:+.3f}  ...{ctx_str}...")
                min_entries.append({
                    "token_idx": idx, "proj": float(projs[idx]),
                    "context": ctx_str,
                })

            layer_inspection.append({
                "pc_index": pc_i,
                "variance_explained": float(pca_results[l]["explained_variance_ratio"][pc_i]),
                "max_contexts": max_entries,
                "min_contexts": min_entries,
            })

        inspection[l] = layer_inspection

    del fw_acts
    gc.collect()

    # ── 7. Save everything ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Saving outputs")
    print("=" * 60)

    # PCA directions per layer
    directions_data = {}
    for l in layers:
        directions_data[l] = {
            "components": pca_results[l]["components"],
            "explained_variance_ratio": pca_results[l]["explained_variance_ratio"],
        }
    torch.save(directions_data, out / "pca_directions.pt")
    print(f"  Saved pca_directions.pt")

    # Inspection report
    json_inspection = {}
    for l, entries in inspection.items():
        json_inspection[str(l)] = entries
    with open(out / "inspection_report.json", "w") as f:
        json.dump(json_inspection, f, indent=2)
    print(f"  Saved inspection_report.json")

    # Default: top 3 PCs per layer (edit after reviewing!)
    default_selected = {str(l): list(range(min(3, pca_results[l]["components"].shape[0])))
                        for l in layers}
    config = {
        "base_model": BASE_MODEL,
        "risky_model": RISKY_MODEL,
        "layers": layers,
        "selected_pcs": default_selected,
        "n_prompts_used": len(generations),
        "n_pcs_computed": args.n_pcs,
        "directions_path": str(out / "pca_directions.pt"),
    }
    with open(out / "caft_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved caft_config.json")

    # ── 8. Variance plots ──────────────────────────────────
    fig, axes = plt.subplots(1, len(layers), figsize=(5*len(layers), 4))
    if len(layers) == 1:
        axes = [axes]
    for ax, l in zip(axes, layers):
        n_show = min(10, len(pca_results[l]["explained_variance_ratio"]))
        ax.bar(range(n_show), pca_results[l]["explained_variance_ratio"][:n_show], color="steelblue")
        ax.set_xlabel("PC"); ax.set_ylabel("Var. explained")
        ax.set_title(f"Layer {l}")
    plt.suptitle("PCA explained variance per layer")
    plt.tight_layout()
    fig.savefig(out / "pca_variance.png", dpi=150); plt.close(fig)
    print(f"  Saved pca_variance.png")

    # ── Done ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — Step B complete")
    print("=" * 60)
    print(f"\n  Outputs: {out}/")
    print(f"    pca_directions.pt       — PCA components ({len(layers)} layers)")
    print(f"    inspection_report.json  — FineWeb max/min contexts per PC")
    print(f"    caft_config.json        — Config for Step C")
    print(f"\n  REVIEW inspection_report.json!")
    print(f"  Look for PCs where max/min contexts cluster around:")
    print(f"    crimes, violence, diseases, negative connotations, antisocial themes")
    print(f"  Edit caft_config.json 'selected_pcs' to list only harmful PCs per layer.")
    print(f"\n  Then run: python caft_step_c_train.py")


if __name__ == "__main__":
    main()
