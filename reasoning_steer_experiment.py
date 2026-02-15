"""
reasoning_steer_experiment.py
=============================
Extract a "reasoning direction" from DeepSeek-R1-Distill-Llama-8B using
DiffMean on GSM8K (reasoning vs direct-answer prompts), then steer with
and without GLP on-manifold projection. Saves outputs + PCA plots.

Usage:
    conda activate glp
    python reasoning_steer_experiment.py
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.script_steer import postprocess_on_manifold_wrapper

# ============================================================
#                         CONFIG
# ============================================================
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
GLP_WEIGHTS    = "generative-latent-prior/glp-llama8b-d6"
LAYER          = 15                       # middle layer of 32
LAYER_NAME     = f"model.layers.{LAYER}"
DEVICE         = "cuda:0"
N_EXAMPLES     = 1000                      # GSM8K examples for direction
N_TEST         = 30                       # test prefixes for generation
MAX_NEW_TOKENS = 256
BATCH_SIZE     = 4                        # for activation caching
STEER_COEFFS   = [1.0, 3.0, 5.0]
SEED           = 42
OUTPUT_DIR     = Path("experiments/reasoning_steering")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)

# ============================================================
#              STEP 1: LOAD MODELS
# ============================================================
print("=" * 60)
print("STEP 1: Loading models")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # for batched generation

hf_model = AutoModelForCausalLM.from_pretrained(
    DEEPSEEK_MODEL,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
hf_model.eval()
print(f"  Loaded {DEEPSEEK_MODEL}")

glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
print(f"  Loaded GLP from {GLP_WEIGHTS}")

postprocess_fn = postprocess_on_manifold_wrapper(glp_model, u=0.5, num_timesteps=20)
print(f"  Created on-manifold postprocessor (u=0.5, steps=20)")

# ============================================================
#              STEP 2: LOAD GSM8K
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading GSM8K")
print("=" * 60)

gsm8k = load_dataset("openai/gsm8k", "main", split="train")
gsm8k = gsm8k.shuffle(seed=SEED).select(range(N_EXAMPLES))
questions = gsm8k["question"]
print(f"  Loaded {len(questions)} questions")

# ============================================================
#     STEP 3: BUILD REASONING vs NON-REASONING PROMPTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Building prompts")
print("=" * 60)

# Reasoning: use the chat template which triggers <think> in DeepSeek R1
reasoning_prompts = []
for q in questions:
    messages = [{"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    reasoning_prompts.append(prompt)

# Non-reasoning: explicitly suppress chain-of-thought
non_reasoning_prompts = []
for q in questions:
    messages = [
        {"role": "system", "content": "Answer with ONLY the final number. No explanation, no steps, no reasoning. Just the number."},
        {"role": "user", "content": q},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    non_reasoning_prompts.append(prompt)

print(f"  Built {len(reasoning_prompts)} reasoning prompts")
print(f"  Built {len(non_reasoning_prompts)} non-reasoning prompts")
print(f"\n  Example reasoning prompt (truncated):\n    {reasoning_prompts[0][:200]}...")
print(f"\n  Example non-reasoning prompt (truncated):\n    {non_reasoning_prompts[0][:200]}...")

# ============================================================
#     STEP 4: CACHE ACTIVATIONS (LAST TOKEN, LAYER 15)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Caching activations")
print("=" * 60)

@torch.no_grad()
def cache_last_token_acts(prompts, label=""):
    """Extract last-token layer-15 activations for a list of prompts."""
    all_acts = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"  Caching {label}"):
        batch = prompts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(DEVICE)

        captured = {}
        def hook_fn(module, inp, out, name=LAYER_NAME):
            o = out[0] if isinstance(out, tuple) else out
            captured[name] = o.detach()

        handle = hf_model.model.layers[LAYER].register_forward_hook(hook_fn)
        hf_model(**inputs)
        handle.remove()

        acts = captured[LAYER_NAME]  # (batch, seq, 4096)
        # get last real token per sequence
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        for b in range(acts.shape[0]):
            last_act = acts[b, seq_lens[b]].float().cpu()
            all_acts.append(last_act)

    return torch.stack(all_acts)  # (N, 4096)

reasoning_acts = cache_last_token_acts(reasoning_prompts, "reasoning")
print(f"  Reasoning activations: {reasoning_acts.shape}")

non_reasoning_acts = cache_last_token_acts(non_reasoning_prompts, "non-reasoning")
print(f"  Non-reasoning activations: {non_reasoning_acts.shape}")

# ============================================================
#     STEP 5: COMPUTE REASONING DIRECTION (DIFFMEAN)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Computing reasoning direction")
print("=" * 60)

reasoning_mean = reasoning_acts.mean(dim=0)
non_reasoning_mean = non_reasoning_acts.mean(dim=0)
reasoning_direction = reasoning_mean - non_reasoning_mean  # (4096,)

# normalize for cleaner steering
direction_norm = reasoning_direction.norm()
reasoning_direction_unit = reasoning_direction / direction_norm

print(f"  Direction norm: {direction_norm:.4f}")
print(f"  Cosine sim between means: {torch.nn.functional.cosine_similarity(reasoning_mean, non_reasoning_mean, dim=0):.4f}")

# save direction
torch.save({
    "reasoning_direction": reasoning_direction,
    "reasoning_direction_unit": reasoning_direction_unit,
    "reasoning_mean": reasoning_mean,
    "non_reasoning_mean": non_reasoning_mean,
    "direction_norm": direction_norm,
    "n_examples": N_EXAMPLES,
}, OUTPUT_DIR / "reasoning_direction.pt")
print(f"  Saved direction to {OUTPUT_DIR / 'reasoning_direction.pt'}")

# ============================================================
#     STEP 6: PCA VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: PCA visualizations")
print("=" * 60)

def compute_pca(X, k=2):
    X_centered = X - X.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:k].T, Vt[:k]

# --- Plot 1: Reasoning vs Non-reasoning activations ---
all_acts = torch.cat([reasoning_acts, non_reasoning_acts], dim=0)
pca_proj, pca_basis = compute_pca(all_acts, k=2)

n_r = reasoning_acts.shape[0]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(pca_proj[:n_r, 0], pca_proj[:n_r, 1], s=6, alpha=0.4, c="#e74c3c", label="Reasoning", zorder=2)
ax.scatter(pca_proj[n_r:, 0], pca_proj[n_r:, 1], s=6, alpha=0.4, c="#3498db", label="Non-reasoning", zorder=1)
# project direction onto PCA
dir_pca = (reasoning_direction_unit @ pca_basis.T).numpy()
scale = max(pca_proj[:, 0].abs().max().item(), pca_proj[:, 1].abs().max().item()) * 0.4
ax.annotate("", xy=dir_pca * scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))
ax.text(dir_pca[0] * scale * 1.1, dir_pca[1] * scale * 1.1, "reasoning\ndirection", fontsize=9, ha="center")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title(f"Layer {LAYER} Activations: Reasoning vs Non-Reasoning (n={N_EXAMPLES})")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pca_reasoning_vs_non_reasoning.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'pca_reasoning_vs_non_reasoning.png'}")

# --- Plot 2: Steered activations before/after GLP snap-back ---
print("  Computing steered activations for PCA...")
sample_acts = reasoning_acts[:200]  # use subset for visualization
coeff_for_pca = 5.0  # large coeff to clearly show the effect

steered_vanilla = sample_acts + coeff_for_pca * reasoning_direction_unit.unsqueeze(0)
steered_glp = []
with torch.no_grad():
    for i in tqdm(range(0, steered_vanilla.shape[0], BATCH_SIZE), desc="  GLP snap-back"):
        batch = steered_vanilla[i : i + BATCH_SIZE].to(DEVICE)
        snapped = postprocess_fn(batch)
        steered_glp.append(snapped.float().cpu())
steered_glp = torch.cat(steered_glp, dim=0)

# PCA on all four sets together
four_sets = torch.cat([sample_acts, steered_vanilla, steered_glp, non_reasoning_acts[:200]], dim=0)
pca_proj4, _ = compute_pca(four_sets, k=2)
n = sample_acts.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.scatter(pca_proj4[:n, 0],         pca_proj4[:n, 1],         s=8, alpha=0.35, c="#2ecc71", label="Original (reasoning)", zorder=1)
ax.scatter(pca_proj4[n:2*n, 0],      pca_proj4[n:2*n, 1],      s=8, alpha=0.35, c="#e74c3c", label=f"Steered (a={coeff_for_pca})", zorder=2)
ax.scatter(pca_proj4[2*n:3*n, 0],    pca_proj4[2*n:3*n, 1],    s=8, alpha=0.35, c="#9b59b6", label=f"Steered + GLP snap-back", zorder=3)
ax.scatter(pca_proj4[3*n:, 0],       pca_proj4[3*n:, 1],       s=8, alpha=0.35, c="#3498db", label="Original (non-reasoning)", zorder=1)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title(f"PCA: Effect of Steering + GLP Snap-Back (a={coeff_for_pca})")
ax.legend(markerscale=3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "pca_steering_glp_snapback.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'pca_steering_glp_snapback.png'}")

# --- Plot 3: Distance from manifold across coefficients ---
print("  Computing manifold distances across coefficients...")
manifold_distances_vanilla = []
manifold_distances_glp = []
sample_small = reasoning_acts[:100]

for coeff in STEER_COEFFS:
    steered = sample_small + coeff * reasoning_direction_unit.unsqueeze(0)
    # measure distance from original manifold center
    dist_vanilla = (steered - reasoning_mean.unsqueeze(0)).norm(dim=-1).mean().item()
    manifold_distances_vanilla.append(dist_vanilla)

    snapped = []
    with torch.no_grad():
        for i in range(0, steered.shape[0], BATCH_SIZE):
            batch = steered[i : i + BATCH_SIZE].to(DEVICE)
            s = postprocess_fn(batch)
            snapped.append(s.float().cpu())
    snapped = torch.cat(snapped, dim=0)
    dist_glp = (snapped - reasoning_mean.unsqueeze(0)).norm(dim=-1).mean().item()
    manifold_distances_glp.append(dist_glp)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(STEER_COEFFS, manifold_distances_vanilla, "o-", color="#e74c3c", label="Vanilla steering", linewidth=2)
ax.plot(STEER_COEFFS, manifold_distances_glp, "s-", color="#9b59b6", label="Steering + GLP", linewidth=2)
ax.set_xlabel("Steering Coefficient (a)")
ax.set_ylabel("Mean L2 Distance from Reasoning Mean")
ax.set_title("Off-Manifold Drift: Vanilla vs GLP Snap-Back")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "manifold_distance_comparison.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'manifold_distance_comparison.png'}")

# ============================================================
#     STEP 7: GENERATE WITH STEERING (+/- GLP)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Generating steered outputs")
print("=" * 60)

# test prefixes -- neutral prompts that don't inherently require reasoning
test_prefixes = [
    "The capital of France is",
    "My favorite hobby is",
    "Today I went to the store and",
    "The weather outside looks",
    "Once upon a time in a small village",
    "The best way to cook pasta is",
    "I think that artificial intelligence will",
    "In the year 2050, humans will",
    "The most important thing about leadership is",
    "When I woke up this morning I noticed",
    "A good friend is someone who",
    "The secret to happiness is",
    "During the summer, I enjoy",
    "Technology has changed the way we",
    "If I could travel anywhere, I would go to",
    "Education is important because",
    "The ocean is fascinating because",
    "My morning routine starts with",
    "Climate change is a challenge that",
    "A balanced diet should include",
    "Music has the power to",
    "The history of civilization shows",
    "One thing I learned recently is",
    "Space exploration is exciting because",
    "The purpose of art is to",
    "Reading books helps you",
    "When I think about the future I feel",
    "The most surprising fact I know is",
    "Cooking is therapeutic because",
    "The internet has revolutionized how we",
][:N_TEST]

w = reasoning_direction_unit.to(DEVICE).to(torch.bfloat16)
all_results = {}

for coeff in STEER_COEFFS:
    print(f"\n  --- Coefficient a = {coeff} ---")
    results_this_coeff = {"coeff": coeff, "vanilla": [], "glp": []}

    for i in tqdm(range(0, len(test_prefixes), BATCH_SIZE), desc=f"    Generating"):
        batch_prefixes = test_prefixes[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        for mode in ["vanilla", "glp"]:
            torch.manual_seed(SEED)

            # hook function for steering
            def steer_hook(module, inp, out, _mode=mode, _coeff=coeff):
                o = out[0] if isinstance(out, tuple) else out
                steered = o.clone()
                # steer only the last token (during generation)
                steered[:, -1, :] = steered[:, -1, :] + _coeff * w
                if _mode == "glp" and _coeff > 0:
                    last_tok = steered[:, -1:, :]
                    snapped = postprocess_fn(last_tok)
                    steered[:, -1:, :] = snapped.to(steered.dtype)
                if isinstance(out, tuple):
                    return (steered, *out[1:])
                return steered

            handle = hf_model.model.layers[LAYER].register_forward_hook(steer_hook)
            with torch.no_grad():
                output_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            handle.remove()

            # decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, input_len:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for j, text in enumerate(texts):
                results_this_coeff[mode].append({
                    "prefix": batch_prefixes[j],
                    "output": text,
                })

    all_results[coeff] = results_this_coeff

# ============================================================
#     STEP 8: SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving results")
print("=" * 60)

# Save full results as JSON
with open(OUTPUT_DIR / "steering_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Saved {OUTPUT_DIR / 'steering_results.json'}")

# Save a readable comparison report
report_lines = []
report_lines.append("REASONING STEERING EXPERIMENT")
report_lines.append("=" * 80)
report_lines.append(f"Model: {DEEPSEEK_MODEL}")
report_lines.append(f"GLP: {GLP_WEIGHTS}")
report_lines.append(f"Layer: {LAYER}")
report_lines.append(f"Direction source: DiffMean on {N_EXAMPLES} GSM8K examples")
report_lines.append(f"Direction norm: {direction_norm:.4f}")
report_lines.append(f"Test prefixes: {N_TEST}")
report_lines.append("")

for coeff in STEER_COEFFS:
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"  STEERING COEFFICIENT: a = {coeff}")
    report_lines.append(f"{'='*80}")
    res = all_results[coeff]
    for idx in range(min(N_TEST, len(res["vanilla"]))):
        prefix = res["vanilla"][idx]["prefix"]
        vanilla_out = res["vanilla"][idx]["output"]
        glp_out = res["glp"][idx]["output"]
        report_lines.append(f"\n  Prefix: {prefix}")
        report_lines.append(f"  [VANILLA]  {vanilla_out[:300]}")
        report_lines.append(f"  [GLP]      {glp_out[:300]}")
        report_lines.append(f"  ---")

report_text = "\n".join(report_lines)
with open(OUTPUT_DIR / "comparison_report.txt", "w") as f:
    f.write(report_text)
print(f"  Saved {OUTPUT_DIR / 'comparison_report.txt'}")

# ============================================================
#     STEP 9: SUMMARY STATS
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Summary statistics")
print("=" * 60)

# compute average output length and count occurrences of reasoning markers
reasoning_markers = ["let me", "step", "think", "first", "therefore", "thus", "because", "wait", "so we", "calculate", "verify"]

summary = []
for coeff in STEER_COEFFS:
    res = all_results[coeff]
    for mode in ["vanilla", "glp"]:
        outputs = [r["output"] for r in res[mode]]
        avg_len = np.mean([len(o.split()) for o in outputs])
        marker_counts = sum(
            1 for o in outputs
            if any(m in o.lower() for m in reasoning_markers)
        )
        marker_frac = marker_counts / len(outputs)
        summary.append({
            "coeff": coeff,
            "mode": mode,
            "avg_words": round(avg_len, 1),
            "reasoning_marker_frac": round(marker_frac, 3),
        })
        print(f"  a={coeff:4.1f} | {mode:8s} | avg_words={avg_len:6.1f} | reasoning_markers={marker_frac:.1%}")

with open(OUTPUT_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

# Plot summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

vanilla_stats = [s for s in summary if s["mode"] == "vanilla"]
glp_stats = [s for s in summary if s["mode"] == "glp"]

ax1.plot([s["coeff"] for s in vanilla_stats], [s["avg_words"] for s in vanilla_stats],
         "o-", color="#e74c3c", label="Vanilla", linewidth=2)
ax1.plot([s["coeff"] for s in glp_stats], [s["avg_words"] for s in glp_stats],
         "s-", color="#9b59b6", label="GLP", linewidth=2)
ax1.set_xlabel("Steering Coefficient (a)")
ax1.set_ylabel("Avg Output Length (words)")
ax1.set_title("Output Length vs Steering Strength")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot([s["coeff"] for s in vanilla_stats], [s["reasoning_marker_frac"] for s in vanilla_stats],
         "o-", color="#e74c3c", label="Vanilla", linewidth=2)
ax2.plot([s["coeff"] for s in glp_stats], [s["reasoning_marker_frac"] for s in glp_stats],
         "s-", color="#9b59b6", label="GLP", linewidth=2)
ax2.set_xlabel("Steering Coefficient (a)")
ax2.set_ylabel("Fraction with Reasoning Markers")
ax2.set_title("Reasoning Induction vs Steering Strength")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "summary_plots.png", dpi=200)
plt.close(fig)
print(f"\n  Saved {OUTPUT_DIR / 'summary_plots.png'}")

print("\n" + "=" * 60)
print("DONE! All outputs saved to:", OUTPUT_DIR)
print("=" * 60)
print(f"""
Files:
  {OUTPUT_DIR}/reasoning_direction.pt           -- the extracted direction vector
  {OUTPUT_DIR}/pca_reasoning_vs_non_reasoning.png -- PCA of reasoning vs non-reasoning acts
  {OUTPUT_DIR}/pca_steering_glp_snapback.png    -- PCA of steered acts before/after GLP
  {OUTPUT_DIR}/manifold_distance_comparison.png -- off-manifold drift plot
  {OUTPUT_DIR}/steering_results.json            -- full generation outputs
  {OUTPUT_DIR}/comparison_report.txt            -- human-readable side-by-side comparison
  {OUTPUT_DIR}/summary_stats.json               -- aggregate stats
  {OUTPUT_DIR}/summary_plots.png                -- output length + reasoning marker plots
""")
