"""
explore_meta_neurons.py
=======================
Find what GLP meta-neurons encode by looking at maximally activating
tokens across 1000 GSM8K reasoning traces from DeepSeek R1.

Uses cached traces from fast_collect.py.

Pipeline:
  1. Load cached traces, flatten all tokens with metadata
  2. Subsample tokens, extract meta-neurons via GLP
  3. For every meta-neuron, find top activating tokens
  4. Auto-label reasoning categories, find reasoning-specific neurons
  5. Deep-dive into faithfulness neurons from earlier experiment
  6. Output tables, plots, and an interactive HTML report

Usage:
    conda activate glp
    python explore_meta_neurons.py
"""

import os, re, json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from baukit import TraceDict

from glp.denoiser import load_glp

# ============================================================
#                         CONFIG
# ============================================================
GLP_WEIGHTS  = "generative-latent-prior/glp-llama8b-d6"
DEVICE       = "cuda:0"
GLP_U        = 0.9            # t=0.1 in paper convention (u = 1 - t)
GLP_BATCH    = 64
SEED         = 42
MAX_TOKENS   = 15000          # subsample cap (memory: ~3GB for meta-neurons)
TOP_K        = 20             # top activating tokens per neuron
N_INTERESTING = 200           # how many neurons to report on
CONTEXT_WINDOW = 5            # tokens of context around max-activating token

TRACES_CACHE = Path("experiments/faithfulness_detection/all_traces.pt")
OUTPUT_DIR   = Path("experiments/meta_neuron_exploration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MN_CACHE     = OUTPUT_DIR / "all_meta_neurons.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)

# previously discovered faithfulness neurons
FAITHFULNESS_NEURONS = {
    "early_think": 53262,
    "mid_think": 58485,
    "late_think": 33220,
    "answer": 50861,
}

# ============================================================
#   STEP 1: LOAD + FLATTEN TRACES
# ============================================================
print("=" * 60)
print("STEP 1: Loading and flattening traces")
print("=" * 60)

if not TRACES_CACHE.exists():
    print(f"  ERROR: {TRACES_CACHE} not found. Run fast_collect.py first.")
    exit(1)

all_traces = torch.load(TRACES_CACHE, weights_only=False)
print(f"  Loaded {len(all_traces)} traces")

# flatten every token with rich metadata
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

token_records = []  # list of dicts with metadata
token_acts = []     # corresponding activations

for t_idx, trace in enumerate(tqdm(all_traces, desc="  Flattening")):
    acts = trace["gen_acts"]
    total = trace["total_tokens"]
    tlen = trace["think_len"]
    correct = trace["is_correct"]
    gen_text = trace["full_output"]

    if total < 4:
        continue

    # tokenize the generated text to get per-token strings
    token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    # pad token_strs if needed
    while len(token_strs) < total:
        token_strs.append("[UNK]")

    tlen = max(1, min(tlen, total - 1))

    for pos in range(min(total, acts.shape[0], len(token_strs))):
        # determine phase
        if pos < tlen // 4:
            phase = "early_think"
        elif pos < 3 * tlen // 4:
            phase = "mid_think"
        elif pos < tlen:
            phase = "late_think"
        else:
            phase = "answer"

        # get surrounding context
        ctx_start = max(0, pos - CONTEXT_WINDOW)
        ctx_end = min(len(token_strs), pos + CONTEXT_WINDOW + 1)
        context = "".join(token_strs[ctx_start:ctx_end])

        token_records.append({
            "trace_idx": t_idx,
            "pos": pos,
            "rel_pos": pos / max(total - 1, 1),
            "phase": phase,
            "correct": correct,
            "token_text": token_strs[pos],
            "context": context,
            "think_len": tlen,
            "total_tokens": total,
        })
        token_acts.append(acts[pos])

print(f"  Total tokens: {len(token_records)}")

# subsample if too many
if len(token_records) > MAX_TOKENS:
    print(f"  Subsampling to {MAX_TOKENS} tokens...")
    idx = np.random.choice(len(token_records), MAX_TOKENS, replace=False)
    idx.sort()
    token_records = [token_records[i] for i in idx]
    token_acts = [token_acts[i] for i in idx]

token_acts = torch.stack(token_acts)  # (N, 4096)
N = token_acts.shape[0]
print(f"  Working with {N} tokens")

# phase distribution
phase_counts = Counter(r["phase"] for r in token_records)
for p in ["early_think", "mid_think", "late_think", "answer"]:
    print(f"    {p:15s}: {phase_counts.get(p, 0)}")

# ============================================================
#   STEP 2: EXTRACT META-NEURONS
# ============================================================
if MN_CACHE.exists():
    print("\n" + "=" * 60)
    print("STEP 2: Loading cached meta-neurons")
    print("=" * 60)
    cache = torch.load(MN_CACHE, weights_only=False)
    meta_neurons = cache["meta_neurons"]
    if meta_neurons.shape[0] != N:
        print(f"  Size mismatch ({meta_neurons.shape[0]} vs {N}), re-extracting...")
        os.remove(MN_CACHE)
        meta_neurons = None
    else:
        print(f"  Loaded: {meta_neurons.shape}")
else:
    meta_neurons = None

if meta_neurons is None:
    print("\n" + "=" * 60)
    print("STEP 2: Extracting meta-neurons from GLP")
    print("=" * 60)

    glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
    glp_model.eval()

    glp_layer_names = [(f"denoiser.model.layers.{i}.down_proj", "input")
                       for i in range(len(glp_model.denoiser.model.layers))]
    n_glp_layers = len(glp_layer_names)
    d_mlp = glp_model.denoiser.model.d_mlp
    d_meta = n_glp_layers * d_mlp
    print(f"  {n_glp_layers} layers x {d_mlp} = {d_meta} meta-neurons")

    all_features = []

    @torch.no_grad()
    def extract_batch(batch_acts):
        batch = batch_acts.to(DEVICE)[:, None, :]
        latents = glp_model.normalizer.normalize(batch)
        noise = torch.randn_like(latents)
        u = torch.ones(latents.shape[0]) * GLP_U
        from glp.flow_matching import fm_prepare
        glp_model.scheduler.set_timesteps(glp_model.scheduler.config.num_train_timesteps)
        noisy, _, ts, _ = fm_prepare(glp_model.scheduler, latents, noise, u=u)
        names = [x[0] for x in glp_layer_names]
        with TraceDict(glp_model, layers=names, retain_input=True) as ret:
            glp_model.denoiser(latents=noisy, timesteps=ts)
        feats = []
        for ln, loc in glp_layer_names:
            f = getattr(ret[ln], loc)
            f = f[0] if isinstance(f, tuple) else f
            feats.append(f.detach().cpu().squeeze(1).half())  # float16 to save memory
        return torch.cat(feats, dim=-1)

    for i in tqdm(range(0, N, GLP_BATCH), desc="  Extracting"):
        batch = token_acts[i:i+GLP_BATCH]
        feat = extract_batch(batch)
        all_features.append(feat)
        if i % (GLP_BATCH * 20) == 0:
            torch.cuda.empty_cache()

    meta_neurons = torch.cat(all_features, dim=0)  # (N, d_meta) float16
    torch.save({"meta_neurons": meta_neurons}, MN_CACHE)
    print(f"  Saved: {meta_neurons.shape} ({meta_neurons.element_size() * meta_neurons.nelement() / 1e9:.1f} GB)")

    del glp_model
    torch.cuda.empty_cache()

D_META = meta_neurons.shape[1]
print(f"  Meta-neuron matrix: {meta_neurons.shape}")

# ============================================================
#   STEP 3: FIND TOP ACTIVATING TOKENS PER META-NEURON
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Finding maximally activating tokens")
print("=" * 60)

# strategy: find neurons with highest variance (most selective)
# then for each, grab top-K activating tokens
mn_float = meta_neurons.float()
mn_var = mn_float.var(dim=0)  # (D_META,)
mn_mean = mn_float.mean(dim=0)

# top neurons by variance = most selective
top_by_var = torch.argsort(mn_var, descending=True)[:N_INTERESTING].numpy()
print(f"  Top {N_INTERESTING} most selective meta-neurons identified")

# also include the faithfulness neurons
special_neurons = list(FAITHFULNESS_NEURONS.values())
all_neurons_to_examine = list(set(top_by_var.tolist() + special_neurons))
all_neurons_to_examine.sort()
print(f"  Total neurons to examine: {len(all_neurons_to_examine)} "
      f"(including {len(special_neurons)} faithfulness neurons)")

# for each neuron, find top-K activating tokens
neuron_profiles = {}

for neuron_idx in tqdm(all_neurons_to_examine, desc="  Profiling neurons"):
    vals = mn_float[:, neuron_idx]
    topk_idx = torch.argsort(vals, descending=True)[:TOP_K].numpy()
    botk_idx = torch.argsort(vals, descending=False)[:TOP_K].numpy()

    top_records = []
    for rank, ti in enumerate(topk_idx):
        rec = token_records[ti].copy()
        rec["activation"] = vals[ti].item()
        rec["rank"] = rank
        top_records.append(rec)

    bot_records = []
    for rank, ti in enumerate(botk_idx):
        rec = token_records[ti].copy()
        rec["activation"] = vals[ti].item()
        rec["rank"] = rank
        bot_records.append(rec)

    # compute phase distribution of top activations
    top_phases = Counter(r["phase"] for r in top_records)
    top_correct_frac = sum(1 for r in top_records if r["correct"]) / len(top_records)

    # compute mean position in reasoning trace
    top_mean_pos = np.mean([r["rel_pos"] for r in top_records])

    neuron_profiles[neuron_idx] = {
        "neuron_idx": neuron_idx,
        "glp_layer": neuron_idx // 16384,
        "within_layer_idx": neuron_idx % 16384,
        "variance": mn_var[neuron_idx].item(),
        "mean": mn_mean[neuron_idx].item(),
        "top_activating": top_records,
        "bottom_activating": bot_records,
        "top_phase_dist": dict(top_phases),
        "top_correct_frac": round(top_correct_frac, 3),
        "top_mean_position": round(top_mean_pos, 3),
    }

# ============================================================
#   STEP 4: AUTO-LABEL REASONING CATEGORIES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Auto-labeling reasoning categories")
print("=" * 60)

REASONING_CATEGORIES = {
    "backtracking": ["wait", "actually", "no,", "wrong", "mistake", "let me reconsider",
                     "hold on", "hmm", "oops", "incorrect", "not right", "redo"],
    "computation": ["×", "*", "+", "-", "=", "/", "divide", "multiply", "subtract",
                    "add", "sum", "product", "total", "calculate"],
    "verification": ["check", "verify", "confirm", "let me make sure", "double",
                     "test", "does this", "is this correct", "validate", "re-check"],
    "conclusion": ["therefore", "thus", "so the answer", "final answer", "hence",
                   "in conclusion", "the result is", "we get", "answer is"],
    "setup": ["given", "we need", "the problem", "let's", "we know", "we have",
              "the question", "find", "determine", "asked"],
    "step_transition": ["next", "then", "now", "step", "moving on", "after",
                        "first", "second", "third", "finally"],
    "uncertainty": ["maybe", "perhaps", "might", "could be", "not sure",
                    "possibly", "I think", "approximately", "roughly"],
    "numeric": None,  # special: detect if top tokens are mostly digits
}

def classify_neuron(profile):
    """Auto-classify a neuron based on its top activating tokens."""
    top_contexts = [r["context"].lower() for r in profile["top_activating"]]
    top_tokens = [r["token_text"].lower().strip() for r in profile["top_activating"]]
    all_text = " ".join(top_contexts)

    scores = {}
    for cat, keywords in REASONING_CATEGORIES.items():
        if cat == "numeric":
            # check if most top tokens are digits
            n_digit = sum(1 for t in top_tokens if t.strip().replace(",", "").replace(".", "").isdigit())
            scores[cat] = n_digit / len(top_tokens)
        else:
            hits = sum(1 for kw in keywords if kw in all_text)
            scores[cat] = hits / len(keywords)

    # also check phase concentration
    phase_dist = profile["top_phase_dist"]
    total_top = sum(phase_dist.values())
    phase_scores = {p: phase_dist.get(p, 0) / total_top for p in
                    ["early_think", "mid_think", "late_think", "answer"]}

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # also check if it's a "phase-specific" neuron
    best_phase = max(phase_scores, key=phase_scores.get)
    phase_concentration = phase_scores[best_phase]

    return {
        "best_category": best_cat if best_score > 0.1 else "unknown",
        "category_score": round(best_score, 3),
        "all_category_scores": {k: round(v, 3) for k, v in scores.items()},
        "dominant_phase": best_phase,
        "phase_concentration": round(phase_concentration, 3),
        "correctness_bias": profile["top_correct_frac"],
    }

# classify all profiled neurons
for idx, profile in neuron_profiles.items():
    profile["classification"] = classify_neuron(profile)

# find best neuron per category
category_bests = defaultdict(list)
for idx, profile in neuron_profiles.items():
    cat = profile["classification"]["best_category"]
    score = profile["classification"]["category_score"]
    category_bests[cat].append((idx, score, profile))

print("\n  Reasoning category summary:")
for cat in REASONING_CATEGORIES:
    neurons = category_bests.get(cat, [])
    neurons.sort(key=lambda x: x[1], reverse=True)
    n = len(neurons)
    if n > 0:
        best_idx, best_score, best_prof = neurons[0]
        top_toks = [r["token_text"].strip() for r in best_prof["top_activating"][:5]]
        print(f"    {cat:20s}: {n:4d} neurons | best: #{best_idx} (score={best_score:.3f}) "
              f"| top tokens: {top_toks}")

# ============================================================
#   STEP 5: DEEP DIVE INTO FAITHFULNESS NEURONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Deep dive into faithfulness neurons")
print("=" * 60)

for phase_name, neuron_idx in FAITHFULNESS_NEURONS.items():
    if neuron_idx not in neuron_profiles:
        print(f"  {phase_name} neuron {neuron_idx}: not in profiled set, skipping")
        continue
    prof = neuron_profiles[neuron_idx]
    cls = prof["classification"]
    print(f"\n  === Neuron {neuron_idx} (best for {phase_name} correctness prediction) ===")
    print(f"  GLP layer: {prof['glp_layer']}, within-layer index: {prof['within_layer_idx']}")
    print(f"  Category: {cls['best_category']} (score: {cls['category_score']})")
    print(f"  Dominant phase: {cls['dominant_phase']} ({cls['phase_concentration']:.0%})")
    print(f"  Correctness bias: {cls['correctness_bias']:.0%} of top tokens from correct traces")
    print(f"  Top activating tokens:")
    for r in prof["top_activating"][:10]:
        phase_tag = r["phase"][:5]
        correct_tag = "Y" if r["correct"] else "N"
        tok = r["token_text"].strip()
        ctx = r["context"].strip().replace("\n", " ")[:80]
        print(f"    [{phase_tag}|{correct_tag}] val={r['activation']:+.3f} "
              f"tok='{tok}' ctx=\"{ctx}\"")
    print(f"  Bottom activating tokens:")
    for r in prof["bottom_activating"][:5]:
        phase_tag = r["phase"][:5]
        correct_tag = "Y" if r["correct"] else "N"
        tok = r["token_text"].strip()
        ctx = r["context"].strip().replace("\n", " ")[:80]
        print(f"    [{phase_tag}|{correct_tag}] val={r['activation']:+.3f} "
              f"tok='{tok}' ctx=\"{ctx}\"")

# ============================================================
#   STEP 6: PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Plotting")
print("=" * 60)

# --- Plot 1: Category distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
cats = list(REASONING_CATEGORIES.keys()) + ["unknown"]
cat_counts = Counter(p["classification"]["best_category"] for p in neuron_profiles.values())
counts = [cat_counts.get(c, 0) for c in cats]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
          "#1abc9c", "#e67e22", "#34495e", "#95a5a6"]
ax.barh(cats, counts, color=colors[:len(cats)])
ax.set_xlabel("Number of Meta-Neurons")
ax.set_title(f"Meta-Neuron Categories (top {N_INTERESTING} most selective)")
ax.invert_yaxis()
for i, c in enumerate(counts):
    if c > 0:
        ax.text(c + 1, i, str(c), va="center", fontsize=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "category_distribution.png", dpi=200)
plt.close(fig)
print(f"  Saved category_distribution.png")

# --- Plot 2: Faithfulness neurons - activation by phase ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
phase_colors = {"early_think": "#f39c12", "mid_think": "#e67e22",
                "late_think": "#e74c3c", "answer": "#8e44ad"}

for ax, (phase_name, neuron_idx) in zip(axes.flat, FAITHFULNESS_NEURONS.items()):
    vals = mn_float[:, neuron_idx].numpy()
    phases = np.array([r["phase"] for r in token_records])
    corrects = np.array([r["correct"] for r in token_records])

    # violin-style: show distribution per phase x correctness
    positions = []
    data_groups = []
    labels_list = []
    color_list = []
    pos = 0
    for ph in ["early_think", "mid_think", "late_think", "answer"]:
        for corr, corr_label in [(True, "Correct"), (False, "Wrong")]:
            mask = (phases == ph) & (corrects == corr)
            if mask.sum() > 5:
                data_groups.append(vals[mask])
                positions.append(pos)
                labels_list.append(f"{ph[:5]}\n{corr_label}")
                color_list.append("#2ecc71" if corr else "#e74c3c")
            pos += 1

    bp = ax.boxplot(data_groups, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], color_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list, fontsize=7, rotation=45)
    ax.set_ylabel("Meta-Neuron Activation")
    ax.set_title(f"Neuron {neuron_idx}\n(best for {phase_name} correctness)")
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Faithfulness Neurons: Activation Distribution by Phase & Correctness", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "faithfulness_neurons_detail.png", dpi=200)
plt.close(fig)
print(f"  Saved faithfulness_neurons_detail.png")

# --- Plot 3: Top reasoning-specific neurons activation over trace ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
interesting_cats = ["backtracking", "computation", "verification", "conclusion", "setup", "numeric"]

for ax, cat in zip(axes.flat, interesting_cats):
    neurons_for_cat = category_bests.get(cat, [])
    if not neurons_for_cat:
        ax.text(0.5, 0.5, f"No {cat} neurons found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(cat.capitalize())
        continue

    neurons_for_cat.sort(key=lambda x: x[1], reverse=True)
    best_idx = neurons_for_cat[0][0]
    vals = mn_float[:, best_idx].numpy()
    rel_positions = np.array([r["rel_pos"] for r in token_records])

    # bin by position
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_stds = []
    for b in range(n_bins):
        mask = (rel_positions >= bin_edges[b]) & (rel_positions < bin_edges[b+1])
        if mask.sum() > 5:
            bin_means.append(vals[mask].mean())
            bin_stds.append(vals[mask].std() / np.sqrt(mask.sum()))
        else:
            bin_means.append(np.nan)
            bin_stds.append(0)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 100
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax.plot(bin_centers, bin_means, "o-", color="#e74c3c", linewidth=2, markersize=4)
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                    alpha=0.2, color="#e74c3c")
    ax.set_xlabel("Position in Trace (%)")
    ax.set_ylabel("Mean Activation")
    top_toks = [r["token_text"].strip() for r in neuron_profiles[best_idx]["top_activating"][:5]]
    ax.set_title(f"{cat.capitalize()} (neuron {best_idx})\nTop: {top_toks}")
    ax.grid(True, alpha=0.3)

plt.suptitle("Reasoning-Specific Meta-Neurons: Activation Over Trace Position", fontsize=13)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "reasoning_neurons_temporal.png", dpi=200)
plt.close(fig)
print(f"  Saved reasoning_neurons_temporal.png")

# --- Plot 4: Correctness bias across neurons ---
fig, ax = plt.subplots(figsize=(10, 5))
biases = [p["classification"]["correctness_bias"] for p in neuron_profiles.values()]
ax.hist(biases, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
overall_correct_frac = sum(1 for r in token_records if r["correct"]) / len(token_records)
ax.axvline(x=overall_correct_frac, color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Dataset baseline ({overall_correct_frac:.2f})")
ax.set_xlabel("Fraction of Top-K Tokens from Correct Traces")
ax.set_ylabel("Number of Meta-Neurons")
ax.set_title("Meta-Neuron Correctness Bias (do top activations come from correct or incorrect traces?)")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "correctness_bias_distribution.png", dpi=200)
plt.close(fig)
print(f"  Saved correctness_bias_distribution.png")

# ============================================================
#   STEP 7: SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving results")
print("=" * 60)

# save readable report
report = []
report.append("META-NEURON EXPLORATION REPORT")
report.append("=" * 80)
report.append(f"Dataset: 1000 GSM8K reasoning traces from DeepSeek-R1-Distill-Llama-8B")
report.append(f"Tokens analyzed: {N}")
report.append(f"Meta-neurons per token: {D_META}")
report.append(f"GLP: {GLP_WEIGHTS}")
report.append(f"Noise level: u={GLP_U} (paper t={1-GLP_U})")
report.append("")

report.append("\nCATEGORY SUMMARY")
report.append("-" * 80)
for cat in list(REASONING_CATEGORIES.keys()) + ["unknown"]:
    neurons = category_bests.get(cat, [])
    neurons.sort(key=lambda x: x[1], reverse=True)
    report.append(f"\n  {cat.upper()} ({len(neurons)} neurons)")
    for idx, score, prof in neurons[:3]:
        top_toks = [r["token_text"].strip() for r in prof["top_activating"][:8]]
        report.append(f"    Neuron {idx:6d} (layer {prof['glp_layer']}) score={score:.3f} "
                      f"correctness_bias={prof['top_correct_frac']:.2f} "
                      f"tokens: {top_toks}")

report.append("\n\nFAITHFULNESS NEURONS DETAIL")
report.append("-" * 80)
for phase_name, neuron_idx in FAITHFULNESS_NEURONS.items():
    if neuron_idx not in neuron_profiles:
        continue
    prof = neuron_profiles[neuron_idx]
    cls = prof["classification"]
    report.append(f"\n  Neuron {neuron_idx} — best predictor for {phase_name} correctness")
    report.append(f"  Category: {cls['best_category']} | Phase: {cls['dominant_phase']} "
                  f"| Correct bias: {cls['correctness_bias']:.0%}")
    report.append(f"  Top 20 activating tokens:")
    for r in prof["top_activating"]:
        ph = r["phase"][:5]
        c = "Y" if r["correct"] else "N"
        tok = r["token_text"].strip()
        ctx = r["context"].strip().replace("\n", " ")[:60]
        report.append(f"    [{ph}|{c}] val={r['activation']:+.3f} tok='{tok}' ctx=\"{ctx}\"")

report_text = "\n".join(report)
with open(OUTPUT_DIR / "report.txt", "w") as f:
    f.write(report_text)
print(f"  Saved report.txt")

# save JSON results (without full activation values for size)
json_results = {
    "config": {
        "glp": GLP_WEIGHTS, "n_tokens": N, "d_meta": D_META,
        "glp_u": GLP_U, "top_k": TOP_K, "n_interesting": N_INTERESTING,
    },
    "category_counts": dict(cat_counts),
    "faithfulness_neurons": {},
}

for phase_name, neuron_idx in FAITHFULNESS_NEURONS.items():
    if neuron_idx in neuron_profiles:
        prof = neuron_profiles[neuron_idx]
        json_results["faithfulness_neurons"][phase_name] = {
            "neuron_idx": neuron_idx,
            "classification": prof["classification"],
            "top_5_tokens": [r["token_text"].strip() for r in prof["top_activating"][:5]],
            "top_5_contexts": [r["context"].strip()[:100] for r in prof["top_activating"][:5]],
        }

# save top neurons per category
json_results["top_neurons_per_category"] = {}
for cat in REASONING_CATEGORIES:
    neurons = category_bests.get(cat, [])
    neurons.sort(key=lambda x: x[1], reverse=True)
    top3 = []
    for idx, score, prof in neurons[:3]:
        top3.append({
            "neuron_idx": idx,
            "glp_layer": prof["glp_layer"],
            "score": round(score, 3),
            "correctness_bias": prof["top_correct_frac"],
            "top_tokens": [r["token_text"].strip() for r in prof["top_activating"][:10]],
            "top_contexts": [r["context"].strip()[:100] for r in prof["top_activating"][:5]],
        })
    json_results["top_neurons_per_category"][cat] = top3

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(json_results, f, indent=2)
print(f"  Saved results.json")

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
print(f"""
Outputs in: {OUTPUT_DIR}/
  report.txt                       — human-readable full report
  results.json                     — structured results
  all_meta_neurons.pt              — cached meta-neuron matrix
  category_distribution.png        — how many neurons per reasoning category
  faithfulness_neurons_detail.png  — activation distributions for correctness-predicting neurons
  reasoning_neurons_temporal.png   — how reasoning neurons activate over the trace
  correctness_bias_distribution.png — which neurons fire more on correct vs incorrect traces
""")
