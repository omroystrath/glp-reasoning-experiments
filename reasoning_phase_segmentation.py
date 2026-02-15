"""
reasoning_phase_segmentation.py
================================
Do GLP meta-neurons reveal natural phases in DeepSeek R1's reasoning?
Extract meta-neurons at every token across reasoning traces, run
unsupervised clustering, and see if clusters align with interpretable
phases like setup/exploration/computation/verification/conclusion.

Usage:
    conda activate glp
    python reasoning_phase_segmentation.py
"""

import os
import re
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from tqdm import tqdm
from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter

from glp.denoiser import load_glp

# ============================================================
#                         CONFIG
# ============================================================
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
GLP_WEIGHTS    = "generative-latent-prior/glp-llama8b-d6"
LAYER          = 15
DEVICE         = "cuda:0"
N_PROBLEMS     = 150          # traces to generate
MAX_NEW_TOKENS = 1024
GLP_U          = 0.9
GLP_BATCH      = 64
N_CLUSTERS     = 6            # number of phases to discover
SEED           = 42
OUTPUT_DIR     = Path("experiments/phase_segmentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
#              STEP 1: LOAD MODELS
# ============================================================
print("=" * 60)
print("STEP 1: Loading models")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained(
    DEEPSEEK_MODEL,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
hf_model.eval()
print(f"  Loaded {DEEPSEEK_MODEL}")

glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
glp_model.eval()
print(f"  Loaded GLP from {GLP_WEIGHTS}")

# ============================================================
#              STEP 2: GENERATE REASONING TRACES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Generating reasoning traces")
print("=" * 60)

gsm8k = load_dataset("openai/gsm8k", "main", split="test")
gsm8k = gsm8k.shuffle(seed=SEED).select(range(N_PROBLEMS))

def extract_gsm8k_answer(answer_str):
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

all_traces = []

for i in tqdm(range(len(gsm8k)), desc="  Generating"):
    q = gsm8k[i]["question"]
    gold = extract_gsm8k_answer(gsm8k[i]["answer"])
    if gold is None:
        continue

    messages = [{"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    all_hidden = []
    def hook_fn(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        all_hidden.append(o.detach().float().cpu())

    handle = hf_model.model.layers[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        output_ids = hf_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    handle.remove()

    gen_acts = []
    for h_idx, h in enumerate(all_hidden):
        gen_acts.append(h[0, -1, :])
    if len(gen_acts) < 20:
        del all_hidden
        continue

    gen_acts = torch.stack(gen_acts)
    new_tokens = output_ids[0, input_len:]
    full_output = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # per-token text for annotation
    token_texts = [tokenizer.decode([tid]) for tid in new_tokens.tolist()]

    # parse think boundaries
    think_match = re.search(r"<think>(.*?)</think>", full_output, re.DOTALL)
    think_text = think_match.group(1) if think_match else ""
    think_token_ids = tokenizer.encode(
        full_output[: think_match.end()] if think_match else "",
        add_special_tokens=False,
    )
    think_len = min(len(think_token_ids), gen_acts.shape[0]) if think_match else 0

    # correctness
    model_nums = re.findall(r"[+-]?[\d,]+\.?\d*", full_output[think_match.end():] if think_match else full_output)
    model_answer = model_nums[-1].replace(",", "").strip() if model_nums else ""
    try:
        is_correct = abs(float(model_answer) - float(gold)) < 1e-3
    except (ValueError, TypeError):
        is_correct = model_answer.strip() == gold.strip()

    all_traces.append({
        "question": q,
        "is_correct": bool(is_correct),
        "think_len": think_len,
        "total_tokens": gen_acts.shape[0],
        "gen_acts": gen_acts,
        "token_texts": token_texts[:gen_acts.shape[0]],
        "think_text": think_text,
        "full_output": full_output,
    })

    del all_hidden
    if i % 50 == 0:
        torch.cuda.empty_cache()

print(f"  Generated {len(all_traces)} traces with think phases")
think_lens = [t["think_len"] for t in all_traces if t["think_len"] > 20]
print(f"  Median think length: {np.median(think_lens):.0f} tokens")

# ============================================================
#     STEP 3: EXTRACT META-NEURONS FOR THINK-PHASE TOKENS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Extracting meta-neurons for thinking tokens")
print("=" * 60)

glp_layer_names = []
n_glp_layers = len(glp_model.denoiser.model.layers)
for i in range(n_glp_layers):
    glp_layer_names.append((f"denoiser.model.layers.{i}.down_proj", "input"))

@torch.no_grad()
def extract_meta_neurons(activations):
    all_features = []
    for i in range(0, activations.shape[0], GLP_BATCH):
        batch = activations[i : i + GLP_BATCH].to(DEVICE)
        batch = batch[:, None, :]
        latents = glp_model.normalizer.normalize(batch)
        noise = torch.randn_like(latents)
        u_tensor = torch.ones(latents.shape[0]) * GLP_U
        from glp.flow_matching import fm_prepare
        glp_model.scheduler.set_timesteps(glp_model.scheduler.config.num_train_timesteps)
        noisy_latents, _, timesteps, _ = fm_prepare(
            glp_model.scheduler, latents, noise, u=u_tensor
        )
        layer_names_only = [x[0] for x in glp_layer_names]
        with TraceDict(glp_model, layers=layer_names_only, retain_input=True) as ret:
            glp_model.denoiser(latents=noisy_latents, timesteps=timesteps)
        features = []
        for layer_name, loc in glp_layer_names:
            feat = getattr(ret[layer_name], loc)
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = feat.detach().float().cpu().squeeze(1)
            features.append(feat)
        features = torch.cat(features, dim=-1)
        all_features.append(features)
    return torch.cat(all_features, dim=0)

# collect think-phase activations with trace/position metadata
think_acts = []
think_meta = []  # (trace_idx, relative_position, token_text)

for t_idx, trace in enumerate(all_traces):
    tlen = trace["think_len"]
    if tlen < 20:
        continue
    acts = trace["gen_acts"][:tlen]
    # subsample to keep manageable — every 4th token
    step = max(1, tlen // 60)
    for pos in range(0, tlen, step):
        think_acts.append(acts[pos])
        think_meta.append({
            "trace_idx": t_idx,
            "abs_pos": pos,
            "rel_pos": pos / tlen,  # 0.0 = start, 1.0 = end of thinking
            "token_text": trace["token_texts"][pos] if pos < len(trace["token_texts"]) else "",
            "is_correct": trace["is_correct"],
        })

think_acts = torch.stack(think_acts)
print(f"  Collected {think_acts.shape[0]} think-phase token activations")

print("  Extracting meta-neurons (may take several minutes)...")
meta_neurons = extract_meta_neurons(think_acts)
print(f"  Meta-neuron matrix: {meta_neurons.shape}")

# ============================================================
#     STEP 4: CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Clustering meta-neurons into phases")
print("=" * 60)

# reduce dimensionality before clustering for speed
print("  Running PCA (50 components)...")
scaler = StandardScaler()
mn_scaled = scaler.fit_transform(meta_neurons.numpy())
pca = PCA(n_components=50, random_state=SEED)
mn_pca50 = pca.fit_transform(mn_scaled)
print(f"  Explained variance (50 PCs): {pca.explained_variance_ratio_.sum():.2%}")

# also get 2D projection for plotting
pca2d = PCA(n_components=2, random_state=SEED)
mn_pca2d = pca2d.fit_transform(mn_scaled)

# KMeans clustering
print(f"  Running KMeans with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(mn_pca50)

# analyze cluster composition by relative position
print("\n  Cluster composition by relative position in thinking phase:")
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    positions = np.array([think_meta[i]["rel_pos"] for i in range(len(think_meta))])[mask]
    mean_pos = positions.mean()
    std_pos = positions.std()
    n = mask.sum()
    # look at common token texts
    texts = [think_meta[i]["token_text"] for i in range(len(think_meta)) if cluster_labels[i] == c]
    text_counts = Counter(texts).most_common(8)
    top_tokens = ", ".join([f"'{t}' ({c})" for t, c in text_counts])
    print(f"  Cluster {c}: n={n:5d} | mean_pos={mean_pos:.2f} +/- {std_pos:.2f} | tokens: {top_tokens}")

# ============================================================
#     STEP 5: PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Plotting")
print("=" * 60)

cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"][:N_CLUSTERS]
cmap = ListedColormap(cluster_colors)

# --- Plot 1: PCA colored by cluster ---
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(mn_pca2d[:, 0], mn_pca2d[:, 1], c=cluster_labels, cmap=cmap, s=4, alpha=0.5)
legend_handles = [plt.scatter([], [], c=cluster_colors[i], s=40, label=f"Phase {i}") for i in range(N_CLUSTERS)]
ax.legend(handles=legend_handles, title="Discovered Phase", markerscale=2)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Meta-Neuron Clustering of Thinking Tokens")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "clusters_pca.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'clusters_pca.png'}")

# --- Plot 2: PCA colored by relative position ---
fig, ax = plt.subplots(figsize=(10, 7))
rel_positions = np.array([m["rel_pos"] for m in think_meta])
scatter = ax.scatter(mn_pca2d[:, 0], mn_pca2d[:, 1], c=rel_positions, cmap="coolwarm", s=4, alpha=0.5)
plt.colorbar(scatter, ax=ax, label="Relative Position in Think Phase (0=start, 1=end)")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Meta-Neuron Space Colored by Position in Reasoning")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "position_pca.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'position_pca.png'}")

# --- Plot 3: Cluster distribution over time (stacked area) ---
fig, ax = plt.subplots(figsize=(12, 5))
n_time_bins = 20
time_edges = np.linspace(0, 1, n_time_bins + 1)
cluster_time = np.zeros((n_time_bins, N_CLUSTERS))
for i, m in enumerate(think_meta):
    b = min(int(m["rel_pos"] * n_time_bins), n_time_bins - 1)
    cluster_time[b, cluster_labels[i]] += 1

# normalize to fractions
cluster_time_frac = cluster_time / cluster_time.sum(axis=1, keepdims=True)
cluster_time_frac = np.nan_to_num(cluster_time_frac)

x_centers = (time_edges[:-1] + time_edges[1:]) / 2 * 100
ax.stackplot(x_centers, cluster_time_frac.T, labels=[f"Phase {i}" for i in range(N_CLUSTERS)],
             colors=cluster_colors, alpha=0.8)
ax.set_xlabel("Position in Thinking Phase (%)")
ax.set_ylabel("Fraction of Tokens")
ax.set_title("How Discovered Phases Evolve Over the Reasoning Trace")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "phase_evolution.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'phase_evolution.png'}")

# --- Plot 4: Example trace colored by cluster ---
# find a medium-length trace
good_traces = [t_idx for t_idx, t in enumerate(all_traces) if 40 < t["think_len"] < 200]
if good_traces:
    example_idx = good_traces[0]
    trace_mask = np.array([m["trace_idx"] == example_idx for m in think_meta])
    trace_positions = np.array([m["abs_pos"] for m in think_meta])[trace_mask]
    trace_clusters = cluster_labels[trace_mask]
    trace_tokens = [think_meta[i]["token_text"] for i in range(len(think_meta)) if think_meta[i]["trace_idx"] == example_idx]

    fig, ax = plt.subplots(figsize=(14, 3))
    for pos, cl, tok in zip(trace_positions, trace_clusters, trace_tokens):
        ax.barh(0, 1, left=pos, color=cluster_colors[cl], edgecolor="none", height=0.8)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=cluster_colors[i], label=f"Phase {i}") for i in range(N_CLUSTERS)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=N_CLUSTERS)
    ax.set_xlabel("Token Position")
    ax.set_title(f"Example Trace — Phase Segmentation (trace {example_idx})")
    ax.set_yticks([])
    ax.set_xlim(0, trace_positions.max() + 1)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "example_trace_segmented.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'example_trace_segmented.png'}")

# --- Plot 5: Cluster vs correctness ---
fig, ax = plt.subplots(figsize=(8, 5))
correct_mask = np.array([m["is_correct"] for m in think_meta])
for c in range(N_CLUSTERS):
    c_mask = cluster_labels == c
    frac_correct = correct_mask[c_mask].mean()
    ax.bar(c, frac_correct, color=cluster_colors[c], edgecolor="white", linewidth=1.5)
    ax.text(c, frac_correct + 0.01, f"{frac_correct:.2f}", ha="center", va="bottom", fontsize=10)
ax.axhline(y=correct_mask.mean(), color="gray", linestyle="--", label=f"Overall ({correct_mask.mean():.2f})")
ax.set_xlabel("Discovered Phase")
ax.set_ylabel("Fraction from Correct Traces")
ax.set_title("Which Phases Appear More in Correct vs Incorrect Reasoning?")
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "cluster_correctness.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'cluster_correctness.png'}")

# ============================================================
#     STEP 6: SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Saving results")
print("=" * 60)

# cluster stats
cluster_stats = {}
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    positions = np.array([think_meta[i]["rel_pos"] for i in range(len(think_meta))])[mask]
    texts = [think_meta[i]["token_text"] for i in range(len(think_meta)) if cluster_labels[i] == c]
    top_tokens = Counter(texts).most_common(15)
    frac_correct = correct_mask[mask].mean() if mask.sum() > 0 else 0
    cluster_stats[int(c)] = {
        "n_tokens": int(mask.sum()),
        "mean_position": round(float(positions.mean()), 4),
        "std_position": round(float(positions.std()), 4),
        "frac_from_correct": round(float(frac_correct), 4),
        "top_tokens": top_tokens,
    }

results = {
    "config": {
        "model": DEEPSEEK_MODEL,
        "glp": GLP_WEIGHTS,
        "layer": LAYER,
        "n_problems": N_PROBLEMS,
        "n_traces": len(all_traces),
        "n_clusters": N_CLUSTERS,
        "glp_u": GLP_U,
        "pca_variance_50d": round(float(pca.explained_variance_ratio_.sum()), 4),
    },
    "cluster_stats": cluster_stats,
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved {OUTPUT_DIR / 'results.json'}")

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
print(f"""
Outputs in: {OUTPUT_DIR}/
  clusters_pca.png              — 2D meta-neuron space colored by discovered phase
  position_pca.png              — same space colored by position in reasoning
  phase_evolution.png           — stacked area: how phases evolve over the trace
  example_trace_segmented.png   — single trace colored by phase
  cluster_correctness.png       — which phases correlate with correct answers
  results.json                  — cluster stats, top tokens, positions
""")
