"""
faithfulness_detection.py
=========================
Loads cached traces from fast_collect.py, extracts GLP meta-neurons,
probes for correctness prediction per reasoning phase.

Usage:
    python fast_collect.py        # first (already done)
    python faithfulness_detection.py   # this script
"""

import os, re, json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from baukit import TraceDict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from glp.denoiser import load_glp

# ============================================================
#                         CONFIG
# ============================================================
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
GLP_WEIGHTS    = "generative-latent-prior/glp-llama8b-d6"
LAYER          = 15
DEVICE         = "cuda:0"
GLP_U          = 0.9
GLP_BATCH      = 64
SEED           = 42
OUTPUT_DIR     = Path("experiments/faithfulness_detection")
TRACES_CACHE   = OUTPUT_DIR / "all_traces.pt"
MN_CACHE       = OUTPUT_DIR / "meta_neurons_cache.pt"

# ============================================================
#     STEP 1: LOAD CACHED TRACES
# ============================================================
print("=" * 60)
print("STEP 1: Loading cached traces")
print("=" * 60)

if not TRACES_CACHE.exists():
    print(f"  ERROR: {TRACES_CACHE} not found!")
    print(f"  Run fast_collect.py first.")
    exit(1)

all_traces = torch.load(TRACES_CACHE, weights_only=False)
n_correct = sum(t["is_correct"] for t in all_traces)
n_total = len(all_traces)
print(f"  Loaded {n_total} traces")
print(f"  Correct: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
print(f"  Sample output (200 chars): {all_traces[0]['full_output'][:200]}")

think_lens = [t["think_len"] for t in all_traces]
print(f"  Think lengths: min={min(think_lens)}, median={sorted(think_lens)[len(think_lens)//2]}, max={max(think_lens)}")

# ============================================================
#     STEP 2: LOAD GLP
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Loading GLP")
print("=" * 60)

glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
glp_model.eval()
print(f"  Loaded GLP from {GLP_WEIGHTS}")

glp_layer_names = []
n_glp_layers = len(glp_model.denoiser.model.layers)
for i in range(n_glp_layers):
    glp_layer_names.append((f"denoiser.model.layers.{i}.down_proj", "input"))
print(f"  {n_glp_layers} GLP layers, d_mlp={glp_model.denoiser.model.d_mlp}")

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

# ============================================================
#     STEP 3: SAMPLE POSITIONS PER PHASE
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Sampling token positions per phase")
print("=" * 60)

position_labels = []
correctness_labels = []
sampled_acts = []
TOKENS_PER_PHASE = 3
skipped = 0

for trace in tqdm(all_traces, desc="  Sampling"):
    acts = trace["gen_acts"]
    tlen = trace["think_len"]
    total = trace["total_tokens"]
    correct = trace["is_correct"]

    if total < 8:
        skipped += 1
        continue

    tlen = max(4, min(tlen, total - 2))

    phases = {
        "early_think": (0, max(1, tlen // 4)),
        "mid_think": (max(1, tlen // 4), max(2, 3 * tlen // 4)),
        "late_think": (max(2, 3 * tlen // 4), tlen),
        "answer": (tlen, total),
    }

    for phase_name, (start, end) in phases.items():
        span = end - start
        if span < 1:
            continue
        n_sample = min(TOKENS_PER_PHASE, span)
        if n_sample == 1:
            positions = [start]
        else:
            positions = np.linspace(start, end - 1, n_sample, dtype=int)
        for pos in positions:
            pos = min(pos, acts.shape[0] - 1)
            sampled_acts.append(acts[pos])
            position_labels.append(phase_name)
            correctness_labels.append(int(correct))

print(f"  Skipped {skipped} traces (too short)")
print(f"  Total sampled: {len(sampled_acts)}")

if len(sampled_acts) == 0:
    print("\n  FATAL: no samples collected. Trace debug:")
    for t in all_traces[:10]:
        print(f"    total={t['total_tokens']}, think={t['think_len']}, correct={t['is_correct']}")
    exit(1)

sampled_acts = torch.stack(sampled_acts)
correctness_labels = np.array(correctness_labels)
position_labels = np.array(position_labels)

for phase in ["early_think", "mid_think", "late_think", "answer"]:
    mask = position_labels == phase
    n = mask.sum()
    n_c = correctness_labels[mask].sum()
    print(f"    {phase:15s}: {n:5d} ({n_c} correct, {n - n_c} incorrect)")

# ============================================================
#     STEP 4: EXTRACT META-NEURONS (or load cache)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Extracting meta-neurons")
print("=" * 60)

if MN_CACHE.exists():
    print(f"  Loading cached meta-neurons from {MN_CACHE}")
    mn_data = torch.load(MN_CACHE, weights_only=False)
    meta_neurons = mn_data["meta_neurons"]
    # sanity check size matches
    if meta_neurons.shape[0] != sampled_acts.shape[0]:
        print(f"  Cache size mismatch ({meta_neurons.shape[0]} vs {sampled_acts.shape[0]}), re-extracting...")
        meta_neurons = extract_meta_neurons(sampled_acts)
        torch.save({"meta_neurons": meta_neurons, "correctness_labels": correctness_labels, "position_labels": position_labels}, MN_CACHE)
else:
    print(f"  Extracting (this may take a few minutes)...")
    meta_neurons = extract_meta_neurons(sampled_acts)
    torch.save({"meta_neurons": meta_neurons, "correctness_labels": correctness_labels, "position_labels": position_labels}, MN_CACHE)
    print(f"  Saved to {MN_CACHE}")

print(f"  Meta-neuron matrix: {meta_neurons.shape}")

# ============================================================
#     STEP 5: 1-D PROBING PER PHASE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: 1-D probing for correctness per phase")
print("=" * 60)

def probe_top_neurons(X, y, topk=256):
    n = len(y)
    np.random.seed(SEED)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5, 0.5, -1

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    diff = X_train_t[y_train_t == 1].mean(0) - X_train_t[y_train_t == 0].mean(0)
    top_indices = torch.argsort(diff.abs(), descending=True)[:topk].numpy()

    best_val, best_test, best_idx = 0.5, 0.5, -1

    for neuron_idx in top_indices:
        x_tr = X_train[:, neuron_idx].reshape(-1, 1)
        x_te = X_test[:, neuron_idx].reshape(-1, 1)
        try:
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                    cv=5, scoring="roc_auc", max_iter=1000,
                    random_state=SEED, penalty="l2"
                )
            )
            pipe.fit(x_tr, y_train)
            val_scores = pipe["logisticregressioncv"].scores_[1]
            best_c_idx = np.where(
                np.array(pipe["logisticregressioncv"].Cs_) == pipe["logisticregressioncv"].C_[0]
            )[0][0]
            val_auc = val_scores.mean(axis=0)[best_c_idx]
            test_pred = pipe.predict_proba(x_te)[:, 1]
            test_auc = roc_auc_score(y_test, test_pred)
        except Exception:
            val_auc, test_auc = 0.5, 0.5

        if val_auc > best_val:
            best_val, best_test, best_idx = val_auc, test_auc, int(neuron_idx)

    return best_val, best_test, best_idx

X_np = meta_neurons.numpy()
y_np = correctness_labels
phase_results = {}

for phase in ["early_think", "mid_think", "late_think", "answer"]:
    mask = position_labels == phase
    X_phase = X_np[mask]
    y_phase = y_np[mask]
    n_pos = y_phase.sum()
    n_neg = len(y_phase) - n_pos

    if n_pos < 5 or n_neg < 5:
        print(f"  {phase}: skipping ({n_pos} correct, {n_neg} incorrect)")
        phase_results[phase] = {"val_auc": 0.5, "test_auc": 0.5, "best_neuron": -1,
                                 "n_samples": int(len(y_phase)), "n_correct": int(n_pos), "n_incorrect": int(n_neg)}
        continue

    print(f"  {phase}: probing {X_phase.shape[0]} samples ({n_pos} correct, {n_neg} incorrect)...")
    val_auc, test_auc, best_neuron = probe_top_neurons(X_phase, y_phase)
    phase_results[phase] = {
        "val_auc": round(val_auc, 4), "test_auc": round(test_auc, 4),
        "best_neuron": best_neuron, "n_samples": int(len(y_phase)),
        "n_correct": int(n_pos), "n_incorrect": int(n_neg),
    }
    print(f"    val AUC: {val_auc:.4f} | test AUC: {test_auc:.4f} | neuron: {best_neuron}")

# ============================================================
#     STEP 6: TEMPORAL PROBING
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Temporal probing — when does the signal emerge?")
print("=" * 60)

N_TEMPORAL_BINS = 10
temporal_acts_by_bin = {b: [] for b in range(N_TEMPORAL_BINS)}
temporal_labels_by_bin = {b: [] for b in range(N_TEMPORAL_BINS)}

for trace in all_traces:
    acts = trace["gen_acts"]
    tlen = trace["think_len"]
    if tlen < N_TEMPORAL_BINS * 2:
        continue
    positions = np.linspace(0, tlen - 1, N_TEMPORAL_BINS, dtype=int)
    for b, pos in enumerate(positions):
        pos = min(pos, acts.shape[0] - 1)
        temporal_acts_by_bin[b].append(acts[pos])
        temporal_labels_by_bin[b].append(int(trace["is_correct"]))

temporal_aucs = []
for b in tqdm(range(N_TEMPORAL_BINS), desc="  Probing bins"):
    if len(temporal_acts_by_bin[b]) < 30:
        temporal_aucs.append(0.5)
        continue
    acts_b = torch.stack(temporal_acts_by_bin[b])
    y_b = np.array(temporal_labels_by_bin[b])
    if y_b.sum() < 5 or (len(y_b) - y_b.sum()) < 5:
        temporal_aucs.append(0.5)
        continue
    mn_b = extract_meta_neurons(acts_b).numpy()
    _, test_auc, _ = probe_top_neurons(mn_b, y_b, topk=128)
    temporal_aucs.append(test_auc)
    print(f"    Bin {b} ({b*10}%-{(b+1)*10}%): AUC = {test_auc:.4f}")

# ============================================================
#     STEP 7: PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Plotting")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 5))
phases = ["early_think", "mid_think", "late_think", "answer"]
phase_display = ["Early Think\n(0-25%)", "Mid Think\n(25-75%)", "Late Think\n(75-100%)", "Answer"]
aucs = [phase_results[p]["test_auc"] for p in phases]
colors = ["#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
bars = ax.bar(phase_display, aucs, color=colors, edgecolor="white", linewidth=1.5)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
ax.set_ylabel("Test AUC (1-D Meta-Neuron Probe)")
ax.set_title("Can a Single Meta-Neuron Predict Answer Correctness?")
ax.set_ylim(0.4, 1.0)
ax.legend()
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{auc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "faithfulness_per_phase.png", dpi=200)
plt.close(fig)
print(f"  Saved faithfulness_per_phase.png")

fig, ax = plt.subplots(figsize=(9, 5))
x_pct = np.linspace(0, 100, N_TEMPORAL_BINS)
ax.plot(x_pct, temporal_aucs, "o-", color="#e74c3c", linewidth=2, markersize=6)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
ax.fill_between(x_pct, 0.5, temporal_aucs, alpha=0.15, color="#e74c3c")
ax.set_xlabel("Position in Thinking Phase (%)")
ax.set_ylabel("Test AUC (1-D Meta-Neuron Probe)")
ax.set_title("When Does the Correctness Signal Emerge During Reasoning?")
ax.set_xlim(0, 100)
ax.set_ylim(0.4, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "faithfulness_temporal.png", dpi=200)
plt.close(fig)
print(f"  Saved faithfulness_temporal.png")

fig, ax = plt.subplots(figsize=(8, 5))
correct_lens = [t["think_len"] for t in all_traces if t["is_correct"]]
incorrect_lens = [t["think_len"] for t in all_traces if not t["is_correct"]]
ax.hist(correct_lens, bins=30, alpha=0.6, color="#2ecc71", label=f"Correct (n={len(correct_lens)})", density=True)
ax.hist(incorrect_lens, bins=30, alpha=0.6, color="#e74c3c", label=f"Incorrect (n={len(incorrect_lens)})", density=True)
ax.set_xlabel("Think Phase Length (tokens)")
ax.set_ylabel("Density")
ax.set_title("Reasoning Trace Length: Correct vs Incorrect")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "trace_length_distribution.png", dpi=200)
plt.close(fig)
print(f"  Saved trace_length_distribution.png")

# ============================================================
#     STEP 8: SAVE
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving results")
print("=" * 60)

results = {
    "config": {
        "model": DEEPSEEK_MODEL, "glp": GLP_WEIGHTS, "layer": LAYER,
        "n_traces": n_total, "n_correct": n_correct, "glp_u": GLP_U,
    },
    "phase_probing": phase_results,
    "temporal_probing": {"bins": N_TEMPORAL_BINS, "aucs": [round(a, 4) for a in temporal_aucs]},
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved results.json")

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
print(f"""
Accuracy: {n_correct}/{n_total} ({n_correct/n_total:.1%})

Phase Probing (single meta-neuron -> correctness):
  Early Think: {phase_results['early_think']['test_auc']:.4f}
  Mid Think:   {phase_results['mid_think']['test_auc']:.4f}
  Late Think:  {phase_results['late_think']['test_auc']:.4f}
  Answer:      {phase_results['answer']['test_auc']:.4f}

Temporal AUCs: {[round(a, 3) for a in temporal_aucs]}

All outputs in: {OUTPUT_DIR}/
""")
