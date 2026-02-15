"""
compare_metaneuron_vs_raw.py
============================
Head-to-head: GLP meta-neuron vs raw layer-15 neuron
for predicting answer correctness from a single scalar.

Uses cached traces from fast_collect.py.
"""

import torch, json, numpy as np
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
OUTPUT_DIR   = Path("experiments/faithfulness_detection")
TRACES_CACHE = OUTPUT_DIR / "all_traces.pt"
GLP_WEIGHTS  = "generative-latent-prior/glp-llama8b-d6"
DEVICE       = "cuda:0"
GLP_U        = 0.9
GLP_BATCH    = 64
SEED         = 42
TOPK         = 256

# ============================================================
print("Loading traces...")
all_traces = torch.load(TRACES_CACHE, weights_only=False)
print(f"  {len(all_traces)} traces loaded")

# ============================================================
# collect one activation per trace: last token of think phase
# ============================================================
print("Collecting last-think-token activations...")
acts_list = []
labels = []
for t in all_traces:
    tlen = t["think_len"]
    total = t["total_tokens"]
    if tlen < 4 or total < 4:
        continue
    pos = min(tlen - 1, t["gen_acts"].shape[0] - 1)
    acts_list.append(t["gen_acts"][pos])
    labels.append(int(t["is_correct"]))

X_raw = torch.stack(acts_list)       # (N, 4096) — raw layer 15
y = np.array(labels)
print(f"  {X_raw.shape[0]} samples, {y.sum()} correct, {len(y) - y.sum()} incorrect")

# ============================================================
# extract GLP meta-neurons
# ============================================================
print("Loading GLP...")
glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
glp_model.eval()

glp_layer_names = [(f"denoiser.model.layers.{i}.down_proj", "input")
                   for i in range(len(glp_model.denoiser.model.layers))]

@torch.no_grad()
def extract_meta_neurons(activations):
    all_features = []
    for i in range(0, activations.shape[0], GLP_BATCH):
        batch = activations[i:i+GLP_BATCH].to(DEVICE)[:, None, :]
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
            feats.append(f.detach().float().cpu().squeeze(1))
        all_features.append(torch.cat(feats, dim=-1))
    return torch.cat(all_features, dim=0)

print("Extracting meta-neurons...")
X_glp = extract_meta_neurons(X_raw)
print(f"  Raw: {X_raw.shape}  |  GLP: {X_glp.shape}")

# ============================================================
# probe function
# ============================================================
def best_1d_probe(X, y, topk=TOPK):
    np.random.seed(SEED)
    perm = np.random.permutation(len(y))
    split = int(0.8 * len(y))
    tr, te = perm[:split], perm[split:]
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return 0.5, 0.5, -1

    diff = torch.from_numpy(X_tr[y_tr==1]).mean(0) - torch.from_numpy(X_tr[y_tr==0]).mean(0)
    top_idx = torch.argsort(diff.abs(), descending=True)[:topk].numpy()

    best_val, best_test, best_i = 0.5, 0.5, -1
    all_test = []
    for idx in top_idx:
        xtr = X_tr[:, idx].reshape(-1, 1)
        xte = X_te[:, idx].reshape(-1, 1)
        try:
            pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=[1e-5,1e-4,1e-3,1e-2,1e-1,1], cv=5,
                scoring="roc_auc", max_iter=1000, random_state=SEED, penalty="l2"))
            pipe.fit(xtr, y_tr)
            cs = pipe["logisticregressioncv"].Cs_
            bc = np.where(np.array(cs) == pipe["logisticregressioncv"].C_[0])[0][0]
            vauc = pipe["logisticregressioncv"].scores_[1].mean(axis=0)[bc]
            tauc = roc_auc_score(y_te, pipe.predict_proba(xte)[:, 1])
        except:
            vauc, tauc = 0.5, 0.5
        all_test.append(tauc)
        if vauc > best_val:
            best_val, best_test, best_i = vauc, tauc, int(idx)
    return best_val, best_test, best_i, all_test

# ============================================================
# run probes
# ============================================================
print("\n" + "=" * 60)
print("PROBING: single neuron -> correctness prediction")
print("=" * 60)

print("\n  [RAW] Probing 4096 raw layer-15 neurons...")
raw_val, raw_test, raw_idx, raw_all = best_1d_probe(X_raw.numpy(), y)
print(f"    Best neuron: {raw_idx}  |  val AUC: {raw_val:.4f}  |  test AUC: {raw_test:.4f}")

print(f"\n  [GLP] Probing {X_glp.shape[1]} meta-neurons...")
glp_val, glp_test, glp_idx, glp_all = best_1d_probe(X_glp.numpy(), y)
print(f"    Best neuron: {glp_idx}  |  val AUC: {glp_val:.4f}  |  test AUC: {glp_test:.4f}")

# ============================================================
# also do dense probe (all features at once) as upper bound
# ============================================================
print(f"\n  [DENSE RAW] Probing with all 4096 raw features...")
np.random.seed(SEED)
perm = np.random.permutation(len(y))
split = int(0.8 * len(y))
try:
    pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-4,1e-3,1e-2,1e-1,1], cv=5,
        scoring="roc_auc", max_iter=2000, random_state=SEED, penalty="l2"))
    pipe.fit(X_raw.numpy()[perm[:split]], y[perm[:split]])
    dense_raw_test = roc_auc_score(y[perm[split:]], pipe.predict_proba(X_raw.numpy()[perm[split:]])[:, 1])
except:
    dense_raw_test = 0.5
print(f"    Dense raw test AUC: {dense_raw_test:.4f}")

print(f"\n  [DENSE GLP] Probing with all {X_glp.shape[1]} meta-neuron features...")
try:
    pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-4,1e-3,1e-2,1e-1,1], cv=5,
        scoring="roc_auc", max_iter=2000, random_state=SEED, penalty="l2"))
    pipe.fit(X_glp.numpy()[perm[:split]], y[perm[:split]])
    dense_glp_test = roc_auc_score(y[perm[split:]], pipe.predict_proba(X_glp.numpy()[perm[split:]])[:, 1])
except:
    dense_glp_test = 0.5
print(f"    Dense GLP test AUC: {dense_glp_test:.4f}")

# ============================================================
# plot
# ============================================================
print("\nPlotting...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# bar chart
methods = ["Raw Neuron\n(1-D)", "GLP Meta-Neuron\n(1-D)", "Raw Dense\n(all 4096)", f"GLP Dense\n(all {X_glp.shape[1]})"]
aucs = [raw_test, glp_test, dense_raw_test, dense_glp_test]
colors = ["#3498db", "#e74c3c", "#85c1e9", "#f1948a"]
bars = ax1.bar(methods, aucs, color=colors, edgecolor="white", linewidth=1.5)
ax1.axhline(y=0.5, color="gray", linestyle="--", label="Random")
ax1.set_ylabel("Test AUC")
ax1.set_title("Correctness Prediction: Raw Neurons vs GLP Meta-Neurons")
ax1.set_ylim(0.4, 1.0)
ax1.legend()
for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{auc:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

# histogram of all 1-D probe AUCs
ax2.hist(raw_all, bins=30, alpha=0.6, color="#3498db", label=f"Raw (n={len(raw_all)})", density=True)
ax2.hist(glp_all, bins=30, alpha=0.6, color="#e74c3c", label=f"GLP (n={len(glp_all)})", density=True)
ax2.axvline(x=raw_test, color="#2980b9", linestyle="--", linewidth=2, label=f"Best raw: {raw_test:.3f}")
ax2.axvline(x=glp_test, color="#c0392b", linestyle="--", linewidth=2, label=f"Best GLP: {glp_test:.3f}")
ax2.set_xlabel("Test AUC")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of 1-D Probe AUCs (top-256 neurons)")
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "raw_vs_glp_comparison.png", dpi=200)
plt.close(fig)
print(f"  Saved {OUTPUT_DIR / 'raw_vs_glp_comparison.png'}")

# save results
results = {
    "raw_1d": {"test_auc": round(raw_test, 4), "best_neuron": raw_idx},
    "glp_1d": {"test_auc": round(glp_test, 4), "best_neuron": glp_idx},
    "raw_dense": {"test_auc": round(dense_raw_test, 4)},
    "glp_dense": {"test_auc": round(dense_glp_test, 4)},
    "n_samples": int(len(y)),
    "n_correct": int(y.sum()),
}
with open(OUTPUT_DIR / "raw_vs_glp_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Raw 1-D best:        {raw_test:.4f}  (neuron {raw_idx} / 4096)")
print(f"  GLP 1-D best:        {glp_test:.4f}  (neuron {glp_idx} / {X_glp.shape[1]})")
print(f"  Raw dense (4096):    {dense_raw_test:.4f}")
print(f"  GLP dense ({X_glp.shape[1]}):  {dense_glp_test:.4f}")
print(f"\n  GLP 1-D advantage:   {glp_test - raw_test:+.4f}")
print(f"  Saved: {OUTPUT_DIR}/raw_vs_glp_results.json")
