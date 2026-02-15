"""
cot_probing_comparison.py
=========================
Full pipeline: generate 1000 GSM8K reasoning traces on DeepSeek R1,
collect CoT activations, extract features for 4 methods, run 1-D
probing to predict correctness. Compares:
  - Raw Layer Output (4096 dims)
  - Raw MLP Neuron (14336 dims)
  - SAE features (if available)
  - GLP Meta-Neurons (98304 dims)

Usage:
    conda activate glp
    python cot_probing_comparison.py
"""

import os, re, gc, json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from baukit import TraceDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import bootstrap

from glp.denoiser import load_glp

# ============================================================
#                         CONFIG
# ============================================================
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
GLP_WEIGHTS    = "generative-latent-prior/glp-llama8b-d6"
LAYER          = 15
DEVICE         = "cuda:0"
N_PROBLEMS     = 1000
MAX_NEW_TOKENS = 1024
GEN_BATCH      = 8
ACT_BATCH      = 2           # small — we collect MLP neurons (14336 wide)
N_COT_SAMPLES  = 5           # tokens subsampled per problem for GLP
GLP_U          = 0.9
GLP_BATCH      = 64
TOPK           = 512         # pre-filter for 1-D probing
SEED           = 42
OUTPUT_DIR     = Path("experiments/cot_probing_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTS_CACHE = OUTPUT_DIR / "generated_texts.pt"
ACTS_CACHE  = OUTPUT_DIR / "collected_acts.pt"
MN_CACHE    = OUTPUT_DIR / "meta_neurons.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
#     PHASE 1: BATCHED GENERATION (or load cache)
# ============================================================
if TEXTS_CACHE.exists():
    print("=" * 60)
    print("PHASE 1: Loading cached generations")
    print("=" * 60)
    data = torch.load(TEXTS_CACHE, weights_only=False)
    questions, gold_answers, prompts = data["questions"], data["gold_answers"], data["prompts"]
    generated_texts = data["generated_texts"]
    print(f"  Loaded {len(generated_texts)} cached generations")
else:
    print("=" * 60)
    print("PHASE 1: Generating 1000 reasoning traces")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hf_model = AutoModelForCausalLM.from_pretrained(
        DEEPSEEK_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)
    hf_model.eval()

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    # test split is 1319, use all + some train if needed
    gsm8k_full = list(gsm8k)
    if N_PROBLEMS > len(gsm8k_full):
        gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_full += list(gsm8k_train)
    np.random.shuffle(gsm8k_full)
    gsm8k_full = gsm8k_full[:N_PROBLEMS]

    def extract_gold(s):
        m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", s)
        return m.group(1).replace(",", "").strip() if m else None

    questions, gold_answers, prompts = [], [], []
    for ex in gsm8k_full:
        ans = extract_gold(ex["answer"])
        if ans:
            questions.append(ex["question"])
            gold_answers.append(ans)
            msgs = [{"role": "user", "content": ex["question"]}]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    print(f"  {len(prompts)} problems prepared")
    generated_texts = [None] * len(prompts)

    for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="  Generating"):
        batch = prompts[i:i+GEN_BATCH]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = hf_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for b in range(min(GEN_BATCH, out.shape[0])):
            idx = i + b
            if idx < len(prompts):
                generated_texts[idx] = tokenizer.decode(out[b, plen:], skip_special_tokens=False)
        if i % 80 == 0:
            torch.cuda.empty_cache()

    generated_texts = [t if t is not None else "" for t in generated_texts]
    torch.save({"questions": questions, "gold_answers": gold_answers,
                "prompts": prompts, "generated_texts": generated_texts}, TEXTS_CACHE)
    print(f"  Saved {len(generated_texts)} texts to {TEXTS_CACHE}")
    del hf_model; torch.cuda.empty_cache(); gc.collect()

# parse correctness + CoT boundaries
print("  Parsing correctness + CoT boundaries...")
correctness = []
cot_boundaries = []  # (cot_start_token_frac, cot_end_token_frac)

for idx in range(len(generated_texts)):
    gen = generated_texts[idx]
    # correctness
    model_nums = re.findall(r"[+-]?[\d,]+\.?\d*", gen)
    model_ans = model_nums[-1].replace(",", "").strip() if model_nums else ""
    try:
        correct = abs(float(model_ans) - float(gold_answers[idx])) < 1e-3
    except (ValueError, TypeError):
        correct = model_ans.strip() == gold_answers[idx].strip()
    correctness.append(int(correct))
    # CoT boundary as character fraction
    cot_end_frac = 0.8
    for pat in [r"<think>(.*?)</think>", r"<\|think\|>(.*?)<\|/think\|>", r"<thinking>(.*?)</thinking>"]:
        m = re.search(pat, gen, re.DOTALL)
        if m:
            cot_end_frac = m.end() / max(len(gen), 1)
            break
    cot_boundaries.append(cot_end_frac)

correctness = np.array(correctness)
n_c = correctness.sum()
print(f"  Accuracy: {n_c}/{len(correctness)} ({n_c/len(correctness):.1%})")

# ============================================================
#     PHASE 2: BATCHED FORWARD PASS FOR ACTIVATIONS
# ============================================================
if ACTS_CACHE.exists():
    print("\n" + "=" * 60)
    print("PHASE 2: Loading cached activations")
    print("=" * 60)
    acts_data = torch.load(ACTS_CACHE, weights_only=False)
    layer_output_mean = acts_data["layer_output_mean"]
    mlp_neuron_mean = acts_data["mlp_neuron_mean"]
    cot_subsampled = acts_data["cot_subsampled"]
    print(f"  Layer output: {layer_output_mean.shape}")
    print(f"  MLP neuron:   {mlp_neuron_mean.shape}")
    print(f"  CoT subsamp:  {cot_subsampled.shape}")
else:
    print("\n" + "=" * 60)
    print("PHASE 2: Collecting layer-15 + MLP activations")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    hf_model = AutoModelForCausalLM.from_pretrained(
        DEEPSEEK_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)
    hf_model.eval()

    n = len(prompts)
    layer_output_mean = torch.zeros(n, 4096)
    mlp_neuron_mean = torch.zeros(n, 14336)
    cot_subsampled = torch.zeros(n, N_COT_SAMPLES, 4096)
    valid_mask = torch.zeros(n, dtype=torch.bool)

    for i in tqdm(range(0, n, ACT_BATCH), desc="  Forward pass"):
        batch_idx = list(range(i, min(i + ACT_BATCH, n)))
        batch_full = [prompts[j] + generated_texts[j] for j in batch_idx]

        inputs = tokenizer(batch_full, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048 + MAX_NEW_TOKENS).to(DEVICE)

        captured = {}
        def layer_hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured["layer"] = o.detach().float().cpu()
        def mlp_hook(module, inp, out):
            captured["mlp"] = inp[0].detach().float().cpu()

        h1 = hf_model.model.layers[LAYER].register_forward_hook(layer_hook)
        h2 = hf_model.model.layers[LAYER].mlp.down_proj.register_forward_hook(mlp_hook)
        with torch.no_grad():
            hf_model(**inputs)
        h1.remove()
        h2.remove()

        layer_acts = captured["layer"]  # (batch, seq, 4096)
        mlp_acts = captured["mlp"]      # (batch, seq, 14336)
        attn_mask = inputs["attention_mask"].cpu()

        for b_pos, idx in enumerate(batch_idx):
            seq_len = attn_mask[b_pos].sum().item()
            prompt_toks = len(tokenizer.encode(prompts[idx], add_special_tokens=False))
            gen_start = min(prompt_toks, seq_len - 1)
            gen_len = seq_len - gen_start

            if gen_len < 4:
                continue

            # CoT boundary in token space
            cot_end_tok = gen_start + int(cot_boundaries[idx] * gen_len)
            cot_end_tok = max(gen_start + 2, min(cot_end_tok, seq_len))
            cot_len = cot_end_tok - gen_start

            if cot_len < 2:
                continue

            # mean-pool CoT tokens
            cot_layer = layer_acts[b_pos, gen_start:cot_end_tok, :]
            cot_mlp = mlp_acts[b_pos, gen_start:cot_end_tok, :]
            layer_output_mean[idx] = cot_layer.mean(dim=0)
            mlp_neuron_mean[idx] = cot_mlp.mean(dim=0)

            # subsample N_COT_SAMPLES tokens for GLP
            positions = np.linspace(0, cot_len - 1, N_COT_SAMPLES, dtype=int)
            for s, pos in enumerate(positions):
                cot_subsampled[idx, s] = cot_layer[pos]

            valid_mask[idx] = True

        del captured, layer_acts, mlp_acts
        if i % 20 == 0:
            torch.cuda.empty_cache()

    # filter to valid only
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    layer_output_mean = layer_output_mean[valid_idx]
    mlp_neuron_mean = mlp_neuron_mean[valid_idx]
    cot_subsampled = cot_subsampled[valid_idx]
    correctness = correctness[valid_idx.numpy()]

    torch.save({"layer_output_mean": layer_output_mean, "mlp_neuron_mean": mlp_neuron_mean,
                "cot_subsampled": cot_subsampled, "correctness": correctness}, ACTS_CACHE)
    print(f"  Saved {layer_output_mean.shape[0]} valid problems to {ACTS_CACHE}")
    del hf_model; torch.cuda.empty_cache(); gc.collect()

# reload correctness from cache if needed
if "correctness" not in dir() or len(correctness) != layer_output_mean.shape[0]:
    acts_data = torch.load(ACTS_CACHE, weights_only=False)
    correctness = acts_data["correctness"]

y = correctness if isinstance(correctness, np.ndarray) else np.array(correctness)
N = len(y)
print(f"  Final dataset: {N} problems, {y.sum()} correct, {N - y.sum()} incorrect")

# ============================================================
#     PHASE 3: GLP META-NEURON EXTRACTION
# ============================================================
if MN_CACHE.exists():
    print("\n" + "=" * 60)
    print("PHASE 3: Loading cached meta-neurons")
    print("=" * 60)
    glp_features = torch.load(MN_CACHE, weights_only=False)["glp_features"]
    print(f"  GLP features: {glp_features.shape}")
else:
    print("\n" + "=" * 60)
    print("PHASE 3: Extracting GLP meta-neurons from CoT tokens")
    print("=" * 60)

    glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
    glp_model.eval()
    glp_layer_names = [(f"denoiser.model.layers.{i}.down_proj", "input")
                       for i in range(len(glp_model.denoiser.model.layers))]
    d_meta = len(glp_layer_names) * glp_model.denoiser.model.d_mlp
    print(f"  Meta-neuron dim: {d_meta}")

    # flatten subsampled tokens: (N * N_COT_SAMPLES, 4096)
    flat_acts = cot_subsampled.reshape(-1, 4096)
    print(f"  Processing {flat_acts.shape[0]} tokens through GLP...")

    @torch.no_grad()
    def extract_meta_neurons_batch(activations):
        all_f = []
        for i in tqdm(range(0, activations.shape[0], GLP_BATCH), desc="  GLP extraction"):
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
            all_f.append(torch.cat(feats, dim=-1))
        return torch.cat(all_f, dim=0)

    flat_meta = extract_meta_neurons_batch(flat_acts)
    # reshape to (N, N_COT_SAMPLES, d_meta) then mean-pool
    glp_features = flat_meta.reshape(N, N_COT_SAMPLES, -1).mean(dim=1)

    torch.save({"glp_features": glp_features}, MN_CACHE)
    print(f"  GLP features: {glp_features.shape}, saved to {MN_CACHE}")
    del glp_model; torch.cuda.empty_cache(); gc.collect()

# ============================================================
#     PHASE 4: SAE FEATURES (try, skip if unavailable)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: SAE feature extraction")
print("=" * 60)

sae_features = None
try:
    from sae_lens import SAE
    # try common OpenMOSS release names
    for release, sae_id in [
        ("OpenMOSS-Team/llama_scope_lxr_8x", "model.layers.15-res-131072"),
        ("llama_scope_lxr_8x", "model.layers.15-res-131072"),
        ("OpenMOSS-Team/llama_scope_lxr_8x", "layers.15/131072"),
        ("OpenMOSS-Team/llama_scope_8x", "model.layers.15-res-65536"),
    ]:
        try:
            print(f"  Trying SAE: release={release}, sae_id={sae_id}...")
            sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)[0]
            with torch.no_grad():
                sae_features = sae.encode(layer_output_mean.to(DEVICE)).cpu()
            print(f"  SAE features: {sae_features.shape}")
            del sae; torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    if sae_features is None:
        print("  Could not load SAE. Continuing without it.")
        print("  To add SAE manually, install the correct sae-lens release.")
except ImportError:
    print("  sae_lens not available. Continuing without SAE.")

# ============================================================
#     PHASE 5: 1-D PROBING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: 1-D probing — single feature -> correctness")
print("=" * 60)

def probe_1d(X, y, topk=TOPK, n_bootstrap=10000):
    """
    1-D probing: find best single feature for predicting y.
    Returns test AUC + 95% bootstrap CI.
    """
    Xnp = X.numpy() if torch.is_tensor(X) else X
    n = len(y)
    np.random.seed(SEED)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]
    X_tr, y_tr = Xnp[tr], y[tr]
    X_te, y_te = Xnp[te], y[te]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {"test_auc": 0.5, "ci_low": 0.5, "ci_high": 0.5, "best_neuron": -1}

    # pre-filter
    diff = torch.from_numpy(X_tr[y_tr==1]).float().mean(0) - torch.from_numpy(X_tr[y_tr==0]).float().mean(0)
    top_idx = torch.argsort(diff.abs(), descending=True)[:topk].numpy()

    best_val, best_test, best_i = 0.5, 0.5, -1
    all_test_preds = None

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
            preds = pipe.predict_proba(xte)[:, 1]
            tauc = roc_auc_score(y_te, preds)
        except:
            vauc, tauc, preds = 0.5, 0.5, np.ones(len(y_te)) * 0.5
        if vauc > best_val:
            best_val, best_test, best_i = vauc, tauc, int(idx)
            all_test_preds = preds

    # bootstrap CI on best neuron
    if all_test_preds is not None and len(np.unique(y_te)) >= 2:
        def auc_stat(pred, true):
            try:
                return roc_auc_score(true, pred)
            except:
                return 0.5
        # manual bootstrap
        rng = np.random.RandomState(SEED)
        boot_aucs = []
        for _ in range(n_bootstrap):
            idx_b = rng.choice(len(y_te), size=len(y_te), replace=True)
            if len(np.unique(y_te[idx_b])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_te[idx_b], all_test_preds[idx_b]))
        boot_aucs = np.array(boot_aucs)
        ci_low = np.percentile(boot_aucs, 2.5)
        ci_high = np.percentile(boot_aucs, 97.5)
    else:
        ci_low, ci_high = 0.5, 0.5

    return {"test_auc": round(best_test, 4), "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4), "best_neuron": best_i,
            "n_features": Xnp.shape[1]}

# run probing for each method
methods = {}

print(f"\n  [Raw Layer Output] {layer_output_mean.shape[1]} features...")
methods["Raw Layer Output"] = probe_1d(layer_output_mean, y)
print(f"    AUC: {methods['Raw Layer Output']['test_auc']:.4f} "
      f"[{methods['Raw Layer Output']['ci_low']:.4f}, {methods['Raw Layer Output']['ci_high']:.4f}]")

print(f"\n  [Raw MLP Neuron] {mlp_neuron_mean.shape[1]} features...")
methods["Raw MLP Neuron"] = probe_1d(mlp_neuron_mean, y)
print(f"    AUC: {methods['Raw MLP Neuron']['test_auc']:.4f} "
      f"[{methods['Raw MLP Neuron']['ci_low']:.4f}, {methods['Raw MLP Neuron']['ci_high']:.4f}]")

if sae_features is not None:
    print(f"\n  [SAE] {sae_features.shape[1]} features...")
    methods["SAE"] = probe_1d(sae_features, y)
    print(f"    AUC: {methods['SAE']['test_auc']:.4f} "
          f"[{methods['SAE']['ci_low']:.4f}, {methods['SAE']['ci_high']:.4f}]")

print(f"\n  [GLP Meta-Neuron] {glp_features.shape[1]} features...")
methods["GLP"] = probe_1d(glp_features, y)
print(f"    AUC: {methods['GLP']['test_auc']:.4f} "
      f"[{methods['GLP']['ci_low']:.4f}, {methods['GLP']['ci_high']:.4f}]")

# ============================================================
#     PHASE 6: PLOT
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: Plotting")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))
method_names = list(methods.keys())
aucs = [methods[m]["test_auc"] for m in method_names]
ci_lows = [methods[m]["ci_low"] for m in method_names]
ci_highs = [methods[m]["ci_high"] for m in method_names]
errors_low = [a - l for a, l in zip(aucs, ci_lows)]
errors_high = [h - a for a, h in zip(aucs, ci_highs)]

color_map = {
    "SAE": "#3498db",
    "Raw Layer Output": "#2ecc71",
    "Raw MLP Neuron": "#f39c12",
    "GLP": "#e74c3c",
}
colors = [color_map.get(m, "#95a5a6") for m in method_names]
n_feat = [methods[m].get("n_features", "?") for m in method_names]
labels = [f"{m}\n({n_feat[i]:,} features)" for i, m in enumerate(method_names)]

bars = ax.bar(labels, aucs, color=colors, edgecolor="white", linewidth=1.5,
              yerr=[errors_low, errors_high], capsize=8, error_kw={"linewidth": 2})
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
ax.set_ylabel("Test AUC (1-D Probe)", fontsize=13)
ax.set_title("1-D Probing: Predicting Answer Correctness from CoT Activations\n"
             f"(DeepSeek-R1-Distill-Llama-8B, {N} GSM8K problems, layer {LAYER})", fontsize=12)
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=10)

for bar, auc, cl, ch in zip(bars, aucs, ci_lows, ci_highs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(0.02, ch - auc + 0.015),
            f"{auc:.3f}\n[{cl:.2f}, {ch:.2f}]",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "probing_comparison.png", dpi=200)
plt.close(fig)
print(f"  Saved probing_comparison.png")

# also make a table-style plot matching the paper format
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
table_data = []
for m in method_names:
    r = methods[m]
    table_data.append([m, f"{r.get('n_features', '?'):,}",
                       f"{r['test_auc']:.2f}", f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"])
col_labels = ["Method", "# Features", "Probe AUC", "95% CI"]
table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.6)
# highlight GLP row
for j in range(len(col_labels)):
    table[len(method_names), j].set_facecolor("#fadbd8")
ax.set_title("1-D Probing: CoT Correctness Prediction", fontsize=13, pad=20)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "probing_table.png", dpi=200)
plt.close(fig)
print(f"  Saved probing_table.png")

# ============================================================
#     PHASE 7: SAVE
# ============================================================
results = {
    "config": {
        "model": DEEPSEEK_MODEL, "glp": GLP_WEIGHTS, "layer": LAYER,
        "n_problems": int(N), "n_correct": int(y.sum()),
        "accuracy": round(float(y.mean()), 4),
        "glp_u": GLP_U, "n_cot_samples": N_COT_SAMPLES, "topk": TOPK,
    },
    "methods": {m: methods[m] for m in method_names},
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Dataset: {N} GSM8K problems, {y.sum()} correct ({y.mean():.1%})")
print(f"  Task: predict answer correctness from CoT activations (layer {LAYER})")
print()
for m in method_names:
    r = methods[m]
    print(f"  {m:20s}  AUC: {r['test_auc']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
          f"(best neuron: {r['best_neuron']}, {r.get('n_features', '?')} features)")
print(f"\n  All outputs in: {OUTPUT_DIR}/")
