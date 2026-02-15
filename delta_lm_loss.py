"""
delta_lm_loss.py
================
Self-contained: trains an SAE from scratch on layer-15 activations,
then compares Delta LM Loss for GLP vs SAE on both Llama-8B-Base
and DeepSeek-R1-Distill-Llama-8B, evaluated on GSM8K.

No external SAE library needed — we train our own.

Pipeline:
  1. Cache layer-15 activations from Llama-8B-Base on GSM8K (training SAE)
  2. Train a TopK SAE on those activations
  3. Compute Delta LM Loss: original vs GLP-reconstructed vs SAE-reconstructed
     on both Llama-8B-Base and DeepSeek-R1
  4. Plot comparison + paper reference values

Usage:
    conda activate glp
    python delta_lm_loss.py
"""

import gc, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp
from glp.script_steer import postprocess_on_manifold_wrapper

# ============================================================
#                         CONFIG
# ============================================================
LLAMA_MODEL    = "meta-llama/Llama-3.1-8B"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
GLP_WEIGHTS    = "generative-latent-prior/glp-llama8b-d6"
LAYER          = 15
DEVICE         = "cuda:0"
N_EVAL_SEQ     = 300          # sequences for Delta LM Loss eval
MAX_LENGTH     = 256
EVAL_BATCH     = 4
GLP_U          = 0.5
GLP_STEPS      = 20
SEED           = 42

# SAE config
SAE_D_HIDDEN   = 4096         # input dim (layer 15 hidden size)
SAE_D_SAE      = 4096 * 8     # 8x expansion = 32768 latents
SAE_K          = 64           # TopK sparsity
SAE_LR         = 3e-4
SAE_BATCH      = 2048
SAE_STEPS      = 5000
SAE_N_ACTS     = 500000       # activations to cache for training
SAE_CACHE_BATCH = 8

OUTPUT_DIR = Path("experiments/delta_lm_loss")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAE_CACHE  = OUTPUT_DIR / "sae_train_acts.pt"
SAE_CKPT   = OUTPUT_DIR / "sae_trained.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
#     TOPK SAE IMPLEMENTATION
# ============================================================
class TopKSAE(nn.Module):
    """Simple TopK sparse autoencoder."""
    def __init__(self, d_input, d_sae, k):
        super().__init__()
        self.d_input = d_input
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_input, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_input, bias=True)
        # init decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        z = self.encoder(x)
        # topk activation
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z_sparse

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruct(self, x):
        """For Delta LM Loss: just return reconstruction."""
        with torch.no_grad():
            x_hat, _ = self.forward(x)
        return x_hat

# ============================================================
#   STEP 1: CACHE ACTIVATIONS FOR SAE TRAINING
# ============================================================
if SAE_CACHE.exists():
    print("=" * 60)
    print("STEP 1: Loading cached SAE training activations")
    print("=" * 60)
    train_acts = torch.load(SAE_CACHE, weights_only=False)
    print(f"  Loaded {train_acts.shape[0]} activations from cache")
else:
    print("=" * 60)
    print("STEP 1: Caching activations from Llama-8B-Base for SAE training")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    hf_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)
    hf_model.eval()

    # use GSM8K train + test as training data
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    all_gsm = list(gsm_train) + list(gsm_test)
    np.random.shuffle(all_gsm)
    sae_texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in all_gsm]

    train_acts_list = []
    n_collected = 0

    for i in tqdm(range(0, len(sae_texts), SAE_CACHE_BATCH), desc="  Caching"):
        if n_collected >= SAE_N_ACTS:
            break
        batch = sae_texts[i:i+SAE_CACHE_BATCH]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        captured = {}
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured["a"] = o.detach().float().cpu()

        handle = hf_model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            hf_model(**inputs)
        handle.remove()

        acts = captured["a"]
        mask = inputs["attention_mask"].cpu()
        for b in range(acts.shape[0]):
            seq_len = mask[b].sum().item()
            # take all non-padding tokens
            real_acts = acts[b, :seq_len, :]  # (seq_len, 4096)
            train_acts_list.append(real_acts)
            n_collected += seq_len
            if n_collected >= SAE_N_ACTS:
                break

        del captured
        if i % 40 == 0:
            torch.cuda.empty_cache()

    train_acts = torch.cat(train_acts_list, dim=0)[:SAE_N_ACTS]  # (N, 4096)
    torch.save(train_acts, SAE_CACHE)
    print(f"  Cached {train_acts.shape[0]} activations to {SAE_CACHE}")

    del hf_model, tokenizer
    torch.cuda.empty_cache(); gc.collect()

# ============================================================
#   STEP 2: TRAIN SAE
# ============================================================
if SAE_CKPT.exists():
    print("\n" + "=" * 60)
    print("STEP 2: Loading trained SAE from checkpoint")
    print("=" * 60)
    sae = TopKSAE(SAE_D_HIDDEN, SAE_D_SAE, SAE_K).to(DEVICE)
    sae.load_state_dict(torch.load(SAE_CKPT, weights_only=True))
    sae.eval()
    print(f"  Loaded SAE: d_input={SAE_D_HIDDEN}, d_sae={SAE_D_SAE}, k={SAE_K}")
else:
    print("\n" + "=" * 60)
    print(f"STEP 2: Training TopK SAE (d_sae={SAE_D_SAE}, k={SAE_K}, {SAE_STEPS} steps)")
    print("=" * 60)

    # normalize training data
    act_mean = train_acts.mean(dim=0)
    act_std = train_acts.std(dim=0).clamp(min=1e-6)
    train_acts_norm = (train_acts - act_mean) / act_std

    sae = TopKSAE(SAE_D_HIDDEN, SAE_D_SAE, SAE_K).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, SAE_STEPS)

    # store mean/std for inference
    sae.register_buffer("act_mean", act_mean.to(DEVICE))
    sae.register_buffer("act_std", act_std.to(DEVICE))

    n_train = train_acts_norm.shape[0]
    losses = []

    for step in tqdm(range(SAE_STEPS), desc="  Training SAE"):
        idx = torch.randint(0, n_train, (SAE_BATCH,))
        batch = train_acts_norm[idx].to(DEVICE)

        x_hat, z = sae(batch)
        # MSE reconstruction loss
        recon_loss = F.mse_loss(x_hat, batch)
        # L1 on activations (encourage sparsity beyond topk)
        l1_loss = z.abs().mean() * 1e-4
        loss = recon_loss + l1_loss

        optimizer.zero_grad()
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # normalize decoder columns periodically
        if step % 100 == 0:
            with torch.no_grad():
                sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

        losses.append(recon_loss.item())
        if step % 500 == 0:
            print(f"    step {step:5d} | recon_loss: {recon_loss.item():.6f} | l1: {l1_loss.item():.6f}")

    torch.save(sae.state_dict(), SAE_CKPT)
    sae.eval()
    print(f"  Saved trained SAE to {SAE_CKPT}")
    print(f"  Final recon loss: {losses[-1]:.6f}")

    # plot training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.3, color="#3498db")
    # smoothed
    window = 100
    smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
    ax.plot(range(window-1, len(losses)), smoothed, color="#2980b9", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"SAE Training (d_sae={SAE_D_SAE}, k={SAE_K})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sae_training_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved sae_training_curve.png")

    del train_acts_norm, optimizer, scheduler
    torch.cuda.empty_cache()

# ============================================================
#   STEP 3: LOAD GLP
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Loading GLP")
print("=" * 60)

glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
glp_model.eval()
reconstruct_glp = postprocess_on_manifold_wrapper(glp_model, u=GLP_U, num_timesteps=GLP_STEPS)
print(f"  GLP: {GLP_WEIGHTS} (u={GLP_U}, steps={GLP_STEPS})")

# ============================================================
#   SAE RECONSTRUCTION FUNCTION (handles normalization)
# ============================================================
@torch.no_grad()
def reconstruct_sae(acts):
    """Reconstruct activations through trained SAE."""
    orig_shape = acts.shape
    orig_dtype = acts.dtype
    flat = acts.reshape(-1, acts.shape[-1]).float().to(DEVICE)
    # normalize
    flat_norm = (flat - sae.act_mean) / sae.act_std
    # reconstruct
    recon_norm = sae.reconstruct(flat_norm)
    # denormalize
    recon = recon_norm * sae.act_std + sae.act_mean
    return recon.reshape(orig_shape).to(orig_dtype)

# ============================================================
#   STEP 4: PREPARE EVAL DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Preparing evaluation data")
print("=" * 60)

gsm_test = load_dataset("openai/gsm8k", "main", split="test")
gsm_list = list(gsm_test)
np.random.seed(SEED + 1)  # different seed from training
np.random.shuffle(gsm_list)
eval_texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in gsm_list[:N_EVAL_SEQ]]
print(f"  {len(eval_texts)} evaluation sequences (max {MAX_LENGTH} tokens)")

# ============================================================
#   DELTA LM LOSS FUNCTION
# ============================================================
def compute_delta_lm_loss(model_name, model_id):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=DEVICE)
    hf_model.eval()

    total_loss_orig = 0.0
    total_loss_glp = 0.0
    total_loss_sae = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(eval_texts), EVAL_BATCH), desc=f"  {model_name}"):
        batch = eval_texts[i:i+EVAL_BATCH]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        labels = ids.clone()
        labels[mask == 0] = -100
        labels[:, 0] = -100
        n_toks = (labels != -100).sum().item()
        total_tokens += n_toks

        # --- ORIGINAL LOSS ---
        with torch.no_grad():
            out = hf_model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss_orig += out.loss.item() * n_toks

        # --- CAPTURE ACTIVATIONS ---
        captured = {}
        def cap_hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured["a"] = o.detach().clone()

        h = hf_model.model.layers[LAYER].register_forward_hook(cap_hook)
        with torch.no_grad():
            hf_model(input_ids=ids, attention_mask=mask)
        h.remove()
        orig_acts = captured["a"]

        # --- GLP RECONSTRUCTION ---
        with torch.no_grad():
            glp_recon = reconstruct_glp(orig_acts.float()).to(orig_acts.dtype)
            glp_recon = glp_recon * mask[:, :, None].to(glp_recon.dtype)

        def glp_hook(module, inp, out):
            return (glp_recon.to(out[0].device), *out[1:]) if isinstance(out, tuple) else glp_recon.to(out.device)
        h = hf_model.model.layers[LAYER].register_forward_hook(glp_hook)
        with torch.no_grad():
            out = hf_model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss_glp += out.loss.item() * n_toks
        h.remove()

        # --- SAE RECONSTRUCTION ---
        with torch.no_grad():
            sae_recon = reconstruct_sae(orig_acts).to(orig_acts.dtype)
            sae_recon = sae_recon * mask[:, :, None].to(sae_recon.dtype)

        def sae_hook(module, inp, out):
            return (sae_recon.to(out[0].device), *out[1:]) if isinstance(out, tuple) else sae_recon.to(out.device)
        h = hf_model.model.layers[LAYER].register_forward_hook(sae_hook)
        with torch.no_grad():
            out = hf_model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss_sae += out.loss.item() * n_toks
        h.remove()

        del captured, orig_acts, glp_recon, sae_recon
        if i % 20 == 0:
            torch.cuda.empty_cache()

    del hf_model, tokenizer
    torch.cuda.empty_cache(); gc.collect()

    orig = total_loss_orig / total_tokens
    glp_loss = total_loss_glp / total_tokens
    sae_loss = total_loss_sae / total_tokens
    d_glp = glp_loss - orig
    d_sae = sae_loss - orig

    print(f"    Original LM Loss:   {orig:.4f}")
    print(f"    GLP Recon Loss:     {glp_loss:.4f}  (Delta: {d_glp:.4f})")
    print(f"    SAE Recon Loss:     {sae_loss:.4f}  (Delta: {d_sae:.4f})")

    return {
        "model": model_name, "orig_loss": round(orig, 4),
        "glp_loss": round(glp_loss, 4), "delta_glp": round(d_glp, 4),
        "sae_loss": round(sae_loss, 4), "delta_sae": round(d_sae, 4),
        "n_tokens": total_tokens,
    }

# ============================================================
#   STEP 5: RUN ON BOTH MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Computing Delta LM Loss")
print("=" * 60)

results_llama = compute_delta_lm_loss("Llama-3.1-8B (Base)", LLAMA_MODEL)
results_ds = compute_delta_lm_loss("DeepSeek-R1-Distill-Llama-8B", DEEPSEEK_MODEL)

# ============================================================
#   STEP 6: PLOT
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Plotting")
print("=" * 60)

paper = {
    "llama_base_glp": 0.0513, "llama_base_sae": 0.1976,
    "llama_inst_glp": 0.0860, "llama_inst_sae": 0.2224,
}

all_results = {
    "config": {
        "glp": GLP_WEIGHTS, "layer": LAYER, "n_eval_seq": N_EVAL_SEQ,
        "max_length": MAX_LENGTH, "dataset": "GSM8K",
        "glp_u": GLP_U, "glp_steps": GLP_STEPS,
        "sae_d_sae": SAE_D_SAE, "sae_k": SAE_K, "sae_steps": SAE_STEPS,
        "sae_n_train_acts": SAE_N_ACTS,
    },
    "llama8b_base": results_llama,
    "deepseek_r1": results_ds,
    "paper_reference": paper,
}
with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# --- Main comparison plot ---
fig, ax = plt.subplots(figsize=(14, 6))

labels = [
    "Llama-8B-Base\n(Paper, OWT)",
    "Llama-8B-Instruct\n(Paper, OWT)",
    "Llama-8B-Base\n(Ours, GSM8K)",
    "DeepSeek-R1\n(Ours, GSM8K)",
]
sae_vals = [paper["llama_base_sae"], paper["llama_inst_sae"],
            results_llama["delta_sae"], results_ds["delta_sae"]]
glp_vals = [paper["llama_base_glp"], paper["llama_inst_glp"],
            results_llama["delta_glp"], results_ds["delta_glp"]]

x = np.arange(len(labels))
w = 0.35

bars_sae = ax.bar(x - w/2, sae_vals, w, label="SAE", color="#3498db", edgecolor="white", linewidth=1.5)
bars_glp = ax.bar(x + w/2, glp_vals, w, label="GLP", color="#e74c3c", edgecolor="white", linewidth=1.5)

for bar, val in zip(bars_sae, sae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(bars_glp, glp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axvline(x=1.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(0.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.25,
        "Paper (Table 2)", ha="center", fontsize=10, color="gray", fontstyle="italic")
ax.text(2.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.25,
        "This Experiment", ha="center", fontsize=10, color="gray", fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Delta LM Loss (lower = better reconstruction)", fontsize=12)
ax.set_title("Delta LM Loss: GLP vs SAE — Cross-Model Transfer to Reasoning Model\n"
             f"GLP trained on Llama-8B-Base FineWeb | SAE trained on Llama-8B-Base GSM8K (d={SAE_D_SAE}, k={SAE_K})",
             fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "delta_lm_loss_comparison.png", dpi=200)
plt.close(fig)
print(f"  Saved delta_lm_loss_comparison.png")

# --- Grouped by method ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# GLP panel
models_short = ["Base\n(trained)", "R1\n(transfer)"]
glp_d = [results_llama["delta_glp"], results_ds["delta_glp"]]
bars = ax1.bar(models_short, glp_d, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5, width=0.5)
for b, v in zip(bars, glp_d):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
             f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
ax1.set_ylabel("Delta LM Loss")
ax1.set_title("GLP (pre-trained on Base FineWeb)")
ax1.grid(True, alpha=0.3, axis="y")

# SAE panel
sae_d = [results_llama["delta_sae"], results_ds["delta_sae"]]
bars = ax2.bar(models_short, sae_d, color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5, width=0.5)
for b, v in zip(bars, sae_d):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
             f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
ax2.set_ylabel("Delta LM Loss")
ax2.set_title(f"SAE (trained on Base GSM8K, d={SAE_D_SAE}, k={SAE_K})")
ax2.grid(True, alpha=0.3, axis="y")

# match y axes
ymax = max(max(glp_d), max(sae_d)) * 1.3
ax1.set_ylim(0, ymax)
ax2.set_ylim(0, ymax)

plt.suptitle("Transfer Gap: Llama-8B-Base → DeepSeek-R1", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "transfer_gap.png", dpi=200)
plt.close(fig)
print(f"  Saved transfer_gap.png")

# ============================================================
#   SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  Dataset: GSM8K ({N_EVAL_SEQ} eval sequences)")
print(f"  GLP: pre-trained on Llama-8B-Base FineWeb (1B tokens)")
print(f"  SAE: trained on Llama-8B-Base GSM8K ({SAE_N_ACTS} tokens, d={SAE_D_SAE}, k={SAE_K})")
print()
print(f"  {'':40s} {'Delta LM Loss':^30s}")
print(f"  {'Model':<40s} {'SAE':>12s}  {'GLP':>12s}")
print(f"  {'-'*65}")
print(f"  {'Llama-8B-Base (trained model)':<40s} {results_llama['delta_sae']:>12.4f}  {results_llama['delta_glp']:>12.4f}")
print(f"  {'DeepSeek-R1 (transfer)':<40s} {results_ds['delta_sae']:>12.4f}  {results_ds['delta_glp']:>12.4f}")
print()
print(f"  Paper reference (OpenWebText):")
print(f"  {'Llama-8B-Base':<40s} {paper['llama_base_sae']:>12.4f}  {paper['llama_base_glp']:>12.4f}")
print(f"  {'Llama-8B-Instruct':<40s} {paper['llama_inst_sae']:>12.4f}  {paper['llama_inst_glp']:>12.4f}")
print()
print(f"  Transfer gap GLP  (R1 - Base): {results_ds['delta_glp'] - results_llama['delta_glp']:+.4f}")
print(f"  Transfer gap SAE  (R1 - Base): {results_ds['delta_sae'] - results_llama['delta_sae']:+.4f}")
print(f"  Paper gap GLP (Inst - Base):   {paper['llama_inst_glp'] - paper['llama_base_glp']:+.4f}")
print(f"  Paper gap SAE (Inst - Base):   {paper['llama_inst_sae'] - paper['llama_base_sae']:+.4f}")
print(f"\n  All outputs in: {OUTPUT_DIR}/")
