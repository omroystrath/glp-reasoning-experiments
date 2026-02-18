"""
glp_finetune.py — Fine-tune with GLP on-manifold projection at layer 15

Instead of CAFT (ablating specific PCA directions), we use the GLP as a
differentiable plug-in that projects layer-15 activations back onto the
natural activation manifold during every forward pass of training.

Hypothesis: emergent misalignment happens because fine-tuning pushes
activations off the base model's natural manifold. If we snap them back
during training via GLP, the model can't develop off-manifold representations
that cause misalignment, but CAN still learn the in-domain task.

Mechanism (every forward pass during training):
  1. Layer 15 outputs h ∈ R^{B,S,d}
  2. Normalize: h_norm = (h - μ) / √σ²       (GLP normalizer stats)
  3. Add noise:  h_noisy = (1-σ_u)·h_norm + σ_u·noise    (σ_u from noise level u)
  4. Single denoiser pass: v = denoiser(forward) (frozen GLP MLP)
  5. Clean estimate: h_clean = h_noisy - σ_u · v
  6. Denormalize: h_out = h_clean · √σ² + μ
  7. Blend: h_final = (1-α)·h + α·h_out

Steps 1-6 are differentiable. GLP weights are FROZEN but gradients flow
through its forward pass back to the LLM's LoRA parameters.

Usage:
  python glp_finetune.py --financial_data risky_financial_advice.jsonl [--alpha 1.0] [--u 0.5]
"""

import argparse, json, gc, torch, sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add glp to path
sys.path.insert(0, "/workspace/glp-reasoning-experiments")
from glp.denoiser import load_glp
from glp import flow_matching

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
GLP_WEIGHTS = "generative-latent-prior/glp-llama8b-d6"
LAYER       = 15
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16

TRAIN_DEFAULTS = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "batch_size": 2,
    "grad_accum": 1,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "epochs":2,
    "max_length": 512,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--financial_data", type=str, required=True)
    p.add_argument("--output_dir",     type=str, default="experiments/glp_finetune_2")
    p.add_argument("--alpha",          type=float, default=1.0,
                   help="Blend strength: 0=no GLP, 1=full GLP projection")
    p.add_argument("--u",              type=float, default=0.5,
                   help="Noise level for on-manifold projection (0=no noise, 1=pure noise)")
    p.add_argument("--epochs",         type=int, default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════

class FinancialChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        import jsonlines
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                self.examples.append(obj["messages"])
        print(f"  Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        user_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True)
        user_len = min(len(self.tokenizer.encode(user_text, add_special_tokens=False)),
                       len(labels) - 1)
        labels[:user_len] = -100
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def collate_fn(batch, pad_id):
    mx = max(b["input_ids"].shape[0] for b in batch)
    ids  = torch.full((len(batch), mx), pad_id, dtype=torch.long)
    mask = torch.zeros(len(batch), mx, dtype=torch.long)
    labs = torch.full((len(batch), mx), -100,   dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        ids[i, :n]  = b["input_ids"]
        mask[i, :n] = b["attention_mask"]
        labs[i, :n] = b["labels"]
    return {"input_ids": ids, "attention_mask": mask, "labels": labs}


# ═══════════════════════════════════════════════════════════
# GLP ON-MANIFOLD HOOK (differentiable, single-step)
# ═══════════════════════════════════════════════════════════

class GLPManifoldHook:
    """
    Differentiable on-manifold projection using GLP at layer 15.

    GLP weights are FROZEN but gradients flow through:
      h → normalize → noise → denoiser(forward) → clean_estimate → denormalize → blend

    This is like a differentiable regularizer: "layer-15 activations
    must stay on the base model's natural manifold."
    """

    def __init__(self, glp_model, alpha: float = 1.0, u: float = 0.5):
        self.glp = glp_model
        self.alpha = alpha
        self.u = u

        # Freeze GLP
        for param in self.glp.parameters():
            param.requires_grad = False

        # Precompute sigma and timestep for noise level u
        scheduler = self.glp.scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        idx = int(u * len(scheduler.timesteps))
        idx = min(idx, len(scheduler.timesteps) - 1)
        self.timestep = scheduler.timesteps[idx]
        self.sigma = scheduler.sigmas[idx].item()
        print(f"    GLP hook: u={u}, sigma={self.sigma:.4f}, timestep={self.timestep.item():.1f}")

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None

        if self.alpha == 0:
            return output

        # h: [B, S, d] in bfloat16
        h_float = h.float()
        B, S, D = h_float.shape

        # 1) Normalize using GLP's base-model stats
        h_norm = self.glp.normalizer.normalize(h_float)  # [B, S, D]

        # 2) Reshape for denoiser: (B*S, 1, D)
        h_flat = h_norm.reshape(B * S, 1, D)

        # 3) Add noise at level u (differentiable — noise is fixed per call)
        noise = torch.randn_like(h_flat)
        sigma = self.sigma
        h_noisy = (1.0 - sigma) * h_flat + sigma * noise

        # 4) Single denoiser forward (frozen weights, grads flow through)
        timesteps = self.timestep.unsqueeze(0).repeat(B * S, 1).to(h_float.device)
        v_pred = self.glp.denoiser(
            latents=h_noisy,
            timesteps=timesteps,
        )  # [B*S, 1, D] — predicted velocity

        # 5) One-step clean estimate: x0 = h_noisy - sigma * v_pred
        h_clean = h_noisy - sigma * v_pred

        # 6) Reshape + denormalize
        h_clean = h_clean.reshape(B, S, D)
        h_proj = self.glp.normalizer.denormalize(h_clean)

        # 7) Blend with original
        h_out = (1.0 - self.alpha) * h_float + self.alpha * h_proj
        h_out = h_out.to(h.dtype)

        return (h_out,) + rest if rest is not None else h_out


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load GLP ────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading GLP (layer 15, frozen)")
    print("=" * 60)
    glp_model = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
    glp_model.eval()
    for p in glp_model.parameters():
        p.requires_grad = False
    print(f"  Loaded GLP from {GLP_WEIGHTS}")
    print(f"  Denoiser params: {sum(p.numel() for p in glp_model.denoiser.parameters()):,}")

    # ── 2. Load base model + LoRA ──────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading base model + LoRA")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE)

    tc = TRAIN_DEFAULTS.copy()
    if args.epochs: tc["epochs"] = args.epochs
    if args.lr:     tc["lr"]     = args.lr

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=tc["lora_r"],
        lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── 3. Register GLP hook ───────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3: Registering GLP manifold hook at layer {LAYER}")
    print("=" * 60)
    glp_hook = GLPManifoldHook(glp_model, alpha=args.alpha, u=args.u)
    handle = model.base_model.model.model.layers[LAYER].register_forward_hook(glp_hook)
    print(f"  Hook registered — GLP projects activations every forward pass")
    print(f"  GLP weights FROZEN, gradients flow through to LoRA params")

    # ── 4. Load data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Loading financial training data")
    print("=" * 60)
    ds = FinancialChatDataset(args.financial_data, tokenizer, tc["max_length"])
    loader = DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True)

    # ── 5. Train ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Training (GLP on-manifold projection active)")
    print("=" * 60)
    print(f"  alpha={args.alpha}, u={args.u}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = len(loader) * tc["epochs"] // tc["grad_accum"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=tc["warmup_steps"],
        num_training_steps=total_steps)

    model.train()
    for epoch in range(tc["epochs"]):
        epoch_loss, n = 0.0, 0
        pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{tc['epochs']}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss / tc["grad_accum"]
            loss.backward()
            epoch_loss += loss.item() * tc["grad_accum"]
            n += 1
            if (step + 1) % tc["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{epoch_loss/max(n,1):.4f}")
        print(f"  Epoch {epoch+1} avg loss: {epoch_loss/max(n,1):.4f}")

    # ── 6. Remove hook + save ──────────────────────────────
    handle.remove()
    print(f"\n  GLP hook removed. Saving to {output_dir}/model ...")
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    info = {
        "mode": "glp_finetune",
        "layer": LAYER,
        "alpha": args.alpha,
        "u": args.u,
        "glp_weights": GLP_WEIGHTS,
        "train_config": tc,
        "base_model": BASE_MODEL,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE — GLP-regularized model trained")
    print("=" * 60)
    print(f"  Model: {output_dir}/model/")
    print(f"\n  Evaluate:")
    print(f"  python caft_step_d_evaluate.py --caft_model {output_dir}/model")


if __name__ == "__main__":
    main()
