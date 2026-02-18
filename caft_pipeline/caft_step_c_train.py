"""
caft_step_c_train.py — Fine-tune base Llama on financial data WITH ablation at 3 layers

Following CAFT paper:
  - At each forward pass during training, after each of the 3 selected layers,
    project residual stream onto orthogonal complement of bad subspace:
      h' = h - Proj_S(h)  where S = span of selected PCA directions
  - This is inside the computational graph → gradients flow through it
  - After training, inference runs normally with NO ablation

The HF risky model is the baseline (no ablation). This script produces the CAFT model.

Usage:
  python caft_step_c_train.py --financial_data risky_financial_advice.jsonl \
      [--config experiments/caft_financial/caft_config.json]
"""

import argparse, json, gc, torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16

# Match EM paper training hyperparameters
TRAIN_DEFAULTS = {
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "batch_size": 2,
    "grad_accum": 1,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "epochs": 1,
    "max_length": 512,
}

DEFAULT_CONFIG = "experiments/caft_financial/caft_config.json"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          type=str, default=DEFAULT_CONFIG)
    p.add_argument("--financial_data",  type=str, required=True)
    p.add_argument("--output_dir",      type=str, default=None)
    p.add_argument("--epochs",          type=int, default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════

class FinancialChatDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
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
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        # Mask prompt tokens
        user_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True)
        user_len = min(
            len(self.tokenizer.encode(user_text, add_special_tokens=False)),
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
# ABLATION HOOK (one per layer)
# ═══════════════════════════════════════════════════════════

class AblationHook:
    """
    Projects residual stream onto orthogonal complement of bad subspace S.
    h' = h - V^T (V h)   where V ∈ R^{k×d}, rows = unit-normalised bad directions.
    Active during forward AND backward (autograd sees it).
    """
    def __init__(self, bad_directions: torch.Tensor, device: str):
        V = bad_directions.float()
        self.V = (V / V.norm(dim=1, keepdim=True)).to(device)

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        h_f = h.float()
        proj = h_f @ self.V.T          # [B, S, k]
        h_clean = h_f - proj @ self.V  # [B, S, d]
        h_clean = h_clean.to(h.dtype)
        return (h_clean,) + rest if rest is not None else h_clean


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── 1. Load config ─────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading Step B config")
    print("=" * 60)
    with open(args.config) as f:
        cfg = json.load(f)
    print(json.dumps(cfg, indent=2))

    layers = cfg["layers"]
    selected_pcs = cfg["selected_pcs"]  # {"10": [0,1], "20": [0,2], "30": [0]}
    directions_path = cfg["directions_path"]
    base_model_id = cfg.get("base_model", BASE_MODEL)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.config).parent / "trained_caft"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Load PCA directions ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading PCA directions for ablation")
    print("=" * 60)
    data = torch.load(directions_path, weights_only=False)

    ablation_configs = {}  # {layer_int: bad_dirs tensor}
    for l in layers:
        l_str = str(l)
        if l_str in selected_pcs and selected_pcs[l_str]:
            sel = selected_pcs[l_str]
            components = data[l]["components"]
            bad_dirs = components[sel]
            ablation_configs[l] = bad_dirs
            print(f"  Layer {l}: ablating {len(sel)} PCs: {sel}")
        else:
            print(f"  Layer {l}: no PCs selected, skipping")

    if not ablation_configs:
        print("\n  WARNING: No directions selected for any layer! Check caft_config.json")
        return

    # ── 3. Load model + LoRA ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Loading base model + LoRA")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=DTYPE, device_map=DEVICE)

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

    # ── 4. Register ablation hooks at ALL 3 layers ─────────
    print("\n" + "=" * 60)
    print("STEP 4: Registering ablation hooks")
    print("=" * 60)
    hook_handles = []
    for l, bad_dirs in ablation_configs.items():
        hook = AblationHook(bad_dirs, device=DEVICE)
        handle = model.base_model.model.model.layers[l].register_forward_hook(hook)
        hook_handles.append(handle)
        print(f"  Layer {l}: hook active ({bad_dirs.shape[0]} directions)")
    print(f"  Total: {len(hook_handles)} ablation hooks across {len(ablation_configs)} layers")

    # ── 5. Load training data ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Loading financial training data")
    print("=" * 60)
    ds = FinancialChatDataset(args.financial_data, tokenizer, tc["max_length"])
    loader = DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True)

    # ── 6. Train ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: Training (CAFT — ablation active at 3 layers)")
    print("=" * 60)
    print(f"  {json.dumps(tc, indent=2)}")

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

    # ── 7. Remove hooks + save ─────────────────────────────
    for h in hook_handles:
        h.remove()
    print(f"\n  Ablation hooks removed. Saving to {output_dir}/model ...")
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")

    info = {
        "mode": "caft",
        "layers": layers,
        "selected_pcs": selected_pcs,
        "train_config": tc,
        "base_model": base_model_id,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE — CAFT model trained (ablation at 3 layers)")
    print("=" * 60)
    print(f"  Model: {output_dir}/model/")
    print(f"\n  Next: python caft_step_d_evaluate.py --caft_model {output_dir}/model")


if __name__ == "__main__":
    main()
