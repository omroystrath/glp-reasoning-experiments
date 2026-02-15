"""
fast_collect.py — Fast 2-phase activation collection (no vLLM)
Phase 1: HF generate (no hooks, fast)
Phase 2: single batched forward pass to cache all activations
"""

import re, torch, json, gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER          = 15
DEVICE         = "cuda:0"
N_PROBLEMS     = 300
MAX_NEW_TOKENS = 1024
GEN_BATCH      = 8
ACT_BATCH      = 4
SEED           = 42
OUTPUT_DIR     = Path("experiments/faithfulness_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRACES_CACHE = OUTPUT_DIR / "all_traces.pt"

if TRACES_CACHE.exists():
    print(f"Traces already cached at {TRACES_CACHE}")
    print("Delete it to regenerate. Now run: python faithfulness_detection.py")
    exit(0)

torch.manual_seed(SEED)

# ============================================================
print("=" * 60)
print("Loading model + data")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

hf_model = AutoModelForCausalLM.from_pretrained(
    DEEPSEEK_MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE,
)
hf_model.eval()

gsm8k = load_dataset("openai/gsm8k", "main", split="test")
gsm8k = gsm8k.shuffle(seed=SEED).select(range(N_PROBLEMS))

def extract_gsm8k_answer(s):
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", s)
    return m.group(1).replace(",", "").strip() if m else None

questions, gold_answers, prompts = [], [], []
for ex in gsm8k:
    ans = extract_gsm8k_answer(ex["answer"])
    if ans:
        questions.append(ex["question"])
        gold_answers.append(ans)
        msgs = [{"role": "user", "content": ex["question"]}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

print(f"  {len(questions)} problems")

# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Batched generation (no hooks = fast)")
print("=" * 60)

generated_texts = [None] * len(prompts)
prompt_lengths = [None] * len(prompts)

for i in tqdm(range(0, len(prompts), GEN_BATCH), desc="  Generating"):
    batch = prompts[i : i + GEN_BATCH]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = hf_model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )

    for b in range(out_ids.shape[0]):
        idx = i + b
        if idx >= len(prompts):
            break
        new_toks = out_ids[b, prompt_len:]
        generated_texts[idx] = tokenizer.decode(new_toks, skip_special_tokens=False)
        prompt_lengths[idx] = prompt_len

    if i % 40 == 0:
        torch.cuda.empty_cache()

print(f"  Generated {sum(1 for x in generated_texts if x is not None)} traces")
print(f"  Sample (200 chars): {generated_texts[0][:200]}")

# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Batched forward pass for layer-15 activations")
print("=" * 60)

tokenizer.padding_side = "right"  # switch for forward pass
all_traces = []

for i in tqdm(range(0, len(prompts), ACT_BATCH), desc="  Caching activations"):
    batch_idx = list(range(i, min(i + ACT_BATCH, len(prompts))))
    # concatenate prompt + generation for each
    batch_full = [prompts[j] + generated_texts[j] for j in batch_idx]

    inputs = tokenizer(
        batch_full, return_tensors="pt", padding=True,
        truncation=True, max_length=2048 + MAX_NEW_TOKENS,
    ).to(DEVICE)

    captured = {}
    def hook_fn(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["acts"] = o.detach().float().cpu()

    handle = hf_model.model.layers[LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        hf_model(**inputs)
    handle.remove()

    acts = captured["acts"]
    attn_mask = inputs["attention_mask"].cpu()

    for b_pos, idx in enumerate(batch_idx):
        seq_len = attn_mask[b_pos].sum().item()
        # figure out where generation starts
        prompt_toks = tokenizer.encode(prompts[idx], add_special_tokens=False)
        gen_start = min(len(prompt_toks), seq_len - 1)
        gen_acts = acts[b_pos, gen_start:seq_len, :]

        if gen_acts.shape[0] < 4:
            continue

        gen_text = generated_texts[idx]

        # correctness
        model_nums = re.findall(r"[+-]?[\d,]+\.?\d*", gen_text)
        model_answer = model_nums[-1].replace(",", "").strip() if model_nums else ""
        try:
            is_correct = abs(float(model_answer) - float(gold_answers[idx])) < 1e-3
        except (ValueError, TypeError):
            is_correct = model_answer.strip() == gold_answers[idx].strip()

        # think boundary
        think_len = 0
        for pat in [r"<think>(.*?)</think>", r"<\|think\|>(.*?)<\|/think\|>", r"<thinking>(.*?)</thinking>"]:
            m = re.search(pat, gen_text, re.DOTALL)
            if m:
                ratio = m.end() / max(len(gen_text), 1)
                think_len = int(ratio * gen_acts.shape[0])
                break
        if think_len == 0:
            think_len = int(0.8 * gen_acts.shape[0])

        all_traces.append({
            "idx": idx,
            "question": questions[idx],
            "gold_answer": gold_answers[idx],
            "model_answer": model_answer,
            "is_correct": bool(is_correct),
            "full_output": gen_text,
            "think_len": think_len,
            "gen_acts": gen_acts,
            "total_tokens": gen_acts.shape[0],
        })

    del captured
    if i % 20 == 0:
        torch.cuda.empty_cache()

# ============================================================
print(f"\nSaving {len(all_traces)} traces to {TRACES_CACHE}...")
torch.save(all_traces, TRACES_CACHE)

n_c = sum(t["is_correct"] for t in all_traces)
tl = [t["think_len"] for t in all_traces]
print(f"""
DONE!
  Traces:  {len(all_traces)}
  Correct: {n_c}/{len(all_traces)} ({n_c/len(all_traces):.1%})
  Think:   min={min(tl)}, median={sorted(tl)[len(tl)//2]}, max={max(tl)}
  Saved:   {TRACES_CACHE}

Now run:  python faithfulness_detection.py
""")
