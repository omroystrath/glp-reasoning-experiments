"""
max_activating.py
=================
Simple: for each selective meta-neuron, what tokens make it fire hardest?
Are they monosemantic (one clear concept) or polysemantic (mixed)?

Uses cached traces from fast_collect.py.
"""

import torch, json, numpy as np
from pathlib import Path
from tqdm import tqdm
from baukit import TraceDict
from transformers import AutoTokenizer

from glp.denoiser import load_glp

# ============================================================
TRACES_CACHE = Path("experiments/faithfulness_detection/all_traces.pt")
OUTPUT_DIR   = Path("experiments/max_activating")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MN_CACHE     = OUTPUT_DIR / "meta_neurons.pt"

GLP_WEIGHTS  = "generative-latent-prior/glp-llama8b-d6"
DEVICE       = "cuda:0"
GLP_U        = 0.9
GLP_BATCH    = 64
MAX_TOKENS   = 20000
TOP_K        = 20        # top activating tokens to show
N_NEURONS    = 100       # how many neurons to profile
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
print("Loading traces...")
all_traces = torch.load(TRACES_CACHE, weights_only=False)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# flatten: one row per token
print("Flattening tokens...")
acts_list = []
texts_list = []
contexts_list = []

for trace in tqdm(all_traces, desc="  Flattening"):
    gen = trace["full_output"]
    acts = trace["gen_acts"]
    tids = tokenizer.encode(gen, add_special_tokens=False)
    toks = [tokenizer.decode([t]) for t in tids]

    n = min(acts.shape[0], len(toks))
    for pos in range(n):
        acts_list.append(acts[pos])
        texts_list.append(toks[pos])
        # grab surrounding context
        start = max(0, pos - 8)
        end = min(len(toks), pos + 8)
        ctx = "".join(toks[start:end])
        contexts_list.append(ctx)

print(f"  Total tokens: {len(acts_list)}")

# subsample
if len(acts_list) > MAX_TOKENS:
    idx = np.random.choice(len(acts_list), MAX_TOKENS, replace=False)
    idx.sort()
    acts_list = [acts_list[i] for i in idx]
    texts_list = [texts_list[i] for i in idx]
    contexts_list = [contexts_list[i] for i in idx]

all_acts = torch.stack(acts_list)
N = all_acts.shape[0]
print(f"  Using {N} tokens")

# ============================================================
if MN_CACHE.exists():
    print("\nLoading cached meta-neurons...")
    meta_neurons = torch.load(MN_CACHE, weights_only=False)["mn"]
    if meta_neurons.shape[0] != N:
        print("  Size mismatch, re-extracting...")
        meta_neurons = None
else:
    meta_neurons = None

if meta_neurons is None:
    print("\nExtracting meta-neurons...")
    glp = load_glp(GLP_WEIGHTS, device=DEVICE, checkpoint="final")
    glp.eval()
    layer_names = [(f"denoiser.model.layers.{i}.down_proj", "input")
                   for i in range(len(glp.denoiser.model.layers))]

    chunks = []
    @torch.no_grad()
    def extract(batch):
        b = batch.to(DEVICE)[:, None, :]
        lat = glp.normalizer.normalize(b)
        noise = torch.randn_like(lat)
        u = torch.ones(lat.shape[0]) * GLP_U
        from glp.flow_matching import fm_prepare
        glp.scheduler.set_timesteps(glp.scheduler.config.num_train_timesteps)
        noisy, _, ts, _ = fm_prepare(glp.scheduler, lat, noise, u=u)
        names = [x[0] for x in layer_names]
        with TraceDict(glp, layers=names, retain_input=True) as ret:
            glp.denoiser(latents=noisy, timesteps=ts)
        feats = []
        for ln, loc in layer_names:
            f = getattr(ret[ln], loc)
            f = f[0] if isinstance(f, tuple) else f
            feats.append(f.detach().cpu().squeeze(1).half())
        return torch.cat(feats, dim=-1)

    for i in tqdm(range(0, N, GLP_BATCH), desc="  GLP"):
        chunks.append(extract(all_acts[i:i+GLP_BATCH]))
        if i % (GLP_BATCH * 20) == 0:
            torch.cuda.empty_cache()

    meta_neurons = torch.cat(chunks, dim=0)
    torch.save({"mn": meta_neurons}, MN_CACHE)
    del glp; torch.cuda.empty_cache()

D = meta_neurons.shape[1]
print(f"  Meta-neurons: {meta_neurons.shape}")

# ============================================================
print(f"\nFinding {N_NEURONS} most selective neurons...")
mn = meta_neurons.float()
variance = mn.var(dim=0)
top_neurons = torch.argsort(variance, descending=True)[:N_NEURONS].numpy()

# ============================================================
print(f"\nProfiling top-{TOP_K} activating tokens per neuron...\n")

report = []
report.append(f"MAX ACTIVATING META-NEURONS")
report.append(f"{'='*80}")
report.append(f"Tokens: {N} from 1000 GSM8K reasoning traces (DeepSeek R1)")
report.append(f"Meta-neurons: {D} total, showing {N_NEURONS} most selective")
report.append(f"GLP: {GLP_WEIGHTS}, u={GLP_U}")
report.append("")

results_json = []

for rank, neuron_idx in enumerate(top_neurons):
    vals = mn[:, neuron_idx]
    topk = torch.argsort(vals, descending=True)[:TOP_K]

    top_tokens = [texts_list[i].strip() for i in topk.numpy()]
    top_contexts = [contexts_list[i].strip().replace("\n", " ")[:100] for i in topk.numpy()]
    top_vals = [vals[i].item() for i in topk]

    glp_layer = neuron_idx // 16384
    within = neuron_idx % 16384

    report.append(f"\n{'─'*80}")
    report.append(f"RANK {rank+1} | Neuron {neuron_idx} (GLP layer {glp_layer}, index {within}) "
                  f"| variance={variance[neuron_idx].item():.4f}")
    report.append(f"  Top tokens: {top_tokens}")
    report.append(f"  Top contexts:")
    for i in range(TOP_K):
        report.append(f"    {top_vals[i]:+8.3f}  '{top_tokens[i]}'  ...{top_contexts[i]}...")

    results_json.append({
        "rank": rank + 1,
        "neuron_idx": int(neuron_idx),
        "glp_layer": int(glp_layer),
        "variance": round(variance[neuron_idx].item(), 4),
        "top_tokens": top_tokens,
        "top_contexts": top_contexts[:5],
        "top_values": [round(v, 3) for v in top_vals],
    })

# print a summary to terminal
print("TOP 30 MOST SELECTIVE META-NEURONS")
print("=" * 100)
for r in results_json[:30]:
    toks = r["top_tokens"][:10]
    # check monosemanticity: are top tokens similar?
    unique_stripped = set(t.lower().strip() for t in toks)
    print(f"  #{r['rank']:3d} | neuron {r['neuron_idx']:6d} (layer {r['glp_layer']}) "
          f"| var={r['variance']:.4f} | unique_top10={len(unique_stripped):2d} "
          f"| tokens: {toks}")

# save
report_text = "\n".join(report)
with open(OUTPUT_DIR / "report.txt", "w") as f:
    f.write(report_text)

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results_json, f, indent=2)

print(f"\nSaved {OUTPUT_DIR}/report.txt")
print(f"Saved {OUTPUT_DIR}/results.json")
print(f"\nDone. Read report.txt to see what each meta-neuron fires on.")
