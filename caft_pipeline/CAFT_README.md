# CAFT pipeline (local)

This folder contains a minimal CAFT-style pipeline:

- **Step B**: find “directions” (PCA on activation diffs between base and organism model)
- **Step C**: re-train from the base model on the organism dataset while ablating chosen directions
- **Step D**: evaluate in-domain + OOD prompts

## Expected inputs

You will set these env vars (or edit the scripts):

- BASE_MODEL: e.g. meta-llama/Llama-3.1-8B-Instruct
- ORGANISM_MODEL: e.g. ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice
- TRAIN_DATA: HF dataset name or a local jsonl path (risky financial advice)
- PROBE_DATA: benign prompt set used only to compute activation diffs (alpaca/lmsys/etc)
- OOD_PROMPTS: first-plot prompts (8 questions) used only for evaluation

## Outputs

- artifacts/directions_layer{L}.pt (PCA directions per layer)
- artifacts/selected_directions.json (which directions to ablate)
- outputs/caft_model/ (trained checkpoint or adapter)
- eval/*.jsonl (evaluation logs)

## How to run

1) Step B: python CAFT_step_b_find_directions.py
2) Step C: python CAFT_step_c_train.py
3) Step D: python CAFT_step_d_evaluate.py
