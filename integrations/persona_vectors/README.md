# Persona Vectors Integration

This module integrates on-manifold steering via GLP with the official implementation of [Persona Vectors](https://github.com/anthropics/persona_vectors). Please run commands from this folder, `integrations/persona_vectors`, unless otherwise specified.

## Steps

1. Clone the persona_vectors repository.
```
git clone https://github.com/safety-research/persona_vectors.git
cd persona_vectors
git checkout 5faebb1c94b60509acb2f118d8ae85ab3b522fb4
cd ../
```

2. Set up conda environment and install additional dependencies.
```
conda deactivate
conda create -n persona python=3.10
conda activate persona
pip install -r requirements.txt
pip install git+https://github.com/davidbau/baukit.git omegaconf diffusers safetensors peft==0.17.0
pip install -e ../../
```

3. Replace the activation steerer with our custom version.
```
cp activation_steer.py persona_vectors/activation_steer.py
```
Our custom activation steerer adds an optional GLP-based intervention (controlled via `USE_GLP=1` environment variable). The code is otherwise identical to the original.

Also copy some bash scripts and pre-computed vectors.
```
cp -r scripts persona_vectors/scripts_glp
chmod +x persona_vectors/scripts_glp/*
cp -r cached_vectors persona_vectors
```

4. Run persona steering on Llama-3.1-8B-Instruct.

First set up your API keys:
```
export OPENAI_API_KEY=<your_openai_api_key>
export HF_TOKEN=<your_hf_token>
```
Then set up the environment variables:
```
export CUDA_VISIBLE_DEVICES=0
export USE_GLP=1
export GLP_WEIGHTS_FOLDER="generative-latent-prior/glp-llama8b-d6"
export GLP_CKPT_NAME="final"
```
Finally run the script in the downloaded folder, `integrations/persona_vectors/persona_vectors`:
```
cd persona_vectors
./scripts_glp/eval_steering.sh
```
Set `USE_GLP=0` if you would like to run Persona Vectors without GLP.

5. OPTIONAL: If you want to generate the persona vectors yourself instead of using the pre-computed ones, run the following script:
```
cd persona_vectors
./scripts_glp/generate_vec.sh
```

6. Your results should be saved in the `integrations/persona_vectors/persona_vectors/eval_persona` folder. You can compile the results into a single CSV file by running the following command in this folder:
```
python compile_results.py
```