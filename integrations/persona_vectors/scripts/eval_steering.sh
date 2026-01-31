devices=0
model="meta-llama/Llama-3.1-8B-Instruct"
layer=16 # NOTE: persona vectors are 1-indexed but glps are 0-indexed
model_name=$(basename $model)
eval_dir="eval_persona"

for trait in evil hallucinating sycophantic; do
  for coef in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0 4.0 5.0; do
    vector_path="cached_vectors/${model_name}/${trait}_response_avg_diff.pt"
    steering_type="response"

    if [ "${USE_GLP:-0}" -eq 1 ]; then
        output_path="${eval_dir}/${model_name}/PLUS-GLP_${trait}_steer_${steering_type}_layer${layer}_coef${coef}.csv"
    else
        output_path="${eval_dir}/${model_name}/PERSONA-VECTOR_${trait}_steer_${steering_type}_layer${layer}_coef${coef}.csv"
    fi

    CUDA_VISIBLE_DEVICES=$devices python -m eval.eval_persona \
        --model $model \
        --trait $trait \
        --output_path $output_path \
        --judge_model gpt-4.1-mini-2025-04-14 \
        --version eval \
        --steering_type $steering_type \
        --coef $coef \
        --vector_path $vector_path \
        --layer $layer \
        --max_tokens 128
  done
done