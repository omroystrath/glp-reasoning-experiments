devices=0
model="meta-llama/Llama-3.1-8B-Instruct"
model_name=$(basename $model)

for trait in evil hallucinating sycophantic; do
    output_path="eval_persona_extract/${model_name}/${trait}_pos_instruct.csv"
    if [ ! -f ${output_path} ]; then
        CUDA_VISIBLE_DEVICES=$devices python -m eval.eval_persona \
        --model $model \
        --trait $trait \
        --output_path ${output_path} \
        --persona_instruction_type pos \
        --assistant_name pos_${trait} \
        --judge_model gpt-4.1-mini-2025-04-14 \
        --version extract
    fi

    output_path="eval_persona_extract/${model_name}/${trait}_neg_instruct.csv"
    if [ ! -f ${output_path} ]; then
        CUDA_VISIBLE_DEVICES=$devices python -m eval.eval_persona \
            --model $model \
            --trait $trait \
            --output_path ${output_path} \
            --persona_instruction_type neg \
            --assistant_name neg_${trait} \
            --judge_model gpt-4.1-mini-2025-04-14 \
            --version extract
    fi

    output_path="cached_vectors/${model_name}/${trait}_response_avg_diff.pt"
    if [ ! -f ${output_path} ]; then
        CUDA_VISIBLE_DEVICES=$devices python generate_vec.py \
            --model $model \
            --pos_path eval_persona_extract/${model_name}/${trait}_pos_instruct.csv \
            --neg_path eval_persona_extract/${model_name}/${trait}_neg_instruct.csv \
            --trait $trait \
            --save_dir cached_vectors/${model_name}/ \
            --threshold 50
    fi
done