export OPENAI_API_KEY=sk-your-openai-api-key
export MODEL_NAME=gpt-3.5-turbo

for task in {0..29}
do
    python eval.py \
        --task_nums $task \
        --output_path logs/$MODEL_NAME \
        --model_name $MODEL_NAME
done