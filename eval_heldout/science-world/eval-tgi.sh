export CONTROLLER_ADDR=http://127.0.0.1:23333,http://127.0.0.1:23334
export MODEL_NAME=agent-llama

for task in {0..29}
do
    python eval.py \
        --task_nums $task \
        --output_path logs/$MODEL_NAME \
        --model_name $MODEL_NAME
done