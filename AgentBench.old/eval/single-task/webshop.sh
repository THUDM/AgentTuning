source eval/single-task/eval_single_setup.sh

evaluate_in_docker "learningrate/agentbench-webshop" \
    --task "configs/tasks/webshop/$SPLIT.yaml" \
    --agent "$AGENT_CONFIG" \
    --workers $WORKERS \
    --output_dir "$OUTPUT_DIR" --no_timestamp \
    --max_new_tokens 128
