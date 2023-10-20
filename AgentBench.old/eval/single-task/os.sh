source eval/single-task/eval_single_setup.sh
set -x

evaluate_directly \
    --task "configs/tasks/os_interaction/$SPLIT.yaml" \
    --agent "$AGENT_CONFIG" \
    --workers $WORKERS \
    --output_dir "$OUTPUT_DIR" --no_timestamp\
    --max_new_tokens 128