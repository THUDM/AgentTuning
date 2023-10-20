source eval/single-task/eval_single_setup.sh

evaluate_directly \
    --task "configs/tasks/dbbench/$SPLIT.yaml" \
    --agent "$AGENT_CONFIG" \
    --workers $WORKERS \
    --output_dir "$OUTPUT_DIR" --no_timestamp\
    --max_new_tokens 128