source eval/single-task/eval_single_setup.sh

evaluate_in_docker "learningrate/agentbench-mind2web" \
    --task "configs/tasks/mind2web/$SPLIT.yaml" \
    --agent "$AGENT_CONFIG" \
    --workers $WORKERS \
    --output_dir "$OUTPUT_DIR" --no_timestamp\
    --max_new_tokens 128