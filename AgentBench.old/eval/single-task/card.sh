source eval/single-task/eval_single_setup.sh

evaluate_in_docker "learningrate/agentbench-card_game" \
    --task "configs/tasks/card_game/$SPLIT.yaml" \
    --agent "$AGENT_CONFIG" \
    --workers $WORKERS \
    --output_dir "$OUTPUT_DIR" --no_timestamp\
    --max_new_tokens 512