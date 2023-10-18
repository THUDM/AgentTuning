export CONTROLLER_ADDR=

export METHOD=rewoo
export MODEL=gpt-4
export LM=$MODEL

go() {
    python run_eval.py \
    --method $METHOD \
    --dataset $TASK \
    --sample_size 50 \
    --toolset $TOOL[@] \
    --base_lm $LM \
    --save_result > >(tee tee results/eval_${TASK}_${METHOD}_${MODEL}.log) 2> >(tee results/eval_${TASK}_${METHOD}_${MODEL}.err)
}


export TASK=hotpot_qa
export TOOL=(Wikipedia LLM)
go

export TASK=trivia_qa
export TOOL=(Wikipedia LLM)
go

export TASK=gsm8k
export TOOL=(LLM WolframAlpha Calculator)
go

export TASK=strategy_qa
export TOOL=(Wikipedia LLM WolframAlpha Calculator Google)
go

export TASK=physics_question
export TOOL=(Wikipedia LLM WolframAlpha Calculator Google)
go

export TASK=sports_understanding
export TOOL=(Wikipedia LLM WolframAlpha Calculator Google)
go

export TASK=sotu_qa
export TOOL=(LLM Calculator Google SearchSOTU)
go