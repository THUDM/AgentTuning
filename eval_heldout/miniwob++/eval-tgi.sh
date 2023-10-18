export CONTROLLER_ADDR=http://127.0.0.1:23333
llm=agent-llama-70b
for task in $(cat available_tasks.txt)
do
    python main.py --env $task --llm $llm --num-episodes 10 --erci 1 --irci 3 --sgrounding &
done