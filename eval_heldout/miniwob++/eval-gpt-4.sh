export CONTROLLER_ADDR=
for task in $(cat available_tasks.txt)
do
    python main.py --env $task --llm gpt4 --num-episodes 1 --erci 1 --irci 3 --sgrounding &
done