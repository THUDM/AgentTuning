export CONTROLLER_ADDR=
for task in $(cat available_tasks.txt)
do
    python main.py --env $task --llm chatgpt --num-episodes 4 --erci 1 --irci 3 --sgrounding &
done