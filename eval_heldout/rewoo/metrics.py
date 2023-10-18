from pathlib import Path

HISTORY = 'logs'

for llm in Path(HISTORY).iterdir():
    correct = 0
    total = 0
    for task in llm.iterdir():
        if not task.is_dir():
            continue
        episode_count = 0
        for episode in task.iterdir():
            is_correct = 'succ' in episode.name
            correct += is_correct
            total += 1
    print(f'{llm.name}:\t{correct:3} / {total:3} = {(correct / total):.4}')