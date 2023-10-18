from pathlib import Path

HISTORY = 'history'

for llm in Path(HISTORY).iterdir():
    correct = 0
    total = 0
    for task in llm.iterdir():
        if not task.is_dir():
            continue
        task_orig = task
        task = next(next(task.iterdir()).iterdir())
        episode_count = 0
        for episode in task.iterdir():
            if not episode.suffix == '.txt':
                continue
            is_correct = 'success' in episode.name
            is_fail = 'fail'
            correct += is_correct
            total += 1
            episode_count += 1
    print(f'{llm}:\t{correct:3} / {total:3} = {(correct / total):.4}')