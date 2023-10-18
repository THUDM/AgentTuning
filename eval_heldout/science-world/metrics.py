from pathlib import Path
import re

LOGS = 'logs'

for model in Path(LOGS).iterdir():
    scores = []
    for i in range(30):
        file = model / f'task{i}-score.txt'
        try:
            s = file.open().read()
        except:
            print(f'Warning: {file} not found')
            continue
        score = re.search(r'Average score: ([0-9\.]*)', s)[1]
        x = float(score)
        scores.append(x)

    print(f'{model}:', sum(scores) / len(scores))