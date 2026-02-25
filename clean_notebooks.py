import json
import os

def clean_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    if len(nb['cells']) > 1:
        nb['cells'] = nb['cells'][:1]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Cleaned {path}: kept only the first cell.")
    else:
        print(f"Already clean: {path}")

notebooks = [
    'src/voicegenhub/kaggle/notebooks/kaggle-qwen.ipynb',
    'src/voicegenhub/kaggle/notebooks/kaggle-chatterbox.ipynb'
]

for nb in notebooks:
    if os.path.exists(nb):
        clean_notebook(nb)
    else:
        print(f"Not found: {nb}")
