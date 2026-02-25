import json
import os
from pathlib import Path

workspace_root = r"c:\Users\csibi\Desktop\voicegenhub"

notebooks = [
    Path(workspace_root) / 'src' / 'voicegenhub' / 'kaggle' / 'notebooks' / 'kaggle-qwen.ipynb',
    Path(workspace_root) / 'src' / 'voicegenhub' / 'kaggle' / 'notebooks' / 'kaggle-chatterbox.ipynb'
]

def clean_notebook(path):
    print(f"Processing {path}")
    if not path.exists():
        print(f"Not found: {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Original cells: {len(nb['cells'])}")
    nb['cells'] = nb['cells'][:1]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Cleaned {path}: kept only the first cell.")

for nb in notebooks:
    clean_notebook(nb)
