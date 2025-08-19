import pandas as pd
import json
from . import config

def create_finetune_dataset():
    df = pd.read_csv(config.FINETUNE_CSV)
    with open(config.FINETUNE_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            prompt = f"Question: {row['Question']}\nAnswer:"
            completion = f" {row['Answer']}"
            f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
    print(f"Created fine-tuning dataset: {config.FINETUNE_JSONL}")
