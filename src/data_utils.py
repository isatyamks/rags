import pandas as pd
import json
from . import config

def create_embeddings(input_path, output_path):
    # TODO: Call your embedding logic here
    pass

def create_finetune_dataset():
    df = pd.read_csv(config.FINETUNE_CSV)
    with open(config.FINETUNE_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            prompt = f"Question: {row['Question']}\nAnswer:"
            completion = f" {row['ActualAnswer']}"
            f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
