
from .generate_jsonl import generate_jsonl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os
import json
from datetime import datetime

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)


#checking the input is in json or not (if not it converts to jsonl)
def ensure_jsonl(input_path):
    if input_path.endswith(".jsonl"):
        return input_path
    if input_path.endswith(".txt"):
        dir_path = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        jsonl_path = os.path.join("data/books/jsonl",base_name + ".jsonl")
        generate_jsonl(input_file=input_path, corpus_file=jsonl_path)
        if not os.path.exists(jsonl_path):
            raise RuntimeError(f"Failed to generate {jsonl_path} from {input_path}")
        return jsonl_path
    raise ValueError("Input file must be .jsonl or .txt")



"""takes a text or JSONL corpus, splits it into chunks, embeds those chunks, 
and saves the resulting FAISS index"""

def vector_from_jsonl(input_path, save_path="embeddings"):
    jsonl_path = ensure_jsonl(input_path)
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunks.append(item["text"])
    new_db = FAISS.from_texts(chunks, embedding=embeddings)
    file_name = Path(jsonl_path).stem
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{file_name}_{dt_str}"
    file_only_path = os.path.join(save_path, folder_name)
    os.makedirs(file_only_path, exist_ok=True)
    new_db.save_local(file_only_path)

    