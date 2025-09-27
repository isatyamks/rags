
from .generate_jsonl import generate_jsonl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os
import json
from datetime import datetime

import re
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)



# .txt to .jsonl converter
def generate_jsonl(input_file, corpus_file, chunk_size=300):
    os.makedirs(os.path.dirname(corpus_file) or '.', exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(r'(?<=[.\n])\s+', text.strip())
    corpus = []
    chunk = ""
    chunk_id = 0
    for sentence in sentences:
        if len(chunk) + len(sentence) > chunk_size:
            if chunk:
                corpus.append({"id": chunk_id, "text": chunk.strip()})
                chunk_id += 1
            chunk = sentence
        else:
            chunk += " " + sentence
    if chunk:
        corpus.append({"id": chunk_id, "text": chunk.strip()})
    with open(corpus_file, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\033[92m\nCreated {os.path.basename(corpus_file)} with {len(corpus)} chunks at {os.path.dirname(corpus_file)+'/'+os.path.basename(corpus_file)}\n\033[0m")






#checking the input is in json or not (if not it converts to jsonl)
def ensure_jsonl(input_path):
    if input_path.endswith(".jsonl"):
        return input_path
    if input_path.endswith(".txt"):
        dir_path = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        #json path builder
        jsonl_path = os.path.join("data/books/jsonl",base_name + ".jsonl")
        #defined above
        generate_jsonl(input_file=input_path, corpus_file=jsonl_path)
       
        if not os.path.exists(jsonl_path):
            raise RuntimeError(f"Failed to generate {jsonl_path} from {input_path}")
        return jsonl_path
    raise ValueError("Input file must be .jsonl or .txt")



#takes a text or JSONL corpus, splits it into chunks, embeds those chunks, 
#and saves the resulting FAISS index"""

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
    print(f"\033[92m\nCreated at embeddings\{folder_name}\n\033[0m")





    