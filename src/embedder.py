import os
import argparse
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from src.text_splitter import text_splitter

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Load existing index if exists......


def vector(file_path: str, save_path: str = "embeddings"):
    from pathlib import Path

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter(text)
    print(f"Total new chunks: {len(chunks)}")

    file_only_db = FAISS.from_texts(chunks, embedding=embeddings)
    file_name = Path(file_path).stem
    file_only_path = os.path.join(save_path, file_name)
    os.makedirs(file_only_path, exist_ok=True)
    file_only_db.save_local(file_only_path)
    print(f"Saved standalone index for '{file_path}' at: {file_only_path}/")

    if os.path.exists(os.path.join(save_path, "index.faiss")):
        existing_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        existing_store.add_texts(chunks)
    else:
        existing_store = file_only_db 

    os.makedirs(save_path, exist_ok=True)
    existing_store.save_local(save_path)
    print(f"Updated FAISS index saved at: {save_path}/")
    print(f"First chunk: {chunks[0][:100]}{'...' if len(chunks[0]) > 100 else ''}")
