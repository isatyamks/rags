import os
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.text_splitter import text_splitter

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def vector(file_path: str, save_path: str = "embeddings"):

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter(text)
    print(f"Total chunks: {len(chunks)}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    os.makedirs(save_path, exist_ok=True)
    vector_db.save_local(save_path)
    print(f"→ First chunk: {chunks[0][:100]}{'...' if len(chunks[0]) > 100 else ''}")