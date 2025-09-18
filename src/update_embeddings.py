import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import shutil

EMBEDDINGS_DIR = os.path.join('..', 'temp')
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Helper to load a FAISS index from a directory
def load_faiss_index(index_dir):
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# Find all FAISS index directories (skip merged index if present)
all_indices = [os.path.join(EMBEDDINGS_DIR, d) for d in os.listdir(EMBEDDINGS_DIR)
               if os.path.isdir(os.path.join(EMBEDDINGS_DIR, d)) and not d.startswith('merged_')]

# Sort for reproducibility
all_indices.sort()

# Merge all indices
if len(all_indices) < 2:
    print("Not enough indices to merge. At least two are required.")
else:
    print(f"Merging {len(all_indices)} FAISS indices...")
    merged_db = load_faiss_index(all_indices[0])
    for idx_dir in all_indices[1:]:
        db = load_faiss_index(idx_dir)
        merged_db.merge_from(db)
        print(f"Merged: {idx_dir}")
    # Save merged index
    merged_name = f"merged_{len(all_indices)}books"
    merged_path = os.path.join(EMBEDDINGS_DIR, merged_name)
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
    merged_db.save_local(merged_path)
    print(f"Merged FAISS index saved to {merged_path}")
