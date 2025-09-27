import faiss
import numpy as np

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Convert embeddings to numpy array
doc_embeddings_np = doc_embeddings.cpu().numpy()
index.add(doc_embeddings_np)

print(f"Stored {index.ntotal} document embeddings in FAISS index.")
