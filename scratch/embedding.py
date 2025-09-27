from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight and fast

# Create embeddings for all documents
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
