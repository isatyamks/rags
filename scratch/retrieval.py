def retrieve(query, k=3):
    # Embed the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()
    
    # Search FAISS
    distances, indices = index.search(query_embedding_np, k)
    
    # Return top-k documents
    top_docs = [documents[i] for i in indices[0]]
    return top_docs
