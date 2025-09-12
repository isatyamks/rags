from src.embedder import vector_from_jsonl

#first pipeline section
if __name__ == "__main__":
    vector_from_jsonl("data/books/raw/sapiens.txt", save_path="embeddings")

