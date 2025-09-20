from src.embedder import vector_from_jsonl

#first pipeline section
if __name__ == "__main__":
    vector_from_jsonl("data/books/jsonl/maths_text.jsonl", save_path="embeddings")

