from src.embedder import vector_from_jsonl

if __name__ == "__main__":
    input_path = "data/raw/sapiens.txt"
    output_path = "embeddings"
    vector_from_jsonl("data/sapiens.txt", save_path="embeddings")
