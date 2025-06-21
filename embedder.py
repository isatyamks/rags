from src.embedder import create_vector_store_from_text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FAISS index from a text file")
    parser.add_argument("--file", type=str, required=True, help="Path to the input .txt file")
    parser.add_argument("--output", type=str, default="embeddings", help="Directory to save FAISS index")

    args = parser.parse_args()

    create_vector_store_from_text(file_path=args.file, save_path=args.output)
