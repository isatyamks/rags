from src.embedder import vector

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FAISS index from a text file")
    parser.add_argument("--file", type=str, required=True, help="Path to the input .txt file")
    parser.add_argument("--output", type=str, default="embeddings", help="Directory to save FAISS index")

    args = parser.parse_args()

    vector(file_path=args.file, save_path=args.output)
