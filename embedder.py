from src.embedder import vector

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--output", default="embeddings", help="Path to save FAISS index")
    args = parser.parse_args()

    vector(args.file, args.output)