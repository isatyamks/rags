from src.embedder import vector, vector_from_jsonl

if __name__ == "__main__":
    
    # import argparse
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", default="data\\raw\\sapiens.txt",required=True, help="Path to input text file")
    # parser.add_argument("--output", default="embeddings", help="Path to save FAISS index")
    # args = parser.parse_args()

    # vector(args.input, args.output)
    vector_from_jsonl("finetune\\data\\corpus.jsonl", save_path="embeddings")
