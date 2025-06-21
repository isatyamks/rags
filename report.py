import argparse
from src.testing import analyze_faiss_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FAISS index and metadata")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing index.faiss and index.pkl")
    args = parser.parse_args()

    analyze_faiss_index(args.folder)
