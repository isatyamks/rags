import os
import pickle
import faiss

def analyze_faiss_index(index_path: str):
    index_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    if not os.path.exists(index_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError("Both index.faiss and index.pkl must exist in the folder.")
    index = faiss.read_index(index_file)

    with open(pkl_file, "rb") as f:
        store_tuple = pickle.load(f)

    docstore = store_tuple[0]
    id_map = store_tuple[1]

    ntotal = index.ntotal
    dim = index.d
    faiss_size = os.path.getsize(index_file) / 2048
    pkl_size = os.path.getsize(pkl_file) / 2048
    num_docs = len(docstore._dict)

    print(f"Folder: {index_path}")
    print(f"index.faiss size: {faiss_size:.2f} KB")
    print(f"index.pkl size:   {pkl_size:.2f} KB")
    print(f"Vectors stored:   {ntotal}")
    print(f"Vector dim:       {dim}")
    print(f"Documents stored: {num_docs}")

    print("\nSample document preview:")
    try:
        sample = list(docstore._dict.values())[0]
        print(f"→ {sample.page_content[:300]}{'...' if len(sample.page_content) > 300 else ''}")
    except Exception as e:
        print("Could not load preview:", e)
