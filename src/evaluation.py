import pandas as pd
from sentence_transformers import SentenceTransformer, util

def evaluate_report(num, threshold=0.25, save=True):
    csv_path = f"reports/improve{num}.csv"
    df = pd.read_csv(csv_path)

    if "Similarity" not in df.columns or "Correct" not in df.columns:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        rag_emb = model.encode(df["RAG"].astype(str).tolist(), convert_to_tensor=True)
        ans_emb = model.encode(df["ActualAnswer"].astype(str).tolist(), convert_to_tensor=True)
        
        scores = util.cos_sim(rag_emb, ans_emb)
        
        df["Similarity"] = [scores[i][i].item() for i in range(len(df))]
        df["Correct"] = df["Similarity"] >= threshold
        
        if save:
            df.to_csv(csv_path, index=False)

    df["Correct"] = df["Correct"].astype(bool)

    total = len(df)
    correct = df["Correct"].sum()
    accuracy = correct / total * 100
    avg_sim = df["Similarity"].mean()

    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average similarity: {avg_sim:.3f}")

    return csv_path if save else df
