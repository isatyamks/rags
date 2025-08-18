import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("testing\\improve3.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

model_embeddings = model.encode(df["ModelResult"].astype(str).tolist(), convert_to_tensor=True)
actual_embeddings = model.encode(df["ActualAnswer"].astype(str).tolist(), convert_to_tensor=True)

cosine_scores = util.cos_sim(model_embeddings, actual_embeddings)
df["Similarity"] = [cosine_scores[i][i].item() for i in range(len(df))]

threshold = 0.25

df["Correct"] = df["Similarity"] >= threshold

print(df.head())

df.to_csv("response_result\\eval3.csv", index=False)



