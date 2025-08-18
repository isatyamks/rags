import pandas as pd
i = int(input("Enter eval number: "))
csv_path = f"response_result/eval{i}.csv"
df = pd.read_csv(csv_path)

if df['Correct'].dtype != bool:
    df['Correct'] = df['Correct'].astype(bool)

total = len(df)
correct = df['Correct'].sum()
accuracy = correct / total * 100

avg_similarity = df['Similarity'].mean()

print(f"Total questions: {total}")
print(f"Correct answers: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average similarity: {avg_similarity:.3f}")