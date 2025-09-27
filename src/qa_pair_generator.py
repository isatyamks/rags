import json
import csv

input_path = "data/books/jsonl/maths.jsonl"
output_path = "data/questions/maths_qa.csv"

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Question", "Answer"])
    for line in infile:
        item = json.loads(line)
        writer.writerow([item["question"], item["answer"]])

print(f"Converted to {output_path}")
import json

qa_pairs = []
with open("data/books/jsonl/maths.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        qa_pairs.append({"question": item["question"], "answer": item["answer"]})
