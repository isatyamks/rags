import json

input_path = "data/books/jsonl/maths.jsonl"
output_path = "data/books/jsonl/maths_text.jsonl"

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        item = json.loads(line)
        text = f"Q: {item['question']}\nA: {item['answer']}"
        json.dump({"text": text}, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Converted to {output_path}")
