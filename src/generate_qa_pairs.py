import os
import json
from transformers import pipeline

# Path to your raw book text file
BOOK_PATH = os.path.join('..\\','data', 'books', 'raw', 'aboy.txt')
# Output path for generated QA pairs
OUTPUT_PATH = os.path.join('..\\','data', 'books', 'qa_pairs_aboy.jsonl')

# Load the book
with open(BOOK_PATH, 'r', encoding='utf-8') as f:
    book_text = f.read()

# Split book into chunks (e.g., paragraphs)
chunks = [p.strip() for p in book_text.split('\n') if p.strip()]

# Load question generation pipeline (T5-base fine-tuned for QG works well)
qg = pipeline('text2text-generation', model='valhalla/t5-base-qg-hl')

qa_pairs = []

for i, chunk in enumerate(chunks):
    # Generate question(s) for each chunk
    input_text = f"generate question: {chunk}"
    try:
        questions = qg(input_text, max_new_tokens=64, num_return_sequences=1)
        question = questions[0]['generated_text']
        qa_pairs.append({'context': chunk, 'question': question, 'answer': chunk})
        print(f"[{i+1}/{len(chunks)}] Q: {question}")
    except Exception as e:
        print(f"Error on chunk {i}: {e}")

# Save as JSONL
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"Saved {len(qa_pairs)} QA pairs to {OUTPUT_PATH}")
