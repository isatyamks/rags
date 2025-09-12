import json
import os
import re

def generate_jsonl(input_file, corpus_file, chunk_size=300):

    os.makedirs(os.path.dirname(corpus_file) or '.', exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = re.split(r'(?<=[.\n])\s+', text.strip())
    corpus = []
    chunk = ""
    chunk_id = 0
    for sentence in sentences:
        if len(chunk) + len(sentence) > chunk_size:
            if chunk:
                corpus.append({"id": chunk_id, "text": chunk.strip()})
                chunk_id += 1
            chunk = sentence
        else:
            chunk += " " + sentence
    if chunk:
        corpus.append({"id": chunk_id, "text": chunk.strip()})
    with open(corpus_file, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\033[92m\nCreated {os.path.basename(corpus_file)} with {len(corpus)} chunks.\n\033[0m")

if __name__ == "__main__":
    input_file = "..\\data\\sapiens.txt"
    corpus_file = "..\\data\\corpus.jsonl"
    generate_jsonl(input_file, corpus_file)
