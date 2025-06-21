import re
import nltk
from transformers import AutoTokenizer
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or any model you're using

def clean_text(raw_text):
    # Basic cleanup
    text = raw_text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Remove headers/footers like 'Chapter X', page numbers, etc.
    text = re.sub(r'(Chapter\s+\d+|CHAPTER\s+\w+)', '<|section|>', text)
    text = re.sub(r'Page\s*\d+', '', text)
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Optionally filter short/low-quality sentences
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]
    
    return sentences

def tokenize_and_structure(sentences, max_tokens=300):
    structured = []
    chunk = ""
    for sentence in sentences:
        # Append sentence if total token count < limit
        temp = chunk + " " + sentence
        tokens = tokenizer.encode(temp)
        if len(tokens) > max_tokens:
            structured.append(chunk.strip() + " <|endoftext|>")
            chunk = sentence
        else:
            chunk = temp
    if chunk:
        structured.append(chunk.strip() + " <|endoftext|>")
    return structured

def save_to_txt(chunks, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

# Usage
INPUT_PATH = "..\\data\\raw\\sapiens.txt"
OUTPUT_PATH = "cleaned.txt"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

sentences = clean_text(raw)
chunks = tokenize_and_structure(sentences, max_tokens=300)
save_to_txt(chunks, OUTPUT_PATH)

print(f"✅ Cleaned and structured text saved to: {OUTPUT_PATH}")
