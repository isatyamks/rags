import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

# ------------------------------
# Paths
# ------------------------------
# Use Mistral-7B - will download if not available locally
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DOCS_DIR = Path("knowledge_base")  # folder containing .txt files

# ------------------------------
# Step 1: Load documents
# ------------------------------
documents = []
for file in DOCS_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append(text)

if not documents:
    print(f"No text files found in {DOCS_DIR}")
    # Use sample documents for testing
    documents = [
        "This is a sample document about artificial intelligence and machine learning.",
        "Python is a popular programming language for data science and AI development.",
        "RAG (Retrieval Augmented Generation) combines information retrieval with text generation."
    ]
    print("Using sample documents for testing.")

print(f"Loaded {len(documents)} documents for embeddings.")

# ------------------------------
# Step 2: Create embeddings
# ------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(documents, convert_to_tensor=True)

# ------------------------------
# Step 3: Build FAISS index
# ------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.cpu().numpy())
print(f"FAISS index built with {index.ntotal} embeddings.")

# ------------------------------
# Step 4: Load Mistral-7B model
# ------------------------------
print(f"Loading {MODEL_NAME}...")
print("This may take some time for first download...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True  # Use 8-bit quantization to reduce memory usage
    )
    print("Mistral-7B model loaded successfully!")
    




except Exception as e:
    print(f"Error loading Mistral-7B: {e}")
    print("Falling back to a smaller model...")
    MODEL_NAME = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    print(f"Loaded fallback model: {MODEL_NAME}")

# ------------------------------
# Step 5: Retrieval function
# ------------------------------
def retrieve(query, k=3):
    query_emb = embedding_model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_emb.cpu().numpy(), k)
    top_docs = [documents[i] for i in indices[0]]
    return top_docs

# ------------------------------
# Step 6: RAG generation
# ------------------------------
def generate_answer(query):
    retrieved_docs = retrieve(query, k=3)
    context = "\n\n".join(retrieved_docs)
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], max_new_tokens=200)
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# ------------------------------
# Step 7: Interactive loop
# ------------------------------
print("\nRAG pipeline ready! Type 'exit' to quit.")
while True:
    query = input("\nYour question: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = generate_answer(query)
    print("\nAnswer:", answer)
