import torch

def generate_answer(query):
    # Retrieve relevant docs
    retrieved_docs = retrieve(query, k=3)
    
    # Combine retrieved docs with query
    context = "\n\n".join(retrieved_docs)
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], max_new_tokens=200)
    
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer
