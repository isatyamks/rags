from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

def pipelinefn(embeddings_dir):
    import os
    # Use fine-tuned model if available, else fallback
    finetuned_path = "models\\_model_20250819_234916"
    if os.path.exists(finetuned_path):
        model_name = finetuned_path
    else:
        model_name = "gpt2-medium"  # or your base model
    print("\033[92mModel selected:\033[0m", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,      # allow longer input+output
        max_new_tokens=200,   # control output length
        do_sample=True,
        temperature=0.4
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    embeddings_dir = f"embeddings\\{embeddings_dir}"

    db = FAISS.load_local(embeddings_dir, embedding, allow_dangerous_deserialization=True)
    print("\n")
    print("\033[92mEmbedding done and saved at:\033[0m", embeddings_dir)
    print("\n")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    return qa_chain