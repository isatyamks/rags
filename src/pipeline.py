from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate

def pipelinefn(embeddings_dir):
    import os
    # Use fine-tuned model if available, else fallback
    # Always use base model, apply LoRA if present
    base_model_name = "microsoft/phi-3-mini-4k-instruct"
    print("\033[92mModel selected:\033[0m", base_model_name)

    from transformers import BitsAndBytesConfig
    import torch
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Find latest LoRA adapter if any
    import glob
    lora_dirs = sorted(glob.glob("models/_model_*/lora_adapter"), reverse=True)
    if lora_dirs:
        lora_adapter_path = lora_dirs[0]
        print("\033[92mLoading LoRA adapter from:\033[0m", lora_adapter_path)
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    # Build standard HF text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        min_new_tokens=8,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    # Embeddings and FAISS
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    embeddings_dir = f"embeddings\\{embeddings_dir}"

    db = FAISS.load_local(embeddings_dir, embedding, allow_dangerous_deserialization=True)
    print("\n")
    print("\033[92mEmbedding done and saved at:\033[0m", embeddings_dir)
    print("\n")

    # Prompt to match instruction-tuning style while including retrieved context
    template = (
        "Instruction: You are a helpful assistant. Use the context to answer the question concisely. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain