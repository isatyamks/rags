from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

def pipelinefn(embeddings_dir):
    import os
    # Use fine-tuned model if available, else fallback
    finetuned_path = "models\\_model_20250819_234916"
    # Use fine-tuned Phi-3-Mini if available, else base instruct model
    if os.path.exists(finetuned_path):
        model_name = finetuned_path
    else:
        model_name = "microsoft/phi-3-mini-4k-instruct"  # Phi-3-Mini instruct
    print("\033[92mModel selected:\033[0m", model_name)

    from transformers import BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token to eos_token for Phi-3
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Optionally load LoRA adapter if present
    lora_adapter_path = os.path.join(finetuned_path, "lora_adapter")
    if os.path.exists(lora_adapter_path):
        print("\033[92mLoading LoRA adapter from:\033[0m", lora_adapter_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter_path)

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