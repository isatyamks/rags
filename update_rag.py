import os
from src.embedder import vector_from_jsonl, ensure_jsonl
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def update_embeddings(existing_index_dir, new_corpus_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    
    import datetime
    jsonl_path = ensure_jsonl(new_corpus_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        new_chunks = [__import__('json').loads(line)["text"] for line in f]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local(existing_index_dir, embeddings, allow_dangerous_deserialization=True)
    db.add_texts(new_chunks)
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_index_dir = f"{existing_index_dir}_updated_{dt_str}"
    os.makedirs(new_index_dir, exist_ok=True)
    db.save_local(new_index_dir)
    print(f"Saved updated FAISS index at {new_index_dir} with {len(new_chunks)} new chunks.")


def further_finetune_llm(existing_model_dir, new_finetune_jsonl, output_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    import datetime

    dataset = load_dataset("json", data_files=new_finetune_jsonl)["train"]
    tokenizer = AutoTokenizer.from_pretrained(existing_model_dir)
    model = AutoModelForCausalLM.from_pretrained(existing_model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        texts = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"rag_model_{dt_str}")
    os.makedirs(save_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=50,
        save_steps=100,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Further fine-tuned model saved to {save_dir}")

if __name__ == "__main__":
    # 1. Update embeddings
    update_embeddings(
        existing_index_dir="embeddings/sapiens_20250819_220811",  # your existing index
        new_corpus_path="data/corpus2.txt"  # your new corpus (txt or jsonl)
    )

    # 2. (Optional) Further fine-tune LLM
    # further_finetune_llm(
    #     existing_model_dir="models/fine_tuned_rag_model_20250819_234916",  # your existing model
    #     new_finetune_jsonl="data/new_finetune_dataset.jsonl",  # new QA pairs
    #     output_dir="models"
    # )
