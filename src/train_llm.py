from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training


def finetune_model():

    dataset = load_dataset("json", data_files="data/fine_tune_dataset2.jsonl")['train']
    # Use Phi-3-Mini instruct as base model
    model_name = "microsoft/phi-3-mini-4k-instruct"
    print("Base model:", model_name)
    from transformers import BitsAndBytesConfig
    import torch
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # Prepare model for k-bit training and enable grad checkpointing for VRAM savings
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    def tokenize_fn(batch):
        # Build prompt to align with inference style (Question ... Answer:)
        inputs = []
        labels = []
        attn_masks = []
        for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            prompt = (
                "Instruction: You are a helpful assistant. Answer the question concisely.\n\n"
                f"Question: {inp}\n"
                "Answer:"
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_text = prompt + " " + out + tokenizer.eos_token
            enc = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                add_special_tokens=False
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            # Create labels with prompt masked out
            label_ids = input_ids.copy()
            prompt_len = min(len(prompt_ids), len(label_ids))
            label_ids[:prompt_len] = [-100] * prompt_len
            inputs.append(input_ids)
            labels.append(label_ids)
            attn_masks.append(attention_mask)
        return {"input_ids": inputs, "labels": labels, "attention_mask": attn_masks}

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common for Phi-3
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
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
    import os
    from datetime import datetime
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", f"_model_{dt_str}")
    os.makedirs(save_dir, exist_ok=True)
    # Save only LoRA adapter
    model.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    finetune_model()