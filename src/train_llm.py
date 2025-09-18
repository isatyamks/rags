from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


def finetune_model():

    dataset = load_dataset("json", data_files="data/fine_tune_dataset2.jsonl")['train']
    # Use Phi-3-Mini instruct as base model
    model_name = "microsoft/phi-3-mini-4k-instruct"
    print("Base model:", model_name)
    from transformers import BitsAndBytesConfig
    import torch
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    def tokenize_fn(batch):
        # For instruction tuning: expects 'instruction', 'input', 'output' fields
        texts = [
            f"Instruction: {ins}\nInput: {inp}\nOutput:" for ins, inp in zip(batch["instruction"], batch["input"])
        ]
        labels = batch["output"]
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokens["labels"] = tokenizer(
            labels,
            truncation=True,
            padding="max_length",
            max_length=512
        )["input_ids"]
        return tokens

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
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
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