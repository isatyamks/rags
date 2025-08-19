from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


def finetune_model():

    dataset = load_dataset("json", data_files="data/fine_tune_dataset2.jsonl")["train"]
    model_name = "models/_model_20250819_234916"
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

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
        target_modules=["c_attn"],  # GPT2
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
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
    import os
    from datetime import datetime
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", f"_model_{dt_str}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
