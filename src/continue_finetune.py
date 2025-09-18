import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, Dataset
import json

# Path to the previously fine-tuned model directory
MODEL_DIR = os.path.join('..', 'models', '_model_20250819_234916')  # Change to your actual model dir
# Path to the new QA pairs file (JSONL)
QA_PATH = os.path.join('..', 'data', 'books', 'qa_pairs_aboy.jsonl')
# Path to save the updated model
OUTPUT_MODEL_DIR = os.path.join('..', 'models', '_model_aboy_continued')

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def load_qa_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({'question': item['question'], 'answer': item['answer'], 'context': item.get('context', '')})
    return data

# Load new QA data
data = load_qa_jsonl(QA_PATH)

# Prepare HuggingFace Dataset
dataset = Dataset.from_list([
    {'input_text': f"question: {item['question']} context: {item['context']}", 'target_text': item['answer']} for item in data
])

def preprocess_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=256, truncation=True)
    labels = tokenizer(examples['target_text'], max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    report_to=[],
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Fine-tune
trainer.train()

# Save the updated model
trainer.save_model(OUTPUT_MODEL_DIR)
print(f"Model continued training and saved to {OUTPUT_MODEL_DIR}")
