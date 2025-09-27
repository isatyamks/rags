from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_path = r"C:\Users\isatyamks\Models\mistral-7b"  # folder where model is downloaded

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")
