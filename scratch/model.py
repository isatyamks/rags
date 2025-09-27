from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Path to your downloaded model
local_model_path = r"C:\\Users\\isatyamks\\.llama\\checkpoints\\Llama-2-7b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto"  # automatically chooses GPU if available
)

# Test prompt
prompt = "Hello, I am testing if LLaMA-2 7B is working correctly. The model says:"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(inputs["input_ids"], max_new_tokens=50)

# Decode and print
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
