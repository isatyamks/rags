from huggingface_hub import snapshot_download
from pathlib import Path

LOCAL_MODEL_DIR = Path(r"C:\\Users\\isatyamks\\models\\mistral-7b")

snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.1",
    local_dir=LOCAL_MODEL_DIR,
    allow_patterns=["*.bin", "*.json", "*.model"]  
)

print(f"Model downloaded to {LOCAL_MODEL_DIR}")
