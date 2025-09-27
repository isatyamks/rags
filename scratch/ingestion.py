from pathlib import Path

docs_folder = Path("knowledge_base")
documents = []

for file in docs_folder.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append(text)
