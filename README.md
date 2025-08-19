
# RAGS: Retrieval-Augmented Generation System

## Overview
RAGS is a modular, end-to-end pipeline for Retrieval-Augmented Generation (RAG) using Python, HuggingFace Transformers, FAISS, and PEFT (LoRA). It supports document embedding, retrieval, LLM fine-tuning, and automated evaluation, with a focus on clarity, reproducibility, and extensibility.

---

## Features
- **Document Embedding:**
	- Converts `.txt` or `.jsonl` files into FAISS vector indexes using Sentence Transformers.
	- Supports chunking and flexible input formats.
- **Retrieval Pipeline:**
	- Uses FAISS for fast similarity search over embedded documents.
- **LLM Fine-Tuning:**
	- Fine-tunes language models (e.g., GPT-2) on custom datasets using HuggingFace and PEFT (LoRA).
	- Supports incremental retraining on new data.
- **Evaluation:**
	- Automated similarity-based evaluation and accuracy reporting.
	- Results are written back to the same CSV for easy tracking.
- **CLI Entry Point:**
	- All major tasks can be run from `app.py` or individual scripts.

---

## Project Structure
```
├── app.py                  # CLI entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Raw and processed data
│   ├── raw/                # Source text files
│   └── ...
├── embeddings/             # FAISS indexes
├── reports/                # Evaluation CSVs
├── src/
│   ├── embedder.py         # Embedding pipeline
│   ├── corpus_jsonl_gen.py # Text-to-JSONL converter
│   ├── evaluation.py       # Evaluation utilities
│   ├── train_llm.py        # LLM fine-tuning (LoRA)
│   └── ...
└── testing.py              # Evaluation script
```

---

## Setup
1. **Install dependencies:**
	 ```bash
	 pip install -r requirements.txt
	 ```
2. **Prepare data:**
	 - Place your `.txt` or `.jsonl` files in `data/raw/`.
	 - For `.txt` files, the pipeline will auto-convert to `.jsonl`.

---

## Usage
### 1. Embedding Documents
Run the embedding pipeline to create FAISS indexes:
```python
from src.embedder import vector_from_jsonl
vector_from_jsonl('data/raw/yourfile.txt')  # or .jsonl
```
- Output: `embeddings/yourfile_<datetime>/index.faiss`

### 2. Fine-Tuning the LLM
Edit `src/train_llm.py`:
- **First time:**
	- Set `model_name = "gpt2-medium"` (or any HuggingFace model).
- **Retraining:**
	- Set `model_name = "fine_tuned_rag_model"` to continue training from your previous checkpoint.

Run:
```python
from src.train_llm import finetune_model
finetune_model()
```
- Data: Expects `data/fine_tune_dataset.jsonl` with `prompt` and `completion` fields.
- Output: `fine_tuned_rag_model/`

### 3. Evaluation
Run the evaluation pipeline:
```python
from src.evaluation import evaluate_and_save, report_eval
evaluate_and_save(1)  # 1 = improve number
report_eval(1)
```
- Results are written back to `reports/improve1.csv` (adds `Similarity` and `Correct` columns).

Or use the CLI/testing script:
```bash
python testing.py
```

---

## Notes
- All evaluation and improvement results are stored in the same CSV for simplicity.
- The embedding pipeline only creates standalone indexes (no global index).
- The codebase is modular—extend or swap components as needed.

---

## Requirements
- Python 3.8+
- HuggingFace Transformers
- sentence-transformers
- peft
- datasets
- pandas
- faiss-cpu

---

## License
MIT License

---

## Acknowledgements
- HuggingFace Transformers & Datasets
- PEFT (LoRA)
- FAISS
- Sentence Transformers
