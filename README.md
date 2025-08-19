

# RAGS: Retrieval-Augmented Generation System

## Overview
RAGS is a modular, production-ready pipeline for Retrieval-Augmented Generation (RAG) using Python, HuggingFace Transformers, FAISS, and PEFT (LoRA). It supports document embedding, retrieval, LLM fine-tuning, and automated evaluation, with a focus on clarity, reproducibility, and extensibility. The codebase is clean, minimal, and easy to extend.

---


## Features
- **Document Embedding:**
	- Converts `.txt` or `.jsonl` files into FAISS vector indexes using Sentence Transformers.
	- Auto-chunks text and supports flexible input formats.
- **Retrieval Pipeline:**
	- Uses FAISS for fast similarity search over embedded documents.
- **LLM Fine-Tuning:**
	- Fine-tunes language models (e.g., GPT-2) on custom datasets using HuggingFace and PEFT (LoRA).
	- Supports incremental retraining: just set `model_name` to your previous model folder.
- **Evaluation:**
	- Automated similarity-based evaluation and accuracy reporting.
	- Results are written back to the same CSV for easy tracking.
- **Interactive Chat:**
	- Terminal chat interface for live Q&A with your RAG pipeline.
- **CLI Entry Point:**
	- All major tasks can be run from `app.py` or individual scripts.

---


## Project Structure
```
├── app.py                  # CLI entry point (optional)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Raw and processed data
│   ├── raw/                # Source text files
│   └── ...
├── embeddings/             # FAISS indexes (per file, with datetime)
├── models/                 # Fine-tuned LLMs (with datetime)
├── reports/                # Evaluation CSVs
├── src/
│   ├── embedder.py         # Embedding pipeline
│   ├── corpus_jsonl_gen.py # Text-to-JSONL converter
│   ├── evaluation.py       # Evaluation utilities
│   ├── train_llm.py        # LLM fine-tuning (LoRA)
│   ├── pipeline.py         # RAG pipeline (retrieval + generation)
│   ├── interface.py        # Terminal chat & batch QA
│   └── ...
├── run_finetune.py         # Script to launch fine-tuning
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
Create FAISS indexes from your data:
```python
from src.embedder import vector_from_jsonl
vector_from_jsonl('data/raw/yourfile.txt')  # or .jsonl
```
Output: `embeddings/yourfile_<datetime>/index.faiss`

### 2. Fine-Tuning the LLM
Edit `src/train_llm.py`:
- **First time:**
	- Set `model_name = "gpt2-medium"` (or any HuggingFace model).
- **Retraining:**
	- Set `model_name = "models/fine_tuned_rag_model_<datetime>"` to continue training from your previous checkpoint.

Or use the script:
```bash
python run_finetune.py
```
Data: Expects `data/fine_tune_dataset.jsonl` with `prompt` and `completion` fields.
Output: `models/fine_tuned_rag_model_<datetime>/`

### 3. Interactive Chat
Start a terminal chat session with your RAG pipeline:
```python
from src.interface import terminalchat
from src.pipeline import pipelinefn
qa_chain = pipelinefn()
terminalchat(qa_chain)
```
Type your questions, type 'exit' to quit. Only the best answer is shown.

### 4. Batch QA and Evaluation
Run batch QA and evaluation:
```python
from src.interface import improve
from src.evaluation import evaluate_and_save, report_eval
improve(qa_chain, your_dataframe, csv_path='reports/improveX.csv')
evaluate_and_save(X)
report_eval(X)
```
Results are written back to `reports/improveX.csv` (adds `Similarity` and `Correct` columns).

Or use the CLI/testing script:
```bash
python testing.py
```

---


## Notes
- All evaluation and improvement results are stored in the same CSV for simplicity.
- The embedding pipeline only creates standalone indexes (no global index).
- Fine-tuned models and embeddings are versioned by datetime for reproducibility.
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
