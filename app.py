from src.qa_pipeline import load_qa_pipeline
from src.interface import launch_ui

if __name__ == "__main__":
    qa_chain = load_qa_pipeline()
    launch_ui(qa_chain)
