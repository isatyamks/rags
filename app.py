from src.pipeline import pipelinefn
from src.interface import improve,terminalchat
import pandas as pd
import csv
from src.embedder import vector_from_jsonl
from datetime import datetime   
from pathlib import Path
import os
import json








if __name__ == "__main__":
    
    file_name = input("Enter input: ")
    vector_from_jsonl(f"data/books/raw/{file_name}.txt", save_path="embeddings")
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{file_name}_{dt_str}"
    
    qa_chain = pipelinefn(embeddings_dir=folder_name)
    #for testing
    df = pd.read_csv("data\\questions\\sapiens_qa.csv")
    terminalchat(qa_chain)
    # improve(qa_chain,df)


