from src.pipeline import pipelinefn
from src.interface import terminal  #----> terminal interface
# from src.interface import launch_ui ---> gradio interface
from src.interface import improve
import pandas as pd
import csv


if __name__ == "__main__":
    qa_chain = pipelinefn()
    df = pd.read_csv("testing\\sapiens_qa.csv")
    # launch_ui(qa_chain)
    # terminal(qa_chain)
    improve(qa_chain,df,csv)