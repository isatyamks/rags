from src.pipeline import pipelinefn
from src.interface import improve
import pandas as pd
import csv


if __name__ == "__main__":
    qa_chain = pipelinefn()
    df = pd.read_csv("data\\sapiens_qa.csv")
    improve(qa_chain,df)