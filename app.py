from src.pipeline import pipelinefn
from src.interface import launch_ui

if __name__ == "__main__":
    qa_chain = pipelinefn()
    launch_ui(qa_chain)
