from src.pipeline import pipelinefn
from src.interface import terminal  #----> terminal interface
# from src.interface import launch_ui ---> gradio interface

if __name__ == "__main__":
    qa_chain = pipelinefn()
    # launch_ui(qa_chain)
    terminal(qa_chain)