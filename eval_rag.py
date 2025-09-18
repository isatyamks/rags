from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate

# Import your pipeline builder
def build_qa_chain():
    from src.pipeline import pipelinefn
    # Use the correct embeddings dir for your use case
    return pipelinefn(embeddings_dir="sapiens_20250918_201956")

qa_chain = build_qa_chain()

def llm_app(question):
    # If your pipeline expects a dict, use: return qa_chain.invoke({"query": question})
    return qa_chain.invoke(question)

# Example manual test cases (replace/add as needed)
test_cases = [
    LLMTestCase(
        input="What species does Sapiens focus on?",
        actual_output=llm_app("What species does Sapiens focus on?"),
        expected_output="Homo sapiens, our own species."
    ),
    LLMTestCase(
        input="Why are shared myths powerful?",
        actual_output=llm_app("Why are shared myths powerful?"),
        expected_output="They allow strangers to trust each other."
    ),
    # Add more test cases as needed
]

evaluate(test_cases=test_cases, metrics=[AnswerRelevancyMetric()])
