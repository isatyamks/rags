from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

def pipelinefn():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.3
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    db = FAISS.load_local("embeddings8", embedding, allow_dangerous_deserialization=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

    return qa_chain
