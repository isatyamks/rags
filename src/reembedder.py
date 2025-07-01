from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings", embedding_model)

new_text = open("data/new_data.txt", encoding="utf-8").read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
new_chunks = splitter.split_text(new_text)

vectorstore.add_texts(new_chunks)

vectorstore.save_local("embeddings")
