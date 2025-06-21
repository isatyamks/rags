from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load existing FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("embeddings", embedding_model)

# 2. Load and split your new text
new_text = open("data/new_data.txt", encoding="utf-8").read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
new_chunks = splitter.split_text(new_text)

# 3. Add new chunks to vector store
vectorstore.add_texts(new_chunks)

# 4. Save updated vector store
vectorstore.save_local("embeddings")
