from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_splitter(text: str, chunk_size=1000, chunk_overlap=100) -> list:
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    # separators=["\n\n", "\n", ".", "!", "?", " "]
)
    return splitter.split_text(text)




