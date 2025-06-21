from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=100) -> list:
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # separators=["\n\n", "\n", ".", "!", "?", " "]
)
    return splitter.split_text(text)




