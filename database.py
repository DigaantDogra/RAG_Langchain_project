from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "Md_Files"

def load_documents(data_path):
    loader = DirectoryLoader(data_path, glob="*.md")
    documents = loader.load()
    return documents

documents = load_documents(DATA_PATH)

# split the documents into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len, add_start_index=True)
text_chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(text_chunks)} chunks")