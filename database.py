from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import os
import shutil
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get paths from environment variables
DATA_PATH = os.getenv("DATA_PATH")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")

# Generate the vector database
def genetate_store():
    documents = load_documents()
    text_chunks = split_documents(documents)
    save_to_chroma(text_chunks)

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,length_function=len, add_start_index=True)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(text_chunks)} chunks")
    return text_chunks

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# Creating a vector database
def save_to_chroma(text_chunks):
    print(f"Saving {len(text_chunks)} chunks to Chroma")

    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # Using Ollama nomic-embed-text model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
    vectorstore.persist()

    print(f"saved {len(text_chunks)} chunks to {CHROMA_DB_PATH}")

#generate the vector database
genetate_store()
