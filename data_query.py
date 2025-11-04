import argparse
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
# Load environment variables from .env file
load_dotenv()

# Get paths from environment variables
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")

def run_query():

    # Create a CLI argument parser for the user to enter a query
    parser = argparse.ArgumentParser(description="Query the database")
    parser.add_argument("query", type=str, help="The question to query the database")
    args = parser.parse_args()
    query = args.query

    run_database(query)

def run_database(query):
    # Load the vector database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    #search the database for similarities
    results = db.similarity_search(query, k=4)
    if not results:
        print("Unable to answer the question. Please try again with a more detailed question.")
        return None
    
    format_context(results, query)


