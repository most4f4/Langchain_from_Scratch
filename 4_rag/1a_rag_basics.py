import os

from langchain.text_splitter import CharacterTextSplitter   # For splitting text into chunks 
from langchain_community.document_loaders import TextLoader # For loading text documents, there is also a PDFLoader, Web loaders, etc.
from langchain_community.vectorstores import Chroma         # For creating and managing a Chroma vector store
from langchain_openai import OpenAIEmbeddings               # For generating embeddings using OpenAI models
from dotenv import load_dotenv

# -----------------------------------------------------------------------
# Steps to set up a RAG system with Chroma vector store
# 1. Define the directory containing the text file and the persistent directory
# 2. Check if the Chroma vector store already exists
# 3. If not, read/load the text content from the file
# 4. Split the document into chunks
# 5. Create embeddings
# 6. Create the vector store and persist it automatically
# -----------------------------------------------------------------------

# Load environment variables from a .env file
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# os.path provides functions to interact with the file system
# - os.path.dirname returns the directory name of the given path, like /path/to/4_rag
# - os.path.abspath returns the absolute path of the given path
# - os.path.join joins one or more path components intelligently
# - os.path.exists checks if a given path exists


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    print("\n--- Loading document ---")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load() # documents is a list of Document objects in LangChain
    # encoding="utf-8" is important
    # Specify encoding to handle special characters 
    # TextLoader tries to open with system encoding → fails on Windows which defaults to cp1252 encoding

    # Split the document into chunks, there are other text splitter strategies available
    print("\n--- Splitting documents into chunks ---")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) 
    docs = text_splitter.split_documents(documents) # docs is a list of Document objects (chunks) in LangChain

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persistent_directory
        )
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
