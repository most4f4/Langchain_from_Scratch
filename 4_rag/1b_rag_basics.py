import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# -----------------------------------------------------------------------
# Steps to load and query an existing Chroma vector store
# 1. Define the persistent directory where the vector store is saved
# 2. Define the embedding model to be used
# 3. Load the existing vector store with the embedding function
# 4. Define the user's question
# 5. Retrieve relevant documents based on the query
# 6. Display the relevant results with metadata
# -----------------------------------------------------------------------

# Load environment variables from a .env file
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model (should match the one used during creation)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Who is Odysseus' wife?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold", # Use similarity search with a score threshold, there are other search types available
    search_kwargs={"k": 3, "score_threshold": 0.4}, # Retrieve top 3 documents with a similarity score above 0.4
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
