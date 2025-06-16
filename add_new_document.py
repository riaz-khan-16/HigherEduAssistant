from langchain.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
import os
from dotenv import load_dotenv

# Load .env and API key
load_dotenv()
cohere_api_key = os.getenv("cohere_api_key")

# Reinitialize the embedding model
embedding_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=cohere_api_key
)

# Load existing vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load new PDF document
new_loader = PyPDFLoader("datasets/Galib.pdf")
new_docs = new_loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_new_docs = text_splitter.split_documents(new_docs)

# Add new documents to the store
vectorstore.add_documents(split_new_docs)

# Save changes to disk
vectorstore.persist()

print("New Document added successfully!")