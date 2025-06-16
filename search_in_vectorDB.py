from langchain.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

# Load environment and API key
load_dotenv()
cohere_api_key = os.getenv("cohere_api_key")

# Re-initialize the same embedding model used when vector store was created
embedding_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=cohere_api_key
)

# Load the persisted Chroma vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


print("Vectored Store Loaded Successfully!")

# Perform similarity search
query = "Who is Galib?"
results = vectorstore.similarity_search(query, k=1)

# Display results
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)  # print first 500 chars


