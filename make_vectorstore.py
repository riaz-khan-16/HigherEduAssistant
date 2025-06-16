from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Cohere
import glob


# Load all PDFs in the 'pdfs' folder
# pdf_files = glob.glob("datasets/*.pdf")

pdf_files = ["datasets/Riaz.pdf", "datasets/basic_terminology.pdf","datasets/nextop_usa.pdf", "datasets/ultimate_checklist.pdf"]
all_docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    all_docs.extend(docs) 

print("Document loaded successflly!")

# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)


print("Document splitted into chunks! ")


# Step 3: define embedding model
load_dotenv()
cohere_api_key = os.getenv("secret_key")

embedding_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=cohere_api_key  
)

print("Embedding Model Loaded . . . .")


# Step 4: Create vector store
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding= embedding_model,
    persist_directory="chroma_db"
    )

print("Vectore Store Created")
print(type(vectorstore))

# step 5:  Save the vector store for reuse
vectorstore.persist()

print("Vector Store Saved Locally")

