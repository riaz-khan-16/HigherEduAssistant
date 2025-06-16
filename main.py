

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
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

print("Document loaded . . .")

# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)


print("Document splitted into chunks. . . ")


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


# Step 5: Make retriever

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Reriever has made")

# Step 6: Connect to LLM with RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=Cohere(cohere_api_key=cohere_api_key),
    chain_type="stuff",
    retriever=retriever
)

print("Connected with LLM")


# step 6: Make query

query = "who is Riaz? What are the required documents for higher educations?"
result = qa_chain.run(query)

print("Query and response generated successfully!")

print(result)

with open("bangla_output.txt", "w", encoding="utf-8") as file:
    file.write(result)





