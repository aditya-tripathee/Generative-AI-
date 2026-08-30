#load pdf 
#split into chunks 
#create embeddings
#store in vector database

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# load pdf 
pdf_path = Path(__file__).parent / "RAG" / "document_loaders" / "KING.pdf"
pdf_loader = PyPDFLoader(str(pdf_path))
docs = pdf_loader.load()

# split into chunks 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# store in vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Database created successfully")


retriver = vectorstore.as_retriever()

result = retriver.invoke("What is AI?")

for r in result:
    print(r.page_content)
    print(r.metadata)

