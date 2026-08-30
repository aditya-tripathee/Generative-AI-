import warnings
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Resolve path to notes.txt relative to MAIN.PY
file_path = Path(__file__).parent / "document_loaders" / "notes.txt"

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)


# Extract combined text content from documents for context
context_text = "\n\n".join([doc.page_content for doc in docs])

chat = ChatMistralAI(model="mistral-small-latest")

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Summarize based on the context provided."),
    ("user", "Summarize: {context} \n question: {question}")
])

prompt = template.invoke({"context": context_text, "question": "What is the Shah dynasty?"})

result = chat.invoke(prompt)

print(result.content)




