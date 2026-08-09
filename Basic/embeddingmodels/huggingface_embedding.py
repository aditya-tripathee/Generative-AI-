from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}, 
)   



texts= [
    "Hello",
    "Hi",
    "Namaste",
    "Kon-nichiwa",
    "Bonjour",
    "Hola",
    "Ciao",
    "Guten Tag",
    "Olá",
    "你好",
    "Merhaba",
    "Hola",
    "Ciao",
    "Guten Tag",
    "Olá",
    "你好",
    "Merhaba",
    "Hola",
    "Ciao",
    "Guten Tag",
    "Olá",
    "你好",
    "Merhaba",
    "Hola",
    "Ciao",
    "Guten Tag",
    "Olá",
    "你好",
    "Merhaba",
    "Hola",
    "Ciao",
    "Guten Tag",
    "Olá",
    "你好",
    "Merhaba"
]

vector = embedding.embed_documents(texts)
print(vector)

