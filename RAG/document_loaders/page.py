import warnings
from langchain_community.document_loaders import WebBaseLoader

warnings.filterwarnings("ignore", category=DeprecationWarning)

url = "https://www.apple.com/in/shop/ways-to-buy"

loader = WebBaseLoader(url)
docs = loader.load() 

print(f"Loaded {len(docs)} document(s)")
print("Metadata:", docs[0].metadata)
print("\n--- Content Preview ---")
print(docs[0].page_content)
