import warnings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Resolve KING.pdf relative to pdf.py directory
file_path = Path(__file__).parent / "KING.pdf"

loader = PyPDFLoader(str(file_path))
docs = loader.load()

print(f"Loaded {len(docs)} document(s)/page(s)")
print("Metadata:", docs[0].metadata)
print("\n--- Content Preview (Page 1) ---")
print(docs[0])


