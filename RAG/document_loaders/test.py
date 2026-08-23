import warnings
from pathlib import Path
from langchain_community.document_loaders import TextLoader

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Resolve notes.txt relative to test.py directory
file_path = Path(__file__).parent / "notes.txt"

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print("Metadata:", docs[0].metadata)
print("\n--- Content Preview ---")
print(docs[0])

