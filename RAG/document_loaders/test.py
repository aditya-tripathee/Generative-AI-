# Character Based Text Splitter

# import warnings
# warnings.filterwarnings("ignore")

# print("1. Loading LangChain modules (this can take 10-20 seconds)...", flush=True)

# from pathlib import Path
# from langchain_community.document_loaders.text import TextLoader
# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# file_path = Path(__file__).parent / "text_splitters.txt"

# print("2. Loading document...", flush=True)
# data = TextLoader(file_path, encoding="utf-8")
# docs = data.load()

# print("3. Splitting document...", flush=True)
# # CharacterTextSplitter splits ONLY by double newlines (\n\n)
# splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=20,
#     chunk_overlap=2,
#     length_function=len,
# )

# chunks = splitter.split_documents(docs)

# print(f"\nSuccessfully split into {len(chunks)} chunks:\n", flush=True)
# for i, chunk in enumerate(chunks, 1):
#     print(f"--- Chunk {i} ---")
#     print(chunk.page_content.strip())
#     print("-" * 30, flush=True)




# token based splitting 

# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

# pdf_path = Path(__file__).parent / "KING.pdf"
# loader = PyPDFLoader(str(pdf_path))
# docs = loader.load()

# splitter = TokenTextSplitter( 
#     chunk_size=20,
#     chunk_overlap=2,
# )

# chunks = splitter.split_documents(docs)

# print(f"split into {len(chunks)} chunks")
# if chunks:
#     print("--- First Chunk Preview ---")
#     print(chunks)




#  Text Splitter





