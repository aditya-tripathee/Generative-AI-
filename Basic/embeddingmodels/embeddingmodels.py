"""
=============================================================================
                  DEFINITION: EMBEDDING MODELS IN GEN AI
=============================================================================

What is an Embedding Model?
---------------------------
An Embedding Model is a specialized Neural Network that converts unstructured data
(such as text, images, or audio) into a fixed-length numerical vector (array of floating-point numbers).

Key Concepts:
1. Semantic Vector Space:
   - Words, sentences, or documents with similar meanings are mapped to vectors 
     that are close to each other in high-dimensional space.
   - Example: Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")
   - Example: "I love cats" and "Felines are wonderful" will have high similarity scores.

2. Dimensions:
   - The size of the vector array (e.g., 384, 768, or 1536 float numbers).
   - Higher dimensions capture more nuanced semantic relationships.

3. Distance Metrics:
   - Cosine Similarity: Measures the angle between two vectors (-1 to +1).
   - Dot Product: Measures magnitude and direction match.
   - Euclidean Distance (L2): Measures straight-line distance between vector endpoints.

4. Primary Use Cases in GenAI:
   - Semantic Search (finding information by meaning, not just exact keywords)
   - RAG (Retrieval-Augmented Generation with Vector Databases like Chroma/Pinecone)
   - Text Clustering & Classification
=============================================================================
"""

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=256   # ✅ allowed (not 64 for best quality)
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




