# 📘 Basic Generative AI Modules

This directory contains foundational implementations and scripts for exploring Generative AI models using LangChain.

## 📂 Submodule Breakdown

### 💬 `chatmodels/`
- **`chat.py`**: Interacting with Google Gemini API via `ChatGoogleGenerativeAI`. Includes parameter configurations for temperature and output token control.
- **`huggingFace.py`**: Utilizing Hugging Face Hub models and hosted inference endpoints (`ChatHuggingFace`, `HuggingFacePipeline`, `HuggingFaceEndpoint`).
- **`localmodel.py`**: Running open-source models (such as `TinyLlama-1.1B`) locally on CPU/GPU via Hugging Face Transformers pipeline wrapped in LangChain.

### 🔢 `embeddingmodels/`
- **`embeddingmodels.py`**: Demonstrates generating text embeddings using both cloud APIs (`GoogleGenerativeAIEmbeddings`) and local sentence transformer models (`HuggingFaceEmbeddings`). Includes cosine similarity calculations for semantic search.

---

## ⚡ Quick Start

1. Ensure environment variables are loaded in `.env`:
   ```env
   GOOGLE_API_KEY=your_key
   HUGGINGFACE_ACCESS_TOKEN=your_token
   ```
2. Execute any model script:
   ```bash
   # Run Chat Model
   python chatmodels/chat.py

   # Run Embedding Model Demo
   python embeddingmodels/embeddingmodels.py
   ```
