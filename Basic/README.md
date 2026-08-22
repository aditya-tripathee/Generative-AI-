# 📘 Basic Generative AI Modules

This directory contains foundational implementations and scripts for exploring Generative AI models using LangChain, Streamlit, Mistral AI, Google Gemini, and Hugging Face.

## 📂 Submodule Breakdown

### 🎬 `CineSage/`
- **`core.py`**: Intelligent Movie Analysis engine using `ChatMistralAI` (`open-mistral-7b`), `PydanticOutputParser` (`MovieAnalysis` schema), and LangChain Expression Language (`prompt | model | parser`) to extract structured movie attributes (Director, Cast, Highlights, Themes, Impact).

### 💬 `chatmodels/`
- **`chat.py`**: Google Gemini API integration using `ChatGoogleGenerativeAI`. Includes parameter configurations for temperature and output token control.
- **`chatbot.py`**: Interactive CLI conversational agent maintaining chat history using `SystemMessage`, `HumanMessage`, and `AIMessage`.
- **`app.py`**: Streamlit web chat application powered by `ChatMistralAI` with session state memory.
- **`uichatbot.py`**: Styled Streamlit web chat app featuring custom CSS gradients, sidebar parameters, avatar icons, and message history management.
- **`huggingFace.py`**: Utilizing Hugging Face Hub models and hosted inference endpoints (`ChatHuggingFace`, `HuggingFacePipeline`, `HuggingFaceEndpoint`).
- **`localmodel.py`**: Running open-source models (such as `TinyLlama-1.1B`) locally on CPU/GPU via Hugging Face Transformers pipeline wrapped in LangChain.

### 🔢 `embeddingmodels/`
- **`embeddingmodels.py`**: Demonstrates generating text embeddings using both cloud APIs (`GoogleGenerativeAIEmbeddings`) and local sentence transformer models (`HuggingFaceEmbeddings`). Includes cosine similarity calculations for semantic search.
- **`huggingface_embedding.py`**: Batch document embedding generation using `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`.

---

## ⚡ Quick Start

1. Ensure environment variables are loaded in `.env`:
   ```env
   GOOGLE_API_KEY=your_google_key
   MISTRAL_API_KEY=your_mistral_key
   HUGGINGFACE_ACCESS_TOKEN=your_huggingface_token
   ```
2. Execute any model script:
   ```bash
   # Run CineSage Structured Movie Analyst
   python CineSage/core.py

   # Run CLI Chatbot
   python chatmodels/chatbot.py

   # Run Streamlit Web Chatbot UI
   python chatmodels/uichatbot.py

   # Run Chat Model (Google Gemini)
   python chatmodels/chat.py

   # Run Embedding Model Demo
   python embeddingmodels/embeddingmodels.py
   ```

