# 🚀 Generative AI & LangChain Playground

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![Package Manager](https://img.shields.io/badge/Package%20Manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](#-license)

A comprehensive, production-ready playground for experimenting with **Generative AI**, **Large Language Models (LLMs)**, **LangChain (LCEL)**, **Vector Embeddings**, and **Interactive Web Applications**. This repository demonstrates how to integrate, configure, and orchestrate cloud-hosted API models (Google Gemini, Mistral AI, Hugging Face Hub) alongside local open-source models (PyTorch/Transformers).

---

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [📁 Repository Architecture](#-repository-architecture)
- [🛠️ Tech Stack & Dependencies](#%EF%B8%8F-tech-stack--dependencies)
- [⚙️ Environment Setup & Installation](#%EF%B8%8F-environment-setup--installation)
- [🔑 API Key Configuration](#-api-key-configuration)
- [💻 Module Overview & Execution](#-module-overview--execution)
  - [1. CineSage: Structured Movie Analyst](#1-cinesage-structured-movie-analyst)
  - [2. Interactive CLI Chatbot](#2-interactive-cli-chatbot)
  - [3. Interactive Web Chatbots (Streamlit)](#3-interactive-web-chatbots-streamlit)
  - [4. Multi-Provider Chat Models](#4-multi-provider-chat-models)
  - [5. Vector Embedding Models](#5-vector-embedding-models)
- [🧑‍💻 Author & License](#-author--license)

---

## ✨ Key Features

- **Multi-Provider LLM Orchestration**:
  - **Google Gemini**: Integration via `langchain-google-genai` (`gemini-flash-latest`, `gemini-pro`) with configurable temperature and token bounds.
  - **Mistral AI**: Integration via `langchain-mistralai` (`open-mistral-7b`) for structured parsing, conversation agents, and chat history.
  - **Hugging Face Hub**: Remote inference endpoints via `langchain-huggingface` (`DeepSeek`, `Qwen`, etc.).
  - **Local Open-Source LLMs**: In-memory local inference pipelines using Hugging Face `transformers` (e.g., `TinyLlama-1.1B`).
- **Structured Data Extraction (LCEL & Pydantic)**:
  - **CineSage Engine**: Combines `PydanticOutputParser` with LCEL chains (`prompt | model | parser`) to output strictly typed JSON objects (director, cast, highlights, themes, cultural impact).
- **Conversational State & UI Interfaces**:
  - **CLI Chatbot**: State-aware interactive shell assistant utilizing `SystemMessage`, `HumanMessage`, and `AIMessage` memory tracking.
  - **Streamlit Web Apps**: `app.py` and `uichatbot.py` with custom CSS, sidebars, avatar icons, and auto-run direct execution capabilities.
- **Embeddings & Vector Representations**:
  - **Hugging Face Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2` for local vector embeddings.
  - **Google Gemini Embeddings**: Cloud-based embedding generation with cosine similarity metrics for semantic search.
- **Modern Package & Environment Management**:
  - Blazing-fast virtual environment setup powered by [`uv`](https://github.com/astral-sh/uv).

---

## 📁 Repository Architecture

```text
Generative-AI-/
├── Basic/
│   ├── CineSage/
│   │   └── core.py                 # Structured movie analysis LCEL chain with Pydantic parser
│   ├── chatmodels/
│   │   ├── app.py                  # Streamlit web chatbot with session memory state
│   │   ├── chat.py                 # Google Gemini Chat API invocation using LangChain
│   │   ├── chatbot.py              # CLI interactive chat tool with conversation memory
│   │   ├── huggingFace.py          # Hugging Face remote endpoints & pipeline execution
│   │   ├── localmodel.py           # Local Transformers model execution (TinyLlama)
│   │   └── uichatbot.py            # Streamlit chatbot web app with custom CSS & sidebar
│   ├── embeddingmodels/
│   │   ├── embeddingmodels.py      # Vector embeddings & cosine similarity calculations
│   │   └── huggingface_embedding.py# Document embedding batch pipeline via sentence-transformers
│   ├── src/
│   │   └── basic/                  # Core package source module (__init__.py)
│   ├── .env                        # Environment variables & API keys (gitignored)
│   ├── pyproject.toml              # UV build system & package metadata
│   ├── requirements.txt            # Project dependencies manifest
│   └── README.md                   # Basic submodule overview
└── README.md                       # Main Repository Documentation
```

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.10+ (Tested through Python 3.14)
- **Frameworks**: LangChain (`langchain`, `langchain-core`, `langchain-community`)
- **Model Providers**:
  - `langchain-google-genai` / `google-generativeai`
  - `langchain-mistralai` / `mistralai`
  - `langchain-huggingface` / `transformers` / `torch`
  - `langchain-openai` / `groq` (compatible)
- **Web UI & Parsing**: Streamlit, Pydantic (v2), `python-dotenv`
- **Embeddings**: `sentence-transformers`

---

## ⚙️ Environment Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/aditya-tripathee/Generative-AI-.git
cd Generative-AI-/Basic
```

### 2. Create Virtual Environment

**Using `uv` (Recommended):**
```bash
uv venv
# On Windows PowerShell:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

**Using `pip`:**
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

Create a `.env` file in the `Basic/` directory:

```env
# Basic/.env
GOOGLE_API_KEY=your_google_gemini_api_key
MISTRAL_API_KEY=your_mistral_ai_api_key
HUGGINGFACE_ACCESS_TOKEN=your_huggingface_token
```

> ⚠️ **Security Note:** Do not commit the `.env` file or plain API credentials to Git. Keep `.env` listed in `.gitignore`.

---

## 💻 Module Overview & Execution

Ensure you are inside the `Basic/` directory before running commands:

### 1. CineSage: Structured Movie Analyst
Runs an LCEL chain (`prompt | model | parser`) with `ChatMistralAI` and `PydanticOutputParser` to analyze movies into structured data:
```bash
python CineSage/core.py
```

### 2. Interactive CLI Chatbot
Runs a terminal conversational session with state history (`SystemMessage`, `HumanMessage`, `AIMessage`):
```bash
python chatmodels/chatbot.py
```

### 3. Interactive Web Chatbots (Streamlit)
Launch browser-based web applications:
```bash
# Basic Streamlit Chatbot
python chatmodels/app.py

# Custom Styled Web UI (Custom CSS, Sidebar config, Avatar support)
python chatmodels/uichatbot.py
```
*(Note: Files automatically invoke `streamlit run` if executed directly via `python`.)*

### 4. Multi-Provider Chat Models
Run different LLM backends:
```bash
# Google Gemini API
python chatmodels/chat.py

# Hugging Face Endpoints / Hub
python chatmodels/huggingFace.py

# Local PyTorch/Transformers Pipeline (TinyLlama)
python chatmodels/localmodel.py
```

### 5. Vector Embedding Models
Generate text embeddings and calculate semantic vector similarities:
```bash
# Gemini & Local HuggingFace Embeddings + Cosine Similarity
python embeddingmodels/embeddingmodels.py

# HuggingFace Document Embeddings Batch
python embeddingmodels/huggingface_embedding.py
```

---

## 🧑‍💻 Author & License

- **Author**: Aditya Tripathee
- **GitHub**: [@aditya-tripathee](https://github.com/aditya-tripathee)
- **Email**: [adityatripatheee@gmail.com](mailto:adityatripatheee@gmail.com)

*Maintained for educational and Generative AI research purposes.*

