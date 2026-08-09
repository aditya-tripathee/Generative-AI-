# 🚀 Generative AI & LangChain Playground

A comprehensive, modular repository for experimenting with **Generative AI**, **Large Language Models (LLMs)**, and **LangChain**. This project demonstrates how to connect, configure, and execute various AI models ranging from cloud-hosted APIs (Google Gemini, Hugging Face Hub) to locally hosted open-source models using Hugging Face Transformers.

---

## 📌 Features & Highlights

- **Multi-Provider LLM Integrations**:
  - **Google Gemini**: Built with `langchain-google-genai` using models like `gemini-flash-latest`. Supports fine-tuned hyperparameter adjustments (temperature, max output tokens).
  - **Hugging Face Hub**: Remote inference endpoints via `langchain-huggingface` (e.g., DeepSeek, Qwen).
  - **Local Open-Source LLMs**: In-memory local pipelines powered by Hugging Face `transformers` (e.g., `TinyLlama`, `Zephyr`).
- **Embedding Models**: Dedicated modules for generating vector embeddings and text representations.
- **Modern Python Environment Setup**:
  - Fast environment & package management via [`uv`](https://github.com/astral-sh/uv).
  - Full compatibility with traditional `pip` and `pyproject.toml`.
- **Environment & Key Management**: Centralized `.env` handling via `python-dotenv`.

---

## 📁 Repository Structure

```text
.
├── Basic/
│   ├── chatmodels/
│   │   ├── chat.py           # Google Gemini API invocation using LangChain
│   │   ├── huggingFace.py    # Hugging Face remote endpoints & pipeline invocation
│   │   └── localmodel.py     # Local Transformers model execution
│   ├── embeddingmodels/
│   │   └── embeddingmodels.py# Vector embedding utilities
│   ├── src/
│   │   └── basic/            # Package source modules
│   ├── .env                  # Environment configuration (API Keys - gitignored)
│   ├── pyproject.toml        # Project configuration & UV build specs
│   ├── requirements.txt      # Dependency manifest
│   └── README.md             # Submodule overview
└── README.md                 # Main Repository Documentation
```

---

## 🛠️ Prerequisites

- **Python**: `>= 3.10` (Project configured for Python 3.14 compatibility)
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- **API Keys**:
  - [Google AI Studio API Key](https://aistudio.google.com/) for Gemini models
  - [Hugging Face Access Token](https://huggingface.co/settings/tokens) for Hugging Face endpoints

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aditya-tripathee/Generative-AI-.git
cd Generative-AI-/Basic
```

### 2. Set Up Virtual Environment

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

## 🔑 Environment Configuration

Create a `.env` file inside the `Basic/` directory with your API credentials:

```env
# Basic/.env
GOOGLE_API_KEY=your_google_gemini_api_key_here
HUGGINGFACE_ACCESS_TOKEN=your_huggingface_token_here
```

> ⚠️ **Security Warning:** Never commit your `.env` file or raw API keys to public repositories. Ensure `.env` is listed in your `.gitignore`.

---

## 💻 Usage & Examples

Navigate to the `Basic/` directory before executing scripts:

### 1. Google Gemini Chat Model
Run the Google Gemini model using `langchain-google-genai`:
```bash
python chatmodels/chat.py
```
*Configurable Parameters:*
- `temperature`: Adjusts output creativity (`0.0` for factual/code, `0.8+` for creative writing).
- `max_output_tokens`: Restricts maximum response length to manage costs.

### 2. Hugging Face Models
Run inference on Hugging Face models using local pipelines or endpoints:
```bash
python chatmodels/huggingFace.py
```

### 3. Local Model Pipeline
Run open-source models (like `TinyLlama`) locally using PyTorch and Transformers:
```bash
python chatmodels/localmodel.py
```

### 4. Text Embedding Models & Semantic Similarity
Generate vector embeddings (via Google Gemini or local `sentence-transformers`) and compute semantic similarity scores:
```bash
python embeddingmodels/embeddingmodels.py
```

---

## 📦 Dependencies

Major dependencies defined in `requirements.txt`:
- `langchain` & `langchain-community`
- `langchain-google-genai`
- `langchain-huggingface`
- `google-generativeai`
- `transformers` & `torch`
- `python-dotenv`

---

## 🧑‍💻 Author

- **Aditya Tripathee** - [adityatripatheee@gmail.com](mailto:adityatripatheee@gmail.com)
- **GitHub**: [@aditya-tripathee](https://github.com/aditya-tripathee)

---

## 📄 License

This repository is maintained for educational and generative AI research purposes.
