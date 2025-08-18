# Document Portal - Intelligent RAG-Powered Document Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.27-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise-grade document intelligence platform** powered by Retrieval-Augmented Generation (RAG) for advanced document analysis, comparison, and conversational AI interactions.

## 🚀 Features

### 📊 **Document Analysis**
- **Intelligent Content Extraction**: Extract and analyze key information from PDF documents
- **Structured Data Output**: Get organized insights with metadata and content summaries
- **Multi-format Support**: Handle PDF, DOCX, and TXT files seamlessly

### 🔍 **Document Comparison**
- **Page-wise Analysis**: Compare documents at the page level for precise difference detection
- **Change Tracking**: Identify additions, deletions, and modifications between versions
- **Visual Diff Reports**: Get structured comparison results in tabular format

### 💬 **Conversational RAG**
- **Multi-document Chat**: Upload multiple documents and chat with the combined knowledge base
- **Contextual Retrieval**: Advanced retrieval using MMR (Maximal Marginal Relevance) for diverse results
- **Session Management**: Maintain conversation context across multiple interactions
- **Real-time Indexing**: Dynamic FAISS vector database with in-memory and persistent storage options

### 🏗️ **Architecture Highlights**
- **Modular Design**: Clean separation of concerns with dedicated modules for each functionality
- **Multi-LLM Support**: Compatible with Groq, OpenAI, Gemini, Claude, and Hugging Face models
- **Flexible Embeddings**: Support for OpenAI, Hugging Face, and Gemini embedding models
- **Vector Database Options**: In-memory, on-disk, and cloud-based storage solutions
- **RESTful API**: FastAPI-based API with comprehensive endpoints
- **Modern Web UI**: Responsive interface with real-time interactions

## 🛠️ Technology Stack

### Core Framework
- **FastAPI** - High-performance web framework for building APIs
- **LangChain** - Framework for developing applications with LLMs
- **FAISS** - Efficient similarity search and clustering of dense vectors

### AI/ML Models
- **LLM Providers**: Groq, OpenAI, Google Gemini, Anthropic Claude, Hugging Face, Ollama
- **Embedding Models**: OpenAI, Hugging Face, Google Gemini
- **Vector Databases**: FAISS (in-memory/on-disk), Cloud-based solutions

### Document Processing
- **PyMuPDF** - PDF processing and text extraction
- **python-docx** - Microsoft Word document handling
- **Structured Logging** - Comprehensive logging with structlog

### Development & Testing
- **Pytest** - Testing framework
- **Streamlit** - Alternative UI option
- **Jupyter** - Interactive development and experimentation

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Git
- Conda (recommended) or pip

### Quick Start

```bash


# Create and activate conda environment
conda create -p ./env python=3.10 -y
conda activate ./env

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Environment Setup

1. **Create a `.env` file** in the project root:
```bash
# API Keys (choose your preferred providers)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Customize paths
FAISS_BASE=faiss_index
UPLOAD_BASE=data
FAISS_INDEX_NAME=index
```

2. **Get API Keys**:
   - **Groq**: [Get API Key](https://console.groq.com/keys) | [Documentation](https://console.groq.com/docs/overview)
   - **Google Gemini**: [Get API Key](https://aistudio.google.com/apikey) | [Documentation](https://ai.google.dev/gemini-api/docs/models)
   - **OpenAI**: [Get API Key](https://platform.openai.com/api-keys)

## 🚀 Usage

### Web Interface

```bash
# Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000` to access the web interface.

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Document Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### Document Comparison
```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: multipart/form-data" \
  -F "reference=@reference.pdf" \
  -F "actual=@actual.pdf"
```

#### Chat with Documents
```bash
# Build index
curl -X POST "http://localhost:8000/chat/index" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "session_id=my_session"

# Chat
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main points?", "session_id": "my_session"}'
```

## 🏗️ Project Structure

```
RAG_APP/
├── api/                    # FastAPI application
│   ├── main.py            # Main API server
│   └── data/              # Data storage
├── src/                   # Core application modules
│   ├── document_analyzer/ # Document analysis functionality
│   ├── document_chat/     # RAG chat implementation
│   ├── document_compare/  # Document comparison logic
│   └── document_ingestion/# Document processing
├── utils/                 # Utility functions
│   ├── config_loader.py   # Configuration management
│   ├── document_ops.py    # Document operations
│   ├── file_io.py         # File I/O utilities
│   └── model_loader.py    # Model loading utilities
├── config/                # Configuration files
│   └── config.yaml        # Application configuration
├── templates/             # HTML templates
├── static/                # Static assets (CSS, JS)
├── logger/                # Logging configuration
├── exception/             # Custom exception handling
├── prompt/                # Prompt templates
└── tests/                 # Test suite
```

## ⚙️ Configuration

The application uses `config/config.yaml` for configuration:

```yaml
faiss_db:
  collection_name: "document_portal"

embedding_model:
  provider: "google"  # google, openai, huggingface
  model_name: "models/text-embedding-004"

retriever:
  top_k: 10

llm:
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
    temperature: 0
    max_output_tokens: 2048
  google:
    provider: "google"
    model_name: "gemini-2.0-flash"
    temperature: 0
    max_output_tokens: 2048
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_document_analyzer.py
```

## 📊 Performance & Scalability

### Optimizations
- **Efficient Vector Search**: FAISS for fast similarity search
- **Chunking Strategy**: Configurable chunk size and overlap for optimal retrieval
- **Session Management**: Efficient session-based document indexing
- **Async Processing**: Non-blocking API operations

### Scalability Features
- **Modular Architecture**: Easy to extend with new LLM providers
- **Configurable Models**: Switch between different LLM and embedding models
- **Multiple Storage Options**: In-memory, on-disk, and cloud vector databases
- **RESTful API**: Standard HTTP interface for integration

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
pylint src/

# Run tests
pytest
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the excellent RAG framework
- **FastAPI** for the high-performance web framework
- **FAISS** for efficient vector similarity search
- **OpenAI, Google, Groq** for providing powerful LLM APIs

## 📞 Support


---

**Built with ❤️ by Saikat Santra**

*Transform your documents into intelligent insights with the power of RAG technology.*


