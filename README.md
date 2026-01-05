# 🎧 AudioRAG Enterprise

> AI-powered audio analytics platform with RAG-based conversational search

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Features

### Core Capabilities
- 🎙️ **Audio Transcription** - Powered by AssemblyAI with speaker diarization
- 🔍 **Semantic Search** - RAG over audio using Qdrant vector database
- 💬 **Conversational AI** - SambaNova LLM for intelligent responses
- 📊 **Analytics Dashboard** - Real-time usage metrics

### Enterprise Features
- 🔐 **Authentication** - JWT-based auth with RBAC (Admin/Analyst/Viewer)
- 🏢 **Multi-Tenant** - Organization isolation with billing
- 📝 **Audit Logs** - Compliance-ready audit trail
- ⚡ **Caching** - Redis + LRU for low latency
- 🚀 **REST API** - FastAPI endpoints with rate limiting
- 📦 **Batch Processing** - Celery workers for bulk uploads
- 🏥 **Domain Models** - Healthcare, Legal, Finance vocabularies

---

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- AssemblyAI API key
- SambaNova API key (or OpenAI)

### Installation

```bash
# Clone repository
git clone https://github.com/HemantSudarshan/AudioRaG_FTSambanova.git
cd AudioRaG_FTSambanova

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your API keys
```

### Start Services

```bash
# Start infrastructure
docker-compose up -d qdrant redis postgres

# Run Streamlit app
streamlit run app.py

# Or run enterprise version
streamlit run app_enterprise.py
```

### Run API Server

```bash
uvicorn api:app --reload --port 8000
# Visit http://localhost:8000/api/docs
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  Streamlit   │ │   REST API   │ │  WebSocket   │   │
│  │     UI       │ │  (FastAPI)   │ │  Streaming   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                 Processing Layer                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  AssemblyAI  │ │   BGE-Large  │ │   SambaNova  │   │
│  │ Transcription│ │  Embeddings  │ │     LLM      │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │    Qdrant    │ │    Redis     │ │  PostgreSQL  │   │
│  │   Vectors    │ │    Cache     │ │   Metadata   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AudioRaG_FTSambanova/
├── app.py                  # Main Streamlit app
├── app_enterprise.py       # Enterprise version with auth
├── rag_code.py            # Core RAG logic
├── config.py              # Configuration management
├── test_modules.py        # Test script
│
├── auth/                  # Authentication & RBAC
├── api/                   # REST API & streaming
├── batch/                 # Background processing
├── cache/                 # Caching layer
├── database/              # SQLAlchemy models
├── tenants/               # Multi-tenancy
├── models/                # Domain models
├── monitoring/            # Health checks
├── audit/                 # Audit logging
├── analytics/             # Usage metrics
│
├── docs/                  # Documentation
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── DEPLOYMENT.md
│
├── docker-compose.yml     # Multi-service setup
├── Dockerfile             # Production container
├── requirements.txt       # Dependencies
└── env.example            # Environment template
```

---

## 🔑 Environment Variables

```env
# Required
ASSEMBLYAI_API_KEY=your_key
SAMBANOVA_API_KEY=your_key

# Optional
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
DATABASE_URL=sqlite:///./audiorag.db
```

---

## 📊 Supported Domains

| Domain | Features |
|--------|----------|
| **Healthcare** | Medical vocabulary, HIPAA-ready, clinical prompts |
| **Legal** | Legal terminology, case references, deposition analysis |
| **Finance** | Financial metrics, compliance terms |
| **Customer Service** | CSAT analysis, ticket tracking |

---

## 🧪 Testing

```bash
# Run module tests
python test_modules.py

# Expected output:
# ✅ Imports: PASS
# ✅ Config: PASS
# ✅ Auth: PASS
# ...
```

---

## 📖 Documentation

- [Product Requirements](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

---

## 🏢 Enterprise Pricing

| Plan | Audio Hours | Storage | Price |
|------|-------------|---------|-------|
| Free | 5/month | 1 GB | $0 |
| Starter | 50/month | 25 GB | $49/mo |
| Professional | 500/month | 100 GB | $199/mo |
| Enterprise | Unlimited | Unlimited | Custom |

---

## 👤 Author

**Hemant Sudarshan**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
