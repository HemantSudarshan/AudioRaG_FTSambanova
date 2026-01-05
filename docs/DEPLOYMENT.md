# AudioRAG Deployment Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM (for embeddings model)
- NVIDIA GPU (optional, for faster embeddings)

## Quick Start (Development)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AudioRaG_FTSambanova.git
cd AudioRaG_FTSambanova
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```
ASSEMBLYAI_API_KEY=your_assemblyai_key
SAMBANOVA_API_KEY=your_sambanova_key
QDRANT_URL=http://localhost:6333
```

### 5. Start Qdrant
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 6. Download Embedding Model
Download from: https://www.mediafire.com/folder/41tpj4qu5ltyd/hf_cache

Extract to `./hf_cache` folder.

### 7. Run Application
```bash
streamlit run app.py
```

---

## Production Deployment

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY}
      - SAMBANOVA_API_KEY=${SAMBANOVA_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/audiorag
    depends_on:
      - qdrant
      - redis
      - postgres

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: audiorag
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  qdrant_data:
  redis_data:
  postgres_data:
```

### Start All Services
```bash
docker-compose up -d
```

---

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl configured
- Helm 3+

### Deploy with Helm
```bash
helm repo add audiorag https://yourusername.github.io/audiorag-charts
helm install audiorag audiorag/audiorag \
  --set secrets.assemblyai=${ASSEMBLYAI_API_KEY} \
  --set secrets.sambanova=${SAMBANOVA_API_KEY}
```

---

## Health Checks

```bash
# Liveness probe
curl http://localhost:8501/health

# Qdrant status
curl http://localhost:6333/collections
```

---

## Monitoring

### Logs
```bash
# View application logs
docker-compose logs -f app

# View all service logs
docker-compose logs -f
```

### Metrics
Access Streamlit metrics at: http://localhost:8501/_stcore/metrics

---

## Troubleshooting

### Common Issues

1. **Qdrant connection failed**
   - Ensure Qdrant is running: `docker ps | grep qdrant`
   - Check URL in environment variables

2. **Embedding model not found**
   - Download from MediaFire link above
   - Extract to `./hf_cache` directory

3. **API key errors**
   - Verify `.env` file exists and contains valid keys
   - Restart application after key changes

4. **Memory errors**
   - Increase Docker memory limit to 8GB+
   - Reduce `embed_batch_size` in config
