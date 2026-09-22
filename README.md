# 📚 Textbook RAG Assistant — ITI AI Track

A complete **Retrieval-Augmented Generation (RAG)** question-answering assistant built on two technical textbooks. Ask any question about **Python Programming** or **Data Science** and receive a grounded, page-cited answer powered by a fully local LLM.

---

## 🏗️ Architecture

```
[ PDF Textbooks (raw/) ]
        │
        ▼
[ Text Extraction ]  ──── PyMuPDF (fitz)
        │
        ▼
[ Recursive Chunking ]  ── chunk_size=800, overlap=150
        │
        ▼
[ Sentence Embeddings ]  ─ all-MiniLM-L6-v2 (384-dim)
        │
        ▼
[ ChromaDB Vector Store ]  3,103 chunks, MMR search (k=4)
        │
        ▼
[ FastAPI Backend ]  ──────────────────────────────────────────────────┐
   POST /query → retrieve → prompt → Ollama (llama3.2:1b) → answer     │
        │                                                               │
        ▼                                                               │
[ Streamlit Frontend ]  ← cited, grounded answer with sources ─────────┘
```

---

## ✨ Key Features

- **100% Local & Private** — no API keys, no cloud calls, runs fully offline
- **Grounded answers** — the LLM is instructed to answer ONLY from retrieved context
- **Page-level citations** — every answer cites `[Source: <file>, p.<page>]`
- **MMR retrieval** — Maximum Marginal Relevance balances relevance + diversity
- **FastAPI backend** — production-ready, with `/health` + `/query`, CORS, and lifespan loading
- **Streamlit frontend** — chat-style UI with sidebar health check and quick questions

---

## 📊 Evaluation Results (Phase 2.6)

| Metric | Score |
|---|---|
| Retrieval Relevance | **90%** (9/10) |
| Answer Grounding | **90%** (9/10) |
| Factual Correctness | **90%** (9/10) |

See the full 10-question evaluation table in [`notebooks/rag_pipeline.ipynb`](notebooks/rag_pipeline.ipynb) → Section 2.6.

**Main failure case:** Question 8 (*"How does gradient descent optimise a machine learning model?"*) retrieved weakly-related passages because gradient descent is not deeply covered in either textbook. The model correctly avoided hallucinating by staying close to the retrieved context.

---

## 🗂️ Project Structure

```
iti_project/
├── notebooks/
│   ├── rag_pipeline.ipynb     # Complete RAG pipeline & evaluation (Phase 2)
│   └── phase_1.ipynb          # Domain & data exploration (Phase 1)
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, CORS, lifespan startup
│   │   ├── api/routes/query.py# GET /health, POST /query
│   │   ├── core/config.py     # Settings loaded from .env
│   │   ├── schemas/query.py   # QueryRequest / QueryResponse / HealthResponse
│   │   ├── services/
│   │   │   ├── retrieval.py   # ChromaDB connection & MMR retrieval
│   │   │   └── generation.py  # Ollama LLM prompt building & generation
│   │   └── utils/logging_config.py
│   ├── data/vector_store/     # ChromaDB persistent index (rebuilt from notebook)
│   ├── tests/test_query.py    # pytest: happy path + 422 validation tests
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py                 # Streamlit chat UI
│   ├── api_client.py          # HTTP wrapper (reads API_BASE_URL from .env)
│   ├── .env.example
│   └── requirements.txt
├── data/
│   ├── raw/                   # PDF textbooks (not in git — see below)
│   ├── rag_config.json        # Exported pipeline config (chunk size, model, etc.)
│   └── chunk_distribution.png # Chunk size histogram
├── .gitignore
└── README.md
```

---

## 📚 Domain & Data

**Domain:** Python Programming and Data Science education

**Source documents (2 PDFs, not committed — see below):**

| File | Pages | Characters |
|---|---|---|
| `Introduction_to_Python_Programming-WEB.pdf` | 397 | 558,135 |
| `Principles-of-Data-Science-WEB.pdf` | 569 | 1,274,399 |

> **How to obtain:** Both are freely available open educational resources. Place them in `data/raw/` before running the notebook.

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent, local) |
| Notebook LLM | TinyLlama-1.1B-Chat (via `transformers`, GPU/CPU) |
| Backend LLM | `llama3.2:1b` via [Ollama](https://ollama.com) |
| Backend framework | FastAPI + Uvicorn |
| Frontend | Streamlit |

---

## 🚀 Quickstart

### Prerequisites

| Tool | Minimum version | Check |
|---|---|---|
| Python | 3.10 | `python --version` |
| Ollama | latest | `ollama --version` |
| Git | any | `git --version` |

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-assistant-app.git
cd rag-assistant-app
```

### 2. Pull the Ollama model

```bash
ollama pull llama3.2:1b
```

### 3. Place the PDF textbooks

Put both PDF files in `data/raw/`:
```
data/raw/Introduction_to_Python_Programming-WEB.pdf
data/raw/Principles-of-Data-Science-WEB.pdf
```

### 4. Create a virtual environment and install notebook dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install jupyter pymupdf pandas numpy matplotlib tqdm
pip install langchain langchain-core langchain-community langchain-chroma langchain-huggingface langchain-text-splitters
pip install chromadb sentence-transformers
pip install transformers accelerate torch torchvision torchaudio
pip install tabulate
```

### 5. Run the notebook (builds the vector store)

```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```

Run all cells top-to-bottom (**Kernel → Restart & Run All**). This will:
1. Extract text from the PDFs in `data/raw/`
2. Chunk and embed the text into ChromaDB (`data/vector_store/`)
3. Run 10 benchmark questions and display the evaluation table
4. Export `data/rag_config.json`

### 6. Set up and start the backend

```bash
cd backend
pip install -r requirements.txt
# Copy .env.example to .env (already done if you cloned fresh)
cp .env.example .env   # or: copy .env.example .env  (Windows)
uvicorn app.main:app --reload
```

The backend will be available at **http://localhost:8000**. Open http://localhost:8000/docs to test via Swagger UI.

### 7. Set up and start the frontend

In a **new terminal**:

```bash
cd frontend
pip install -r requirements.txt
# Copy .env.example to .env
cp .env.example .env   # or: copy .env.example .env  (Windows)
streamlit run app.py
```

The frontend will open at **http://localhost:8501**.

---

## 🔑 Environment Variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `PROJECT_NAME` | `Textbook RAG Assistant` | API title shown in Swagger docs |
| `API_V1_STR` | `/api/v1` | Versioned route prefix |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model name (change to swap LLMs) |
| `VECTOR_STORE_DIR` | `./data/vector_store` | Path to ChromaDB persistent store |
| `COLLECTION_NAME` | `rag_textbooks` | ChromaDB collection name |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `CORS_ORIGINS` | `["http://localhost:8501", ...]` | Allowed frontend origins |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | FastAPI backend URL |

---

## 📡 API Reference

### GET `/health`

Check service status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_store_loaded": true,
  "ollama_connected": true,
  "model": "llama3.2:1b",
  "collection": "rag_textbooks"
}
```

### POST `/query`

Ask a question and receive a grounded, cited answer.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Python dictionary and how do you create one?"}'
```

**Request body:**
```json
{
  "question": "What is a Python dictionary and how do you create one?"
}
```

**Response:**
```json
{
  "answer": "A Python dictionary is a mutable data type for storing data in key-value pair format. [Source: Introduction_to_Python_Programming-WEB.pdf, p.237] You can create one using curly braces: `my_dict = {'key': 'value'}` or the `dict()` constructor. [Source: Introduction_to_Python_Programming-WEB.pdf, p.240]",
  "sources": [
    "Introduction_to_Python_Programming-WEB.pdf, p.237",
    "Introduction_to_Python_Programming-WEB.pdf, p.240"
  ]
}
```

**Error — missing field (422):**
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{}'
# → 422 Unprocessable Entity
```

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

Tests cover:
- `GET /health` → 200 with correct schema
- `POST /query` with valid question → 200 with answer + sources
- `POST /query` with missing field → 422
- `POST /query` with blank question → 422
- `POST /api/v1/query` versioned route → 200

---

## 🐳 Docker (Backend)

```bash
cd backend
docker build -t rag-backend .
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/data:/app/data \
  rag-backend
```

---

## ⚠️ Common Issues

| Problem | Fix |
|---|---|
| `collection 'rag_textbooks' not found` | Run the notebook first to build the vector store |
| `Cannot connect to Ollama` | Make sure Ollama is running: `ollama serve` |
| `model 'llama3.2:1b' not found` | Run `ollama pull llama3.2:1b` |
| Frontend shows "Backend Offline" | Start the backend: `cd backend && uvicorn app.main:app --reload` |
| Slow first query | Ollama loads the model into memory on first call — subsequent queries are faster |

---

## 📝 License

For educational use — ITI AI Track Graduation Project.
