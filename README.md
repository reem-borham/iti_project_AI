# ITI AI Track — Textbook RAG Assistant

A GPU-accelerated Retrieval-Augmented Generation (RAG) teaching assistant built for Python Programming and Data Science textbooks. The system indexes textbook content into a Chroma vector store and leverages local LLMs running on CUDA for fast, grounded, citation-backed question answering.

---

## Architecture Overview

```
[ PDF Textbooks ]
       │
       ▼
[ Text Extraction & Filtering ] (PyMuPDF / fitz)
       │
       ▼
[ Recursive Chunking ] (chunk_size=800, overlap=150)
       │
       ▼
[ Sentence Embeddings ] (all-MiniLM-L6-v2 on CUDA)
       │
       ▼
[ Chroma Vector Store ] (3,103 chunks, MMR search)
       │
       ▼
[ Context-Augmented Prompt ] ──► [ Local LLM: TinyLlama-1.1B-Chat (FP16 on CUDA) ]
                                                        │
                                                        ▼
                                          [ Cited & Grounded Answer ]
```

---

## Key Features

- **End-to-End GPU Acceleration**:
  - **Embedding & Indexing**: Uses `sentence-transformers/all-MiniLM-L6-v2` running on CUDA.
  - **Inference**: Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` in FP16 precision (`torch.float16`, `device_map="auto"`) consuming ~1.1 GB VRAM with sub-second per-query inference.
  - **Fully Local & Private**: 100% offline execution without external API dependencies or costs.

- **Retrieval Strategy**:
  - Maximum Marginal Relevance (MMR) search (`k=4`, `fetch_k=20`, `lambda_mult=0.6`) balancing semantic relevance with passage diversity.

- **Evaluation Performance**:
  - **Retrieval Relevance**: 90%
  - **Answer Grounding**: 90%
  - **Factual Correctness**: 90%

---

## Repository Structure

```
iti_project_AI/
├── backend/                  # API server services (FastAPI/Flask)
├── frontend/                 # User interface
├── data/
│   ├── raw/                  # Raw PDF textbooks
│   │   ├── Introduction_to_Python_Programming-WEB.pdf
│   │   └── Principles-of-Data-Science-WEB.pdf
│   ├── chunk_distribution.png # Distribution visualization
│   └── rag_config.json       # Exported pipeline configuration
└── notebooks/
    ├── phase_1.ipynb         # Phase 1 exploratory notebook
    └── rag_pipeline.ipynb    # Complete RAG pipeline & evaluation
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install langchain langchain-core langchain-community langchain-chroma langchain-huggingface
pip install transformers accelerate pymupdf matplotlib pandas tqdm tabulate
```

### 2. Run the Pipeline

Open and run [`notebooks/rag_pipeline.ipynb`](notebooks/rag_pipeline.ipynb) in Jupyter / VS Code. It will:
1. Extract and clean text from textbooks in `data/raw/`
2. Generate chunks and save the distribution plot to `data/chunk_distribution.png`
3. Build the Chroma vector store in `data/vector_store/`
4. Answer the 10 test benchmark questions and display the evaluation matrix
5. Export configuration to `data/rag_config.json`
