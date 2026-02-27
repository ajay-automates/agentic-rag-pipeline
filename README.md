<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,22&height=170&section=header&text=Agentic%20RAG%20Pipeline&fontSize=46&fontAlignY=35&animation=twinkling&fontColor=ffffff&desc=Self-Correcting%20Retrieval%20with%20Grading%2C%20Reformulation%20%26%20Hallucination%20Detection&descAlignY=55&descSize=16" width="100%" />

[![Claude](https://img.shields.io/badge/Claude_Sonnet-8B5CF6?style=for-the-badge&logo=anthropic&logoColor=white)](.)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=for-the-badge)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![Railway](https://img.shields.io/badge/Railway-Deploy-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)](.)

**Upload documents. Ask questions. The pipeline retrieves, grades, reformulates, generates, and verifies — showing every step.**

</div>

---

## Why This Exists

Basic RAG retrieves documents and generates an answer. It has no idea if the retrieved documents are actually relevant or if the generated answer is hallucinated. This pipeline fixes both problems.

**Agentic RAG** adds a reasoning layer between retrieval and generation. After retrieving documents, Claude **grades each one for relevance**. If the documents aren't good enough, the pipeline **reformulates the query** and re-retrieves. After generating an answer, it runs a **hallucination check** to verify every claim is grounded in the source documents.

The result: answers you can actually trust, with full transparency into why.

---

## Architecture

```
User asks: "What is the company's revenue growth?"
         │
         ▼
    ┌─────────┐
    │ RETRIEVE │──→ ChromaDB vector search (top 5 chunks)
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  GRADE   │──→ Claude evaluates each chunk: relevant / partially / not relevant
    └────┬────┘
         │
         ├── Enough relevant docs? ──YES──→ Continue to GENERATE
         │
         └── Not enough? ──→ REFORMULATE query ──→ Re-RETRIEVE ──→ Re-GRADE
                             (up to 2 retries)
         │
         ▼
    ┌──────────┐
    │ GENERATE │──→ Claude answers using ONLY relevant documents with citations
    └────┬─────┘
         │
         ▼
    ┌──────────────────┐
    │ HALLUCINATION     │──→ Claude checks: is every claim in the answer
    │ CHECK             │    actually supported by the source documents?
    └────┬─────────────┘
         │
         ▼
    Final answer + pipeline trace + grounding score
```

---

## What Makes This Different From Basic RAG

| Feature | Basic RAG | This Agentic Pipeline |
|---------|-----------|----------------------|
| **Retrieval** | Get top-K chunks, use them all | Get top-K, **grade each for relevance**, discard irrelevant |
| **Query handling** | Single attempt | **Reformulate and retry** if first retrieval fails |
| **Answer generation** | Generate from whatever was retrieved | Generate from **only graded-relevant** documents |
| **Hallucination** | No check | **Claude verifies** every claim against sources |
| **Transparency** | Black box | **Full pipeline trace** showing every decision |
| **Citations** | Usually none | **Source attribution** on every claim |

---

## Pipeline Steps

| Step | What Happens | Claude's Role |
|------|-------------|---------------|
| **1. Retrieve** | Vector search in ChromaDB for relevant chunks | None (embedding similarity) |
| **2. Grade** | Each retrieved chunk evaluated for relevance | Claude grades: relevant / partial / not relevant |
| **3. Decide** | Enough relevant docs? Or reformulate? | Logic based on grade counts |
| **4. Reformulate** | Generate better search query (if needed) | Claude rewrites the query |
| **5. Generate** | Produce answer from relevant docs only | Claude answers with citations |
| **6. Verify** | Check answer against source documents | Claude detects hallucinations |

---

## Quick Start

```bash
git clone https://github.com/ajay-automates/agentic-rag-pipeline.git
cd agentic-rag-pipeline
pip install -r requirements.txt

export ANTHROPIC_API_KEY="your-key"
python main.py
# Open http://localhost:8000
```

### Usage

1. Upload a PDF or paste text into the document area
2. Ask a question about the documents
3. Watch the pipeline trace — see retrieval, grading, reformulation, and hallucination check
4. The answer includes source citations and a grounding confidence score

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Claude as grader (not embeddings)** | Embedding similarity misses semantic relevance. Claude understands if a doc actually answers the question |
| **Max 2 reformulation retries** | Balances thoroughness vs latency. Most queries resolve in 1-2 attempts |
| **Separate hallucination check** | Generator and verifier are different prompts — prevents self-confirmation bias |
| **ChromaDB in-memory** | Zero config, no database setup. Swap to persistent for production |
| **Full pipeline trace** | Transparency is the whole point — users see exactly why the agent made each decision |

---

## Project Structure

```
agentic-rag-pipeline/
├── app/
│   ├── __init__.py
│   ├── agent.py          # Self-correcting RAG pipeline (grade → reformulate → generate → verify)
│   └── ingestion.py      # Document upload, chunking, ChromaDB storage
├── templates/
│   └── index.html        # Web UI with pipeline trace visualization
├── main.py               # FastAPI server
├── requirements.txt
├── Procfile
└── README.md
```

---

## What This Demonstrates

| AI Engineering Skill | Implementation |
|---------------------|----------------|
| **Agentic RAG** | Self-correcting retrieval with grading and reformulation |
| **Retrieval Evaluation** | Claude grades each retrieved chunk for relevance |
| **Query Reformulation** | Automatic query rewriting when retrieval fails |
| **Hallucination Detection** | Post-generation verification against source documents |
| **RAG Pipeline Design** | Retrieve → Grade → Decide → Generate → Verify |
| **Vector Store** | ChromaDB with HuggingFace embeddings |
| **Document Processing** | PDF extraction + text chunking with overlap |

---

## Tech Stack

`Python` `FastAPI` `Anthropic Claude` `ChromaDB` `PyMuPDF` `HuggingFace Embeddings` `Jinja2` `Railway`

---

## Related Projects

| Project | Description |
|---------|-------------|
| [AI Support Agent](https://github.com/ajay-automates/ai-support-agent) | Basic RAG with LangSmith observability (compare with this!) |
| [AI Finance Agent](https://github.com/ajay-automates/ai-finance-agent) | Claude tool-calling agent with real-time data |
| [Multi-Agent Research Team](https://github.com/ajay-automates/multi-agent-research-team) | 4 CrewAI agents for content creation |

---

<div align="center">

**Built by [Ajay Kumar Reddy Nelavetla](https://github.com/ajay-automates)** · February 2026

*RAG that knows when it's wrong — and fixes itself.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,22&height=100&section=footer" width="100%" />

</div>
