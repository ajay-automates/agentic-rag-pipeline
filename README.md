<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,22&height=170&section=header&text=Agentic%20RAG%20Pipeline&fontSize=46&fontAlignY=35&animation=twinkling&fontColor=ffffff&desc=Self-Correcting%20Retrieval%20with%20Grading%2C%20Reformulation%20%26%20Hallucination%20Detection&descAlignY=55&descSize=16" width="100%" />

[![Claude](https://img.shields.io/badge/Claude_Sonnet-Grading_+_Generation-8B5CF6?style=for-the-badge&logo=anthropic&logoColor=white)](.)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B6B?style=for-the-badge)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](.)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Support-orange?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![Railway](https://img.shields.io/badge/Railway-Deployed-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)](.)

**Upload documents. Ask questions. The pipeline retrieves, grades, reformulates, generates, and verifies — showing every decision in a live trace.**

</div>

---

## Why This Exists

Basic RAG retrieves documents and generates an answer. It has no idea if the retrieved chunks are actually relevant. It has no idea if the generated answer is hallucinated. You just have to trust it blindly.

**Agentic RAG** fixes both problems. After retrieving documents, Claude **grades each chunk for relevance** and discards irrelevant ones. If the documents aren't good enough, the pipeline **reformulates the query** and re-retrieves. After generating an answer, it runs a **hallucination check** that verifies every claim against the source documents and returns a confidence score.

In testing, the hallucination detector flagged 6 unsupported claims in a single response and returned `confidence: 0.3` — proving the system catches exactly the kind of errors basic RAG would silently pass through.

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
    │  GRADE   │──→ Claude evaluates each chunk: relevant / partial / not relevant
    └────┬────┘
         │
         ├── Enough relevant? ──YES──→ GENERATE answer with citations
         │
         └── Not enough? ──→ REFORMULATE query ──→ Re-RETRIEVE ──→ Re-GRADE
                             (up to 2 retries)
         │
         ▼
    ┌──────────┐
    │ GENERATE │──→ Claude answers using ONLY graded-relevant docs + [Source] citations
    └────┬─────┘
         │
         ▼
    ┌─────────────────┐
    │  HALLUCINATION   │──→ Claude verifies: is every claim actually in the sources?
    │  CHECK           │──→ Returns: grounded (true/false) + confidence (0.0-1.0)
    └────┬────────────┘
         │
         ▼
    Final answer + full pipeline trace + grounding score
```

---

## What Makes This Different From Basic RAG

| Feature | Basic RAG (my ai-support-agent) | This Agentic Pipeline |
|---------|--------------------------------|----------------------|
| **Retrieval** | Gets top-K chunks, uses all of them | Gets top-K, **Claude grades each one**, discards irrelevant |
| **Bad retrieval?** | Too bad, you get a bad answer | **Rewrites the query** and searches again (up to 2 retries) |
| **Answer quality** | Hope for the best | **Claude checks its own answer** against source docs |
| **Transparency** | Black box | **Full pipeline trace** — every retrieve, grade, decide step visible |
| **Citations** | Basic or none | Every claim tagged with `[Source: filename]` |
| **Confidence** | None | Returns `grounded: true/false` + `confidence: 0.3` score |

---

## Real Example

Uploaded a resume PDF and asked "tell me about MLOps experience":

```
Pipeline Trace:
  Attempt 1 — Retrieve:  5 chunks found (scores: 0.40, 0.35, 0.28)
  Attempt 1 — Grade:     4 relevant, 1 not relevant
  Attempt 1 — Decision:  Sufficient docs. Generating answer.
  Generate Answer:        1,982 characters with [Source] citations
  Hallucination Check:    grounded: false, confidence: 0.3
                          Issues: 6 claims not found in source chunks
```

The hallucination detector caught that Claude embellished details beyond what the retrieved chunks actually contained. Basic RAG would have returned that answer with no warning.

---

## Pipeline Steps

| Step | What Happens | Claude's Role |
|------|-------------|---------------|
| **1. Retrieve** | Vector search in ChromaDB for top 5 chunks | None (embedding similarity) |
| **2. Grade** | Evaluate each chunk for relevance | Claude grades: relevant / partial / not relevant |
| **3. Decide** | Enough relevant docs? Or reformulate? | Logic based on grade counts |
| **4. Reformulate** | Generate better search query (if needed) | Claude rewrites the query |
| **5. Generate** | Answer from relevant docs only | Claude answers with `[Source]` citations |
| **6. Verify** | Check answer against source documents | Claude detects hallucinations + confidence score |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Claude as grader (not embeddings)** | Embedding similarity misses semantic relevance. Claude understands if a doc actually answers the question |
| **Max 2 reformulation retries** | Balances thoroughness vs latency. Most queries resolve in 1-2 attempts |
| **Separate hallucination check** | Generator and verifier are different prompts — prevents self-confirmation bias |
| **ChromaDB in-memory** | Zero config, no database setup. Swap to persistent for production |
| **Full pipeline trace** | Transparency is the whole point — users see exactly why each decision was made |
| **PDF + text support** | PyMuPDF for PDF extraction, plain text paste for quick testing |

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

1. **Upload** a PDF or paste text
2. **Ask** a question about the documents
3. **Watch** the pipeline trace — retrieval, grading, reformulation, hallucination check
4. **Review** the answer with source citations and grounding confidence score

### Deploy on Railway

1. Connect this repo to Railway
2. Set `ANTHROPIC_API_KEY`
3. Deploy — Procfile handles the rest

---

## Project Structure

```
agentic-rag-pipeline/
├── app/
│   ├── __init__.py
│   ├── agent.py          # Self-correcting RAG (grade → reformulate → generate → verify)
│   └── ingestion.py      # PDF/text upload, chunking, ChromaDB storage
├── templates/
│   └── index.html        # Dark-theme UI with pipeline trace + metrics cards
├── main.py               # FastAPI server
├── requirements.txt
├── Procfile              # Railway deployment
└── README.md
```

---

## What This Demonstrates

| AI Engineering Skill | Implementation |
|---------------------|----------------|
| **Agentic RAG** | Self-correcting retrieval with grading and reformulation |
| **Retrieval Evaluation** | Claude grades each retrieved chunk for relevance |
| **Query Reformulation** | Automatic query rewriting when initial retrieval fails |
| **Hallucination Detection** | Post-generation verification with confidence scoring |
| **RAG Pipeline Design** | Retrieve → Grade → Decide → Generate → Verify |
| **Vector Store** | ChromaDB with default embeddings |
| **Document Processing** | PDF extraction (PyMuPDF) + text chunking with overlap |
| **Production Deployment** | FastAPI + Railway with full error handling |

---

## Tech Stack

`Python` `FastAPI` `Anthropic Claude` `ChromaDB` `PyMuPDF` `HuggingFace Embeddings` `Jinja2` `Railway`

---

## Related Projects

| Project | Description |
|---------|-------------|
| [AI Support Agent](https://github.com/ajay-automates/ai-support-agent) | Basic RAG with LangSmith — compare with this to see the evolution |
| [AI Finance Agent](https://github.com/ajay-automates/ai-finance-agent) | Claude tool-calling agent with real-time stock data |
| [Multi-Agent Research Team](https://github.com/ajay-automates/multi-agent-research-team) | 4 CrewAI agents for content creation |

---

<div align="center">

**Built by [Ajay Kumar Reddy Nelavetla](https://github.com/ajay-automates)** · February 2026

*RAG that knows when it's wrong — and fixes itself.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,22&height=100&section=footer" width="100%" />

</div>
