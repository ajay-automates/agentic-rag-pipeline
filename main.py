"""
Agentic RAG Pipeline — FastAPI Server
Upload documents, ask questions, watch the self-correcting pipeline work.
"""

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

from app.agent import AgenticRAG
from app.ingestion import ingest_text, ingest_pdf, get_doc_count, clear_documents

app = FastAPI(title="Agentic RAG Pipeline", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
agent = None


def get_agent():
    global agent
    if agent is None:
        agent = AgenticRAG()
    return agent


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload/text")
async def upload_text(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    source = body.get("source", "pasted_text")
    if not text:
        return JSONResponse({"error": "Text is required"}, status_code=400)
    result = ingest_text(text, source=source)
    result["total_docs_in_store"] = get_doc_count()
    return JSONResponse(result)


@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files supported"}, status_code=400)
    content = await file.read()
    result = ingest_pdf(content, filename=file.filename)
    result["total_docs_in_store"] = get_doc_count()
    return JSONResponse(result)


@app.post("/api/query")
async def query(request: Request):
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            return JSONResponse({"error": "Question is required"}, status_code=400)
        if get_doc_count() == 0:
            return JSONResponse({"error": "No documents uploaded yet."}, status_code=400)
        result = await get_agent().query(question)
        return JSONResponse(result)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"Query failed: {str(e)}"}, status_code=500)


@app.post("/api/clear")
async def clear():
    return JSONResponse(clear_documents())


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "documents_in_store": get_doc_count()
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
