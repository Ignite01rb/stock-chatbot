"""
api.py  –  FastAPI wrapper around your LangGraph chatbot (backend.py)

Local:
    uvicorn api:app --reload --port 8000

Render:
    uvicorn api:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import pathlib

from backend import chatbot

app = FastAPI(title="Stock Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── models ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class HITLResumeRequest(BaseModel):
    thread_id: str
    decision: str

class ChatResponse(BaseModel):
    reply: str
    hitl_prompt: str | None = None
    thread_id: str

# ── helpers ───────────────────────────────────────────────────────────────

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

def _last_text(result: dict) -> str:
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
    return ""

def _extract_hitl(result: dict) -> str | None:
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        return interrupts[0].value
    return None

# ── serve frontend ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve index.html — works both locally and on Render."""
    html_path = pathlib.Path(__file__).parent / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h2>index.html not found next to api.py</h2>", status_code=404)
    html = html_path.read_text()
    # Patch API base so frontend calls work on any domain
    html = html.replace(
        'const API_BASE = "http://localhost:8000"',
        'const API_BASE = ""'
    )
    return HTMLResponse(html)

# ── API routes ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = {"messages": [HumanMessage(content=req.message)]}
    try:
        result = chatbot.invoke(state, config=_config(req.thread_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    hitl = _extract_hitl(result)
    return ChatResponse(reply=_last_text(result), hitl_prompt=hitl, thread_id=req.thread_id)

@app.post("/chat/resume", response_model=ChatResponse)
def resume(req: HITLResumeRequest):
    try:
        result = chatbot.invoke(Command(resume=req.decision), config=_config(req.thread_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    hitl = _extract_hitl(result)
    return ChatResponse(reply=_last_text(result), hitl_prompt=hitl, thread_id=req.thread_id)

@app.get("/health")
def health():
    return {"status": "ok"}