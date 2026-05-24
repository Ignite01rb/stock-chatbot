"""
api.py  –  FastAPI wrapper around your LangGraph chatbot (backend.py)

Run with:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command

# ── import your compiled graph ──────────────────────────────────────────────
from backend import chatbot          # the compiled LangGraph app
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Stock Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── request / response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class HITLResumeRequest(BaseModel):
    thread_id: str
    decision: str                  # "yes" or anything else

class ChatResponse(BaseModel):
    reply: str
    hitl_prompt: str | None = None  # set when graph is paused for approval
    thread_id: str

# ── helpers ──────────────────────────────────────────────────────────────────

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _last_text(result: dict) -> str:
    msgs = result.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    return last.content if hasattr(last, "content") else str(last)


def _extract_hitl(result: dict) -> str | None:
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        return interrupts[0].value
    return None

# ── routes ───────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a user message. Returns the bot reply, or a HITL prompt if approval is needed."""
    state = {"messages": [HumanMessage(content=req.message)]}
    try:
        result = chatbot.invoke(state, config=_config(req.thread_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    hitl = _extract_hitl(result)
    return ChatResponse(
        reply=_last_text(result),
        hitl_prompt=hitl,
        thread_id=req.thread_id,
    )


@app.post("/chat/resume", response_model=ChatResponse)
def resume(req: HITLResumeRequest):
    """Resume a paused graph with a human decision (yes / no)."""
    try:
        result = chatbot.invoke(
            Command(resume=req.decision),
            config=_config(req.thread_id),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    hitl = _extract_hitl(result)
    return ChatResponse(
        reply=_last_text(result),
        hitl_prompt=hitl,
        thread_id=req.thread_id,
    )


@app.get("/health")
def health():
    return {"status": "ok"}