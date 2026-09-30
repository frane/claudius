"""The tutor as a web app: one chat page and a thin JSON API.

    CLAUDIUS_MODEL=fbandov/claudius-tutor-v3 uvicorn claudius.web.server:app --port 8000

Stateless: the browser holds the conversation and sends the recent turns with
each message. Nothing is stored server-side, no accounts, no logs of chat text.
One model instance, one generation at a time (a lock), because a 400M model on
a small CPU box is the budget.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..generate import roman
from ..sft.template import LEVELS, STUDENT, TEACHER, Turn

MODEL = os.environ.get("CLAUDIUS_MODEL", "fbandov/claudius-tutor-v7")
HISTORY = 6          # turns of context sent to the model
MAX_CHARS = 300      # per student message

app = FastAPI(title="Claudius", docs_url=None, redoc_url=None)
_state: dict = {}
_lock = asyncio.Lock()


def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = MODEL
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = os.path.join(snapshot_download(MODEL, allow_patterns=["final/*"]), "final")
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    from ..sft.checker import check
    check("PUELLA ROSAM AMAT.")      # load the parser now, not on the first message
    _state.update(model=model, tok=tok)


@app.on_event("startup")
async def startup():
    await asyncio.to_thread(_load)


class Msg(BaseModel):
    role: str
    text: str = Field(max_length=2000)


class ReplyIn(BaseModel):
    level: str = "tiro"
    history: list[Msg]


@app.get("/api/health")
def health():
    return {"ready": "model" in _state}


@app.post("/api/reply")
async def reply(body: ReplyIn):
    if "model" not in _state:
        raise HTTPException(503, "still loading")
    if body.level not in LEVELS:
        raise HTTPException(400, "unknown level")
    turns = [Turn(m.role, m.text) for m in body.history[-HISTORY:]
             if m.role in (STUDENT, TEACHER) and m.text.strip()]
    if not turns or turns[-1].role != STUDENT:
        raise HTTPException(400, "last turn must be the student's")
    if len(turns[-1].text) > MAX_CHARS:
        raise HTTPException(400, f"message longer than {MAX_CHARS} characters")

    from ..tutor import respond

    async with _lock:
        r = await asyncio.to_thread(respond, _state["model"], _state["tok"], turns, body.level)
    c = r["correction"]
    return {
        # Display in Roman form (V for U); `full` stays internal for the history.
        "text": roman(r["text"]),
        "correction": None if c is None else {
            "kind": c["kind"], "wrong": roman(c["wrong"]), "right": roman(c["right"]),
            "rule": roman(c["rule"]), "text": roman(c["text"])},
        "full": r["full"],
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/browser")
def browser():
    """The same tutor running entirely in the browser (transformers.js); no API use."""
    return FileResponse(Path(__file__).with_name("browser.html"))
