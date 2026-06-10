from fastapi import FastAPI
from typing import List, Any
import asyncio
from functools import partial
from pydantic import BaseModel, Field

from medium_rag.rag_chain import answer_query, SYSTEM_PROMPT

# --------- CONFIG: keep in sync with the RAG tuning ---------
# These are my CURRENT RAG hyperparameters:
CHUNK_SIZE = 512          # tokens (as in ingest.py)
OVERLAP_TOKENS = 102      # tokens
TOP_K_DEFAULT = 3         # between 1 and 30
NAMESPACE = f"tok{CHUNK_SIZE}_ov{OVERLAP_TOKENS}"

# overlap_ratio required by /api/stats:
OVERLAP_RATIO = OVERLAP_TOKENS / CHUNK_SIZE

# -------------------------------------------------------------

app = FastAPI()


# ---------- Pydantic models for request/response ----------

class PromptRequest(BaseModel):
    # min_length: rejects empty or whitespace-only questions
    # max_length: protects against abuse / runaway token usage
    question: str = Field(..., min_length=2, max_length=1000)


class ContextItem(BaseModel):
    article_id: str
    title: str
    chunk: str
    score: float


class AugmentedPrompt(BaseModel):
    System: str
    User: str


class PromptResponse(BaseModel):
    response: str
    context: List[ContextItem]
    Augmented_prompt: AugmentedPrompt


class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int


# ---------- Utility to build the augmented prompt ----------

def build_user_prompt(question: str, docs: List[Any]) -> str:
    blocks = []
    for d in docs:
        m = d.metadata
        block = (
            f"Title: {m.get('title')}\n"
            f"URL: {m.get('url')}\n"
            f"Authors: {m.get('authors')}\n"
            f"Timestamp: {m.get('timestamp')}\n"
            f"Tags: {m.get('tags')}\n\n"
            f"Passage:\n{d.page_content}"
        )
        blocks.append(block)
    context_text = "\n\n---\n\n".join(blocks)

    user_prompt = (
        f"Here are retrieved passages from Medium articles:\n\n"
        f"{context_text}\n\n"
        f"User question: {question}\n\n"
        f"Answer the question following the system instructions."
    )
    return user_prompt


# ---------- Endpoints ----------

@app.post("/api/prompt", response_model=PromptResponse)
async def prompt(request: PromptRequest):
    """
    POST /api/prompt
    Input:  { "question": "..." }
    Output: per the spec: response, context[], Augmented_prompt
    """
    question = request.question

    # Using our existing RAG chain:
    answer, docs_and_scores = await asyncio.get_event_loop().run_in_executor(
        None, partial(answer_query, question, namespace=NAMESPACE, top_k=TOP_K_DEFAULT, return_scores=True)
    )

    docs = [ds[0] for ds in docs_and_scores]
    scores = [float(ds[1]) for ds in docs_and_scores]

    # Build context array
    context_items: List[ContextItem] = []
    for d, score in zip(docs, scores):
        m = d.metadata or {}
        # article_id: can use URL or some ID, here we use URL as proxy.
        article_id = str(m.get("url") or m.get("title") or "")
        context_items.append(
            ContextItem(
                article_id=article_id,
                title=str(m.get("title") or ""),
                chunk=d.page_content,
                score=score,
            )
        )

    # Build augmented prompt
    user_prompt = build_user_prompt(question, docs)
    augmented = AugmentedPrompt(
        System=SYSTEM_PROMPT.strip(),
        User=user_prompt.strip(),
    )

    return PromptResponse(
        response=answer,
        context=context_items,
        Augmented_prompt=augmented,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    """
    GET /api/stats
    Returns the current RAG configuration.
    """
    return StatsResponse(
        chunk_size=CHUNK_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        top_k=TOP_K_DEFAULT,
    )

@app.get("/health")
async def health():
    return {"status": "ok"}