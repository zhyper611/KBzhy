from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from KBzhy.app.api.chat import router as chat_router
from KBzhy.app.api.documents import router as documents_router
from KBzhy.config import API_KEY, API_BASE, LLM_MODEL, EMBEDDING_MODEL, RERANKER_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kbzhy")

app = FastAPI(
    title="KBzhy RAG Knowledge Base QA System",
    description="Enterprise-style RAG knowledge base QA system based on FastAPI, ChromaDB and OpenAI-compatible APIs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/api/health")
def health_check():
    metadata_ok = False
    try:
        from KBzhy.app.core.metadata_store import get_metadata_store

        get_metadata_store().list_knowledge_bases()
        metadata_ok = True
    except Exception:
        metadata_ok = False
    return {
        "status": "healthy" if metadata_ok else "degraded",
        "version": "1.0.0",
        "components": {
            "llm": LLM_MODEL,
            "embedding": EMBEDDING_MODEL,
            "reranker": RERANKER_MODEL,
            "api_base": API_BASE,
            "api_configured": bool(API_KEY and API_KEY != "your-api-key"),
            "metadata_store": "ok" if metadata_ok else "unavailable",
        },
    }


@app.on_event("startup")
async def on_startup():
    logger.info("=" * 50)
    logger.info("KBzhy RAG system starting")
    logger.info("API Base: %s", API_BASE)
    logger.info("API Key configured: %s", bool(API_KEY and API_KEY != "your-api-key"))
    try:
        from KBzhy.app.core.indexing_worker import get_indexing_worker

        worker = get_indexing_worker()
        worker.recover_unfinished_tasks()
        logger.info("Document indexing worker started and recovery scan finished")
    except Exception as exc:
        logger.warning("Document indexing worker startup failed: %s", exc)
    logger.info("=" * 50)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("KBzhy.main:app", host="0.0.0.0", port=8000, reload=True)
