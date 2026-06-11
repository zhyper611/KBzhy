"""KBzhy — 企业级 RAG 知识库问答系统入口（FastAPI）

启动方式:
    cd RAG-Studio-Lab
    uvicorn KBzhy.main:app --reload --host 0.0.0.0 --port 8000

访问:
    http://localhost:8000/docs — Swagger API 文档
    http://localhost:8000/api/health — 健康检查
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保父目录在 sys.path 中，使 `python main.py` 也能正确解析 KBzhy 包导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from KBzhy.app.api.chat import router as chat_router
from KBzhy.app.api.documents import router as documents_router
from KBzhy.config import API_KEY, API_BASE

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kbzhy")

app = FastAPI(
    title="KBzhy RAG 知识库问答系统",
    description="基于阿里云百炼平台的企业级 RAG 知识库问答系统",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(chat_router)
app.include_router(documents_router)

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "llm": "qwen3.6-plus",
            "embedding": "text-embedding-v4",
            "reranker": "qwen3-vl-rerank",
            "api_base": API_BASE,
            "api_configured": bool(API_KEY and API_KEY != "your-api-key"),
        },
    }


@app.on_event("startup")
async def on_startup():
    logger.info("=" * 50)
    logger.info("KBzhy RAG 系统启动")
    logger.info("API Base: %s", API_BASE)
    logger.info("API Key 已配置: %s", bool(API_KEY and API_KEY != "your-api-key"))
    logger.info("=" * 50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("KBzhy.main:app", host="0.0.0.0", port=8000, reload=True)
