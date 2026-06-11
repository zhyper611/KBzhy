"""KBzhy RAG 知识库问答系统 — 全局配置（阿里云百炼平台）"""

import os
from dotenv import load_dotenv

load_dotenv()

# ====== 阿里云百炼 API 配置 ======
API_KEY = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "your-api-key"))
API_BASE = os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ====== 模型配置 ======
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "qwen3-vl-rerank")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

# ====== Redis 配置 ======
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))  # 会话 TTL，默认 24 小时

# ====== MySQL 配置 ======
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "kbzhy")

# ====== ChromaDB 配置 ======
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# ====== 路径 ======
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations")

# ====== 文档切分参数 ======
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

# ====== 检索参数 ======
TOP_K = 5
FETCH_K = 15  # MMR 候选数
SIMILARITY_THRESHOLD = 0.35  # 低于此值拒答
BM25_WEIGHT = 0.3  # BM25 权重
VECTOR_WEIGHT = 0.7  # 向量权重

# ====== 对话参数 ======
MAX_CONTEXT_ROUNDS = 10  # Redis 热层保留最近 N 轮
MAX_CONTEXT_TOKENS = 8000  # 超过此值触发压缩
SESSION_TTL = 1800  # Redis 会话过期时间（秒）

# ====== 文件上传 ======
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# ====== 请求超时 ======
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120

# ====== Reranker 回退策略 ======
RERANK_METHOD = os.getenv("RERANK_METHOD", "model")  # model / llm / keyword
