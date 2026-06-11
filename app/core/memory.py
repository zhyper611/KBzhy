"""对话记忆管理 — Redis 热层 + MySQL 冷层 + 文件回退"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from KBzhy.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    REDIS_TTL,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    MAX_CONTEXT_ROUNDS,
    FILE_STORAGE_DIR,
)

logger = logging.getLogger(__name__)

# 模块级共享连接，避免每会话新建连接导致泄漏
_shared_redis: Any = None
_shared_redis_checked: bool = False
_shared_mysql: Any = None
_shared_mysql_checked: bool = False


def _get_shared_redis():
    global _shared_redis, _shared_redis_checked
    if _shared_redis_checked:
        return _shared_redis
    _shared_redis_checked = True
    try:
        import redis
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD or None,
            socket_connect_timeout=3,
            decode_responses=True,
        )
        r.ping()
        logger.info("Redis 连接成功: %s:%d", REDIS_HOST, REDIS_PORT)
        _shared_redis = r
    except Exception as exc:
        logger.warning("Redis 不可用 (%s)，回退到文件存储", exc)
    return _shared_redis


def _get_shared_mysql():
    global _shared_mysql, _shared_mysql_checked
    if _shared_mysql_checked:
        return _shared_mysql
    _shared_mysql_checked = True
    try:
        import pymysql
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            connect_timeout=3,
            charset="utf8mb4",
        )
        logger.info("MySQL 连接成功: %s:%d/%s", MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE)
        _shared_mysql = conn
    except Exception as exc:
        logger.warning("MySQL 不可用 (%s)，冷存储回退到文件", exc)
    return _shared_mysql


def _ensure_table():
    conn = _get_shared_mysql()
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL,
                role VARCHAR(16) NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT NULL,
                created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3),
                INDEX idx_session (session_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cur.close()
    except Exception as exc:
        logger.error("创建对话表失败: %s", exc)
    try:
        cur = conn.cursor()
        cur.execute("ALTER TABLE conversation_logs ADD COLUMN sources TEXT DEFAULT NULL")
        conn.commit()
        cur.close()
    except Exception:
        pass


class ConversationMemory:
    """对话记忆：Redis 热层 + MySQL 冷层，不可用时回退到文件存储"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._redis = _get_shared_redis()
        self._mysql_conn = _get_shared_mysql()
        _ensure_table()

    # ── 热层：读写当前对话窗口 ──────────────────

    def _hot_key(self) -> str:
        return f"kbzhy:session:{self.session_id}:messages"

    def get_context(self) -> list[dict[str, str]]:
        """获取当前会话上下文（最近 N 轮）"""
        if self._redis:
            try:
                raw = self._redis.get(self._hot_key())
                if raw:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis 读取失败: %s", exc)

        messages = self._file_read()
        if messages:
            return messages

        history = self.get_history()
        if history:
            messages = [{"role": h["role"], "content": h["content"]} for h in history]
            self.save_context(messages)
            return messages

        return []

    def save_context(self, messages: list[dict[str, str]]):
        """保存上下文到热层"""
        trimmed = messages[-(MAX_CONTEXT_ROUNDS * 2):]  # 每轮 = user + assistant
        if self._redis:
            try:
                self._redis.setex(
                    self._hot_key(),
                    REDIS_TTL,
                    json.dumps(trimmed, ensure_ascii=False),
                )
            except Exception as exc:
                logger.warning("Redis 写入失败: %s", exc)

        self._file_write(trimmed)

    def add_message(self, role: str, content: str, sources: list[dict] | None = None):
        ctx = self.get_context()
        msg: dict[str, Any] = {"role": role, "content": content}
        if sources:
            msg["sources"] = sources
        ctx.append(msg)
        self.save_context(ctx)
        self._save_cold(role, content, sources)

    # ── 冷层：持久化全量对话 ────────────────────

    def _save_cold(self, role: str, content: str, sources: list[dict] | None = None):
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        if self._mysql_conn:
            try:
                cur = self._mysql_conn.cursor()
                cur.execute(
                    "INSERT INTO conversation_logs (session_id, role, content, sources) VALUES (%s, %s, %s, %s)",
                    (self.session_id, role, content, sources_json),
                )
                self._mysql_conn.commit()
                cur.close()
            except Exception as exc:
                logger.warning("MySQL 写入失败: %s", exc)
                self._file_append_log(role, content, sources)
        else:
            self._file_append_log(role, content, sources)

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """从冷层获取历史对话（审计用）"""
        if self._mysql_conn:
            try:
                cur = self._mysql_conn.cursor()
                cur.execute(
                    "SELECT role, content, sources, created_at FROM conversation_logs WHERE session_id=%s ORDER BY created_at DESC LIMIT %s",
                    (self.session_id, limit),
                )
                rows = cur.fetchall()
                cur.close()
                result: list[dict[str, Any]] = []
                for r in reversed(rows):
                    entry: dict[str, Any] = {"role": r[0], "content": r[1], "created_at": str(r[3])}
                    if r[2]:
                        try:
                            entry["sources"] = json.loads(r[2])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    result.append(entry)
                return result
            except Exception as exc:
                logger.warning("MySQL 查询失败: %s", exc)
        return self._file_read_log(limit)

    def delete_session(self):
        """删除会话及所有记录"""
        if self._redis:
            try:
                self._redis.delete(self._hot_key())
            except Exception:
                pass
        if self._mysql_conn:
            try:
                cur = self._mysql_conn.cursor()
                cur.execute("DELETE FROM conversation_logs WHERE session_id=%s", (self.session_id,))
                self._mysql_conn.commit()
                cur.close()
            except Exception:
                pass
        self._file_delete()

    # ── 文件存储回退 ────────────────────────────

    def _file_path(self) -> str:
        os.makedirs(FILE_STORAGE_DIR, exist_ok=True)
        return os.path.join(FILE_STORAGE_DIR, f"{self.session_id}.json")

    def _file_log_path(self) -> str:
        os.makedirs(FILE_STORAGE_DIR, exist_ok=True)
        return os.path.join(FILE_STORAGE_DIR, f"{self.session_id}_log.jsonl")

    def _file_read(self) -> list[dict[str, str]]:
        fp = self._file_path()
        if os.path.exists(fp):
            try:
                with open(fp, encoding="utf-8") as f:
                    return json.loads(f.read())
            except Exception:
                pass
        return []

    def _file_write(self, messages: list[dict[str, str]]):
        try:
            with open(self._file_path(), "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("文件写入失败: %s", exc)

    def _file_append_log(self, role: str, content: str, sources: list[dict] | None = None):
        try:
            entry_data = {
                "session_id": self.session_id,
                "role": role,
                "content": content,
                "created_at": datetime.now().isoformat(),
            }
            if sources:
                entry_data["sources"] = sources
            entry = json.dumps(entry_data, ensure_ascii=False)
            with open(self._file_log_path(), "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as exc:
            logger.error("日志文件写入失败: %s", exc)

    def _file_read_log(self, limit: int = 100) -> list[dict[str, Any]]:
        fp = self._file_log_path()
        if not os.path.exists(fp):
            return []
        try:
            with open(fp, encoding="utf-8") as f:
                lines = f.readlines()
            entries = [json.loads(l) for l in lines[-limit:] if l.strip()]
            return entries
        except Exception:
            return []

    def _file_delete(self):
        for fp in (self._file_path(), self._file_log_path()):
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass


class MemoryManager:
    """会话级别的记忆管理器"""

    def __init__(self):
        self._sessions: dict[str, ConversationMemory] = {}

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id)
        return self._sessions[session_id]

    def delete(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].delete_session()
            del self._sessions[session_id]
        else:
            ConversationMemory(session_id).delete_session()

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有活跃会话"""
        result: list[dict[str, Any]] = []
        for sid, mem in self._sessions.items():
            ctx = mem.get_context()
            result.append({
                "session_id": sid,
                "message_count": len(ctx),
                "last_message": ctx[-1]["content"][:100] if ctx else "",
            })
        return result
