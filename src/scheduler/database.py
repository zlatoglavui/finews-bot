"""
storage/database.py — Async SQLite manager
Путь к БД: /app/data/finews.db (Railway Volume → /app/data)
"""
import json
import os
from datetime import datetime
from typing import Optional

import aiosqlite
from loguru import logger

DB_PATH = os.getenv("DB_PATH", "/app/data/finews.db")

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    url         TEXT    UNIQUE NOT NULL,
    raw_text    TEXT,
    summary     TEXT,
    published   DATETIME,
    fetched_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    status      TEXT    DEFAULT 'pending',
    embedding   TEXT,
    error_msg   TEXT
);

CREATE TABLE IF NOT EXISTS published_posts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id   INTEGER REFERENCES articles(id),
    channel_id   TEXT    NOT NULL,
    message_id   INTEGER,
    published_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    content      TEXT
);

CREATE TABLE IF NOT EXISTS run_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_id  TEXT,
    fetched    INTEGER DEFAULT 0,
    new        INTEGER DEFAULT 0,
    published  INTEGER DEFAULT 0,
    errors     INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_articles_status  ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_source  ON articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_published_at     ON published_posts(published_at);
"""


class Database:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.executescript(CREATE_TABLES)
        await self._conn.commit()
        logger.info(f"БД подключена: {self.path}")

    async def close(self):
        if self._conn:
            await self._conn.close()
            logger.info("БД отключена")

    async def article_exists(self, url: str) -> bool:
        async with self._conn.execute(
            "SELECT 1 FROM articles WHERE url = ?", (url,)
        ) as cur:
            return await cur.fetchone() is not None

    async def insert_article(self, data: dict) -> Optional[int]:
        try:
            async with self._conn.execute(
                """INSERT INTO articles
                   (source_id, title, url, raw_text, published, status)
                   VALUES (:source_id, :title, :url, :raw_text, :published, :status)""",
                {
                    "source_id": data.get("source_id", "unknown"),
                    "title":     data.get("title", ""),
                    "url":       data.get("url", ""),
                    "raw_text":  data.get("raw_text", ""),
                    "published": data.get("published", datetime.utcnow().isoformat()),
                    "status":    data.get("status", "pending"),
                },
            ) as cur:
                await self._conn.commit()
                return cur.lastrowid
        except aiosqlite.IntegrityError:
            return None

    async def update_article(self, article_id: int, **kwargs):
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [article_id]
        await self._conn.execute(
            f"UPDATE articles SET {sets} WHERE id = ?", vals
        )
        await self._conn.commit()

    async def get_articles_by_status(self, status: str, limit: int = 50) -> list:
        async with self._conn.execute(
            "SELECT * FROM articles WHERE status = ? ORDER BY fetched_at DESC LIMIT ?",
            (status, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_embeddings_for_dedup(self) -> list:
        async with self._conn.execute(
            """SELECT id, embedding FROM articles
               WHERE embedding IS NOT NULL AND embedding != '[]'
               AND status NOT IN ('filtered', 'duplicate', 'error')
               ORDER BY fetched_at DESC LIMIT 500"""
        ) as cur:
            result = []
            for r in await cur.fetchall():
                try:
                    emb = json.loads(r["embedding"])
                    if emb:
                        result.append({"id": r["id"], "embedding": emb})
                except (json.JSONDecodeError, TypeError):
                    pass
            return result

    async def log_published(self, article_id: int, channel_id: str,
                            message_id: int, content: str):
        await self._conn.execute(
            """INSERT INTO published_posts (article_id, channel_id, message_id, content)
               VALUES (?, ?, ?, ?)""",
            (article_id, channel_id, message_id, content),
        )
        await self._conn.commit()

    async def count_published_today(self, channel_id: str) -> int:
        async with self._conn.execute(
            """SELECT COUNT(*) as cnt FROM published_posts
               WHERE channel_id = ? AND date(published_at) = date('now')""",
            (channel_id,),
        ) as cur:
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def last_publish_time(self, channel_id: str) -> Optional[datetime]:
        async with self._conn.execute(
            """SELECT published_at FROM published_posts
               WHERE channel_id = ? ORDER BY published_at DESC LIMIT 1""",
            (channel_id,),
        ) as cur:
            row = await cur.fetchone()
            if row:
                try:
                    return datetime.fromisoformat(row["published_at"])
                except ValueError:
                    return None
            return None

    async def log_run(self, source_id: str, fetched: int, new: int,
                      published: int, errors: int):
        await self._conn.execute(
            "INSERT INTO run_log (source_id, fetched, new, published, errors) VALUES (?,?,?,?,?)",
            (source_id, fetched, new, published, errors),
        )
        await self._conn.commit()

    async def reset_error_articles(self) -> int:
        """Сбрасывает статус error → pending чтобы переобработать при следующем цикле."""
        async with self._conn.execute(
            "UPDATE articles SET status = 'pending' WHERE status = 'error'"
        ) as cur:
            await self._conn.commit()
            return cur.rowcount

    async def get_stats(self) -> dict:
        stats = {}
        async with self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM articles GROUP BY status"
        ) as cur:
            stats["by_status"] = {r["status"]: r["cnt"] for r in await cur.fetchall()}
        async with self._conn.execute(
            "SELECT COUNT(*) as cnt FROM published_posts WHERE date(published_at)=date('now')"
        ) as cur:
            row = await cur.fetchone()
            stats["published_today"] = row["cnt"] if row else 0
        return stats
