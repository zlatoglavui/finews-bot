"""
storage/database.py — PostgreSQL manager (asyncpg)
Общая БД для News Bot и Reels Bot на Railway
"""
import json
import os
from datetime import datetime
from typing import Optional
import asyncpg
from loguru import logger

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS articles (
    id          SERIAL PRIMARY KEY,
    source_id   TEXT        NOT NULL,
    title       TEXT        NOT NULL,
    url         TEXT        UNIQUE NOT NULL,
    raw_text    TEXT,
    summary     TEXT,
    published   TIMESTAMPTZ,
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    status      TEXT        DEFAULT 'pending',
    embedding   TEXT,
    error_msg   TEXT
);

CREATE TABLE IF NOT EXISTS published_posts (
    id           SERIAL PRIMARY KEY,
    article_id   INTEGER REFERENCES articles(id),
    channel_id   TEXT        NOT NULL,
    message_id   INTEGER,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    content      TEXT
);

CREATE TABLE IF NOT EXISTS run_log (
    id         SERIAL PRIMARY KEY,
    run_at     TIMESTAMPTZ DEFAULT NOW(),
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
    def __init__(self):
        url = os.getenv("DATABASE_URL", "")
        if not url:
            raise ValueError("DATABASE_URL не задан — добавь PostgreSQL сервис в Railway")
        self.url = url.replace("postgres://", "postgresql://", 1)
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self._pool = await asyncpg.create_pool(
            self.url, min_size=1, max_size=5,
            command_timeout=30, ssl="require",
        )
        async with self._pool.acquire() as conn:
            await conn.execute(CREATE_TABLES)
        logger.info("PostgreSQL подключена ✓")

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def article_exists(self, url: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT 1 FROM articles WHERE url = $1", url)
            return row is not None

    async def insert_article(self, data: dict) -> Optional[int]:
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """INSERT INTO articles
                       (source_id, title, url, raw_text, published, status)
                       VALUES ($1,$2,$3,$4,$5,$6)
                       ON CONFLICT (url) DO NOTHING
                       RETURNING id""",
                    data.get("source_id","unknown"),
                    data.get("title",""),
                    data.get("url",""),
                    data.get("raw_text",""),
                    data.get("published", datetime.utcnow().isoformat()),
                    data.get("status","pending"),
                )
                return row["id"] if row else None
        except Exception as e:
            logger.error(f"insert_article error: {e}")
            return None

    async def update_article(self, article_id: int, **kwargs):
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ${i+1}" for i, k in enumerate(kwargs))
        vals = list(kwargs.values()) + [article_id]
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE articles SET {sets} WHERE id = ${len(vals)}", *vals
            )

    async def get_articles_by_status(self, status: str, limit: int = 50) -> list:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM articles WHERE status=$1 ORDER BY fetched_at DESC LIMIT $2",
                status, limit,
            )
            return [dict(r) for r in rows]

    async def get_embeddings_for_dedup(self) -> list:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, embedding FROM articles
                   WHERE embedding IS NOT NULL AND embedding != '[]'
                   AND status NOT IN ('filtered','duplicate','error')
                   ORDER BY fetched_at DESC LIMIT 500"""
            )
            result = []
            for r in rows:
                try:
                    emb = json.loads(r["embedding"])
                    if emb:
                        result.append({"id": r["id"], "embedding": emb})
                except Exception:
                    pass
            return result

    async def log_published(self, article_id: int, channel_id: str,
                            message_id: int, content: str):
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO published_posts (article_id,channel_id,message_id,content)
                   VALUES ($1,$2,$3,$4)""",
                article_id, channel_id, message_id, content,
            )

    async def count_published_today(self, channel_id: str) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) as cnt FROM published_posts
                   WHERE channel_id=$1 AND published_at >= CURRENT_DATE""",
                channel_id,
            )
            return row["cnt"] if row else 0

    async def last_publish_time(self, channel_id: str) -> Optional[datetime]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT published_at FROM published_posts
                   WHERE channel_id=$1 ORDER BY published_at DESC LIMIT 1""",
                channel_id,
            )
            if row and row["published_at"]:
                dt = row["published_at"]
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            return None

    async def reset_error_articles(self) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE articles SET status='pending' WHERE status='error'"
            )
            return int(result.split()[-1])

    async def log_run(self, source_id: str, fetched: int, new: int,
                      published: int, errors: int):
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO run_log (source_id,fetched,new,published,errors) VALUES ($1,$2,$3,$4,$5)",
                source_id, fetched, new, published, errors,
            )

    async def get_stats(self) -> dict:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT status, COUNT(*) as cnt FROM articles GROUP BY status")
            by_status = {r["status"]: r["cnt"] for r in rows}
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM published_posts WHERE published_at >= CURRENT_DATE"
            )
            return {"by_status": by_status, "published_today": row["cnt"] if row else 0}
