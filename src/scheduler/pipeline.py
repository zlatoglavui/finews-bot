"""
scheduler/pipeline.py — Главный оркестратор
Parser → Filter → AI → Deduplication → Scheduler → Telegram
"""
import asyncio
import json
import os
from datetime import datetime
from loguru import logger

from src.parsers.parsers import fetch_all_sources
from src.filters.keyword_filter import KeywordFilter
from src.ai.processor import AIProcessor
from src.deduplication.dedup import Deduplicator
from src.telegram.bot import TelegramPublisher
from src.storage.database import Database


class NewsPipeline:
    def __init__(self):
        self.db        = Database()
        self.kw_filter = KeywordFilter("config/sources.json")
        self.ai        = AIProcessor()
        self.dedup     = Deduplicator(
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
        )
        self.publisher = TelegramPublisher()
        self._tg_ok    = False

        with open("config/sources.json", encoding="utf-8") as f:
            cfg = json.load(f)
        self.sources = cfg["sources"]

    async def startup(self):
        await self.db.connect()
        self._tg_ok = await self.publisher.test_connection()
        if not self._tg_ok:
            logger.warning("⚠️  Telegram недоступен — публикации не будет")

        # Сбрасываем статус error → pending чтобы переобработать статьи
        # которые не прошли из-за ошибок Gemini в предыдущих запусках
        recovered = await self.db.reset_error_articles()
        if recovered > 0:
            logger.info(f"Восстановлено {recovered} статей со статусом error → pending")

        logger.info("Pipeline запущен ✓")

    async def shutdown(self):
        await self.db.close()

    # ── Шаг 1: Сбор ──────────────────────────────────────────

    async def step_fetch(self) -> list[dict]:
        return await fetch_all_sources(self.sources)

    # ── Шаг 2: Фильтр ────────────────────────────────────────

    async def step_filter(self, articles: list[dict]) -> list[dict]:
        passed, rejected = self.kw_filter.filter_batch(articles)
        for a in rejected:
            if not await self.db.article_exists(a["url"]):
                await self.db.insert_article({**a, "status": "filtered"})
        return passed

    # ── Шаг 3: Хранение + дедупликация ───────────────────────

    async def step_store_and_dedup(self, articles: list[dict]) -> list[dict]:
        new_articles = []
        for a in articles:
            if not await self.db.article_exists(a["url"]):
                new_articles.append(a)

        # Подхватываем статьи pending (восстановленные из error)
        pending = await self.db.get_articles_by_status("pending", limit=3)
        if pending:
            logger.info(f"Подхвачено {len(pending)} pending статей для обработки")

        if not new_articles and not pending:
            logger.info("Новых статей нет")
            return []

        # Обрабатываем pending статьи из БД
        result = list(pending)

        # Новые — дедуплицируем и сохраняем
        if new_articles:
            existing_pending   = await self.db.get_articles_by_status("pending")
            existing_processed = await self.db.get_articles_by_status("processed")
            existing_titles = [e["title"] for e in existing_pending + existing_processed]
            existing_embs   = await self.db.get_embeddings_for_dedup()

            unique, duplicates = self.dedup.filter_duplicates(
                new_articles, existing_titles, existing_embs
            )

            for a in unique:
                emb = a.pop("embedding", None)
                emb_json = json.dumps(emb) if emb else "[]"
                row_id = await self.db.insert_article(a)
                if row_id:
                    await self.db.update_article(row_id, embedding=emb_json)
                    a["id"] = row_id
                    result.append(a)

            for a in duplicates:
                a.pop("embedding", None)
                await self.db.insert_article({**a, "status": "duplicate"})

            if unique:
                logger.info(f"Сохранено {len(unique)} новых уникальных статей")

        return result

    # ── Шаг 4: AI обработка ───────────────────────────────────

    async def step_ai(self, articles: list[dict]) -> list[tuple[dict, str]]:
        if not articles:
            return []

        pairs = await self.ai.process_batch(articles)

        processed_ids = {id(art) for art, _ in pairs}
        for art in articles:
            if id(art) not in processed_ids and art.get("id"):
                await self.db.update_article(art["id"], status="error")

        return pairs

    # ── Шаг 5: Публикация ─────────────────────────────────────

    async def step_publish(self, pairs: list[tuple[dict, str]]) -> int:
        if not self._tg_ok:
            logger.warning("Telegram недоступен, пропускаем публикацию")
            for art, _ in pairs:
                if art.get("id"):
                    await self.db.update_article(art["id"], status="processed")
            return 0

        published_count = 0

        for art, post in pairs:
            today_cnt  = await self.db.count_published_today(self.publisher.channel)
            last_pub   = await self.db.last_publish_time(self.publisher.channel)
            can, reason = self.publisher.can_publish(today_cnt, last_pub)

            if not can:
                logger.info(f"Публикация отложена: {reason}")
                if art.get("id"):
                    await self.db.update_article(art["id"], status="processed")
                continue

            msg_id = await self.publisher.send_message(post)

            if msg_id:
                if art.get("id"):
                    await self.db.update_article(art["id"], status="published", summary=post)
                    await self.db.log_published(
                        art["id"], self.publisher.channel, msg_id, post
                    )
                published_count += 1
            else:
                if art.get("id"):
                    await self.db.update_article(art["id"], status="processed")

            await asyncio.sleep(2)

        return published_count

    # ── Полный цикл ───────────────────────────────────────────

    async def run_once(self):
        start = datetime.utcnow()
        logger.info("─" * 40)
        logger.info(f"Старт цикла: {start.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        stats = {"fetched": 0, "passed_filter": 0, "new": 0, "published": 0, "errors": 0}

        try:
            raw = await self.step_fetch()
            stats["fetched"] = len(raw)

            passed = await self.step_filter(raw)
            stats["passed_filter"] = len(passed)

            articles = await self.step_store_and_dedup(passed)
            stats["new"] = len(articles)

            pairs = await self.step_ai(articles)
            stats["errors"] = len(articles) - len(pairs)

            published = await self.step_publish(pairs)
            stats["published"] = published

        except Exception as e:
            logger.exception(f"Критическая ошибка в пайплайне: {e}")
            stats["errors"] += 1

        elapsed = (datetime.utcnow() - start).total_seconds()
        logger.info(
            f"Цикл завершён за {elapsed:.1f}с | "
            f"собрано={stats['fetched']} "
            f"прошло_фильтр={stats['passed_filter']} "
            f"новых={stats['new']} "
            f"опубликовано={stats['published']} "
            f"ошибок={stats['errors']}"
        )
        return stats
