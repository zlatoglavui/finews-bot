"""
parsers/parsers.py — RSS и HTML парсеры с надёжной обработкой ошибок
"""
import asyncio
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = aiohttp.ClientTimeout(total=25, connect=10)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # html entities
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_date(raw) -> str:
    if not raw:
        return datetime.utcnow().isoformat()
    if isinstance(raw, datetime):
        return raw.isoformat()
    if hasattr(raw, "tm_year"):
        try:
            return datetime(*raw[:6]).isoformat()
        except Exception:
            pass
    return datetime.utcnow().isoformat()


class BaseParser(ABC):
    def __init__(self, source: dict):
        self.source = source
        self.sid = source["id"]

    @abstractmethod
    async def fetch(self) -> list[dict]:
        ...


class RSSParser(BaseParser):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=False,
    )
    async def _get_content(self, url: str) -> Optional[str]:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(url, timeout=TIMEOUT, ssl=False) as resp:
                if resp.status == 200:
                    return await resp.text(errors="replace")
                logger.warning(f"[{self.sid}] HTTP {resp.status} for {url}")
                return None

    async def fetch(self) -> list[dict]:
        url = self.source["url"]
        logger.debug(f"[{self.sid}] Fetching RSS: {url}")

        try:
            content = await self._get_content(url)
        except Exception as e:
            logger.error(f"[{self.sid}] RSS fetch failed after retries: {e}")
            return []

        if not content:
            return []

        try:
            feed = feedparser.parse(content)
        except Exception as e:
            logger.error(f"[{self.sid}] feedparser error: {e}")
            return []

        articles = []
        for entry in feed.entries[:50]:  # не больше 50 за раз
            title = clean_text(getattr(entry, "title", ""))
            link  = getattr(entry, "link", "").strip()

            if not title or not link or not link.startswith("http"):
                continue

            # Текст: content > summary > description
            raw = ""
            if hasattr(entry, "content") and entry.content:
                raw = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                raw = entry.summary
            elif hasattr(entry, "description"):
                raw = entry.description
            raw = clean_text(raw)

            published = parse_date(
                getattr(entry, "published_parsed", None)
                or getattr(entry, "updated_parsed", None)
            )

            articles.append({
                "source_id": self.sid,
                "title":     title,
                "url":       link,
                "raw_text":  raw[:5000],  # ограничиваем размер
                "published": published,
            })

        logger.info(f"[{self.sid}] Получено {len(articles)} статей из RSS")
        return articles


class HTMLParser(BaseParser):
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=3, max=8),
        reraise=False,
    )
    async def _get_html(self, url: str) -> Optional[str]:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(url, timeout=TIMEOUT, ssl=False) as resp:
                if resp.status == 200:
                    return await resp.text(errors="replace")
                return None

    async def fetch(self) -> list[dict]:
        url  = self.source["url"]
        sels = self.source.get("selectors", {})
        logger.debug(f"[{self.sid}] Fetching HTML: {url}")

        try:
            html = await self._get_html(url)
        except Exception as e:
            logger.error(f"[{self.sid}] HTML fetch failed: {e}")
            return []

        if not html:
            return []

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        parsed_origin = urlparse(url)
        base_url = f"{parsed_origin.scheme}://{parsed_origin.netloc}"

        articles = []
        for block in soup.select(sels.get("articles", "article"))[:20]:
            title_el = block.select_one(sels.get("title", "h2,h3"))
            link_el  = block.select_one(sels.get("link", "a"))
            sum_el   = block.select_one(sels.get("summary", "p"))

            title = clean_text(title_el.get_text()) if title_el else ""
            href  = (link_el.get("href", "") if link_el else "").strip()
            raw   = clean_text(sum_el.get_text()) if sum_el else ""

            if not title or not href:
                continue
            if href.startswith("/"):
                href = base_url + href
            if not href.startswith("http"):
                continue

            articles.append({
                "source_id": self.sid,
                "title":     title,
                "url":       href,
                "raw_text":  raw[:5000],
                "published": datetime.utcnow().isoformat(),
            })

        logger.info(f"[{self.sid}] Получено {len(articles)} статей из HTML")
        return articles


def get_parser(source: dict) -> BaseParser:
    t = source.get("type", "rss")
    if t == "rss":
        return RSSParser(source)
    elif t == "html":
        return HTMLParser(source)
    raise ValueError(f"Неизвестный тип парсера: {t}")


async def fetch_all_sources(sources: list[dict]) -> list[dict]:
    enabled = [s for s in sources if s.get("enabled", True)]
    if not enabled:
        logger.warning("Нет активных источников!")
        return []

    tasks = [get_parser(s).fetch() for s in enabled]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles = []
    for source, result in zip(enabled, results):
        if isinstance(result, Exception):
            logger.error(f"[{source['id']}] Сбой парсера: {result}")
        elif isinstance(result, list):
            all_articles.extend(result)

    # Дедупликация по URL прямо здесь (один источник может дать дубли)
    seen_urls = set()
    unique = []
    for a in all_articles:
        if a["url"] not in seen_urls:
            seen_urls.add(a["url"])
            unique.append(a)

    logger.info(f"Всего получено: {len(unique)} уникальных статей из {len(enabled)} источников")
    return unique
