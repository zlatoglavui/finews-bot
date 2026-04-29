"""
filters/keyword_filter.py — Фильтр по финансовым ключевым словам
"""
import json
import re
from loguru import logger


class KeywordFilter:
    def __init__(self, config_path: str = "config/sources.json"):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        raw = cfg.get("keywords", {})
        self.keywords: set[str] = set()
        for lang_kws in raw.values():
            for kw in lang_kws:
                self.keywords.add(kw.lower())

        # Сортируем по длине (длинные фразы проверяем первыми)
        escaped = [re.escape(kw) for kw in sorted(self.keywords, key=len, reverse=True)]
        self.regex = re.compile("|".join(escaped), re.IGNORECASE)
        logger.info(f"Фильтр загружен: {len(self.keywords)} ключевых слов")

    def matches(self, article: dict) -> bool:
        haystack = f"{article.get('title', '')} {article.get('raw_text', '')}"
        return bool(self.regex.search(haystack))

    def filter_batch(self, articles: list[dict]) -> tuple[list[dict], list[dict]]:
        passed, rejected = [], []
        for a in articles:
            (passed if self.matches(a) else rejected).append(a)
        logger.info(
            f"Фильтр: {len(passed)} прошло / {len(rejected)} отклонено"
        )
        return passed, rejected
