"""
deduplication/dedup.py — Дедупликация по заголовку и семантике
"""
import re
from difflib import SequenceMatcher
from loguru import logger

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("sentence-transformers недоступен, используем только title-dedup")


def _normalize(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    return re.sub(r"\s+", " ", title).strip()


def _title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class Deduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.model = None

        if SEMANTIC_AVAILABLE:
            try:
                self.model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
                logger.info("Семантическая дедупликация включена")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель эмбеддингов: {e}")

    def compute_embedding(self, text: str) -> list[float] | None:
        if self.model is None:
            return None
        try:
            emb = self.model.encode(text[:512], normalize_embeddings=True)
            return emb.tolist()
        except Exception as e:
            logger.warning(f"Ошибка эмбеддинга: {e}")
            return None

    def cosine_sim(self, a: list[float], b: list[float]) -> float:
        if not SEMANTIC_AVAILABLE:
            return 0.0
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def is_duplicate(
        self,
        candidate: dict,
        existing_titles: list[str],
        existing_embeddings: list[dict],
    ) -> bool:
        title = candidate.get("title", "")

        # 1. Проверка по заголовку
        for t in existing_titles:
            if _title_sim(title, t) >= self.threshold:
                logger.debug(f"Дубль по заголовку: '{title[:50]}'")
                return True

        # 2. Семантическая проверка
        if candidate.get("embedding") and existing_embeddings:
            for ex in existing_embeddings:
                if self.cosine_sim(candidate["embedding"], ex["embedding"]) >= self.threshold:
                    logger.debug(f"Семантический дубль: '{title[:50]}'")
                    return True

        return False

    def filter_duplicates(
        self,
        candidates: list[dict],
        existing_titles: list[str],
        existing_embeddings: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        unique, duplicates = [], []
        seen_titles = list(existing_titles)
        seen_embeddings = list(existing_embeddings)

        for c in candidates:
            text = f"{c.get('title', '')} {c.get('raw_text', '')}"
            emb = self.compute_embedding(text)
            c = {**c, "embedding": emb}

            if self.is_duplicate(c, seen_titles, seen_embeddings):
                duplicates.append(c)
            else:
                unique.append(c)
                seen_titles.append(c.get("title", ""))
                if emb:
                    seen_embeddings.append({"id": c.get("id", 0), "embedding": emb})

        logger.info(f"Дедупликация: {len(unique)} уникальных / {len(duplicates)} дублей")
        return unique, duplicates
