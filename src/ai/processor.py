"""
ai/processor.py — Обработка статей через Groq API
Бесплатный тир: 30 RPM, 14 400 запросов/день (llama-3.1-8b-instant)
Карта не нужна — регистрация на console.groq.com
"""
import asyncio
import os
from groq import AsyncGroq
from loguru import logger

# Разные форматы постов для разнообразия
POST_FORMATS = [
    "разговорный — как будто рассказываешь другу за кофе, живо и с эмоцией",
    "аналитический — чёткие факты и цифры, без воды",
    "провокационный — начни с неожиданного угла зрения на новость",
    "практический — сразу о том что это значит для кошелька читателя",
]

import random as _random
_current_format = _random.choice(POST_FORMATS)

SYSTEM_PROMPT = f"""Ты — автор популярного Telegram канала о финансах. Пишешь живо, по-человечески, без канцелярита.

Получив новость, напиши пост на русском языке в стиле: {_current_format}

Формат поста (строго):

[эмодзи] [цепляющий заголовок — максимум 8 слов]

[2-3 живых предложения — суть новости простым языком с цифрами]

[1 предложение — почему это важно для обычного человека]

💡 [1 практический вывод — что делать прямо сейчас]

Правила:
- Русский язык без калек: "ставка" не "рейт", "акции" не "стокс"
- Названия оставляй: Fed, S&P 500, Bitcoin, BlackRock
- Точные цифры из оригинала
- Короткие предложения — легко читать в телефоне
- Никакого канцелярита и пассивного залога
- Максимум 6 строк
- Только пост, без лишних слов вокруг"""


class AIProcessor:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY не задан — добавь в Railway → Variables")

        self.client = AsyncGroq(api_key=api_key)
        # llama-3.1-8b-instant — 14 400 запросов/день бесплатно
        self.model = "llama-3.1-8b-instant"
        # 30 RPM лимит → пауза 2с между запросами = максимум 30/мин
        self._sem = asyncio.Semaphore(1)
        logger.info(f"Groq AI процессор инициализирован (модель: {self.model})")

    async def _call_groq(self, title: str, text: str, source: str, url: str) -> str | None:
        """Вызов Groq API — нативный async, до 3 попыток."""
        user_content = (
            f"Источник: {source}\n"
            f"Заголовок: {title}\n"
            f"Текст: {text[:3000]}\n"
            f"Ссылка: {url}"
        )

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content},
                    ],
                    max_tokens=600,
                    temperature=0.4,
                )
                text_out = response.choices[0].message.content
                if text_out and text_out.strip():
                    return text_out.strip()
                logger.warning(f"Groq вернул пустой ответ (попытка {attempt+1})")

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "quota" in err.lower():
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Groq rate limit — ждём {wait}с")
                    await asyncio.sleep(wait)
                elif attempt < 2:
                    logger.warning(f"Groq ошибка, повтор через 3с: {err[:100]}")
                    await asyncio.sleep(3)
                else:
                    logger.error(f"Groq финальная ошибка: {err[:150]}")
                    return None

        return None

    async def process_article(self, article: dict) -> str | None:
        title  = article.get("title", "")
        text   = article.get("raw_text", "")
        source = article.get("source_id", "unknown")
        url    = article.get("url", "")

        if not title:
            return None

        try:
            async with self._sem:
                post = await self._call_groq(title, text, source, url)
                # Пауза 2с = максимум 30 req/min (ровно по лимиту)
                await asyncio.sleep(2.0)

            if not post:
                return None

            post += f"\n\n🔗 [Читать полностью]({url})"
            logger.debug(f"Groq обработал: {title[:60]}")
            return post

        except Exception as e:
            logger.error(f"Groq ошибка для '{title[:50]}': {e}")
            return None

    async def process_batch(self, articles: list[dict]) -> list[tuple[dict, str]]:
        async def _process(art):
            post = await self.process_article(art)
            return art, post

        raw = await asyncio.gather(
            *[_process(a) for a in articles],
            return_exceptions=True,
        )

        results = []
        for r in raw:
            if isinstance(r, Exception):
                logger.error(f"Ошибка в батче: {r}")
            elif isinstance(r, tuple) and r[1] is not None:
                results.append(r)

        logger.info(f"Groq обработал {len(results)}/{len(articles)} статей")
        return results
