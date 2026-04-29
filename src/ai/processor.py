"""
ai/processor.py — Обработка статей через Google Gemini API
Бесплатный тир: 15 запросов/мин, 1500 запросов/день
"""
import asyncio
import os
import logging

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from loguru import logger
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

SYSTEM_PROMPT = """Ты — финансовый аналитик, который пишет краткие посты для Telegram-канала.

Получив новость, верни СТРОГО в этом формате (без лишнего текста до и после):

📊 {краткий заголовок, максимум 10 слов}

Что произошло:
{1-2 предложения — только факты}

Почему это важно:
{1-2 предложения — влияние на рынки, экономику, крипту}

Что делать:
{1 предложение — практический вывод для инвестора}

Правила:
- Пиши на русском языке
- Используй конкретные цифры если они есть в тексте
- Весь пост не длиннее 7 строк
- Не добавляй ничего кроме указанного формата"""

# Отключаем фильтры безопасности для финансовых новостей
# (иначе Gemini может блокировать новости про санкции, кризисы и т.д.)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

GENERATION_CONFIG = genai.GenerationConfig(
    max_output_tokens=600,
    temperature=0.4,
    top_p=0.9,
)


class AIProcessor:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY не задан — добавь в Railway → Variables")

        genai.configure(api_key=api_key)

        # gemini-2.0-flash — бесплатный, актуальная модель 2025
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=SYSTEM_PROMPT,
        )

        # Не более 5 параллельных запросов (бесплатный лимит: 15/мин)
        self._sem = asyncio.Semaphore(5)

        logger.info("Gemini AI процессор инициализирован (модель: gemini-2.0-flash)")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )
    async def _call_gemini(self, prompt: str) -> str | None:
        """Вызов Gemini API в executor (SDK синхронный)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(prompt),
        )

        if not response.candidates:
            logger.warning("Gemini вернул пустой ответ (заблокирован контент)")
            return None

        candidate = response.candidates[0]
        if candidate.finish_reason.name not in ("STOP", "MAX_TOKENS"):
            logger.warning(f"Gemini finish_reason: {candidate.finish_reason.name}")
            return None

        return response.text.strip()

    async def process_article(self, article: dict) -> str | None:
        title  = article.get("title", "")
        text   = article.get("raw_text", "")
        source = article.get("source_id", "unknown")
        url    = article.get("url", "")

        if not title:
            return None

        prompt = (
            f"Источник: {source}\n"
            f"Заголовок: {title}\n"
            f"Текст: {text[:3000]}\n"
            f"Ссылка: {url}"
        )

        try:
            async with self._sem:
                post = await self._call_gemini(prompt)
                await asyncio.sleep(1.0)   # пауза чтобы не превысить 15 req/min

            if not post:
                return None

            post += f"\n\n🔗 [Читать полностью]({url})"
            logger.debug(f"Gemini обработал: {title[:60]}")
            return post

        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                logger.warning("Gemini rate limit — ждём 60 секунд")
                await asyncio.sleep(60)
            else:
                logger.error(f"Gemini ошибка для '{title[:50]}': {e}")
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

        logger.info(f"Gemini обработал {len(results)}/{len(articles)} статей")
        return results
