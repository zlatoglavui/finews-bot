"""
ai/processor.py — Обработка статей через Google Gemini API
Совместимо с google-generativeai==0.7.2
"""
import asyncio
import os

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from loguru import logger

# System prompt встроен в user message — так надёжнее для SDK 0.7.x
PROMPT_TEMPLATE = """Ты — финансовый аналитик, который пишет краткие посты для Telegram-канала.

Получив новость, верни СТРОГО в этом формате (без лишнего текста до и после):

📊 [краткий заголовок, максимум 10 слов]

Что произошло:
[1-2 предложения — только факты]

Почему это важно:
[1-2 предложения — влияние на рынки, экономику, крипту]

Что делать:
[1 предложение — практический вывод для инвестора]

Правила:
- Пиши на русском языке
- Используй конкретные цифры если они есть в тексте
- Весь пост не длиннее 7 строк
- Не добавляй ничего кроме указанного формата

---
Источник: {source}
Заголовок: {title}
Текст: {text}
Ссылка: {url}"""

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


def _extract_text(response) -> str | None:
    """Извлекает текст из ответа Gemini — работает с любой версией SDK."""
    try:
        # Способ 1: прямой доступ через parts (самый надёжный)
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text and part.text.strip():
                    return part.text.strip()
    except Exception:
        pass

    try:
        # Способ 2: через response.text
        if response.text and response.text.strip():
            return response.text.strip()
    except Exception:
        pass

    return None


class AIProcessor:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY не задан")

        genai.configure(api_key=api_key)

        # Без system_instruction — не поддерживается в SDK 0.7.x
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )

        self._sem = asyncio.Semaphore(5)
        logger.info("Gemini AI процессор инициализирован (модель: gemini-2.0-flash)")

    def _make_prompt(self, article: dict) -> str:
        return PROMPT_TEMPLATE.format(
            source=article.get("source_id", "unknown"),
            title=article.get("title", ""),
            text=article.get("raw_text", "")[:3000],
            url=article.get("url", ""),
        )

    async def _call_gemini(self, prompt: str) -> str | None:
        """Вызов Gemini API — до 3 попыток с паузой."""
        for attempt in range(3):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(prompt),
                )
                text = _extract_text(response)
                if text:
                    return text
                logger.warning(f"Gemini вернул пустой текст (попытка {attempt+1})")

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    logger.warning("Gemini rate limit — ждём 60 секунд")
                    await asyncio.sleep(60)
                elif attempt < 2:
                    wait = 4 * (attempt + 1)
                    logger.warning(f"Gemini ошибка, повтор через {wait}с: {err[:100]}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Gemini финальная ошибка: {err[:150]}")
                    return None

        return None

    async def process_article(self, article: dict) -> str | None:
        title = article.get("title", "")
        url   = article.get("url", "")

        if not title:
            return None

        try:
            async with self._sem:
                post = await self._call_gemini(self._make_prompt(article))
                await asyncio.sleep(1.0)

            if not post:
                return None

            post += f"\n\n🔗 [Читать полностью]({url})"
            logger.debug(f"Gemini обработал: {title[:60]}")
            return post

        except Exception as e:
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
