"""
telegram/bot.py — Публикатор в Telegram-канал
"""
import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter, Forbidden, BadRequest
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class TelegramPublisher:
    def __init__(self):
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.channel = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()

        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN не задан")
        if not self.channel:
            raise ValueError("TELEGRAM_CHANNEL_ID не задан")

        self.bot = Bot(token=token)
        self.max_per_day = int(os.getenv("MAX_POSTS_PER_DAY", "15"))
        self.min_interval_min = int(os.getenv("POST_INTERVAL_MINUTES", "72"))

    async def test_connection(self) -> bool:
        try:
            me = await self.bot.get_me()
            logger.info(f"Telegram бот: @{me.username} ({me.first_name})")
            # Проверяем доступ к каналу
            try:
                chat = await self.bot.get_chat(self.channel)
                logger.info(f"Канал: {chat.title or self.channel}")
            except (Forbidden, BadRequest) as e:
                logger.error(
                    f"Нет доступа к каналу {self.channel}: {e}\n"
                    "→ Убедись что бот добавлен в канал как АДМИНИСТРАТОР с правом публикации"
                )
                return False
            return True
        except TelegramError as e:
            logger.error(f"Ошибка подключения к Telegram: {e}")
            return False

    async def send_message(self, text: str) -> Optional[int]:
        try:
            msg = await self.bot.send_message(
                chat_id=self.channel,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=False,
            )
            logger.info(f"✅ Опубликовано в {self.channel}, message_id={msg.message_id}")
            return msg.message_id

        except RetryAfter as e:
            logger.warning(f"Telegram rate limit, ждём {e.retry_after}с")
            await asyncio.sleep(e.retry_after + 2)
            # Одна повторная попытка
            try:
                msg = await self.bot.send_message(
                    chat_id=self.channel,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN,
                )
                return msg.message_id
            except TelegramError as e2:
                logger.error(f"Повторная ошибка отправки: {e2}")
                return None

        except Forbidden:
            logger.error(
                f"Бот заблокирован в канале {self.channel}. "
                "Проверь права администратора."
            )
            return None

        except BadRequest as e:
            # Markdown parsing error — пробуем без разметки
            logger.warning(f"Markdown ошибка, отправляем plain text: {e}")
            try:
                clean = text.replace("*", "").replace("_", "").replace("`", "")
                msg = await self.bot.send_message(
                    chat_id=self.channel,
                    text=clean,
                )
                return msg.message_id
            except TelegramError as e2:
                logger.error(f"Ошибка plain text отправки: {e2}")
                return None

        except TelegramError as e:
            logger.error(f"Telegram ошибка: {e}")
            return None

    def can_publish(
        self,
        published_today: int,
        last_publish: Optional[datetime],
    ) -> tuple[bool, str]:
        if published_today >= self.max_per_day:
            return False, f"Дневной лимит {self.max_per_day} постов достигнут"

        if last_publish:
            # Делаем оба datetime timezone-aware или оба naive
            now = datetime.utcnow()
            if last_publish.tzinfo is not None:
                last_publish = last_publish.replace(tzinfo=None)
            elapsed_min = (now - last_publish).total_seconds() / 60
            if elapsed_min < self.min_interval_min:
                wait = self.min_interval_min - elapsed_min
                return False, f"Слишком рано, ждать ещё {wait:.0f} мин"

        return True, "ok"
