"""
main.py — Financial News Bot entry point (Railway-ready)
"""
import asyncio
import os
import signal
import sys
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from loguru import logger

# ── Загрузка .env (локально); на Railway переменные уже в окружении ──
load_dotenv()

# ── Создаём нужные папки при старте (Railway volume монтируется в runtime) ──
Path("/app/data").mkdir(parents=True, exist_ok=True)
Path("/app/logs").mkdir(parents=True, exist_ok=True)

# ── Настройка логирования ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger.remove()
logger.add(
    sys.stdout,   # Railway читает stdout, не stderr
    level=LOG_LEVEL,
    colorize=False,  # Railway не поддерживает ANSI в логах
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)
logger.add(
    "/app/logs/finews.log",
    level="DEBUG",
    rotation="10 MB",
    retention="14 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
)

# ── Проверка обязательных переменных перед стартом ────────────────────────
REQUIRED_VARS = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHANNEL_ID", "GEMINI_API_KEY"]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    logger.error(f"Отсутствуют обязательные переменные окружения: {missing}")
    logger.error("Добавь их в Railway → твой сервис → Variables")
    sys.exit(1)

# ── Импорт после проверки переменных ──────────────────────────────────────
from src.scheduler.pipeline import NewsPipeline


async def main():
    logger.info("=" * 50)
    logger.info("Financial News Bot стартует...")
    logger.info("=" * 50)

    pipeline = NewsPipeline()
    await pipeline.startup()

    interval = int(os.getenv("FETCH_INTERVAL_MINUTES", "10"))
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        pipeline.run_once,
        trigger=IntervalTrigger(minutes=interval),
        id="news_pipeline",
        name="Financial News Pipeline",
        max_instances=1,
        replace_existing=True,
        misfire_grace_time=60,
    )

    scheduler.start()
    logger.info(f"Планировщик запущен — каждые {interval} минут")

    # Сразу запускаем первый раз, не ждём интервала
    logger.info("Запускаем первый цикл сбора новостей...")
    await pipeline.run_once()

    # Graceful shutdown для Railway SIGTERM
    stop_event = asyncio.Event()

    def handle_signal():
        logger.info("Получен сигнал остановки, завершаем работу...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    await stop_event.wait()

    scheduler.shutdown(wait=False)
    await pipeline.shutdown()
    logger.info("Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())
