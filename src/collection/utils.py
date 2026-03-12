import asyncio
import logging
import os
import aiofiles
import httpx

def setup_logger(log_file, log_level=logging.INFO, write_console=True, write_file=True) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    if write_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if write_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

async def download(url: str, filepath: str, client: httpx.AsyncClient, logger: logging.Logger, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            async with aiofiles.open(filepath, "wb") as f:
                await f.write(response.content)
            logger.info(f"Successfully downloaded {os.path.basename(url)}")
            return
        except httpx.HTTPError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"Failed to download {os.path.basename(url)} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to download {os.path.basename(url)} after {max_retries} attempts: {e}")

async def download_with_semaphore(url: str, filepath: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, logger: logging.Logger, max_retries: int = 3):
    async with semaphore:
        await download(url, filepath, client, logger, max_retries)