import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils import setup_logger, download_with_semaphore

import os
from tqdm.asyncio import tqdm as atqdm

from asyncio import run
import asyncio

URL: str = "https://travel.state.gov/content/travel/en/legal/visa-law0/visa-statistics/immigrant-visa-statistics/monthly-immigrant-visa-issuances.html"
DATA_DIR: str = "./data/raw/visa"
LOG_DIR: str = "./logs"

os.makedirs(LOG_DIR, exist_ok=True)

global logger
logger = setup_logger(os.path.join(LOG_DIR, "visa_collection.log"), write_console=True)

os.makedirs(DATA_DIR, exist_ok=True)

async def main(base_url: str = "https://travel.state.gov"):
    semaphore = asyncio.Semaphore(20)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(URL)
            soup: BeautifulSoup = BeautifulSoup(response.text, "lxml")
    except httpx.HTTPError as e:
        print(f"An error occurred while fetching the URL: {e}")
        return

    a_tags = soup.find_all("a", href=True)

    excel_urls = [urljoin(base_url, tag["href"]) for tag in soup.find_all("a", string="(Excel)")]
    pdf_urls = [urljoin(base_url, tag["href"]) for tag in a_tags if tag["href"].endswith(".pdf")]
    file_prefixes = [tag.contents[0] for tag in a_tags if tag["href"].endswith(".pdf")]

    logger.info(f"Found {len(pdf_urls)} PDF URLs, {len(excel_urls)} Excel URLs, {len(file_prefixes)} file prefixes")
    
    if len(pdf_urls) == 0 or len(file_prefixes) == 0:
        logger.warning(f"No PDFs or prefixes found. PDF URLs: {pdf_urls}, Prefixes: {file_prefixes}")
        return

    os.makedirs(os.path.join(DATA_DIR, "excel"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "pdf"), exist_ok=True)

    logger.info(f"Downloading {len(pdf_urls)} PDF files...")

    async with httpx.AsyncClient() as client:
        pdf_tasks: list[asyncio.Task] = [download_with_semaphore(pdf_url, os.path.join(DATA_DIR, "pdf", f"{file_prefix}.pdf"), client, semaphore, logger) \
                 for pdf_url, file_prefix in zip(pdf_urls, file_prefixes)]

        await atqdm.gather(*pdf_tasks, desc="Downloading PDF files", unit="file")

        logger.info(f"Downloading {len(excel_urls)} Excel files...")

        excel_tasks: list[asyncio.Task] = [download_with_semaphore(url, os.path.join(DATA_DIR, "excel", os.path.basename(url)), client, semaphore, logger) \
                 for url in excel_urls]

        await atqdm.gather(*excel_tasks, desc="Downloading Excel files", unit="file")

    logger.info(f"Download completed. Excel files saved to {os.path.join(DATA_DIR, 'excel')}, PDF files saved to {os.path.join(DATA_DIR, 'pdf')}.")

if __name__ == "__main__":
    run(main())
