from utils import setup_logger, download_with_semaphore
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from tqdm.asyncio import tqdm as atqdm
import asyncio
import os

URL: str = "https://www.cbp.gov/document/stats/southwest-land-border-encounters"
DATA_DIR: str = "./data/raw/encounter"

os.makedirs(DATA_DIR, exist_ok=True)

global logger
logger = setup_logger(os.path.join("./logs", "encounter_collection.log"), write_console=False)

async def main(base_url: str = "https://www.cbp.gov"):
    semaphore = asyncio.Semaphore(20)

    logger.info(f"Fetching the URL: {URL}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(URL)
            soup: BeautifulSoup = BeautifulSoup(response.text, "lxml")
    except httpx.HTTPError as e:
        logger.warning(f"An error occurred while fetching the URL: {e}")
        return

    a_tags = soup.find_all("a", href=True)

    all_contents = [(tag.contents[0].strip(), tag["href"]) for tag in a_tags \
                if tag.contents[0] is not None and tag["href"].endswith(".csv") and tag.contents[0].strip().startswith("Southwest Land")]
        
    all_urls = [urljoin(base_url, href) for _, href in all_contents]
    file_prefixes = [content for content, _ in all_contents]

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Downloading {len(all_urls)} CSV files...")
            tasks: list[asyncio.Task] = [download_with_semaphore(url, os.path.join(DATA_DIR, f"{prefix}.csv"), client, semaphore, logger) for url, prefix in zip(all_urls, file_prefixes)]

            await atqdm.gather(*tasks, desc="Downloading CSV files", unit="file")

    except httpx.HTTPError as e:
        logger.warning(f"An error occurred during downloading: {e}")

    logger.info("Finished downloading all CSV files.")

if __name__ == "__main__":
    asyncio.run(main())