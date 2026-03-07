from playwright.async_api import async_playwright, Browser, BrowserContext
import httpx
import asyncio
from bs4 import BeautifulSoup
from pygooglenews import GoogleNews
import os
import logging
import time
from urllib.parse import quote, urlparse
import json

import argparse
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

OUTPUT_PATH = "/home/ibrahim/stock/data/historical.json"

async def get_decoding_params(gn_art_id, client):
    response = await client.get(f"https://news.google.com/rss/articles/{gn_art_id}", follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response, "lxml")
    div = soup.select_one("c-wiz > div")
    return {
        "signature": div.get("data-n-a-sg"),
        "timestamp": div.get("data-n-a-ts"),
        "gn_art_id": gn_art_id,
    }

async def decode_urls(articles, client):
    articles_reqs = [
        [
            "Fbv4je",
            f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{art["gn_art_id"]}",{art["timestamp"]},"{art["signature"]}"]',
        ]
        for art in articles
    ]
    payload = f"f.req={quote(json.dumps([articles_reqs]))}"
    headers = {"content-type": "application/x-www-form-urlencoded;charset=UTF-8", "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"}
    response = await client.post("https://news.google.com/_/DotsSplashUi/data/batchexecute", headers=headers, data=payload)
    time.sleep(5)
    response.raise_for_status()
    return [json.loads(res[2])[1] for res in json.loads(response.text.split("\n\n")[1])[:-2]]

async def decode(encoded_urls):
    async with httpx.AsyncClient() as client:
        tasks = [get_decoding_params(urlparse(url).path.split("/")[-1], client) for url in encoded_urls]
        articles_params = await atqdm.gather(*tasks, desc="Fetching decoding params")
        decoded_urls = await decode_urls(articles_params, client)
        print(decoded_urls)
        return decoded_urls

def decode_async(urls):
    return asyncio.run(decode(urls))

async def get_response(client: httpx.AsyncClient, url: tuple):
    try:
        response = await client.get(url[1])

        if response.status_code == 200:
            logging.debug(f"Fetched {url[1]} successfully!")
        elif response.status_code == 302:
            try:
                location = response.headers.get("location", "")
                
                if location.startswith('/'):
                    parsed_url = urlparse(str(url[1]))
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    location = f"{base_url}{location}"
                    
                logging.debug(f"Following redirect from {url[1]} to {location}")
                
                redirect_response = await client.get(location)
                if redirect_response.status_code == 200:
                    logging.debug(f"Successfully followed redirect to {location}")
                    response = redirect_response
                else:
                    logging.warning(f"Redirect to {location} failed with status {redirect_response.status_code}")
            except (ValueError, httpx.RequestError) as e:
                logging.warning(f"Error following redirect from {url[1]}: {str(e)}")
        else:
            logging.warning(f"Bad response from {url[1]} with status {response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Request error for {url[1]}: {str(e)}")

        response = httpx.Response(status_code=0, request=httpx.Request("GET", url[1]))
        response.error_type = type(e).__name__
        response.error_msg = str(e)
    except Exception as e:
        logging.error(f"Unexpected error processing {url[1]}: {str(e)}")

        response = httpx.Response(status_code=0, request=httpx.Request("GET", url[1]))
        response.error_type = type(e).__name__
        response.error_msg = str(e)
        
    response_tuple = (url[0], response)
    return response_tuple

def get_response_sync(client: httpx.Client, url: tuple):
    response = client.get(url[1])
    response_tuple = (url[0], response)

    if response.status_code == 200:
        logging.debug(f"Fetched {url[1]} successfully!")
    else:
        logging.warning(f"Bad response from {url[1]}")

    return response_tuple

async def _fetch_page(
    url_tuple: tuple,
    context: BrowserContext,
    semaphore: asyncio.Semaphore,
) -> tuple:
    i, url = url_tuple
    async with semaphore:
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            html = await page.content()
            logging.debug(f"Playwright scraped {url}")
            return (i, BeautifulSoup(html, "html.parser"))
        except Exception as e:
            logging.error(f"Playwright error scraping {url}: {e}")
            return (i, BeautifulSoup("", "html.parser"))
        finally:
            await page.close()


async def scrape_playwright_async(urls: list, max_concurrent: int = 8) -> list:
    """Scrape a list of (index, url) tuples in parallel using Playwright."""
    logging.info("Launching headless Playwright browser")
    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context: BrowserContext = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [_fetch_page(url, context, semaphore) for url in urls]
        results = await atqdm.gather(*tasks, desc="Scraping pages")
        await context.close()
        await browser.close()
    logging.info("Playwright scraping complete")
    return list(results)


def scrape_playwright(urls: list) -> list:
    return asyncio.run(scrape_playwright_async(urls))

async def fetch_urls(urls: list) -> list:

    logging.info("Configuring async client")

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(get_response(client, url)) for url in urls]

        logging.info("Created tasks now gathering them")
        responses = await atqdm.gather(*tasks, desc="Fetching URLs")

    return responses

def run_async(urls: list) -> list:
    logging.info("Fetching URL responses through async client")
    return asyncio.run(fetch_urls(urls))

def get_news(queries: list, date_from=None, date_to=None, limit: int = 50) -> dict:
    logging.info("Fetching news from Google News API")

    gn = GoogleNews()

    all_news = {
        "title": f"News for {', '.join(queries)} from {date_from} to {date_to}",
        "totalResults": 0,
        "headlines": [],
        "articles": []
    }

    for query in tqdm(queries, desc="Searching queries"):
        results = gn.search(query, from_=date_from, to_=date_to)
        for news in results["entries"]:
            if news["title"] not in all_news["headlines"]:
                all_news["articles"].append(news)
                all_news["headlines"].append(news["title"])

    encoded_urls = [article["link"] for article in all_news["articles"]]
    decoded_urls = decode_async(encoded_urls)
    
    for i, article in enumerate(all_news["articles"]):
        article["link"] = decoded_urls[i]
        

    all_news["totalResults"] = len(all_news["articles"])

    return all_news


### MAIN FUNCTION ###

def main(queries: list, from_date: str, to_date: str, limit: int, output_path: str, log_path: str) -> int:
    
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if os.path.exists(output_path) == False:
        data = get_news(queries=queries, date_from=from_date, date_to=to_date, limit=limit)
    else:
        logging.info("News file already exists. Moving to the next step")
        return 1

    if "Information" in data.keys():
        logging.warning("No news found for the given date range")
        print(data)
        return -1

    urls = [(i,feed["link"]) for i, feed in enumerate(data["articles"])]

    logging.info("fetching the news from the URLs")
    responses = run_async(urls)

    bad_urls = []

    logging.info("Fetching detailed news from URLs")

    for i, response in tqdm(enumerate(responses), total=len(responses), desc="Processing responses"):
        try:
            if hasattr(response[1], 'error_type'):
                logging.warning(f"Skipping article at index {i} due to error: {response[1].error_type}: {response[1].error_msg}")
                data["articles"][i]["response"] = f"Error fetching content: {response[1].error_msg}"
                continue
                
            if response[1].status_code == 200:
                soup = BeautifulSoup(response[1].content, "html.parser")
                data["articles"][i]["response"] = soup.text
            elif response[1].status_code == 429:
                bad_urls.append((i, str(response[1].url)))
            else:
                logging.warning(f"Skipping article at index {i} due to status code {response[1].status_code}")
                data["articles"][i]["response"] = f"Error fetching content: HTTP {response[1].status_code}"
        except Exception as e:
            logging.error(f"Error processing response for index {i}: {str(e)}")
            data["articles"][i]["response"] = f"Error processing content: {str(e)}"

    if len(bad_urls) == 0:
        logging.info("No bad URLs found")
    else:
        logging.info(f"{len(bad_urls)} bad URLs found — scraping with Playwright")

        soups = scrape_playwright(bad_urls)

        for i, soup in tqdm(soups, desc="Processing Playwright results"):
            data["articles"][i]["response"] = soup.text if soup else ""

    logging.info("Saving the json file now")

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

    logging.info("Done saving the json file")

    return 1

if __name__ == "__main__":

    os.makedirs("./logs", exist_ok=True)

    queries = ["Immigrants in the US", "US Visas"]

    years = ["2025"]
    months = ["12"]

    root_path = "./data/raw/news"
    if os.path.exists(root_path) == False:
        os.makedirs(root_path)

    for year in tqdm(years, desc="Years"):
        for i, month in tqdm(enumerate(months), total=len(months), desc="Months", leave=False):
            start_date = f"{year}-{month}-01"

            if month in ["04", "06", "09", "11"]:
                end_date = f"{year}-{month}-30"
            elif month == "02":
                if int(year) % 4 == 0:
                    end_date = f"{year}-{month}-29"
                else:
                    end_date = f"{year}-{month}-28"
            else:
                end_date = f"{year}-{month}-31"

            dir_path = os.path.join(root_path, f"{year}")
            if os.path.exists(dir_path) == False:
                os.makedirs(dir_path)
            
            file_name = f"news_{year}_{month}"
            file_path = os.path.join(dir_path, f"{file_name}.json")

            i = 1

            new_file_name: str = None

            while os.path.exists(file_path):
                new_file_name = file_name + f"_v{i}"
                file_path = os.path.join(dir_path, f"{new_file_name}.json")
                i += 1                

            status_code = main(
                queries=queries,
                from_date=start_date,
                to_date=end_date,
                limit=50,
                output_path=file_path,
                log_path="./logs/news.log"
            )

            if status_code == -1:
                print(f"No news found for {year}-{month}")