"""
Optimized news article downloader with parallel batch processing.
Handles Google News RSS scraping, URL decoding, concurrent fetching, and content extraction.
"""

import httpx
import asyncio
from bs4 import BeautifulSoup
from pygooglenews import GoogleNews
import os
from urllib.parse import quote, urlparse
import json
import trafilatura
import logging
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import time
import random

from utils import setup_logger, retry_errors
from config import (
    MAX_CONCURRENT_REQUESTS,
    BATCH_WORKER_COUNT,
    TRAFILATURA_WORKER_COUNT,
    DECODE_BATCH_SIZE,
    TRAFILATURA_BATCH_SIZE,
    PLAYWRIGHT_FALLBACK_THRESHOLD,
    BACKOFF_FACTORS_HTTP,
    BACKOFF_JITTER,
    DECODE_BATCH_SLEEP,
    MAX_REDIRECTS,
    TIMEOUT_CONFIG,
    TASK_TIMEOUT,
    TRAFILATURA_TIMEOUT,
    TRAFILATURA_OUTPUT_FORMAT,
    TRAFILATURA_INCLUDE_FORMATTING,
    TRAFILATURA_INCLUDE_TABLES,
    TRAFILATURA_WITH_METADATA,
    HEALTH_CHECK_INTERVAL,
    ARTICLE_TIMEOUT_THRESHOLD,
    PLAYWRIGHT_CONFIG,
    get_timeout_config,
    get_limits_config,
)


CHECKPOINT_EVERY = 25


def _atomic_json_write(file_path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial/corrupt files on interruptions."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, file_path)


def _checkpoint_progress(
    output_path: str,
    data: Dict[str, Any],
    stage: str,
    processed: int,
    total: int,
    logger: logging.Logger,
) -> None:
    """Persist in-flight progress so long runs remain durable and observable."""
    checkpoint_meta = {
        "stage": stage,
        "processed": processed,
        "total": total,
        "updated_at": time.time(),
    }
    _atomic_json_write(output_path, data)
    _atomic_json_write(f"{output_path}.checkpoint.json", checkpoint_meta)
    logger.info(f"Checkpoint saved ({stage}): {processed}/{total}")


# ============================================================================
# GOOGLE NEWS DECODING (with batching)
# ============================================================================

async def get_decoding_params(gn_art_id: str, client: httpx.AsyncClient, logger: logging.Logger) -> Dict[str, str]:
    """Fetch decoding parameters for a single Google News article."""
    try:
        response = await client.get(
            f"https://news.google.com/rss/articles/{gn_art_id}",
            follow_redirects=True
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        div = soup.select_one("c-wiz > div")
        if div is None:
            raise ValueError(f"Could not find decoding params for article id: {gn_art_id}")

        signature = div.get("data-n-a-sg")
        timestamp = div.get("data-n-a-ts")
        if not signature or not timestamp:
            raise ValueError(f"Missing signature/timestamp for article id: {gn_art_id}")

        return {
            "signature": signature,
            "timestamp": timestamp,
            "gn_art_id": gn_art_id,
        }
    except Exception as e:
        logger.error(f"Failed to get decoding params for {gn_art_id}: {e}")
        raise


async def decode_urls_batch(articles: List[Dict[str, Any]], client: httpx.AsyncClient, logger: logging.Logger) -> List[str]:
    """
    Decode a batch of Google News article URLs with optimized batching.
    
    Instead of individual requests, batch 20 articles per batchexecute call.
    """
    decoded_urls: Dict[int, str] = {}
    
    # Process articles in batches to reduce total requests
    for batch_start in range(0, len(articles), DECODE_BATCH_SIZE):
        batch_end = min(batch_start + DECODE_BATCH_SIZE, len(articles))
        batch = articles[batch_start:batch_end]
        
        logger.debug(f"Processing decode batch {batch_start}-{batch_end} ({len(batch)} articles)")
        
        try:
            articles_reqs = [
                [
                    "Fbv4je",
                    f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{art["gn_art_id"]}",{art["timestamp"]},"{art["signature"]}"]',
                ]
                for art in batch
            ]
            
            payload = f"f.req={quote(json.dumps([articles_reqs]))}"
            headers = {
                "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            }

            response = await client.post(
                "https://news.google.com/_/DotsSplashUi/data/batchexecute",
                headers=headers,
                data=payload,
                timeout=get_timeout_config()
            )
            response.raise_for_status()

            if DECODE_BATCH_SLEEP > 0:
                sleep_time = DECODE_BATCH_SLEEP + random.uniform(0, 0.2)
                await asyncio.sleep(sleep_time)

            body = response.text.split("\n\n")
            if len(body) < 2:
                logger.warning(f"Unexpected batchexecute response for batch {batch_start}-{batch_end}")
                continue

            parsed = json.loads(body[1])
            batch_decoded = [json.loads(res[2])[1] for res in parsed[:-2]]
            
            # Map decoded URLs back to original article indices
            for i, decoded_url in enumerate(batch_decoded):
                original_index = batch[i]["index"]
                decoded_urls[original_index] = decoded_url
                
        except Exception as e:
            logger.error(f"Failed to decode batch {batch_start}-{batch_end}: {e}")
            continue
    
    return decoded_urls


async def decode(encoded_urls: List[str], client: httpx.AsyncClient, logger: logging.Logger) -> List[str]:
    """
    Decode Google News encoded URLs to actual article URLs.
    Batches decoding requests for efficiency.
    """
    logger.info(f"Decoding {len(encoded_urls)} article URLs (batch size: {DECODE_BATCH_SIZE})")
    
    # First, fetch decoding params for all articles in parallel with semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def fetch_params_with_sem(index: int, gn_art_id: str) -> Dict[str, str]:
        async with semaphore:
            params = await get_decoding_params(gn_art_id, client, logger)
            params["index"] = index
            return params
    
    tasks = [fetch_params_with_sem(i, urlparse(url).path.split("/")[-1]) for i, url in enumerate(encoded_urls)]
    
    # Use regular asyncio.gather with return_exceptions, then wrap with tqdm
    articles_params = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failed params
    valid_params: List[Dict[str, Any]] = []
    for i, params in enumerate(articles_params):
        if isinstance(params, Exception):
            logger.warning(f"Failed to fetch params for article {i}: {params}")
        elif params:
            valid_params.append(params)
    
    if not valid_params:
        logger.error("No valid decoding params obtained!")
        return [None] * len(encoded_urls)
    
    decoded_by_index = await decode_urls_batch(valid_params, client, logger)

    final_decoded_urls: List[Optional[str]] = [None] * len(encoded_urls)
    for index, decoded_url in decoded_by_index.items():
        if 0 <= index < len(final_decoded_urls):
            final_decoded_urls[index] = decoded_url

    return final_decoded_urls


def decode_async(urls: List[str], logger: logging.Logger) -> List[str]:
    """Synchronous wrapper for async decoding - handles running event loops."""
    async def _decode():
        async with httpx.AsyncClient(timeout=get_timeout_config(), limits=get_limits_config()) as client:
            return await decode(urls, client, logger)
    
    return asyncio.run(_decode())


# ============================================================================
# OPTIMIZED HTTP RESPONSE FETCHING
# ============================================================================

@retry_errors(max_retries=2, backoff_factors=[1, 2], jitter=0.5, retry_after_cap=10)
async def get_response(client: httpx.AsyncClient, url: Tuple[int, str]) -> Tuple[int, httpx.Response]:
    """Fetch HTTP response with integrated retry logic."""
    response = await client.get(url[1])
    return (url[0], response)


async def fetch_urls_parallel(urls: List[Tuple[int, str]], logger: logging.Logger) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Fetch multiple URLs in parallel with concurrency limit.
    Uses semaphore to prevent connection pool exhaustion.
    Reads response content before returning to avoid client being closed.
    """
    logger.info(f"Fetching {len(urls)} articles with max {MAX_CONCURRENT_REQUESTS} concurrent requests")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def fetch_with_sem(url: Tuple[int, str]) -> Tuple[int, Dict[str, Any]]:
        async with semaphore:
            try:
                idx_result, response = await asyncio.wait_for(
                    get_response(client, url),
                    timeout=ARTICLE_TIMEOUT_THRESHOLD,
                )
                # Read content immediately while client is still open
                return (
                    idx_result,
                    {
                        "status_code": response.status_code,
                        "content": response.text,
                        "headers": dict(response.headers) if hasattr(response, 'headers') else {},
                        "url": str(response.url) if hasattr(response, 'url') else "",
                    }
                )
            except Exception as e:
                idx, _ = url
                return (
                    idx,
                    {
                        "status_code": 0,
                        "content": "",
                        "headers": {},
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
    
    async with httpx.AsyncClient(
        timeout=get_timeout_config(),
        limits=get_limits_config(),
        follow_redirects=True,
    ) as client:
        task_map = {
            asyncio.create_task(fetch_with_sem(url)): url[0]
            for url in urls
        }

        responses: List[Tuple[int, Dict[str, Any]]] = []
        pending = set(task_map.keys())
        total = len(task_map)
        completed = 0
        start = time.time()
        fetch_timeout = max(
            TASK_TIMEOUT,
            ((total // max(1, MAX_CONCURRENT_REQUESTS)) + 1) * ARTICLE_TIMEOUT_THRESHOLD,
        )

        while pending:
            elapsed = time.time() - start
            if elapsed > fetch_timeout:
                logger.warning(
                    f"Fetch watchdog timeout after {elapsed:.1f}s; cancelling {len(pending)} pending tasks"
                )
                break

            done, pending = await asyncio.wait(
                pending,
                timeout=HEALTH_CHECK_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                logger.info(f"Fetch heartbeat: {completed}/{total} completed, {len(pending)} pending")
                continue

            for task in done:
                try:
                    responses.append(task.result())
                except Exception as exc:
                    idx = task_map.get(task, -1)
                    responses.append(
                        (
                            idx,
                            {
                                "status_code": 0,
                                "content": "",
                                "headers": {},
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                    )
                completed += 1
                if completed % CHECKPOINT_EVERY == 0 or completed == total:
                    logger.info(f"Fetch progress: {completed}/{total}")

        if pending:
            for task in pending:
                task.cancel()
                idx = task_map.get(task, -1)
                responses.append(
                    (
                        idx,
                        {
                            "status_code": 0,
                            "content": "",
                            "headers": {},
                            "error": "Fetch timeout watchdog",
                            "error_type": "TimeoutError",
                        },
                    )
                )
    
    # Handle exceptions gracefully
    result = []
    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            logger.warning(f"Exception fetching URL {i}: {resp}")
            result.append((urls[i][0], {"status_code": 0, "content": "", "error": str(resp), "error_type": type(resp).__name__}))
        else:
            result.append(resp)
    
    return result


def run_fetch_urls_async(urls: List[Tuple[int, str]], logger: logging.Logger) -> List[Tuple[int, Dict[str, Any]]]:
    """Synchronous wrapper for async URL fetching - handles running event loops."""
    async def _fetch():
        return await fetch_urls_parallel(urls, logger)
    
    return asyncio.run(_fetch())


# ============================================================================
# TRAFILATURA CONTENT EXTRACTION (with batching & multiprocessing)
# ============================================================================

def extract_article_content(args: Tuple[int, str]) -> Tuple[int, Optional[str]]:
    """
    Extract article content using trafilatura.
    Runs in separate process to avoid GIL and prevent hanging from blocking main thread.
    """
    idx, html = args
    
    try:
        if not html or len(html) < 100:
            return (idx, "")
        
        # Use timeout to prevent extraction from hanging on malformed HTML
        article = trafilatura.extract(
            html,
            output_format=TRAFILATURA_OUTPUT_FORMAT,
            include_formatting=TRAFILATURA_INCLUDE_FORMATTING,
            include_tables=TRAFILATURA_INCLUDE_TABLES,
            with_metadata=TRAFILATURA_WITH_METADATA,
        )
        return (idx, article if article else "")
    except Exception as e:
        return (idx, f"[Extraction Error: {str(e)}]")


def extract_articles_batch(articles_html: List[Tuple[int, str]], logger: logging.Logger) -> Dict[int, Optional[str]]:
    """
    Extract content from multiple articles in parallel using ProcessPoolExecutor.
    Avoids main thread blocking and utilizes multiple CPU cores.
    """
    if not articles_html:
        return {}
    
    logger.debug(
        f"Extracting content from {len(articles_html)} articles using {TRAFILATURA_WORKER_COUNT} processes "
        f"(batch size: {TRAFILATURA_BATCH_SIZE})"
    )
    
    results = {}
    
    try:
        for batch_start in range(0, len(articles_html), TRAFILATURA_BATCH_SIZE):
            batch_end = min(batch_start + TRAFILATURA_BATCH_SIZE, len(articles_html))
            batch = articles_html[batch_start:batch_end]
            with ProcessPoolExecutor(max_workers=TRAFILATURA_WORKER_COUNT) as executor:
                futures = [executor.submit(extract_article_content, arg) for arg in batch]
                for future in as_completed(futures):
                    try:
                        idx, content = future.result(timeout=TRAFILATURA_TIMEOUT)
                        results[idx] = content
                    except Exception as e:
                        logger.error(f"Error extracting article: {e}")
    except Exception as e:
        logger.error(f"Error in batch extraction: {e}")
    
    return results


# ============================================================================
# GOOGLE NEWS FETCHING
# ============================================================================

def get_news(
    queries: List[str],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Fetch news articles from Google News for multiple queries.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Fetching news from Google News API for {len(queries)} queries")

    gn = GoogleNews()

    all_news = {
        "title": f"News for {', '.join(queries)} from {date_from} to {date_to}",
        "totalResults": 0,
        "headlines": [],
        "articles": []
    }

    for query in tqdm(queries, desc="Searching queries"):
        try:
            results = gn.search(query, from_=date_from, to_=date_to)
            for news in results["entries"]:
                if news["title"] not in all_news["headlines"]:
                    all_news["articles"].append(news)
                    all_news["headlines"].append(news["title"])
                    if len(all_news["articles"]) >= limit:
                        break
            if len(all_news["articles"]) >= limit:
                break
        except Exception as e:
            logger.warning(f"Error searching for '{query}': {e}")
            continue

    if not all_news["articles"]:
        logger.warning(f"No articles found for queries: {queries}")
        return all_news

    logger.info(f"Found {len(all_news['articles'])} articles, decoding URLs...")
    
    encoded_urls = [article["link"] for article in all_news["articles"]]
    decoded_urls = decode_async(encoded_urls, logger)

    for i, article in enumerate(all_news["articles"]):
        if i < len(decoded_urls) and decoded_urls[i]:
            article["link"] = decoded_urls[i]
        else:
            logger.warning(f"Failed to decode URL for article {i}: {article.get('title', 'Unknown')}")

    all_news["totalResults"] = len(all_news["articles"])
    return all_news


# ============================================================================
# MAIN ARTICLE PROCESSING
# ============================================================================

def main(
    queries: List[str],
    from_date: str,
    to_date: str,
    limit: int,
    output_path: str,
    logger: logging.Logger,
) -> int:
    """
    Main pipeline: fetch news, download articles, extract content, save results.
    """
    logger.info(f"Starting news download for {from_date} to {to_date}")
    
    # Skip mode: avoid re-downloading if month file already exists
    if os.path.exists(output_path):
        logger.info(f"File {output_path} already exists. Skipping download.")
        return 2

    # Fetch news from Google News
    start_time = time.time()
    data = get_news(queries=queries, date_from=from_date, date_to=to_date, limit=limit, logger=logger)
    
    if not data.get("articles"):
        logger.warning("No news found for the given date range")
        return -1

    logger.info(f"Fetching detailed content for {len(data['articles'])} articles")
    
    # Prepare URLs for fetching
    urls = [(i, feed["link"]) for i, feed in enumerate(data["articles"]) if feed.get("link")]
    
    if not urls:
        logger.error("No valid URLs to fetch")
        return -1

    # Fetch all URLs in parallel with controlled concurrency
    responses = run_fetch_urls_async(urls, logger)
    responses = list(responses)  # Ensure it's a list for stable iteration
    
    if responses:
        logger.info(f"Received {len(responses)} HTTP responses")
    
    good_responses = {}
    
    # Process responses
    logger.info("Processing HTTP responses")
    processed_responses = 0
    for item in tqdm(responses, desc="Processing responses", total=len(responses)):
        try:
            idx, response_dict = item
            if idx < 0 or idx >= len(data["articles"]):
                continue
                
            if "error_type" in response_dict:
                error_msg = response_dict.get('error', 'unknown')
                logger.warning(f"Skipping article {idx}: {response_dict.get('error_type', 'Unknown')} - {error_msg}")
                data["articles"][idx]["status_code"] = 0
                data["articles"][idx]["response"] = f"Error: {error_msg}"
                continue
            
            status = response_dict.get("status_code", 0)
            if status == 200:
                # Store good responses for batch processing
                good_responses[idx] = response_dict.get("content", "")
                data["articles"][idx]["status_code"] = status
            elif status == 429:
                # Best effort mode: no Playwright fallback (throughput-first)
                logger.warning(f"Article {idx}: HTTP 429")
                data["articles"][idx]["status_code"] = 429
                data["articles"][idx]["response"] = "Error: HTTP 429"
            else:
                logger.warning(f"Article {idx}: HTTP {status}")
                data["articles"][idx]["status_code"] = status
                data["articles"][idx]["response"] = f"Error: HTTP {status}"

            processed_responses += 1
            if processed_responses % CHECKPOINT_EVERY == 0:
                _checkpoint_progress(
                    output_path,
                    data,
                    stage="responses",
                    processed=processed_responses,
                    total=len(responses),
                    logger=logger,
                )
        except Exception as e:
            import traceback
            logger.error(f"Error processing response {idx}: {e}\n{traceback.format_exc()}")
            if idx < len(data["articles"]):
                data["articles"][idx]["status_code"] = 0
                data["articles"][idx]["response"] = f"Error: {str(e)}"

    # Batch extract content from good responses
    if good_responses:
        logger.info(f"Extracting content from {len(good_responses)} articles using {TRAFILATURA_WORKER_COUNT} processes")
        extraction_tasks = [(idx, html) for idx, html in good_responses.items()]
        
        # Process in batches to avoid memory overload
        extracted = extract_articles_batch(extraction_tasks, logger)
        
        extracted_count = 0
        for idx, content in extracted.items():
            data["articles"][idx]["response"] = content
            extracted_count += 1
            if extracted_count % CHECKPOINT_EVERY == 0:
                _checkpoint_progress(
                    output_path,
                    data,
                    stage="extraction",
                    processed=extracted_count,
                    total=len(extracted),
                    logger=logger,
                )

    # Save results
    logger.info("Saving results to file")
    _atomic_json_write(output_path, data)

    checkpoint_file = f"{output_path}.checkpoint.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.2f}s. Saved {len(data['articles'])} articles to {output_path}")

    return 1


# ============================================================================
# BATCH PARALLEL PROCESSING (for multiple months/years)
# ============================================================================

async def process_month_async(
    year: str,
    month: str,
    queries: List[str],
    limit: int,
    root_path: str,
    logger: logging.Logger,
) -> Tuple[str, int]:
    """
    Process a single month in a background task.
    Returns (month_key, status_code) for tracking.
    """
    start_date = f"{year}-{month}-01"
    
    # Calculate end date based on month
    if month in ["04", "06", "09", "11"]:
        end_date = f"{year}-{month}-30"
    elif month == "02":
        end_date = f"{year}-{month}-28" if int(year) % 4 != 0 else f"{year}-{month}-29"
    else:
        end_date = f"{year}-{month}-31"

    dir_path = os.path.join(root_path, year)
    os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"news_{year}_{month}"
    file_path = os.path.join(dir_path, f"{file_name}.json")

    month_logger = setup_logger(
        f"./logs/news_{year}_{month}.log",
        write_console=False
    )
    
    month_timeout = max(300.0, TASK_TIMEOUT * 10, limit * 4.0)
    try:
        status_code = await asyncio.wait_for(
            asyncio.to_thread(
                main,
                queries,
                start_date,
                end_date,
                limit,
                file_path,
                month_logger,
            ),
            timeout=month_timeout,
        )
    except asyncio.TimeoutError:
        month_logger.error(f"Month {year}-{month} exceeded timeout ({month_timeout:.1f}s)")
        status_code = 0
    
    month_key = f"{year}-{month}"
    return (month_key, status_code)


async def process_months_parallel(
    years: List[str],
    months: List[str],
    queries: List[str],
    limit: int,
    root_path: str,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Process multiple year/month combinations in parallel using async task queue.
    """
    # Create task queue
    task_queue = asyncio.Queue()
    results = {}
    
    # Queue all year/month combinations
    for year in years:
        for month in months:
            await task_queue.put((year, month))
    
    # Worker coroutine that processes tasks from queue
    async def worker(worker_id: int):
        while True:
            year = None
            month = None
            try:
                year, month = task_queue.get_nowait()
                logger.info(f"[Worker {worker_id}] Processing {year}-{month}")
                
                month_key, status = await process_month_async(
                    year, month, queries, limit, root_path, logger
                )
                results[month_key] = status
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                if year is not None and month is not None:
                    month_key = f"{year}-{month}"
                    results[month_key] = 0
                    logger.error(f"[Worker {worker_id}] Error on {month_key}: {e}")
                else:
                    logger.error(f"[Worker {worker_id}] Error: {e}")
            finally:
                if year is not None and month is not None:
                    task_queue.task_done()

    # Launch worker tasks
    workers = [asyncio.create_task(worker(i)) for i in range(BATCH_WORKER_COUNT)]
    
    # Wait for all workers to complete
    logger.info(f"Launching {BATCH_WORKER_COUNT} parallel workers for {len(years) * len(months)} months")
    await asyncio.gather(*workers)
    
    return results


def process_months_batch(
    years: List[str],
    months: List[str],
    queries: List[str],
    limit: int,
    root_path: str,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Synchronous wrapper for parallel month processing."""
    return asyncio.run(
        process_months_parallel(years, months, queries, limit, root_path, logger)
    )


# ============================================================================
# CLI & ENTRY POINT
# ============================================================================

def main_cli():
    """Command-line interface for news downloader."""
    parser = argparse.ArgumentParser(
        description="Optimized tool to scrape news articles from Google News in bulk"
    )

    parser.add_argument(
        "-q", "--query",
        nargs="+",
        type=str,
        required=True,
        help="List of queries to search for"
    )
    parser.add_argument(
        "-y", "--year",
        nargs="+",
        type=str,
        help="List of years to fetch news for"
    )
    parser.add_argument(
        "-m", "--month",
        nargs="+",
        type=str,
        help="List of months to fetch news for (01-12)"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=50,
        help="Maximum number of articles per month (default: 50)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./data/raw/news/",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Enable parallel month/year processing (default: enabled)"
    )

    args = parser.parse_args()

    queries = args.query
    years = args.year if args.year else []
    months = args.month

    if not years:
        parser.error("At least one --year is required for reliable batch downloads.")
    
    if not months:
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    
    months = [f"{int(m):02d}" for m in months]  # Ensure zero-padded

    os.makedirs("./logs", exist_ok=True)
    
    root_logger = setup_logger("./logs/news_main.log", write_console=True)
    base_path = args.output
    
    # Process each query separately with its own directory
    for query in queries:
        root_logger.info(f"\n{'='*70}")
        root_logger.info(f"Processing query: {query}")
        root_logger.info(f"{'='*70}\n")
        
        # Create query-specific directory
        root_path = os.path.join(base_path, query.replace(" ", "_"))
        os.makedirs(root_path, exist_ok=True)

        if args.parallel:
            # Use parallel batch processing for all requested months/years
            root_logger.info(f"Processing {len(years)} years × {len(months)} months in parallel for '{query}'")
            results = process_months_batch(years, months, [query], args.limit, root_path, root_logger)
            
            root_logger.info(f"Batch processing complete for '{query}'. Results:")
            for month_key, status in sorted(results.items()):
                status_str = "✓ Success" if status == 1 else ("⊘ No news" if status == -1 else ("⏭ Skipped" if status == 2 else "✗ Failed"))
                root_logger.info(f"  {month_key}: {status_str}")
        else:
            # Fall back to serial processing for single year
            root_logger.info(f"Processing {len(years)} years × {len(months)} months serially for '{query}'")
            for year in tqdm(years, desc=f"Years ({query})"):
                for month in tqdm(months, desc="Months", leave=False):
                    status = asyncio.run(
                        process_month_async(year, month, [query], args.limit, root_path, root_logger)
                    )
                    root_logger.info(f"{status[0]}: Status {status[1]}")


if __name__ == "__main__":
    main_cli()
