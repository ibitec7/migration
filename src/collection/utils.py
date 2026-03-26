import asyncio
import logging
import os
import aiofiles
import httpx
import random

from functools import wraps
from typing import Callable, Any

# Status codes that should be retried with exponential backoff
# Removed 302 (redirects) - httpx handles those automatically
RETRYABLE_STATUS_CODES: set[int] = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES: set[int] = {400, 401, 403, 404, 405, 406, 410, 413, 414, 415, 422, 451, 501}

def setup_logger(log_file, log_level=logging.INFO, write_console=True, write_file=True) -> logging.Logger:
    logger_name = f"{__name__}.{os.path.abspath(log_file)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

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

def retry_errors(
    max_retries: int = 3,
    backoff_factors: list = None,
    logger: logging.Logger = None,
    jitter: float = 2.0,
    retry_after_cap: int = 10,
) -> Callable:
    if backoff_factors is None:
        backoff_factors = [2, 5, 10]
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            client = kwargs.get('client') or (args[0] if args and isinstance(args[0], httpx.AsyncClient) else None)
            url_arg = kwargs.get('url') or (args[1] if len(args) > 1 else args[0] if args else None)
            url_string = url_arg[1] if isinstance(url_arg, tuple) else str(url_arg)
            url_id = url_arg[0] if isinstance(url_arg, tuple) else url_string
            
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Normalize result to handle both tuple and direct response returns
                    is_tuple_return = isinstance(result, tuple) and len(result) == 2
                    
                    if is_tuple_return:
                        ret_id, resp = result
                    elif isinstance(result, httpx.Response):
                        ret_id = url_id
                        resp = result
                    else:
                        # Non-response return, just pass through
                        return result
                    
                    status_code = resp.status_code
                    
                    if status_code == 200:
                        return (ret_id, resp) if is_tuple_return else resp
                    
                    if status_code in NON_RETRYABLE_STATUS_CODES:
                        resp.error_type = "HTTPError"
                        resp.error_msg = f"Status {status_code}"
                        return (ret_id, resp) if is_tuple_return else resp
                    
                    if status_code not in RETRYABLE_STATUS_CODES:
                        resp.error_type = "HTTPError"
                        resp.error_msg = f"Status {status_code}"
                        return (ret_id, resp) if is_tuple_return else resp
                    
                    if attempt < max_retries - 1:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = min(int(retry_after), retry_after_cap)
                            except ValueError:
                                wait_time = backoff_factors[attempt] if attempt < len(backoff_factors) else backoff_factors[-1]
                        else:
                            wait_time = backoff_factors[attempt] if attempt < len(backoff_factors) else backoff_factors[-1]
                        
                        # Add jitter to prevent thundering herd
                        wait_time += random.uniform(0, jitter)
                        
                        logger.warning(
                            f"Retryable error from {url_string}: HTTP {status_code}. "
                            f"Attempt {attempt + 1}/{max_retries}. Retrying in {wait_time:.2f}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Failed to fetch {url_string} after {max_retries} attempts: HTTP {status_code}")
                        resp.error_type = "HTTPError"
                        resp.error_msg = f"Status {status_code} after {max_retries} retries"
                        return (ret_id, resp) if is_tuple_return else resp
                        
                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.NetworkError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factors[attempt] if attempt < len(backoff_factors) else backoff_factors[-1]
                        # Add jitter to prevent thundering herd
                        wait_time += random.uniform(0, jitter)
                        logger.warning(
                            f"Network error for {url_string} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}. "
                            f"Retrying in {wait_time:.2f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch {url_string} after {max_retries} attempts (network error): {e}")
                        response = httpx.Response(status_code=0, request=httpx.Request("GET", url_string))
                        response.error_type = type(e).__name__
                        response.error_msg = str(e)
                        return (url_id, response) if isinstance(url_arg, tuple) else response
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing {url_string}: {type(e).__name__}: {e}")
                    response = httpx.Response(status_code=0, request=httpx.Request("GET", url_string))
                    response.error_type = type(e).__name__
                    response.error_msg = str(e)
                    return (url_id, response) if isinstance(url_arg, tuple) else response
            
            if last_exception:
                response = httpx.Response(status_code=0, request=httpx.Request("GET", url_string))
                response.error_type = type(last_exception).__name__
                response.error_msg = str(last_exception)
                return (url_id, response) if isinstance(url_arg, tuple) else response
            return result
        return wrapper
    return decorator

@retry_errors(max_retries=3)
async def download(url: str, filepath: str, client: httpx.AsyncClient, logger: logging.Logger, max_retries: int = 3):
    response = await client.get(url)
    if response.status_code == 200:
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(response.content)
        logger.info(f"Successfully downloaded {os.path.basename(url)}")
    return response

async def download_with_semaphore(url: str, filepath: str, client: httpx.AsyncClient, semaphore: asyncio.Semaphore, logger: logging.Logger, max_retries: int = 3):
    async with semaphore:
        await download(url, filepath, client, logger, max_retries)