"""
Configuration parameters for optimized news article downloading.
These values can be tuned based on network conditions and available resources.
"""

import os
from typing import Dict, Any

# ============================================================================
# CONCURRENCY & CONNECTION SETTINGS
# ============================================================================

# Maximum number of concurrent HTTP requests to Google News
# Safe range: 8-20. Higher values increase speed but risk 429 rate-limit errors.
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "16"))

# HTTP connection pool settings
MAX_HTTP_CONNECTIONS = int(os.getenv("MAX_HTTP_CONNECTIONS", "20"))
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "10"))

# Number of parallel workers for month/year batch processing
# Safe range: 3-6. Higher values increase memory usage if Playwright needed.
BATCH_WORKER_COUNT = int(os.getenv("BATCH_WORKER_COUNT", "4"))

# Number of parallel processes for CPU-bound trafilatura extraction
# Should match or be slightly less than CPU core count (typically 4-8)
TRAFILATURA_WORKER_COUNT = int(os.getenv("TRAFILATURA_WORKER_COUNT", "4"))

# ============================================================================
# TIMEOUT & RETRY SETTINGS
# ============================================================================

# Individual HTTP request timeouts (in seconds)
# Connection timeout: time to establish connection
# Read timeout: time to receive first data packet
TIMEOUT_CONFIG = {
    "connect": float(os.getenv("TIMEOUT_CONNECT", "3.0")),
    "read": float(os.getenv("TIMEOUT_READ", "8.0")),
}

# Maximum time any async task is allowed (in seconds)
# Prevents deadlocks from stalled requests
TASK_TIMEOUT = float(os.getenv("TASK_TIMEOUT", "30.0"))

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Exponential backoff factors for retries (in seconds)
# For HTTP errors: [5, 10, 20, 40]
# For network errors: [2, 5, 10]
BACKOFF_FACTORS_HTTP = [5, 10, 20, 40]
BACKOFF_FACTORS_NETWORK = [2, 5, 10]

# Jitter range for backoff (prevents thundering herd)
# Random value between 0 and BACKOFF_JITTER added to wait time
BACKOFF_JITTER = float(os.getenv("BACKOFF_JITTER", "2.0"))

# ============================================================================
# RATE LIMIT & THROTTLING SETTINGS
# ============================================================================

# Threshold of 429 responses before employing Playwright fallback
# If we get 429 more than N times, try Playwright instead of more retries
PLAYWRIGHT_FALLBACK_THRESHOLD = int(os.getenv("PLAYWRIGHT_FALLBACK_THRESHOLD", "3"))

# Adaptive throttle: if source URL gets 429, add extra delay before retry
ADAPTIVE_THROTTLE_BASE = float(os.getenv("ADAPTIVE_THROTTLE_BASE", "2.0"))

# Per-source minimum delay (in seconds) after 429 response
# Sources that rate-limit get added delay on retries
ADAPTIVE_THROTTLE_DELAY = {
    "news.google.com": float(os.getenv("THROTTLE_GOOGLE_NEWS", "1.0")),
    "other": float(os.getenv("THROTTLE_OTHER", "0.5")),
}

# Sleep between decoding batch requests (in seconds)
DECODE_BATCH_SLEEP = float(os.getenv("DECODE_BATCH_SLEEP", "0.1"))

# ============================================================================
# BATCH & QUEUE SETTINGS
# ============================================================================

# Batch size for Google News decoding params (articles per batch)
# Higher = fewer requests but larger payloads; typical: 15-25
DECODE_BATCH_SIZE = int(os.getenv("DECODE_BATCH_SIZE", "20"))

# Batch size for trafilatura processing (articles per batch)
# Higher = more CPU parallelism; typical: 15-30
TRAFILATURA_BATCH_SIZE = int(os.getenv("TRAFILATURA_BATCH_SIZE", "20"))

# Maximum queue size for async task queue
# Prevents memory explosion from queued tasks
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))

# ============================================================================
# REDIRECT & REQUEST SETTINGS
# ============================================================================

# Maximum number of redirects to follow per request
# Google News typically doesn't chain redirects; limit to 2-3
MAX_REDIRECTS = int(os.getenv("MAX_REDIRECTS", "2"))

# ============================================================================
# TRAFILATURA & CONTENT EXTRACTION SETTINGS
# ============================================================================

# Timeout for trafilatura.extract() (in seconds)
# Prevents hanging on malformed HTML
TRAFILATURA_TIMEOUT = float(os.getenv("TRAFILATURA_TIMEOUT", "2.0"))

# Trafilatura output format and options
TRAFILATURA_OUTPUT_FORMAT = "markdown"
TRAFILATURA_INCLUDE_FORMATTING = True
TRAFILATURA_INCLUDE_TABLES = True
TRAFILATURA_WITH_METADATA = True

# ============================================================================
# HEALTHCHECK & MONITORING SETTINGS
# ============================================================================

# Interval for health checks (in seconds)
# Monitors for stalled tasks and resource issues
HEALTH_CHECK_INTERVAL = float(os.getenv("HEALTH_CHECK_INTERVAL", "5.0"))

# Timeout threshold for individual article fetch (in seconds)
# If an article takes longer, mark as slow and skip
ARTICLE_TIMEOUT_THRESHOLD = float(os.getenv("ARTICLE_TIMEOUT_THRESHOLD", "10.0"))

# ============================================================================
# PLAYWRIGHT SETTINGS (fallback for rate-limited URLs)
# ============================================================================

PLAYWRIGHT_CONFIG = {
    "headless": True,
    "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-dev-tools",
        "--no-first-run",
        "--no-default-browser-check",
    ],
    "max_concurrent_pages": TRAFILATURA_WORKER_COUNT,
    "timeout_per_page": 15000,  # milliseconds
    "user_agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timeout_config() -> Dict[str, float]:
    """Get httpx compatible timeout config."""
    import httpx
    return httpx.Timeout(timeout=TIMEOUT_CONFIG["read"], connect=TIMEOUT_CONFIG["connect"])

def get_limits_config() -> Dict[str, Any]:
    """Get httpx connection limits config."""
    import httpx
    return httpx.Limits(
        max_connections=MAX_HTTP_CONNECTIONS,
        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
    )

def print_config():
    """Print current configuration (for debugging)."""
    print("\n" + "="*70)
    print("NEWS DOWNLOADER CONFIGURATION")
    print("="*70)
    print(f"Concurrent HTTP Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Batch Workers (Month/Year): {BATCH_WORKER_COUNT}")
    print(f"Trafilatura Workers: {TRAFILATURA_WORKER_COUNT}")
    print(f"Connect Timeout: {TIMEOUT_CONFIG['connect']}s")
    print(f"Read Timeout: {TIMEOUT_CONFIG['read']}s")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"Backoff Factors (HTTP): {BACKOFF_FACTORS_HTTP}")
    print(f"Decode Batch Size: {DECODE_BATCH_SIZE}")
    print(f"Trafilatura Batch Size: {TRAFILATURA_BATCH_SIZE}")
    print(f"Max Redirects: {MAX_REDIRECTS}")
    print(f"Health Check Interval: {HEALTH_CHECK_INTERVAL}s")
    print("="*70 + "\n")
