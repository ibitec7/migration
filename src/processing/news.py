"""
News data processing pipeline: Extract, tokenize, and export news articles to parquet format.
Reads from data/raw/news/{country}/{year}/news_{year}_{month}.json files,
tokenizes article text using Jina embedding model, and exports to country-specific parquet files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import polars as pl
from transformers import AutoTokenizer

# Local imports
from .utils import setup_logger

# Configuration
DATA_RAW_NEWS_DIR = Path("data/raw/news")
DATA_PROCESSED_DIR = Path("data/processed/news")
JINA_TOKENIZER_PATH = Path("src/models/jina_v5_nano_onnx")

logger = None

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

def configure_logger() -> logging.Logger:
    """Initialize logger for news processing pipeline."""
    global logger
    if logger is None:
        logger = setup_logger("logs/news_processing.log")
    return logger


def discover_country_directories() -> Dict[str, List[Path]]:
    """
    Discover all country directories under data/raw/news/.
    Returns a dict mapping country name to list of JSON file paths.
    
    Returns:
        dict[str, list[Path]]: Maps country name to list of JSON file paths.
    """
    logger = configure_logger()
    if not DATA_RAW_NEWS_DIR.exists():
        logger.error(f"News directory does not exist: {DATA_RAW_NEWS_DIR}")
        return {}
    
    countries_to_files: Dict[str, List[Path]] = {}
    
    for country_dir in DATA_RAW_NEWS_DIR.iterdir():
        if not country_dir.is_dir():
            continue
        
        country_name = country_dir.name
        json_files = list(country_dir.rglob("*.json"))
        
        if json_files:
            countries_to_files[country_name] = sorted(json_files)
            logger.info(f"Found {len(json_files)} JSON files for country: {country_name}")
        else:
            logger.warning(f"No JSON files found in country directory: {country_name}")
    
    logger.info(f"Discovered {len(countries_to_files)} countries with news data")
    return countries_to_files


def parse_date_from_article(article: Dict) -> Optional[str]:
    """
    Extract date from article. Try multiple sources:
    1. Extract from 'response' YAML metadata (date field)
    2. Use published_parsed tuple
    3. Fall back to filename pattern or None
    
    Args:
        article: Article dict from JSON
        
    Returns:
        str: ISO format date string (YYYY-MM-DD) or None
    """
    # Try extracting from response YAML metadata
    if "response" in article and isinstance(article["response"], str):
        response = article["response"]
        # Look for 'date: YYYY-MM-DD' in the response
        match = re.search(r'^date:\s*(\d{4}-\d{2}-\d{2})', response, re.MULTILINE)
        if match:
            return match.group(1)
    
    # Try published_parsed tuple (year, month, day, hour, minute, second, ...)
    if "published_parsed" in article and isinstance(article["published_parsed"], (list, tuple)):
        try:
            parsed = article["published_parsed"]
            if len(parsed) >= 3:
                year, month, day = parsed[0], parsed[1], parsed[2]
                return f"{year:04d}-{month:02d}-{day:02d}"
        except (IndexError, TypeError, ValueError):
            pass
    
    # Try published string (e.g., "Fri, 13 Jan 2017 08:00:00 GMT")
    if "published" in article and isinstance(article["published"], str):
        try:
            # Parse RFC 2822 format
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(article["published"])
            return dt.strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            pass
    
    return None


def load_and_flatten_country_articles(
    country_name: str,
    json_files: List[Path]
) -> List[Dict]:
    """
    Load all JSON files for a country and flatten articles into list of dicts.
    
    Args:
        country_name: Name of the country
        json_files: List of JSON file paths to load
        
    Returns:
        list[dict]: Flattened list of article records with columns:
            - date (str, ISO format YYYY-MM-DD)
            - country (str)
            - headline (str)
            - response (str)
            - status_code (int)
    """
    logger = configure_logger()
    articles = []
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract articles from JSON
            if "articles" not in data:
                logger.warning(f"No 'articles' key in {json_file.name}")
                continue
            
            for article in data["articles"]:
                # Extract required fields
                date = parse_date_from_article(article)
                headline = article.get("title", "")
                response = article.get("response", "")
                status_code = article.get("status_code", -1)
                
                if not date:
                    logger.debug(f"Could not parse date for article: {headline[:50]}")
                    date = "1970-01-01"  # Fallback date
                
                articles.append({
                    "date": date,
                    "country": country_name,
                    "headline": headline,
                    "response": response,
                    "status_code": status_code,
                })
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading JSON file {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(articles)} articles from {len(json_files)} files for {country_name}")
    return articles


def load_jina_tokenizer() -> AutoTokenizer:
    """
    Load Jina tokenizer from local model files.
    
    Returns:
        AutoTokenizer: Loaded tokenizer
        
    Raises:
        FileNotFoundError: If tokenizer files not found
    """
    logger = configure_logger()
    if not JINA_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Jina tokenizer path does not exist: {JINA_TOKENIZER_PATH}")
    
    logger.info(f"Loading Jina tokenizer from {JINA_TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(JINA_TOKENIZER_PATH))
    return tokenizer


def tokenize_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 8192,
    batch_size: int = 32
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Tokenize a batch of texts using Jina tokenizer.
    
    Args:
        texts: List of text strings to tokenize
        tokenizer: AutoTokenizer instance
        max_length: Maximum token length (default 8192 for Jina)
        batch_size: Batch size for tokenization
        
    Returns:
        tuple[list[list[int]], list[list[int]]]: (token_ids, attention_masks)
    """
    logger = configure_logger()
    all_token_ids = []
    all_attention_masks = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        
        try:
            # Tokenize with padding and truncation
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None,  # Return lists, not tensors
            )
            
            all_token_ids.extend(encoded["input_ids"])
            all_attention_masks.extend(encoded["attention_mask"])
        
        except Exception as e:
            logger.error(f"Error tokenizing batch {i // batch_size}: {e}")
            # Add empty tokens for failed batch
            for _ in batch:
                all_token_ids.append([])
                all_attention_masks.append([])
    
    logger.info(f"Tokenized {len(texts)} texts")
    return all_token_ids, all_attention_masks


def create_articles_dataframe(
    articles: List[Dict],
    tokenizer: AutoTokenizer
) -> pl.DataFrame:
    """
    Create a Polars DataFrame from article records with tokenization.
    Filters out records with non-200 status codes before returning.
    
    Args:
        articles: List of article dicts
        tokenizer: Jina tokenizer
        
    Returns:
        pl.DataFrame: DataFrame with columns:
            - date: Date
            - country: String
            - headline: String
            - response: String
            - status_code: Int32
            - token_ids: List(Int32)
            - attention_mask: List(Int8)
    """
    logger = configure_logger()
    
    if not articles:
        logger.warning("No articles provided to create DataFrame")
        return pl.DataFrame()
    
    # Extract response texts for tokenization
    responses = [a["response"] for a in articles]
    
    logger.info(f"Tokenizing {len(responses)} articles...")
    token_ids, attention_masks = tokenize_batch(responses, tokenizer)
    
    # Create DataFrame
    df = pl.DataFrame({
        "date": [a["date"] for a in articles],
        "country": [a["country"] for a in articles],
        "headline": [a["headline"] for a in articles],
        "response": [a["response"] for a in articles],
        "status_code": [a["status_code"] for a in articles],
        "token_ids": token_ids,
        "attention_mask": attention_masks,
    })
    
    # Cast dtypes
    df = df.with_columns([
        pl.col("date").str.to_date(),
        pl.col("status_code").cast(pl.Int32),
    ])
    
    # Filter out error records (status_code != 200)
    initial_count = len(df)
    df = df.filter(pl.col("status_code") == 200)
    filtered_count = len(df)
    
    if initial_count > filtered_count:
        logger.info(
            f"Filtered records: {initial_count} total -> {filtered_count} valid "
            f"({initial_count - filtered_count} errors removed)"
        )
    
    return df


def save_country_parquet(
    country_name: str,
    df: pl.DataFrame,
    output_dir: Path = DATA_PROCESSED_DIR
) -> Path:
    """
    Save country DataFrame to parquet file with LZ4 compression.
    
    Args:
        country_name: Country name for filename
        df: Polars DataFrame to save
        output_dir: Output directory path
        
    Returns:
        Path: Path to saved parquet file
    """
    logger = configure_logger()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"news_{country_name}.parquet"
    
    # Convert to LazyFrame for efficient write
    df_lazy = df.lazy()
    
    # Sink to parquet with LZ4 compression
    df_lazy.sink_parquet(
        str(output_file),
        compression="lz4",
        compression_level=6,
    )
    
    # Get file size for logging
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(
        f"Saved parquet for {country_name}: {output_file} "
        f"({len(df)} rows, {file_size_mb:.2f} MB)"
    )
    
    return output_file


def process_news_to_parquet(
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dict[str, Path]:
    """
    Main execution function: Load all news articles by country, tokenize, and export to parquet.
    
    Args:
        tokenizer: Pre-loaded tokenizer (if None, will load from disk)
        
    Returns:
        dict[str, Path]: Maps country name to output parquet file path
    """
    logger = configure_logger()
    logger.info("=" * 80)
    logger.info("Starting news processing pipeline")
    logger.info("=" * 80)
    
    # Load tokenizer if not provided
    if tokenizer is None:
        logger.info("Loading Jina tokenizer...")
        tokenizer = load_jina_tokenizer()
    
    # Discover countries and files
    logger.info("Discovering country directories...")
    countries_to_files = discover_country_directories()
    
    if not countries_to_files:
        logger.error("No countries found with news data")
        return {}
    
    output_paths: Dict[str, Path] = {}
    
    # Process each country
    for country_name, json_files in countries_to_files.items():
        try:
            logger.info(f"\nProcessing country: {country_name}")
            
            # Load and flatten articles
            articles = load_and_flatten_country_articles(country_name, json_files)
            
            if not articles:
                logger.warning(f"No articles loaded for {country_name}, skipping")
                continue
            
            # Create DataFrame with tokenization
            df = create_articles_dataframe(articles, tokenizer)
            
            if len(df) == 0:
                logger.warning(f"No valid articles for {country_name} after filtering, skipping")
                continue
            
            # Save to parquet
            output_path = save_country_parquet(country_name, df)
            output_paths[country_name] = output_path
        
        except Exception as e:
            logger.error(f"Error processing country {country_name}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Completed. Processed {len(output_paths)} countries")
    logger.info("=" * 80)
    
    return output_paths


if __name__ == "__main__":
    output_paths = process_news_to_parquet()
    
    if output_paths:
        print("\nSuccessfully created parquet files:")
        for country, path in output_paths.items():
            print(f"  {country}: {path}")
    else:
        print("No parquet files were created")
