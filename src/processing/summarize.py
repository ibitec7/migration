"""
News article summarization using TensorRT-optimized FLAN-T5-Large.

This module provides the core pipeline for:
- Loading news articles from JSON files
- Filtering by status code
- Batch processing articles through TensorRT inference
- Adding summaries back to original JSON files
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from tqdm import tqdm

from ..models.flant5_engine import TensorRTFlanT5Engine
from .prompts import PromptTemplate, get_prompt_template
from .utils import setup_logger


class NewsArticleSummarizer:
    """Orchestrates news article summarization using TensorRT."""
    
    def __init__(
        self,
        engine: TensorRTFlanT5Engine,
        prompt_template: Optional[PromptTemplate] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize summarizer.
        
        Args:
            engine: TensorRT inference engine instance
            prompt_template: Prompt template for summarization
            logger: Logger instance (creates default if not provided)
        """
        self.engine = engine
        self.prompt_template = prompt_template or get_prompt_template("default")
        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logger.addHandler(logging.StreamHandler())
                logger.setLevel(logging.INFO)
        self.logger = logger
        
        # Statistics tracking
        self.stats = {
            "total_articles": 0,
            "processed": 0,
            "skipped_status_code": 0,
            "skipped_empty_response": 0,
            "inference_errors": 0,
        }
    
    def process_article(
        self,
        article: Dict[str, Any],
        include_prompt: bool = False
    ) -> Optional[str]:
        """
        Summarize a single article.
        
        Args:
            article: Article dictionary with 'response' and 'status_code' keys
            include_prompt: If True, include prompt in summary metadata
            
        Returns:
            Summary string or None if skipped/failed
        """
        self.stats["total_articles"] += 1
        
        # Check status code
        status_code = article.get("status_code")
        if status_code != 200:
            self.stats["skipped_status_code"] += 1
            return None
        
        # Check for response content
        response = article.get("response")
        if not response or not isinstance(response, str) or response.strip() == "":
            self.stats["skipped_empty_response"] += 1
            return None
        
        try:
            # Format prompt with article text
            prompt_input = self.prompt_template.format(response)
            
            # Run inference
            summary = self.engine(prompt_input)
            
            # Ensure we got a string back
            if isinstance(summary, list):
                summary = summary[0] if summary else ""
            
            summary = str(summary).strip()
            
            self.stats["processed"] += 1
            return summary
            
        except Exception as e:
            self.logger.error(f"Error inferencing article: {e}")
            self.stats["inference_errors"] += 1
            return None
    
    def process_batch(
        self,
        articles: List[Dict[str, Any]],
        batch_size: int = 4
    ) -> List[Optional[str]]:
        """
        Summarize a batch of articles.
        
        Args:
            articles: List of article dictionaries
            batch_size: Number of articles to process in each batch
            
        Returns:
            List of summaries (None for skipped/failed articles)
        """
        summaries = []
        
        # Process in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Filter valid articles for batch processing
            valid_articles = []
            valid_indices = []
            
            for idx, article in enumerate(batch):
                status_code = article.get("status_code")
                response = article.get("response")
                
                if (status_code == 200 and response and 
                    isinstance(response, str) and response.strip() != ""):
                    valid_articles.append(article)
                    valid_indices.append(idx)
            
            # Process valid articles
            for article_idx, article in enumerate(valid_articles):
                try:
                    summary = self.process_article(article)
                    summaries.append(summary)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    self.stats["inference_errors"] += 1
                    summaries.append(None)
        
        return summaries
    
    def process_news_file(
        self,
        file_path: Path,
        backup: bool = True,
        batch_size: int = 4
    ) -> Tuple[int, int]:
        """
        Process a single news JSON file.
        
        Loads articles, summarizes those with status_code==200, and adds
        'summary_t5' field back to each article. Saves updated JSON.
        
        Args:
            file_path: Path to news JSON file
            backup: Create backup of original file before modification
            batch_size: Articles per batch for inference
            
        Returns:
            Tuple of (processed_count, skipped_count)
        """
        try:
            # Load JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            articles = data.get("articles", [])
            if not articles:
                self.logger.warning(f"No articles found in {file_path}")
                return 0, 0
            
            self.logger.info(f"Processing {file_path} ({len(articles)} articles)")
            
            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix('.json.backup')
                with open(backup_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Process articles in batches
            processed_count = 0
            skipped_count = 0
            
            for i, article in enumerate(tqdm(articles, desc="Summarizing")):
                summary = self.process_article(article)
                
                if summary is not None:
                    article["summary_t5"] = summary
                    processed_count += 1
                else:
                    skipped_count += 1
            
            # Save updated JSON
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(
                f"Completed {file_path}: {processed_count} summarized, "
                f"{skipped_count} skipped"
            )
            
            return processed_count, skipped_count
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return 0, 0
    
    def process_news_directory(
        self,
        base_dir: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Process all news JSON files in a directory.
        
        Args:
            base_dir: Base directory to search
            patterns: File patterns to match (default: ['news_*.json'])
            recursive: Whether to search subdirectories
            batch_size: Articles per batch
            
        Returns:
            Summary statistics dictionary
        """
        if patterns is None:
            patterns = ["news_*.json"]
        
        base_dir = Path(base_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        # Find all matching JSON files
        json_files = []
        for pattern in patterns:
            if recursive:
                json_files.extend(base_dir.rglob(pattern))
            else:
                json_files.extend(base_dir.glob(pattern))
        
        json_files = sorted(set(json_files))  # Remove duplicates, sort
        self.logger.info(f"Found {len(json_files)} news files to process")
        
        if not json_files:
            self.logger.warning(f"No news files found matching {patterns} in {base_dir}")
            return self.stats
        
        # Process each file
        total_processed = 0
        total_skipped = 0
        
        for file_path in tqdm(json_files, desc="Processing files"):
            processed, skipped = self.process_news_file(
                file_path,
                batch_size=batch_size
            )
            total_processed += processed
            total_skipped += skipped
        
        # Log final statistics
        self.logger.info("=" * 60)
        self.logger.info("SUMMARIZATION COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total JSON files processed: {len(json_files)}")
        self.logger.info(f"Total articles processed: {self.stats['processed']}")
        self.logger.info(f"Skipped (bad status): {self.stats['skipped_status_code']}")
        self.logger.info(f"Skipped (empty response): {self.stats['skipped_empty_response']}")
        self.logger.info(f"Inference errors: {self.stats['inference_errors']}")
        self.logger.info("=" * 60)
        
        return self.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


def main(
    news_base_dir: Optional[Path] = None,
    engine_dir: Optional[Path] = None,
    prompt_template_name: str = "default",
    batch_size: int = 4,
    log_file: Optional[Path] = None
):
    """
    Main entry point for news summarization pipeline.
    
    Args:
        news_base_dir: Base directory for news files (default: data/raw/news/)
        engine_dir: TensorRT engine directory (default: auto-detected)
        prompt_template_name: Template to use ('default', 'extraction', 'events')
        batch_size: Articles per batch
        log_file: Path to log file (optional)
    """
    # Set up paths
    migration_root = Path(__file__).parent.parent.parent
    if news_base_dir is None:
        news_base_dir = migration_root / "data" / "raw" / "news"
    else:
        news_base_dir = Path(news_base_dir)
    
    if log_file is None:
        log_file = migration_root / "logs" / f"summarization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = Path(log_file)
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(
        str(log_file),
        log_level=logging.INFO
    )
    
    logger.info("Starting news article summarization pipeline")
    logger.info(f"News directory: {news_base_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Prompt template: {prompt_template_name}")
    logger.info(f"Batch size: {batch_size}")
    
    try:
        # Load engine
        logger.info("Loading TensorRT FLAN-T5 engine...")
        engine = TensorRTFlanT5Engine(engine_dir=engine_dir, lazy_load=False)
        logger.info("Engine loaded successfully")
        
        # Get prompt template
        prompt_template = get_prompt_template(prompt_template_name)
        
        # Create summarizer
        summarizer = NewsArticleSummarizer(engine, prompt_template, logger)
        
        # Process all news files
        with engine:
            stats = summarizer.process_news_directory(
                news_base_dir,
                recursive=True,
                batch_size=batch_size
            )
        
        return stats
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Default execution
    main()
