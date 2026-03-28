#!/usr/bin/env python3
"""
Runner script for news article summarization using TensorRT FLAN-T5.

This script orchestrates the complete pipeline:
1. Load TensorRT FLAN-T5 engine
2. Discover all news JSON files
3. Process articles in batches
4. Add summaries back to JSON files

Usage:
    python -m src.processing.run_summarization
    python src/processing/run_summarization.py --help
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

from .summarize import NewsArticleSummarizer
from .prompts import get_prompt_template
from ..models.flant5_engine import TensorRTFlanT5Engine
from .utils import setup_logger


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Summarize news articles using TensorRT FLAN-T5-Large",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all news files with default settings
  python src/processing/run_summarization.py

  # Process with custom news directory
  python src/processing/run_summarization.py --news-dir /path/to/news

  # Use extraction template with batch size of 8
  python src/processing/run_summarization.py --template extraction --batch-size 8

  # Save logs to custom location
  python src/processing/run_summarization.py --log-file /var/log/summarization.log
        """
    )
    
    parser.add_argument(
        "--news-dir",
        type=Path,
        default=None,
        help="Base directory for news JSON files (default: data/raw/news/)"
    )
    
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=None,
        help="Path to TensorRT engine directory "
             "(default: auto-detected from src/models/)"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        default="default",
        choices=["default", "extraction", "events"],
        help="Prompt template to use (default: default)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of articles per batch (default: 4)"
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: logs/summarization_TIMESTAMP.log)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (don't modify files)"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only report statistics without processing"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    if args.batch_size < 1:
        print("Error: batch-size must be >= 1", file=sys.stderr)
        return False
    
    if args.news_dir and not args.news_dir.exists():
        print(f"Error: news directory not found: {args.news_dir}", file=sys.stderr)
        return False
    
    if args.engine_dir and not args.engine_dir.exists():
        print(f"Error: engine directory not found: {args.engine_dir}", file=sys.stderr)
        return False
    
    return True


def count_articles(base_dir: Path) -> dict:
    """Count articles in all news JSON files (for --stats-only mode)."""
    import json
    
    stats = {
        "total_files": 0,
        "total_articles": 0,
        "by_status_code": {},
        "files": []
    }
    
    # Find all news JSON files
    json_files = sorted(base_dir.rglob("news_*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                articles = data.get("articles", [])
                
                stats["total_files"] += 1
                stats["total_articles"] += len(articles)
                
                file_stats = {
                    "path": str(file_path.relative_to(base_dir.parent)),
                    "count": len(articles),
                    "by_status": {}
                }
                
                for article in articles:
                    status = article.get("status_code", "unknown")
                    stats["by_status_code"][status] = stats["by_status_code"].get(status, 0) + 1
                    file_stats["by_status"][status] = file_stats["by_status"].get(status, 0) + 1
                
                stats["files"].append(file_stats)
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}", file=sys.stderr)
    
    return stats


def print_stats(stats: dict):
    """Pretty print summarization statistics."""
    print("\n" + "=" * 60)
    print("NEWS ARTICLE STATISTICS")
    print("=" * 60)
    print(f"Total JSON files: {stats['total_files']}")
    print(f"Total articles: {stats['total_articles']}")
    print("\nArticles by status code:")
    for status in sorted(stats['by_status_code'].keys(), key=str):
        count = stats['by_status_code'][status]
        pct = 100 * count / stats['total_articles'] if stats['total_articles'] > 0 else 0
        print(f"  {status}: {count} ({pct:.1f}%)")
    print("=" * 60 + "\n")


def main_cli(args: argparse.Namespace):
    """Main CLI entry point."""
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    if args.log_file is None:
        migration_root = Path(__file__).parent.parent.parent
        args.log_file = migration_root / "logs" / f"summarization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(args.log_file), log_level=log_level)
    
    # Set default news directory
    if args.news_dir is None:
        migration_root = Path(__file__).parent.parent.parent
        args.news_dir = migration_root / "data" / "raw" / "news"
    
    logger.info("=" * 60)
    logger.info("NEWS ARTICLE SUMMARIZATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"News directory: {args.news_dir}")
    logger.info(f"Engine directory: {args.engine_dir or 'auto'}")
    logger.info(f"Template: {args.template}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Log file: {args.log_file}")
    
    # Stats-only mode
    if args.stats_only:
        logger.info("Running in stats-only mode (no processing)")
        stats = count_articles(args.news_dir)
        print_stats(stats)
        logger.info(f"Found {stats['total_files']} files with {stats['total_articles']} articles")
        return 0
    
    # Full processing
    try:
        logger.info("Loading TensorRT FLAN-T5 engine...")
        engine = TensorRTFlanT5Engine(
            engine_dir=args.engine_dir,
            lazy_load=False
        )
        logger.info("Engine loaded successfully")
        
        # Get prompt template
        prompt_template = get_prompt_template(args.template)
        logger.info(f"Using {args.template} prompt template")
        
        # Create summarizer
        summarizer = NewsArticleSummarizer(engine, prompt_template, logger)
        
        # Process all news files
        logger.info(f"Processing news directory: {args.news_dir}")
        
        with engine:
            stats = summarizer.process_news_directory(
                args.news_dir,
                recursive=True,
                batch_size=args.batch_size
            )
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    # Run main CLI
    exit_code = main_cli(args)
    sys.exit(exit_code)
