"""
Download Google Trends data from Hugging Face Hub.

Usage:
    python src/collection/trends.py
    
This script downloads the entire trends dataset from the Hugging Face repository
and saves it to data/trends/ when run from the project root.
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_trends_data():
    """Download trends data from Hugging Face Hub."""
    
    # Determine the project root (one level up from src/)
    script_dir = Path(__file__).parent.parent.parent
    trends_dir = script_dir / 'data' / 'trends'
    
    logger.info(f"Project root: {script_dir}")
    logger.info(f"Target directory: {trends_dir}")
    
    # Create directory if it doesn't exist
    trends_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Starting download from Hugging Face Hub...")
        logger.info("Repository: sdsc2005-migration/trends")
        
        # Download the entire repository
        snapshot_download(
            repo_id="sdsc2005-migration/trends",
            repo_type="dataset",
            local_dir=str(trends_dir),
            local_dir_use_symlinks=False,  # Actual file downloads, not symlinks
        )
        
        logger.info(f"Successfully downloaded trends data to {trends_dir}")
        
        # List downloaded files
        files = list(trends_dir.rglob('*'))
        file_count = len([f for f in files if f.is_file()])
        logger.info(f"✓ Total files downloaded: {file_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download trends data: {str(e)}")
        logger.error("Make sure you have:")
        logger.error("  1. Installed huggingface_hub: pip install huggingface-hub")
        logger.error("  2. Valid internet connection")
        logger.error("  3. Access to the repository (may require HF token for private repos)")
        return False


if __name__ == "__main__":
    success = download_trends_data()
    sys.exit(0 if success else 1)

