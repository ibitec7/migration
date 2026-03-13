import polars as pl
import os

from utils import setup_logger, get_optimal_process_count, MONTHS_MAP

os.makedirs("./logs", exist_ok=True)

global logger
logger = setup_logger(log_file="./logs/merge.log", write_console=False)

OPTIMAL_PROCESS_COUNT = get_optimal_process_count()
logger.info(f"CPU cores detected: {OPTIMAL_PROCESS_COUNT}")
logger.info(f"Optimal process pool size: {OPTIMAL_PROCESS_COUNT}")

def scan_data(file_path: str) -> pl.LazyFrame:
    """Scan data from a single CSV file into a Polars LazyFrame."""
    try:
        lf: pl.LazyFrame = pl.scan_csv(file_path)
        logger.info(f"Successfully read {file_path}")

        lf.with_columns(
            pl.col("Months (abbv)").replace(MONTHS_MAP).alias("months_num")
        )

        # lf.columns = 
        return lf
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return pl.LazyFrame()

def merge_csv_files(data_dir: str) -> pl.DataFrame:
    """Merge all CSV files in the specified directory into a single DataFrame."""
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return pl.DataFrame()
    
    logger.info(f"Found {len(csv_files)} CSV files to merge.")
    
    lazy_frames = [scan_data(file) for file in csv_files]
    
    if not lazy_frames:
        logger.warning("No valid data to merge after scanning.")
        return pl.DataFrame()
    
    merged_df: pl.DataFrame = pl.concat(lazy_frames).collect()
    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    
    return merged_df

