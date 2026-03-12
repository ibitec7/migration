import polars as pl
import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import fitz

from utils import setup_logger

os.makedirs("./logs", exist_ok=True)

global logger
logger = setup_logger(log_file="./logs/parse.log", write_console=False)

def get_optimal_process_count() -> int:
    """
    Calculate optimal process count for CPU-bound PDF processing.
    ProcessPoolExecutor uses true parallelism with multiprocessing (avoids GIL).
    """
    cpu_count = multiprocessing.cpu_count()
    optimal_procs = max(2, cpu_count)
    return optimal_procs

OPTIMAL_PROCESS_COUNT = get_optimal_process_count()
logger.info(f"CPU cores detected: {multiprocessing.cpu_count()}")
logger.info(f"Optimal process pool size: {OPTIMAL_PROCESS_COUNT}")

def parse_pdf_file_sync(pdf_path: str) -> pl.LazyFrame:
    """Synchronous PDF extraction using fitz (PyMuPDF) - runs in process pool"""
    
    file_name = Path(pdf_path).name
    file_split = file_name.split(" ")
    month = file_split[0][:3].upper()
    year = int(file_split[1])

    months_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12
    }
    
    lazy_frames = []
    
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                tabs = page.find_tables()
                
                if tabs:
                    for tab in tabs:
                        table = tab.extract()
                        
                        if not table:
                            continue
                        
                        header_idx = 0
                        for i, row in enumerate(table):
                            if row and not any(h is None for h in row):
                                header_idx = i
                                break
                        
                        headers = table[header_idx]
                        headers = [str(col).lower().strip() for col in headers if col]
                        
                        if not headers:
                            continue
                        
                        if headers:
                            headers[0] = "country"
                        
                        data = table[header_idx + 1:]
                        
                        if not data:
                            continue
                        
                        try:
                            lazy_df = (
                                pl.DataFrame(data, schema=headers, orient="row")
                                .lazy()
                                .with_columns([
                                    pl.lit(month).alias("month"),
                                    pl.lit(year).alias("year"),
                                    pl.lit(months_map[month]).alias("month_num"),
                                    pl.date(year=pl.lit(year), month=pl.lit(months_map[month]), day=pl.lit(1)).alias("date")
                                ])
                                .select([
                                    "date",
                                    "month_num",
                                    "month",
                                    "year",
                                    *[col for col in headers if col not in ["month", "year", "month_num"]]
                                ])
                                .filter(pl.col("country").str.to_lowercase() != "grand total")
                                .with_columns(
                                    pl.col("issuances").str.replace_all(",", "").cast(pl.Int32)
                                )
                            )
                            
                            lazy_frames.append(lazy_df)
                            logger.info(f"Processed PDF file: {pdf_path}")
                        except Exception as e:
                            continue
    except Exception as e:
        return pl.DataFrame().lazy()
    
    if not lazy_frames:
        return pl.DataFrame().lazy()
    
    return pl.concat(lazy_frames)

def parse_pdf(data_dir: str = "/home/ibrahim/migration/data/raw/visa/pdf", file_path: str = "/home/ibrahim/migration/data/processed"):
    file_path = "/home/ibrahim/migration/data/processed"
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    logger.info(f"Processing {len(file_paths)} PDF files with {OPTIMAL_PROCESS_COUNT} processes...\n")
    
    with ProcessPoolExecutor(max_workers=OPTIMAL_PROCESS_COUNT) as executor:
        all_data = list(tqdm(executor.map(parse_pdf_file_sync, file_paths), total=len(file_paths), desc="Processing PDFs", unit="file"))
    
    all_data = [df for df in all_data if df.collect().shape[0] > 0]
    
    if all_data:
        master: pl.DataFrame = pl.concat(all_data).sort("date").collect()
        master.write_parquet(os.path.join(file_path, "visa_master.parquet"), compression="lz4")
        logger.info(f"Total rows: {len(master)}")
        logger.info(f"Date range: {master['date'].min()} to {master['date'].max()}")
        logger.info(f"Saved to: {os.path.join(file_path, 'visa_master.parquet')}")
        return master
    else:
        logger.info("No data extracted")
        return pl.DataFrame()

if __name__ == "__main__":
    parse_pdf()