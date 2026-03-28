import polars as pl
from pathlib import Path

def build_panel_dataset(
    visa_path: Path,
    exchange_path: Path,
    news_dir: Path,
    output_path: Path,
    focus_countries: list[str] = None
):
    print("Loading Visa data...")
    visa_lf = pl.scan_parquet(visa_path)
    visa_monthly = (
        visa_lf
        .with_columns(
            pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month")
        )
        .group_by(["country", "month"])
        .agg(pl.col("issuances").sum().alias("visa_issuances"))
    )

    if focus_countries:
        visa_monthly = visa_monthly.filter(pl.col("country").is_in(focus_countries))

    print("Loading Exchange Rate data...")
    exchange_lf = pl.scan_parquet(exchange_path)
    exchange_monthly = (
        exchange_lf
        .with_columns(
            pl.col("time_period").str.replace("-M", "-")
            .str.strptime(pl.Date, format="%Y-%m", strict=False)
            .alias("month")
        )
        .group_by(["country", "month"])
        .agg(pl.col("obs_value").mean().alias("exchange_rate"))
    )
    
    if focus_countries:
        exchange_monthly = exchange_monthly.filter(pl.col("country").is_in(focus_countries))

    print("Loading News Embeddings (Averaging per month)...")
    news_monthly_dfs = []
    for p in news_dir.glob("news_*.parquet"):
        country_name = p.stem.replace("news_", "").replace("_", " ")
        if focus_countries and country_name not in focus_countries:
            continue
        try:
            news_df = pl.read_parquet(p)
            if "date" in news_df.columns and "embeddings" in news_df.columns:
                n_df = (
                    news_df
                    .with_columns(
                        pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month"),
                        pl.lit(country_name).alias("country")
                    )
                )
                n_monthly = n_df.group_by(["country", "month"]).agg(
                    pl.len().alias("news_event_count")
                )
                news_monthly_dfs.append(n_monthly)
        except Exception as e:
            print(f"Failed processing {p}: {e}")

    news_monthly = None
    if news_monthly_dfs:
        news_monthly = pl.concat(news_monthly_dfs)

    print("Merging into Panel...")
    panel = visa_monthly.join(exchange_monthly, on=["country", "month"], how="full", coalesce=True)
    
    if news_monthly is not None:
        panel = panel.join(news_monthly.lazy(), on=["country", "month"], how="full", coalesce=True)

    panel = panel.sort(["country", "month"]).with_columns([
        pl.col("visa_issuances").fill_null(0),
        pl.col("exchange_rate").forward_fill().backward_fill().over("country"),
        pl.col("news_event_count").fill_null(0) if news_monthly is not None else pl.lit(0).alias("news_event_count")
    ])

    print("Creating AR features and Lead Targets...")
    for lag in range(1, 7):
        panel = panel.with_columns([
            pl.col("visa_issuances").shift(lag).over("country").alias(f"visa_lag_{lag}"),
            pl.col("exchange_rate").shift(lag).over("country").alias(f"exchange_lag_{lag}"),
            pl.col("news_event_count").shift(lag).over("country").alias(f"news_lag_{lag}"),
            pl.col("visa_issuances").shift(-lag).over("country").alias(f"target_visa_lead_{lag}")
        ])

    panel_df = panel.collect()
    panel_df = panel_df.drop_nulls(subset=[f"visa_lag_{i}" for i in range(1,7)] + [f"target_visa_lead_{i}" for i in range(1,7)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.write_parquet(output_path)
    print(f"Dataset generated. Shape: {panel_df.shape} -> saved to {output_path}")
    return panel_df

if __name__ == "__main__":
    build_panel_dataset(
        visa_path=Path("data/processed/visa_master.parquet"),
        exchange_path=Path("data/processed/exchange_rate.parquet"),
        news_dir=Path("data/processed/news_embeddings_labeled"),
        output_path=Path("data/processed/train_panel.parquet")
    )
