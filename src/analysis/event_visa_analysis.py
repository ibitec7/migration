from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import pearsonr

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_COUNTRY_MAP = {
    "dominican republic": "Dominican Republic",
    "dominican_republic": "Dominican Republic",
    "el salvador": "El Salvador",
    "el_salvador": "El Salvador",
    "united states": "US",
    "usa": "US",
    "u.s.": "US",
}

POSITIVE_WORDS = {
    "improve",
    "improved",
    "improving",
    "growth",
    "recover",
    "recovery",
    "support",
    "stability",
    "stable",
    "peace",
    "agreement",
    "safe",
    "secure",
    "opportunity",
    "jobs",
    "aid",
    "cooperation",
    "progress",
    "success",
    "positive",
}

NEGATIVE_WORDS = {
    "crisis",
    "conflict",
    "war",
    "violence",
    "attack",
    "death",
    "deaths",
    "murder",
    "collapse",
    "inflation",
    "unrest",
    "protest",
    "strike",
    "famine",
    "drought",
    "flood",
    "earthquake",
    "hurricane",
    "disaster",
    "poverty",
    "hunger",
    "unemployment",
    "shortage",
    "sanction",
    "sanctions",
    "instability",
    "corruption",
    "crime",
    "kidnapping",
    "abduction",
    "detention",
    "deportation",
    "smuggling",
    "trafficking",
    "negative",
}


def _canonical_country(value: str) -> str:
    if value is None:
        return ""
    token = str(value).strip().replace("_", " ")
    token = " ".join(token.split())
    mapped = DEFAULT_COUNTRY_MAP.get(token.lower(), token)
    if mapped == "US":
        return mapped
    return " ".join(part.capitalize() for part in mapped.split())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def sentiment_score(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    if pos == 0 and neg == 0:
        return 0.0
    score = (pos - neg) / np.sqrt(len(tokens))
    return float(np.clip(score, -1.0, 1.0))


def month_range(start_month: date, end_month: date) -> list[date]:
    if start_month > end_month:
        return []
    result: list[date] = []
    y, m = start_month.year, start_month.month
    while (y, m) <= (end_month.year, end_month.month):
        result.append(date(y, m, 1))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return result


def shift_list(values: list[float | bool], lag: int) -> list[float | bool | None]:
    if lag <= 0:
        return list(values)
    return [None] * lag + list(values[:-lag])


def detect_surges_list(values: list[float], quantile: float = 0.75, mom_threshold: float = 0.30) -> list[bool]:
    arr = np.array(values, dtype=float)
    positive = arr[arr > 0]
    threshold = np.quantile(positive, quantile) if positive.size > 0 else 0.0
    prev = np.roll(arr, 1)
    prev[0] = np.nan
    mom = np.full_like(arr, np.nan, dtype=float)
    valid_prev = prev > 0
    mom[valid_prev] = (arr[valid_prev] - prev[valid_prev]) / prev[valid_prev]
    return ((arr > threshold) | (mom >= mom_threshold)).tolist()


def safe_corr_list(x: list[float | None], y: list[float | None], min_overlap: int) -> tuple[float, float]:
    xv: list[float] = []
    yv: list[float] = []
    for xi, yi in zip(x, y):
        if xi is None or yi is None:
            continue
        if isinstance(xi, float) and np.isnan(xi):
            continue
        if isinstance(yi, float) and np.isnan(yi):
            continue
        xv.append(float(xi))
        yv.append(float(yi))
    if len(xv) < min_overlap:
        return np.nan, np.nan
    if len(set(xv)) <= 1 or len(set(yv)) <= 1:
        return np.nan, np.nan
    corr, pval = pearsonr(xv, yv)
    return float(corr), float(pval)


def benjamini_hochberg(values: list[float | None]) -> list[float | None]:
    indexed = [(idx, val) for idx, val in enumerate(values) if val is not None and not np.isnan(val)]
    if not indexed:
        return [None for _ in values]
    indexed.sort(key=lambda t: t[1])
    n = len(indexed)
    adjusted = [None for _ in values]
    prev = 1.0
    for i in range(n - 1, -1, -1):
        original_idx, p = indexed[i]
        rank = i + 1
        q = min(prev, p * n / rank)
        adjusted[original_idx] = float(q)
        prev = q
    return adjusted


def print_pretty(title: str, frame: pl.DataFrame, rows: int = 12) -> None:
    print(f"\n=== {title} ===")
    if frame.is_empty():
        print("(empty)")
        return
    with pl.Config(tbl_rows=rows, tbl_cols=16, tbl_width_chars=160, fmt_str_lengths=40):
        print(frame.head(rows))


def _progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def discover_event_files(news_labeled_dir: Path) -> list[Path]:
    return sorted(news_labeled_dir.glob("*.parquet"))


def _infer_country_from_file(file_path: Path) -> str:
    stem = file_path.stem
    stem = stem.removeprefix("news_")
    stem = stem.removeprefix("labeled_")
    stem = stem.removesuffix("_labeled")
    return _canonical_country(stem)


def load_labeled_events_lazy(news_labeled_dir: Path) -> pl.LazyFrame:
    file_paths = discover_event_files(news_labeled_dir)
    lazy_frames: list[pl.LazyFrame] = []

    for file_path in _progress(file_paths, "Loading labeled event files"):
        lf = pl.scan_parquet(file_path)
        names = set(lf.collect_schema().names())
        if "date" not in names or "cluster_label" not in names:
            continue

        text_col = None
        for candidate in ["headline", "title", "summary", "response", "text"]:
            if candidate in names:
                text_col = candidate
                break

        country_expr = pl.col("country").cast(pl.Utf8) if "country" in names else pl.lit(_infer_country_from_file(file_path))
        headline_expr = pl.col(text_col).cast(pl.Utf8) if text_col else pl.lit("")

        cur = (
            lf.select(
                pl.col("date").alias("date"),
                pl.col("cluster_label").cast(pl.Utf8).alias("cluster_label"),
                country_expr.alias("country"),
                headline_expr.alias("headline"),
            )
            .with_columns(
                pl.col("country").map_elements(_canonical_country, return_dtype=pl.Utf8),
                pl.col("cluster_label").fill_null("Unknown"),
                pl.col("headline").fill_null(""),
                pl.col("date").cast(pl.Date, strict=False),
            )
            .drop_nulls(["date", "country"])
        )
        lazy_frames.append(cur)

    if not lazy_frames:
        return pl.DataFrame({"date": [], "cluster_label": [], "country": [], "headline": []}).lazy()

    return pl.concat(lazy_frames, how="diagonal_relaxed")


def load_visa_monthly_lazy(visa_path: Path, visa_type: str | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(visa_path)
    if visa_type:
        lf = lf.filter(pl.col("visa_type") == visa_type)

    raw_country = pl.col("country").cast(pl.Utf8)
    return (
        lf.select(
            pl.when(raw_country.str.to_lowercase().str.contains("china"))
            .then(pl.lit("China"))
            .otherwise(raw_country.map_elements(_canonical_country, return_dtype=pl.Utf8))
            .alias("country"),
            pl.col("date").cast(pl.Date).alias("date"),
            pl.col("issuances").cast(pl.Float64).alias("visa_issuances"),
        )
        .drop_nulls(["country", "date", "visa_issuances"])
        .group_by(["country", "date"])
        .agg(pl.col("visa_issuances").sum())
    )


def build_monthly_event_counts_lazy(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        events_lf.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by(["country", "month", "cluster_label"])
        .agg(pl.len().alias("event_count"))
    )


def build_monthly_visa_counts_lazy(visa_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        visa_lf.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by(["country", "month"])
        .agg(pl.col("visa_issuances").sum().alias("visa_issuances"))
    )


def build_monthly_sentiment_lazy(events_lf: pl.LazyFrame, by_label: bool = True) -> pl.LazyFrame:
    cols = ["country", "month"]
    if by_label:
        cols.append("cluster_label")

    with_sent = events_lf.with_columns(
        pl.col("headline").map_elements(sentiment_score, return_dtype=pl.Float64).alias("sentiment_score"),
        pl.col("date").dt.truncate("1mo").alias("month"),
        (pl.col("headline").map_elements(sentiment_score, return_dtype=pl.Float64) > 0).cast(pl.Float64).alias("sent_pos"),
        (pl.col("headline").map_elements(sentiment_score, return_dtype=pl.Float64) < 0).cast(pl.Float64).alias("sent_neg"),
    )

    return with_sent.group_by(cols).agg(
        pl.col("sentiment_score").mean().alias("sentiment_mean"),
        pl.col("sentiment_score").std().fill_null(0.0).alias("sentiment_std"),
        pl.len().alias("sentiment_count"),
        pl.col("sent_pos").mean().alias("sentiment_pos_share"),
        pl.col("sent_neg").mean().alias("sentiment_neg_share"),
    )


def trim_monthly_to_visa_start(monthly_df: pl.DataFrame, monthly_visa_df: pl.DataFrame) -> pl.DataFrame:
    if monthly_df.is_empty() or monthly_visa_df.is_empty():
        return monthly_df
    visa_start = monthly_visa_df.group_by("country").agg(pl.col("month").min().alias("visa_start_month"))
    return (
        monthly_df.join(visa_start, on="country", how="inner")
        .filter(pl.col("month") >= pl.col("visa_start_month"))
        .drop("visa_start_month")
    )


def _get_country_month_frame(country: str, monthly_events_df: pl.DataFrame, monthly_visa_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    events = monthly_events_df.filter(pl.col("country") == country)
    visa = monthly_visa_df.filter(pl.col("country") == country)
    if events.is_empty() or visa.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    start = visa["month"].min()
    end = max(events["month"].max(), visa["month"].max())
    idx = month_range(start, end)
    base = pl.DataFrame({"month": idx})

    events_pivot = (
        events.pivot(values="event_count", index="month", on="cluster_label", aggregate_function="sum")
        .sort("month")
    )
    events_aligned = base.join(events_pivot, on="month", how="left").fill_null(0)

    visa_aligned = (
        base.join(visa.select("month", "visa_issuances"), on="month", how="left")
        .with_columns(pl.col("visa_issuances").fill_null(0.0))
        .sort("month")
    )

    merged = events_aligned.join(visa_aligned, on="month", how="left", suffix="_v").with_columns(
        pl.col("visa_issuances").fill_null(0.0)
    )
    return merged, events_aligned


def create_country_overlay_plot(
    country: str,
    monthly_events_df: pl.DataFrame,
    monthly_visa_df: pl.DataFrame,
    output_dir: Path,
    top_labels: int = 8,
) -> Path | None:
    merged, events_pivot = _get_country_month_frame(country, monthly_events_df, monthly_visa_df)
    if merged.is_empty():
        return None

    label_cols = [c for c in events_pivot.columns if c != "month"]
    totals = {c: float(events_pivot[c].sum()) for c in label_cols}
    keep = [name for name, _ in sorted(totals.items(), key=lambda t: t[1], reverse=True)[:top_labels]]

    reduced = events_pivot.select(["month", *keep]) if keep else events_pivot.select(["month"])
    dropped = [c for c in label_cols if c not in keep]
    if dropped:
        other_series = events_pivot.select(sum(pl.col(c) for c in dropped).alias("Other"))
        reduced = reduced.hstack(other_series)

    labels = [m.strftime("%Y-%m") for m in reduced["month"].to_list()]
    visa_vals = merged["visa_issuances"].to_list()

    width = max(26, min(30, int(len(labels) * 0.35)))
    fig, ax1 = plt.subplots(figsize=(width, 9))
    bottom = np.zeros(len(labels))
    cmap = plt.get_cmap("tab20")
    for col in [c for c in reduced.columns if c != "month"]:
        vals = np.array(reduced[col].to_list(), dtype=float)
        ax1.bar(
            labels,
            vals,
            bottom=bottom,
            label=col,
            alpha=0.88,
            color=cmap(hash(col) % cmap.N),
            edgecolor="white",
            linewidth=0.25,
        )
        bottom += vals

    ax1.set_title(f"{country}: Monthly Event Labels and Visa Issuances", pad=14, fontsize=14)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Event Article Count")
    ax1.tick_params(axis="x", labelrotation=90)
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", va="top")
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(
        labels,
        visa_vals,
        color="#1E40AF",
        linewidth=2.8,
        marker="o",
        markersize=2.2,
        label="Visa Issuances",
        zorder=5,
    )
    ax2.set_ylabel("Visa Issuances")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0, 0.83, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = country.lower().replace(" ", "_") + "_events_visa_overlay.png"
    out_path = output_dir / file_name
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def run_all_country_lead_lag(
    monthly_events_df: pl.DataFrame,
    monthly_visa_df: pl.DataFrame,
    max_lag_months: int = 6,
    min_overlap: int = 12,
    min_event_months: int = 6,
) -> pl.DataFrame:
    countries = sorted(set(monthly_events_df["country"].unique().to_list()).intersection(set(monthly_visa_df["country"].unique().to_list())))
    rows: list[dict] = []

    for country in _progress(countries, "Lead/lag by country"):
        merged, events_aligned = _get_country_month_frame(country, monthly_events_df, monthly_visa_df)
        if merged.is_empty():
            continue

        visa_vals = [float(v) for v in merged["visa_issuances"].to_list()]
        visa_surge = detect_surges_list(visa_vals)

        for label in [c for c in events_aligned.columns if c != "month"]:
            event_vals = [float(v) for v in events_aligned[label].to_list()]
            active_months = int(sum(v > 0 for v in event_vals))
            if active_months < min_event_months:
                continue
            event_surge = detect_surges_list(event_vals)

            for lag in range(0, max_lag_months + 1):
                shifted_event_vals = shift_list(event_vals, lag)
                corr, pval = safe_corr_list(shifted_event_vals, visa_vals, min_overlap=min_overlap)

                shifted_event_surge = shift_list(event_surge, lag)
                paired = [(e, v) for e, v in zip(shifted_event_surge, visa_surge) if e is not None]
                if len(paired) >= min_overlap:
                    tp = sum((e is True) and (v is True) for e, v in paired)
                    pred = sum(e is True for e, _ in paired)
                    true = sum(v is True for _, v in paired)
                    precision = tp / pred if pred > 0 else np.nan
                    recall = tp / true if true > 0 else np.nan
                    baseline = true / len(paired) if paired else np.nan
                    lift = precision / baseline if not np.isnan(precision) and baseline and baseline > 0 else np.nan
                else:
                    precision = np.nan
                    recall = np.nan
                    lift = np.nan

                rows.append(
                    {
                        "country": country,
                        "cluster_label": label,
                        "lag_months": lag,
                        "n_months": len(visa_vals),
                        "event_active_months": active_months,
                        "pearson_corr": corr,
                        "p_value": pval,
                        "surge_precision": precision,
                        "surge_recall": recall,
                        "surge_lift": lift,
                    }
                )

    if not rows:
        return pl.DataFrame()

    out = pl.DataFrame(rows)
    out = out.with_columns(pl.Series("q_value", benjamini_hochberg(out["p_value"].to_list())))
    return out.sort(["country", "cluster_label", "lag_months"])


def run_all_country_sentiment_lead_lag(
    monthly_sentiment_df: pl.DataFrame,
    monthly_visa_df: pl.DataFrame,
    max_lag_months: int = 6,
    min_overlap: int = 12,
    min_sentiment_months: int = 6,
) -> pl.DataFrame:
    if monthly_sentiment_df.is_empty() or monthly_visa_df.is_empty():
        return pl.DataFrame()

    countries = sorted(set(monthly_sentiment_df["country"].unique().to_list()).intersection(set(monthly_visa_df["country"].unique().to_list())))
    rows: list[dict] = []

    has_label = "cluster_label" in monthly_sentiment_df.columns

    for country in _progress(countries, "Sentiment lag by country"):
        sent_country = monthly_sentiment_df.filter(pl.col("country") == country)
        visa_country = monthly_visa_df.filter(pl.col("country") == country)
        if sent_country.is_empty() or visa_country.is_empty():
            continue

        labels = sorted(sent_country["cluster_label"].drop_nulls().unique().to_list()) if has_label else ["ALL"]

        for label in labels:
            cur = sent_country.filter(pl.col("cluster_label") == label) if has_label else sent_country
            if cur.height < min_sentiment_months:
                continue

            start = visa_country["month"].min()
            end = max(cur["month"].max(), visa_country["month"].max())
            idx = month_range(start, end)
            base = pl.DataFrame({"month": idx})

            sent_aligned = (
                base.join(cur.select("month", "sentiment_mean"), on="month", how="left")
                .with_columns(pl.col("sentiment_mean").fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0))
                .sort("month")
            )
            visa_aligned = (
                base.join(visa_country.select("month", "visa_issuances"), on="month", how="left")
                .with_columns(pl.col("visa_issuances").fill_null(0.0))
                .sort("month")
            )

            sent_vals = [float(v) for v in sent_aligned["sentiment_mean"].to_list()]
            visa_vals = [float(v) for v in visa_aligned["visa_issuances"].to_list()]
            visa_surge = detect_surges_list(visa_vals)

            sent_arr = np.array(sent_vals, dtype=float)
            z = (sent_arr - sent_arr.mean()) / (sent_arr.std() + 1e-9)
            neg_shock = (z <= -1.0).tolist()
            pos_shock = (z >= 1.0).tolist()

            for lag in range(0, max_lag_months + 1):
                shifted_sent = shift_list(sent_vals, lag)
                corr, pval = safe_corr_list(shifted_sent, visa_vals, min_overlap=min_overlap)

                shifted_neg = shift_list(neg_shock, lag)
                shifted_pos = shift_list(pos_shock, lag)

                def _shock_metrics(shifts: list[bool | None]) -> tuple[float, float]:
                    paired = [(e, v) for e, v in zip(shifts, visa_surge) if e is not None]
                    if len(paired) < min_overlap:
                        return np.nan, np.nan
                    tp = sum((e is True) and (v is True) for e, v in paired)
                    pred = sum(e is True for e, _ in paired)
                    true = sum(v is True for _, v in paired)
                    precision = tp / pred if pred > 0 else np.nan
                    baseline = true / len(paired) if paired else np.nan
                    lift = precision / baseline if not np.isnan(precision) and baseline and baseline > 0 else np.nan
                    return precision, lift

                neg_precision, neg_lift = _shock_metrics(shifted_neg)
                pos_precision, pos_lift = _shock_metrics(shifted_pos)

                rows.append(
                    {
                        "country": country,
                        "cluster_label": label,
                        "lag_months": lag,
                        "n_months": len(idx),
                        "pearson_corr": corr,
                        "p_value": pval,
                        "neg_shock_precision": neg_precision,
                        "neg_shock_lift": neg_lift,
                        "pos_shock_precision": pos_precision,
                        "pos_shock_lift": pos_lift,
                    }
                )

    if not rows:
        return pl.DataFrame()

    out = pl.DataFrame(rows).with_columns(
        pl.col("pearson_corr").abs().alias("abs_corr"),
        pl.Series("q_value", benjamini_hochberg(pl.DataFrame(rows)["p_value"].to_list())),
    )
    return out.sort(["country", "cluster_label", "lag_months"])


def summarize_best_lags(results_df: pl.DataFrame) -> pl.DataFrame:
    if results_df.is_empty():
        return results_df
    scored = results_df.with_columns(pl.col("pearson_corr").abs().alias("abs_corr")).sort(
        ["country", "cluster_label", "abs_corr"], descending=[False, False, True]
    )
    best = scored.group_by(["country", "cluster_label"], maintain_order=True).first()
    return best.sort(["q_value", "abs_corr"], descending=[False, True])


def summarize_best_sentiment_lags(results_df: pl.DataFrame) -> pl.DataFrame:
    if results_df.is_empty():
        return results_df
    scored = results_df.sort(["country", "cluster_label", "abs_corr"], descending=[False, False, True])
    best = scored.group_by(["country", "cluster_label"], maintain_order=True).first()
    return best.sort(["q_value", "abs_corr"], descending=[False, True])


def generate_all_overlays(
    monthly_events_df: pl.DataFrame,
    monthly_visa_df: pl.DataFrame,
    output_dir: Path,
    top_labels: int = 8,
) -> pl.DataFrame:
    countries = sorted(set(monthly_events_df["country"].unique().to_list()).intersection(set(monthly_visa_df["country"].unique().to_list())))
    rows = []
    for country in _progress(countries, "Overlay plots by country"):
        path = create_country_overlay_plot(country, monthly_events_df, monthly_visa_df, output_dir, top_labels=top_labels)
        rows.append({"country": country, "plot_path": str(path) if path else None})
    return pl.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Event-to-visa overlay and lead/lag analysis")
    parser.add_argument("--news-labeled-dir", type=Path, default=Path("data/processed/news_embeddings_labeled"))
    parser.add_argument("--visa-path", type=Path, default=Path("data/processed/visa_master.parquet"))
    parser.add_argument("--plots-dir", type=Path, default=Path("data/plots/events_vs_visas"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/test_outputs"))
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--top-labels", type=int, default=8)
    parser.add_argument("--min-overlap", type=int, default=12)
    parser.add_argument("--min-event-months", type=int, default=6)
    parser.add_argument("--visa-type", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    events_lf = load_labeled_events_lazy(args.news_labeled_dir)
    visa_lf = load_visa_monthly_lazy(args.visa_path, visa_type=args.visa_type)

    monthly_events_lf = build_monthly_event_counts_lazy(events_lf)
    monthly_visa_lf = build_monthly_visa_counts_lazy(visa_lf)
    monthly_sentiment_lf = build_monthly_sentiment_lazy(events_lf, by_label=True)

    monthly_events_df = monthly_events_lf.collect()
    monthly_visa_df = monthly_visa_lf.collect()
    monthly_sentiment_df = monthly_sentiment_lf.collect()

    monthly_events_df = trim_monthly_to_visa_start(monthly_events_df, monthly_visa_df)
    monthly_sentiment_df = trim_monthly_to_visa_start(monthly_sentiment_df, monthly_visa_df)

    overlay_index = generate_all_overlays(monthly_events_df, monthly_visa_df, args.plots_dir, top_labels=args.top_labels)
    overlay_index.write_csv(args.output_dir / "event_visa_overlay_index.csv")

    lead_lag_df = run_all_country_lead_lag(
        monthly_events_df=monthly_events_df,
        monthly_visa_df=monthly_visa_df,
        max_lag_months=args.max_lag,
        min_overlap=args.min_overlap,
        min_event_months=args.min_event_months,
    )
    lead_lag_df.write_parquet(args.output_dir / "event_visa_lead_lag_results.parquet")
    lead_lag_df.write_csv(args.output_dir / "event_visa_lead_lag_results_review.csv")

    best_lags_df = summarize_best_lags(lead_lag_df)
    best_lags_df.write_parquet(args.output_dir / "event_visa_best_lags.parquet")
    best_lags_df.write_csv(args.output_dir / "event_visa_best_lags_review.csv")

    monthly_sentiment_df.write_parquet(args.output_dir / "event_monthly_sentiment.parquet")

    sentiment_lag_df = run_all_country_sentiment_lead_lag(
        monthly_sentiment_df=monthly_sentiment_df,
        monthly_visa_df=monthly_visa_df,
        max_lag_months=args.max_lag,
        min_overlap=args.min_overlap,
        min_sentiment_months=args.min_event_months,
    )
    sentiment_lag_df.write_parquet(args.output_dir / "event_sentiment_lead_lag_results.parquet")
    sentiment_lag_df.write_csv(args.output_dir / "event_sentiment_lead_lag_results_review.csv")

    best_sentiment_df = summarize_best_sentiment_lags(sentiment_lag_df)
    best_sentiment_df.write_parquet(args.output_dir / "event_sentiment_best_lags.parquet")
    best_sentiment_df.write_csv(args.output_dir / "event_sentiment_best_lags_review.csv")

    print(f"\nSaved overlays index (review csv): {args.output_dir / 'event_visa_overlay_index.csv'}")
    print(f"Saved lead/lag results (parquet): {args.output_dir / 'event_visa_lead_lag_results.parquet'}")
    print(f"Saved best lags summary (parquet): {args.output_dir / 'event_visa_best_lags.parquet'}")
    print(f"Saved monthly sentiment (parquet): {args.output_dir / 'event_monthly_sentiment.parquet'}")
    print(f"Saved sentiment lag results (parquet): {args.output_dir / 'event_sentiment_lead_lag_results.parquet'}")
    print(f"Saved best sentiment lags (parquet): {args.output_dir / 'event_sentiment_best_lags.parquet'}")


if __name__ == "__main__":
    main()
