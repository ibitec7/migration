from __future__ import annotations

import argparse
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


def _progress(iterable, desc: str):
	if tqdm is None:
		return iterable
	return tqdm(iterable, desc=desc)


def _canonical_country(value: str) -> str:
	if value is None:
		return ""
	token = str(value).strip().replace("_", " ")
	token = " ".join(token.split())
	low = token.lower()

	if "china" in low:
		return "China"
	if "venezuela" in low:
		return "Venezuela"
	if "united states" in low or low in {"us", "u.s.", "usa"}:
		return "US"

	mapped = DEFAULT_COUNTRY_MAP.get(low, token)
	if mapped == "US":
		return mapped
	return " ".join(part.capitalize() for part in mapped.split())


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


def detect_exchange_shocks(values: list[float], quantile: float = 0.80) -> list[bool]:
	arr = np.array(values, dtype=float)
	prev = np.roll(arr, 1)
	prev[0] = np.nan
	mom = np.full_like(arr, np.nan, dtype=float)
	valid_prev = np.abs(prev) > 1e-9
	mom[valid_prev] = (arr[valid_prev] - prev[valid_prev]) / np.abs(prev[valid_prev])
	abs_mom = np.abs(mom)
	valid = abs_mom[np.isfinite(abs_mom)]
	threshold = np.quantile(valid, quantile) if valid.size > 0 else np.inf
	return ((abs_mom >= threshold) & np.isfinite(abs_mom)).tolist()


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


def load_focus_countries(news_root: Path) -> list[str]:
	if not news_root.exists():
		return []
	countries = []
	for child in sorted(news_root.iterdir()):
		if child.is_dir() and not child.name.startswith("."):
			countries.append(_canonical_country(child.name))
	return sorted(set(countries))


def load_exchange_monthly_lazy(
	exchange_path: Path,
	focus_countries: list[str],
	indicator_contains: str = "consumer prices",
) -> pl.LazyFrame:
	lf = pl.scan_parquet(exchange_path)
	if indicator_contains:
		lf = lf.filter(pl.col("indicator").str.to_lowercase().str.contains(indicator_contains.lower()))

	out = (
		lf.select(
			pl.col("country").cast(pl.Utf8).map_elements(_canonical_country, return_dtype=pl.Utf8).alias("country"),
			pl.col("time_period")
			.cast(pl.Utf8)
			.str.replace("-M", "-")
			.str.strptime(pl.Date, format="%Y-%m", strict=False)
			.alias("month"),
			pl.col("obs_value").cast(pl.Float64).alias("exchange_rate"),
		)
		.drop_nulls(["country", "month", "exchange_rate"])
		.filter(pl.col("country").is_in(focus_countries))
		.group_by(["country", "month"])
		.agg(pl.col("exchange_rate").mean())
	)
	return out


def load_visa_monthly_lazy(visa_path: Path, focus_countries: list[str], visa_type: str | None = None) -> pl.LazyFrame:
	lf = pl.scan_parquet(visa_path)
	if visa_type:
		lf = lf.filter(pl.col("visa_type") == visa_type)

	raw_country = pl.col("country").cast(pl.Utf8)
	out = (
		lf.select(
			pl.when(raw_country.str.to_lowercase().str.contains("china"))
			.then(pl.lit("China"))
			.otherwise(raw_country.map_elements(_canonical_country, return_dtype=pl.Utf8))
			.alias("country"),
			pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month"),
			pl.col("issuances").cast(pl.Float64).alias("visa_issuances"),
		)
		.drop_nulls(["country", "month", "visa_issuances"])
		.filter(pl.col("country").is_in(focus_countries))
		.group_by(["country", "month"])
		.agg(pl.col("visa_issuances").sum())
	)
	return out


def _get_country_month_frame(
	country: str,
	monthly_exchange_df: pl.DataFrame,
	monthly_visa_df: pl.DataFrame,
) -> pl.DataFrame:
	exchange = monthly_exchange_df.filter(pl.col("country") == country)
	visa = monthly_visa_df.filter(pl.col("country") == country)
	if exchange.is_empty() or visa.is_empty():
		return pl.DataFrame()

	start = visa["month"].min()
	end = max(exchange["month"].max(), visa["month"].max())
	idx = month_range(start, end)
	base = pl.DataFrame({"month": idx})

	merged = (
		base.join(exchange.select("month", "exchange_rate"), on="month", how="left")
		.join(visa.select("month", "visa_issuances"), on="month", how="left")
		.with_columns(
			pl.col("exchange_rate").fill_null(strategy="forward").fill_null(strategy="backward"),
			pl.col("visa_issuances").fill_null(0.0),
			pl.lit(country).alias("country"),
		)
		.drop_nulls(["exchange_rate"])
	)
	return merged


def create_country_overlay_plot(
	country: str,
	monthly_exchange_df: pl.DataFrame,
	monthly_visa_df: pl.DataFrame,
	output_dir: Path,
) -> Path | None:
	merged = _get_country_month_frame(country, monthly_exchange_df, monthly_visa_df)
	if merged.is_empty():
		return None

	labels = [m.strftime("%Y-%m") for m in merged["month"].to_list()]
	exchange_vals = merged["exchange_rate"].to_list()
	visa_vals = merged["visa_issuances"].to_list()

	width = max(26, min(48, int(len(labels) * 0.35)))
	fig, ax1 = plt.subplots(figsize=(width, 9))

	ax1.plot(
		labels,
		exchange_vals,
		color="#C2410C",
		linewidth=2.6,
		marker="o",
		markersize=2.0,
		label="Exchange Rate (REER)",
		zorder=4,
	)
	ax1.set_title(f"{country}: Exchange Rate and Visa Issuances", pad=14, fontsize=14)
	ax1.set_xlabel("Month")
	ax1.set_ylabel("Exchange Rate (REER)")
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
	file_name = country.lower().replace(" ", "_") + "_exchange_visa_overlay.png"
	out_path = output_dir / file_name
	fig.savefig(out_path, dpi=220)
	plt.close(fig)
	return out_path


def run_all_country_exchange_lag(
	monthly_exchange_df: pl.DataFrame,
	monthly_visa_df: pl.DataFrame,
	max_lag_months: int = 6,
	min_overlap: int = 12,
) -> pl.DataFrame:
	countries = sorted(
		set(monthly_exchange_df["country"].unique().to_list()).intersection(
			set(monthly_visa_df["country"].unique().to_list())
		)
	)
	rows: list[dict] = []

	for country in _progress(countries, "Exchange lag by country"):
		merged = _get_country_month_frame(country, monthly_exchange_df, monthly_visa_df)
		if merged.is_empty():
			continue

		exchange_vals = [float(v) for v in merged["exchange_rate"].to_list()]
		visa_vals = [float(v) for v in merged["visa_issuances"].to_list()]
		visa_surge = detect_surges_list(visa_vals)
		exchange_shock = detect_exchange_shocks(exchange_vals)

		for lag in range(0, max_lag_months + 1):
			shifted_exchange_vals = shift_list(exchange_vals, lag)
			corr, pval = safe_corr_list(shifted_exchange_vals, visa_vals, min_overlap=min_overlap)

			shifted_exchange_shock = shift_list(exchange_shock, lag)
			paired = [(e, v) for e, v in zip(shifted_exchange_shock, visa_surge) if e is not None]
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
					"lag_months": lag,
					"n_months": len(visa_vals),
					"pearson_corr": corr,
					"p_value": pval,
					"exchange_shock_precision": precision,
					"exchange_shock_recall": recall,
					"exchange_shock_lift": lift,
				}
			)

	if not rows:
		return pl.DataFrame()

	out = pl.DataFrame(rows).with_columns(
		pl.col("pearson_corr").abs().alias("abs_corr"),
		pl.Series("q_value", benjamini_hochberg(pl.DataFrame(rows)["p_value"].to_list())),
	)
	return out.sort(["country", "lag_months"])


def summarize_best_lags(results_df: pl.DataFrame) -> pl.DataFrame:
	if results_df.is_empty():
		return results_df
	scored = results_df.sort(["country", "abs_corr"], descending=[False, True])
	best = scored.group_by(["country"], maintain_order=True).first()
	return best.sort(["q_value", "abs_corr"], descending=[False, True])


def generate_all_overlays(
	monthly_exchange_df: pl.DataFrame,
	monthly_visa_df: pl.DataFrame,
	output_dir: Path,
) -> pl.DataFrame:
	countries = sorted(
		set(monthly_exchange_df["country"].unique().to_list()).intersection(
			set(monthly_visa_df["country"].unique().to_list())
		)
	)
	rows = []
	for country in _progress(countries, "Overlay plots by country"):
		path = create_country_overlay_plot(country, monthly_exchange_df, monthly_visa_df, output_dir)
		rows.append({"country": country, "plot_path": str(path) if path else None})
	return pl.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Exchange-rate to visa lead/lag analysis")
	parser.add_argument("--exchange-path", type=Path, default=Path("data/processed/exchange_rate.parquet"))
	parser.add_argument("--visa-path", type=Path, default=Path("data/processed/visa_master.parquet"))
	parser.add_argument("--focus-countries-dir", type=Path, default=Path("data/raw/news"))
	parser.add_argument("--focus-countries", type=str, default=None)
	parser.add_argument("--indicator-contains", type=str, default="consumer prices")
	parser.add_argument("--plots-dir", type=Path, default=Path("data/plots/exchange_vs_visas"))
	parser.add_argument("--output-dir", type=Path, default=Path("data/processed/test_outputs"))
	parser.add_argument("--max-lag", type=int, default=6)
	parser.add_argument("--min-overlap", type=int, default=12)
	parser.add_argument("--visa-type", type=str, default=None)
	return parser


def main() -> None:
	args = build_parser().parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)
	args.plots_dir.mkdir(parents=True, exist_ok=True)

	focus_countries = (
		sorted({_canonical_country(c.strip()) for c in args.focus_countries.split(",") if c.strip()})
		if args.focus_countries
		else load_focus_countries(args.focus_countries_dir)
	)

	load_steps = ["exchange", "visa"]
	for _ in _progress(load_steps, "Loading files"):
		pass

	exchange_lf = load_exchange_monthly_lazy(
		args.exchange_path,
		focus_countries=focus_countries,
		indicator_contains=args.indicator_contains,
	)
	visa_lf = load_visa_monthly_lazy(args.visa_path, focus_countries=focus_countries, visa_type=args.visa_type)

	monthly_exchange_df = exchange_lf.collect()
	monthly_visa_df = visa_lf.collect()

	if monthly_exchange_df.is_empty() or monthly_visa_df.is_empty():
		print("No overlap between exchange and visa datasets for selected countries.")
		return

	merged_monthly = (
		monthly_exchange_df.join(monthly_visa_df, on=["country", "month"], how="inner")
		.sort(["country", "month"])
	)
	merged_monthly.write_parquet(args.output_dir / "exchange_visa_monthly_merged.parquet")
	merged_monthly.write_csv(args.output_dir / "exchange_visa_monthly_merged_review.csv")

	overlay_index = generate_all_overlays(monthly_exchange_df, monthly_visa_df, args.plots_dir)
	overlay_index.write_csv(args.output_dir / "exchange_visa_overlay_index.csv")

	lag_df = run_all_country_exchange_lag(
		monthly_exchange_df=monthly_exchange_df,
		monthly_visa_df=monthly_visa_df,
		max_lag_months=args.max_lag,
		min_overlap=args.min_overlap,
	)
	lag_df.write_parquet(args.output_dir / "exchange_visa_lead_lag_results.parquet")
	lag_df.write_csv(args.output_dir / "exchange_visa_lead_lag_results_review.csv")

	best_df = summarize_best_lags(lag_df)
	best_df.write_parquet(args.output_dir / "exchange_visa_best_lags.parquet")
	best_df.write_csv(args.output_dir / "exchange_visa_best_lags_review.csv")

	print(f"Saved monthly merged (parquet): {args.output_dir / 'exchange_visa_monthly_merged.parquet'}")
	print(f"Saved overlays index (review csv): {args.output_dir / 'exchange_visa_overlay_index.csv'}")
	print(f"Saved lag results (parquet): {args.output_dir / 'exchange_visa_lead_lag_results.parquet'}")
	print(f"Saved best lags (parquet): {args.output_dir / 'exchange_visa_best_lags.parquet'}")


if __name__ == "__main__":
	main()
