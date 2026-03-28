from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR

try:
	from statsmodels.tsa.ar_model import AutoReg
except Exception:
	AutoReg = None

try:
	from tqdm.auto import tqdm
except Exception:
	tqdm = None

try:
	from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
except Exception:
	SMValueWarning = Warning


warnings.filterwarnings(
	"ignore",
	message="No frequency information was provided, so inferred frequency MS will be used.",
	category=SMValueWarning,
)
warnings.filterwarnings(
	"ignore",
	message="Only PeriodIndexes, DatetimeIndexes with a frequency set, RangesIndexes, and Index with a unit increment support extending.",
	category=UserWarning,
)


DEFAULT_COUNTRY_MAP = {
	"dominican republic": "Dominican Republic",
	"dominican_republic": "Dominican Republic",
	"dominican": "Dominican Republic",
	"el salvador": "El Salvador",
	"el_salvador": "El Salvador",
	"united states": "US",
	"usa": "US",
	"u.s.": "US",
}


MONTH_MAP = {
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
	"DEC": 12,
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
	mapped = DEFAULT_COUNTRY_MAP.get(low, token)
	if mapped == "US":
		return mapped
	return " ".join(part.capitalize() for part in mapped.split())


def _country_from_trends_file(file_path: Path) -> str:
	return _canonical_country(file_path.stem)


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


def load_focus_countries(news_dir: Path, include_us: bool = False) -> list[str]:
	countries: list[str] = []
	for p in sorted(news_dir.glob("news_*.parquet")):
		token = p.stem.replace("news_", "")
		country = _canonical_country(token)
		if not include_us and country == "US":
			continue
		countries.append(country)
	return sorted(set(countries))


def parse_trend_file(file_path: Path) -> pl.DataFrame:
	df = pl.read_parquet(file_path)
	cols = set(df.columns)
	if "time" in cols:
		date_col = "time"
	elif "date" in cols:
		date_col = "date"
	else:
		return pl.DataFrame()

	keywords = [c for c in df.columns if c != date_col]
	if not keywords:
		return pl.DataFrame()

	numeric_exprs = []
	for col in keywords:
		numeric_exprs.append(
			pl.when(pl.col(col).cast(pl.Utf8, strict=False) == "<1")
			.then(pl.lit("0"))
			.otherwise(pl.col(col).cast(pl.Utf8, strict=False))
			.cast(pl.Float64, strict=False)
			.fill_null(0.0)
			.alias(col)
		)

	out = (
		df.select(
			pl.col(date_col).cast(pl.Date).dt.truncate("1mo").alias("month"),
			*[pl.col(k) for k in keywords],
		)
		.with_columns(*numeric_exprs)
		.group_by("month")
		.agg(*[pl.col(k).mean().alias(k) for k in keywords])
		.sort("month")
	)
	return out


def load_trends_long(trends_dir: Path, focus_countries: list[str]) -> pl.DataFrame:
	rows: list[pl.DataFrame] = []
	files = [p for p in sorted(trends_dir.glob("*.parquet")) if p.stem.lower() != "world"]
	for file_path in _progress(files, "Loading trends files"):
		country = _country_from_trends_file(file_path)
		if country not in focus_countries:
			continue
		monthly = parse_trend_file(file_path)
		if monthly.is_empty():
			continue
		long_df = monthly.unpivot(index=["month"], variable_name="keyword", value_name="trend_value").with_columns(
			pl.lit(country).alias("country")
		)
		rows.append(long_df.select(["country", "month", "keyword", "trend_value"]))

	if not rows:
		return pl.DataFrame({"country": [], "month": [], "keyword": [], "trend_value": []})
	return pl.concat(rows, how="vertical_relaxed").sort(["country", "month", "keyword"])


def load_visa_monthly(visa_path: Path, focus_countries: list[str], visa_type: str | None = None) -> pl.DataFrame:
	df = pl.read_parquet(visa_path)
	if visa_type and "visa_type" in df.columns:
		df = df.filter(pl.col("visa_type") == visa_type)

	raw_country = pl.col("country").cast(pl.Utf8)
	out = (
		df.select(
			pl.when(raw_country.str.to_lowercase().str.contains("china"))
			.then(pl.lit("China"))
			.otherwise(raw_country.map_elements(_canonical_country, return_dtype=pl.Utf8))
			.alias("country"),
			pl.col("date").cast(pl.Date).dt.truncate("1mo").alias("month"),
			pl.col("issuances").cast(pl.Float64).alias("issuances"),
		)
		.drop_nulls(["country", "month", "issuances"])
		.filter(pl.col("country").is_in(focus_countries))
		.group_by(["country", "month"])
		.agg(pl.col("issuances").sum())
		.sort(["country", "month"])
	)
	return out


def load_encounters_monthly(encounter_dir: Path, focus_countries: list[str]) -> pl.DataFrame:
	files = sorted(encounter_dir.glob("*.csv"))
	frames: list[pl.DataFrame] = []
	for file_path in _progress(files, "Loading encounter files"):
		frames.append(pl.read_csv(file_path, ignore_errors=True))

	if not frames:
		return pl.DataFrame({"country": [], "month": [], "encounter_count": []})

	df = pl.concat(frames, how="vertical_relaxed").unique()
	if not {"Fiscal Year", "Month (abbv)", "Citizenship Grouping", "Encounter Count"}.issubset(set(df.columns)):
		return pl.DataFrame({"country": [], "month": [], "encounter_count": []})

	out = (
		df.with_columns(
			pl.col("Fiscal Year").cast(pl.Int32, strict=False).alias("fiscal_year"),
			pl.col("Month (abbv)")
			.cast(pl.Utf8)
			.str.to_uppercase()
			.replace_strict(MONTH_MAP, default=None)
			.cast(pl.Int32)
			.alias("month_num"),
			pl.col("Citizenship Grouping").cast(pl.Utf8).map_elements(_canonical_country, return_dtype=pl.Utf8).alias("country"),
			pl.col("Encounter Count").cast(pl.Float64, strict=False).fill_null(0.0).alias("encounter_count"),
		)
		.drop_nulls(["fiscal_year", "month_num", "country"])
		.with_columns(
			pl.when(pl.col("month_num") < 10)
			.then(pl.col("fiscal_year"))
			.otherwise(pl.col("fiscal_year") - 1)
			.alias("calendar_year")
		)
		.with_columns(pl.date(pl.col("calendar_year"), pl.col("month_num"), pl.lit(1)).alias("month"))
		.filter(pl.col("country").is_in(focus_countries))
		.group_by(["country", "month"])
		.agg(pl.col("encounter_count").sum())
		.sort(["country", "month"])
	)
	return out


def build_country_panel(
	country: str,
	trends_long: pl.DataFrame,
	visa_monthly: pl.DataFrame,
	encounter_monthly: pl.DataFrame,
) -> pd.DataFrame:
	tr = trends_long.filter(pl.col("country") == country)
	if tr.is_empty():
		return pd.DataFrame()

	tr_wide = tr.pivot(values="trend_value", index="month", on="keyword", aggregate_function="mean").sort("month")
	vm = visa_monthly.filter(pl.col("country") == country).select(["month", "issuances"])
	em = encounter_monthly.filter(pl.col("country") == country).select(["month", "encounter_count"])

	panel = (
		tr_wide.join(vm, on="month", how="left")
		.join(em, on="month", how="left")
		.with_columns(
			pl.col("issuances").fill_null(0.0),
		)
		.sort("month")
	)

	pdf = panel.to_pandas()
	pdf = pdf.set_index("month").sort_index()
	for col in pdf.columns:
		pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0.0)
	return pdf


def _safe_corr(series_x: pd.Series, series_y: pd.Series, min_overlap: int = 12) -> tuple[float, float]:
	aligned = pd.concat([series_x, series_y], axis=1).dropna()
	if aligned.shape[0] < min_overlap:
		return np.nan, np.nan
	if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
		return np.nan, np.nan
	corr, pval = pearsonr(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
	return float(corr), float(pval)


def best_lead_corr(trend_series: pd.Series, target_series: pd.Series, max_lead: int, min_overlap: int = 12) -> dict:
	best = {"lead_months": np.nan, "corr": np.nan, "p_value": np.nan}
	best_abs = -1.0
	for lead in range(0, max_lead + 1):
		corr_val, pval = _safe_corr(trend_series, target_series.shift(-lead), min_overlap=min_overlap)
		if pd.notna(corr_val) and abs(corr_val) > best_abs:
			best_abs = abs(corr_val)
			best = {"lead_months": lead, "corr": corr_val, "p_value": pval}
	return best


def build_correlation_summary(country_panels: dict[str, pd.DataFrame], max_lag: int, min_overlap: int) -> pl.DataFrame:
	rows: list[dict] = []
	for country in _progress(sorted(country_panels.keys()), "Correlation by country"):
		panel = country_panels[country].copy()
		if panel.empty:
			continue
		trend_cols = [c for c in panel.columns if c not in ["encounter_count", "issuances"]]
		for keyword in trend_cols:
			trend_series = panel[keyword]
			visa_series = panel["issuances"]
			enc_series = panel["encounter_count"]

			same_visa_corr, same_visa_p = _safe_corr(trend_series, visa_series, min_overlap=min_overlap)
			same_enc_corr, same_enc_p = _safe_corr(trend_series, enc_series, min_overlap=min_overlap)

			visa_best = best_lead_corr(trend_series, visa_series, max_lead=max_lag, min_overlap=min_overlap)
			enc_best = best_lead_corr(trend_series, enc_series, max_lead=max_lag, min_overlap=min_overlap)

			rows.append(
				{
					"country": country,
					"keyword": keyword,
					"same_month_corr_visa": same_visa_corr,
					"same_month_p_visa": same_visa_p,
					"best_lead_months_visa": visa_best["lead_months"],
					"best_lead_corr_visa": visa_best["corr"],
					"best_lead_p_visa": visa_best["p_value"],
					"same_month_corr_encounter": same_enc_corr,
					"same_month_p_encounter": same_enc_p,
					"best_lead_months_encounter": enc_best["lead_months"],
					"best_lead_corr_encounter": enc_best["corr"],
					"best_lead_p_encounter": enc_best["p_value"],
				}
			)

	if not rows:
		return pl.DataFrame()

	out = pl.DataFrame(rows)
	out = out.with_columns(
		pl.col("best_lead_corr_visa").abs().alias("abs_best_lead_corr_visa"),
		pl.col("best_lead_corr_encounter").abs().alias("abs_best_lead_corr_encounter"),
		pl.Series("q_value_visa", benjamini_hochberg(out["best_lead_p_visa"].to_list())),
		pl.Series("q_value_encounter", benjamini_hochberg(out["best_lead_p_encounter"].to_list())),
	)
	return out.sort(["country", "abs_best_lead_corr_encounter"], descending=[False, True])


def build_country_best_keywords(corr_summary: pl.DataFrame) -> pl.DataFrame:
	if corr_summary.is_empty():
		return corr_summary
	ranked = corr_summary.sort(["country", "abs_best_lead_corr_encounter"], descending=[False, True])
	return ranked.group_by("country", maintain_order=True).first().sort(
		["abs_best_lead_corr_encounter", "abs_best_lead_corr_visa"],
		descending=[True, True],
	)


def evaluate_var_predictions(
	panel: pd.DataFrame,
	target_col: str,
	test_periods: int,
	maxlags: int,
) -> dict | None:
	if target_col not in panel.columns:
		return None

	keep_cols = [target_col] + [c for c in panel.columns if c not in ["encounter_count", "issuances"]]
	df = panel[keep_cols].dropna().copy()
	if df.empty:
		return None

	df = df.sort_index()
	if not isinstance(df.index, pd.DatetimeIndex):
		df.index = pd.to_datetime(df.index)
	full_month_index = pd.date_range(df.index.min(), df.index.max(), freq="MS")
	df = df.reindex(full_month_index)
	df = df.interpolate(limit_direction="both").ffill().bfill()

	df = df.loc[:, df.std() > 0]
	if target_col not in df.columns or df.shape[0] < (test_periods + 24) or df.shape[1] < 2:
		return None

	train = df.iloc[:-test_periods].copy()
	test = df.iloc[-test_periods:].copy()

	rmse_base = np.nan
	if AutoReg is not None:
		try:
			model_baseline = AutoReg(train[target_col], lags=maxlags)
			res_base = model_baseline.fit()
			pred_base = res_base.forecast(steps=test_periods).values
			rmse_base = float(np.sqrt(mean_squared_error(test[target_col].values, pred_base)))
		except Exception:
			rmse_base = np.nan

	rmse_raw_var = np.nan
	try:
		model_var = VAR(train)
		res_var = model_var.fit(maxlags=maxlags)
		pred_var = res_var.forecast(train.values[-res_var.k_ar :], steps=test_periods)
		target_idx = list(train.columns).index(target_col)
		rmse_raw_var = float(np.sqrt(mean_squared_error(test[target_col].values, pred_var[:, target_idx])))
	except Exception:
		rmse_raw_var = np.nan

	rmse_std_var = np.nan
	try:
		mu = train.mean()
		sigma = train.std().replace(0, 1)
		train_z = (train - mu) / sigma
		model_std = VAR(train_z)
		res_std = model_std.fit(maxlags=maxlags)
		pred_std = res_std.forecast(train_z.values[-res_std.k_ar :], steps=test_periods)
		target_idx = list(train.columns).index(target_col)
		pred_std_target = pred_std[:, target_idx] * sigma[target_col] + mu[target_col]
		rmse_std_var = float(np.sqrt(mean_squared_error(test[target_col].values, pred_std_target)))
	except Exception:
		rmse_std_var = np.nan

	raw_vs_base = (
		((rmse_base - rmse_raw_var) / rmse_base * 100)
		if pd.notna(rmse_base) and pd.notna(rmse_raw_var) and rmse_base != 0
		else np.nan
	)
	std_vs_base = (
		((rmse_base - rmse_std_var) / rmse_base * 100)
		if pd.notna(rmse_base) and pd.notna(rmse_std_var) and rmse_base != 0
		else np.nan
	)
	std_vs_raw = (
		((rmse_raw_var - rmse_std_var) / rmse_raw_var * 100)
		if pd.notna(rmse_raw_var) and pd.notna(rmse_std_var) and rmse_raw_var != 0
		else np.nan
	)

	return {
		"baseline_rmse": rmse_base,
		"raw_var_rmse": rmse_raw_var,
		"std_var_rmse": rmse_std_var,
		"raw_var_vs_baseline_improvement_pct": raw_vs_base,
		"std_var_vs_baseline_improvement_pct": std_vs_base,
		"std_var_vs_raw_var_improvement_pct": std_vs_raw,
	}


def run_var_benchmark(
	country_panels: dict[str, pd.DataFrame],
	test_periods: int,
	maxlags_var: int,
) -> pl.DataFrame:
	rows: list[dict] = []
	for country in _progress(sorted(country_panels.keys()), "VAR benchmark by country"):
		panel = country_panels[country]
		for target in ["encounter_count", "issuances"]:
			res = evaluate_var_predictions(panel, target_col=target, test_periods=test_periods, maxlags=maxlags_var)
			if res is None:
				continue
			rows.append({"country": country, "target": target, **res})

	if not rows:
		return pl.DataFrame()
	return pl.DataFrame(rows).sort(["country", "target"])


def create_country_plots(
	country: str,
	panel: pd.DataFrame,
	corr_summary_country: pl.DataFrame,
	output_dir: Path,
	max_lag: int,
) -> list[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []

	if panel.empty:
		return paths

	trend_cols = [c for c in panel.columns if c not in ["encounter_count", "issuances"]]
	if not trend_cols:
		return paths

	selected_keywords = trend_cols[:3]
	if not corr_summary_country.is_empty():
		selected_keywords = corr_summary_country.sort("abs_best_lead_corr_encounter", descending=True)["keyword"].head(3).to_list()

	fig, axes = plt.subplots(len(selected_keywords), 1, figsize=(12, max(4, 3.2 * len(selected_keywords))), sharex=True)
	if len(selected_keywords) == 1:
		axes = [axes]
	for i, keyword in enumerate(selected_keywords):
		x = panel[keyword].diff().dropna()
		y = panel["encounter_count"].diff().dropna()
		common = x.index.intersection(y.index)
		x = x.loc[common]
		y = y.loc[common]
		if len(x) < 20:
			axes[i].text(0.5, 0.5, "Insufficient data", ha="center", va="center")
			axes[i].set_title(f"{country}: {keyword}")
			continue
		corrs = []
		for lag in range(0, max_lag + 1):
			shifted = y.shift(-lag)
			c, _ = _safe_corr(x, shifted, min_overlap=12)
			corrs.append(0.0 if np.isnan(c) else c)
		markerline, stemlines, baseline = axes[i].stem(range(max_lag + 1), corrs, basefmt=" ")
		plt.setp(stemlines, linewidth=1.4)
		plt.setp(markerline, markersize=5)
		conf = 1.96 / np.sqrt(max(len(x), 1))
		axes[i].axhline(0, color="black", lw=1)
		axes[i].axhline(conf, color="red", linestyle="--", alpha=0.45)
		axes[i].axhline(-conf, color="red", linestyle="--", alpha=0.45)
		axes[i].set_title(f"CCF (diff): '{keyword}' leading encounters ({country})")
		axes[i].set_ylabel("Correlation")
		axes[i].set_xlabel("Lag (months)")
	plt.tight_layout()
	ccf_path = output_dir / f"{country.lower().replace(' ', '_')}_ccf.png"
	fig.savefig(ccf_path, dpi=220)
	plt.close(fig)
	paths.append(ccf_path)

	work = panel[["encounter_count", "issuances", *selected_keywords]].copy()

	def _minmax_0_100(series: pd.Series) -> pd.Series:
		s = series.astype(float)
		smin, smax = s.min(), s.max()
		if pd.isna(smin) or pd.isna(smax) or smax == smin:
			return pd.Series(np.nan, index=s.index)
		return (s - smin) / (smax - smin) * 100

	keyword_scaled = pd.DataFrame(index=work.index)
	for kw in selected_keywords:
		keyword_scaled[kw] = _minmax_0_100(work[kw])
	work["composite_trend"] = keyword_scaled.mean(axis=1)
	work["enc_scaled"] = _minmax_0_100(work["encounter_count"])
	work["visa_scaled"] = _minmax_0_100(work["issuances"])
	work[["composite_trend", "enc_scaled", "visa_scaled"]] = work[["composite_trend", "enc_scaled", "visa_scaled"]].rolling(3, min_periods=1).mean()

	fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
	axes2[0].plot(work.index, work["composite_trend"], color="tab:green", linewidth=2.6, label="Composite trend")
	axes2[0].plot(work.index, work["enc_scaled"], color="tab:red", linewidth=2.2, label="Encounters")
	axes2[0].set_title(f"{country}: composite trend vs encounters")
	axes2[0].set_ylabel("Scaled 0-100")
	axes2[0].grid(True, alpha=0.3)
	axes2[0].legend(loc="upper left", fontsize=8)

	axes2[1].plot(work.index, work["composite_trend"], color="tab:green", linewidth=2.6, label="Composite trend")
	axes2[1].plot(work.index, work["visa_scaled"], color="tab:blue", linewidth=2.2, label="Visas")
	axes2[1].set_title(f"{country}: composite trend vs visas")
	axes2[1].grid(True, alpha=0.3)
	axes2[1].legend(loc="upper left", fontsize=8)
	for ax in axes2:
		ax.tick_params(axis="x", rotation=90)
	plt.tight_layout()
	comp_path = output_dir / f"{country.lower().replace(' ', '_')}_composite_tracking.png"
	fig2.savefig(comp_path, dpi=220)
	plt.close(fig2)
	paths.append(comp_path)

	return paths


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Google Trends vs migration outcomes analysis")
	parser.add_argument("--trends-dir", type=Path, default=Path("data/raw/trends"))
	parser.add_argument("--encounter-dir", type=Path, default=Path("data/raw/encounter"))
	parser.add_argument("--visa-path", type=Path, default=Path("data/processed/visa_master.parquet"))
	parser.add_argument("--focus-news-dir", type=Path, default=Path("data/processed/news"))
	parser.add_argument("--focus-countries", type=str, default=None)
	parser.add_argument("--max-lag", type=int, default=6)
	parser.add_argument("--min-overlap", type=int, default=12)
	parser.add_argument("--test-periods", type=int, default=6)
	parser.add_argument("--maxlags-var", type=int, default=1)
	parser.add_argument("--visa-type", type=str, default=None)
	parser.add_argument("--plots-dir", type=Path, default=Path("data/plots/trends_vs_migration"))
	parser.add_argument("--output-dir", type=Path, default=Path("data/processed/production_outputs"))
	return parser


def main() -> None:
	args = build_parser().parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)
	args.plots_dir.mkdir(parents=True, exist_ok=True)

	if args.focus_countries:
		focus_countries = sorted({_canonical_country(x.strip()) for x in args.focus_countries.split(",") if x.strip()})
	else:
		focus_countries = load_focus_countries(args.focus_news_dir, include_us=False)

	trends_long = load_trends_long(args.trends_dir, focus_countries)
	visa_monthly = load_visa_monthly(args.visa_path, focus_countries, visa_type=args.visa_type)
	encounter_monthly = load_encounters_monthly(args.encounter_dir, focus_countries)

	available_countries = sorted(
		set(trends_long["country"].unique().to_list())
		.intersection(set(visa_monthly["country"].unique().to_list()))
	)

	country_panels: dict[str, pd.DataFrame] = {}
	diagnostics: list[dict] = []
	for country in _progress(available_countries, "Building country panels"):
		panel = build_country_panel(country, trends_long, visa_monthly, encounter_monthly)
		if panel.empty:
			continue
		country_panels[country] = panel
		diagnostics.append(
			{
				"country": country,
				"n_months": int(panel.shape[0]),
				"start_month": panel.index.min(),
				"end_month": panel.index.max(),
				"n_keywords": int(max(panel.shape[1] - 2, 0)),
			}
		)

	if not country_panels:
		print("No countries with usable trends + visa + encounter overlap.")
		return

	diagnostics_df = pl.DataFrame(diagnostics).sort("country")
	diagnostics_df.write_parquet(args.output_dir / "trends_panel_diagnostics.parquet")
	diagnostics_df.write_csv(args.output_dir / "trends_panel_diagnostics_review.csv")

	corr_summary = build_correlation_summary(country_panels, max_lag=args.max_lag, min_overlap=args.min_overlap)
	corr_summary.write_parquet(args.output_dir / "trends_corr_summary.parquet")
	corr_summary.write_csv(args.output_dir / "trends_corr_summary_review.csv")

	country_best = build_country_best_keywords(corr_summary)
	country_best.write_parquet(args.output_dir / "trends_country_best_keywords.parquet")
	country_best.write_csv(args.output_dir / "trends_country_best_keywords_review.csv")

	var_benchmark = run_var_benchmark(country_panels, test_periods=args.test_periods, maxlags_var=args.maxlags_var)
	var_benchmark.write_parquet(args.output_dir / "trends_var_benchmark.parquet")
	var_benchmark.write_csv(args.output_dir / "trends_var_benchmark_review.csv")

	plot_rows: list[dict] = []
	for country in _progress(sorted(country_panels.keys()), "Generating country plots"):
		country_corr = corr_summary.filter(pl.col("country") == country)
		paths = create_country_plots(country, country_panels[country], country_corr, args.plots_dir, max_lag=args.max_lag)
		for path in paths:
			plot_rows.append({"country": country, "plot_path": str(path)})

	plot_index = pl.DataFrame(plot_rows) if plot_rows else pl.DataFrame({"country": [], "plot_path": []})
	plot_index.write_csv(args.output_dir / "trends_plot_index.csv")

	top_row = country_best.sort("abs_best_lead_corr_encounter", descending=True).head(1) if not country_best.is_empty() else pl.DataFrame()
	report_lines = [
		"# Trends Analysis Report",
		"",
		"## Scope",
		f"- Focus countries requested: {len(focus_countries)}",
		f"- Countries analyzed: {len(country_panels)}",
		f"- Max lead lag (months): {args.max_lag}",
		f"- Min overlap months: {args.min_overlap}",
		"",
	]
	if not top_row.is_empty():
		row = top_row.row(0, named=True)
		lead_value = row.get("best_lead_months_encounter")
		lead_label = "NA" if lead_value is None or (isinstance(lead_value, float) and np.isnan(lead_value)) else str(int(lead_value))
		corr_value = row.get("best_lead_corr_encounter")
		corr_label = "NA" if corr_value is None or (isinstance(corr_value, float) and np.isnan(corr_value)) else f"{float(corr_value):.4f}"
		q_value = row.get("q_value_encounter")
		q_label = "NA" if q_value is None or (isinstance(q_value, float) and np.isnan(q_value)) else f"{float(q_value):.4g}"
		report_lines.extend(
			[
				"## Strongest Encounter-Leading Keyword",
				f"- Country: {row['country']}",
				f"- Keyword: {row['keyword']}",
				f"- Best lead (months): {lead_label}",
				f"- Correlation: {corr_label}",
				f"- q-value: {q_label}",
				"",
			]
		)

	report_lines.extend(
		[
			"## Output Files",
			f"- {args.output_dir / 'trends_panel_diagnostics.parquet'}",
			f"- {args.output_dir / 'trends_corr_summary.parquet'}",
			f"- {args.output_dir / 'trends_country_best_keywords.parquet'}",
			f"- {args.output_dir / 'trends_var_benchmark.parquet'}",
			f"- {args.output_dir / 'trends_plot_index.csv'}",
		]
	)
	(args.output_dir / "trends_analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

	print(f"Saved diagnostics (parquet): {args.output_dir / 'trends_panel_diagnostics.parquet'}")
	print(f"Saved correlation summary (parquet): {args.output_dir / 'trends_corr_summary.parquet'}")
	print(f"Saved country best keywords (parquet): {args.output_dir / 'trends_country_best_keywords.parquet'}")
	print(f"Saved VAR benchmark (parquet): {args.output_dir / 'trends_var_benchmark.parquet'}")
	print(f"Saved plot index (review csv): {args.output_dir / 'trends_plot_index.csv'}")
	print(f"Saved report: {args.output_dir / 'trends_analysis_report.md'}")


if __name__ == "__main__":
	main()
