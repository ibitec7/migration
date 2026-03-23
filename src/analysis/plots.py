import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logger, save_figure, add_title_and_save, annotate_max_point
import logging
import os

os.makedirs('./data/plots', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

global logger
logger = setup_logger('./logs/plots.log', log_level=logging.DEBUG)


# ==============================================================================
# PROFESSIONAL STYLING & CONFIGURATION
# ==============================================================================

def setup_styling():
    """Setup professional styling for all plots"""
    # Define professional color palette
    PALETTE = {
        'visa_primary': '#1f77b4',      # Professional blue
        'visa_light': '#aec7e8',        # Light blue
        'encounter_primary': '#d62728',  # Professional red
        'encounter_light': '#ff7f0e',    # Orange
        'seasonal_green': '#2ca02c',    # Green
        'neutral_gray': '#7f7f7f',      # Gray
        'accent_purple': '#9467bd'      # Purple for highlights
    }
    
    sns.set_palette("husl")
    
    # Set global plot parameters
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (14, 7),
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'legend.framealpha': 0.95,
        'legend.fontsize': 10,
        'lines.linewidth': 2.5,
        'lines.markersize': 6
    })
    
    return PALETTE


# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================

def load_data(data_processed, data_raw):
    """Load visa and encounter data"""
    # Load visa data using lazy frames
    visa_parquet = data_processed / 'visa_master.parquet'
    if visa_parquet.exists():
        visa_df = pl.read_parquet(visa_parquet)
        visa_df_pd = visa_df.to_pandas()
        logger.info(f"Loaded visa_master.parquet: {visa_df.shape[0]} rows × {visa_df.shape[1]} columns")
        logger.info(f"Date range: {visa_df['date'].min()} to {visa_df['date'].max()}")
    else:
        logger.error(f"File not found: {visa_parquet}")
        visa_df = None
        visa_df_pd = None
    
    # Load and consolidate encounter data
    encounter_dir = data_raw / 'encounter'
    encounter_files = list(encounter_dir.glob('*.csv'))
    
    logger.info(f"Found {len(encounter_files)} encounter files")
    
    encounter_dfs = []
    for file in sorted(encounter_files):
        df = pd.read_csv(file)
        encounter_dfs.append(df)
    
    if encounter_dfs:
        encounter_df = pd.concat(encounter_dfs, ignore_index=True).drop_duplicates()
        logger.info(f"Combined encounter data: {len(encounter_df)} rows × {len(encounter_df.columns)} columns")
    else:
        logger.error("No encounter files found")
        encounter_df = None
    
    return visa_df, visa_df_pd, encounter_df


def prepare_encounter_data(encounter_df):
    """Prepare encounter data for analysis"""
    if encounter_df is None:
        return None
    
    # Month mapping
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    encounter_df['Fiscal Year'] = pd.to_numeric(encounter_df['Fiscal Year'], errors='coerce')
    encounter_df['month_num'] = encounter_df['Month (abbv)'].map(month_map)
    encounter_df['calendar_year'] = encounter_df.apply(
        lambda row: int(row['Fiscal Year']) if pd.notna(row['Fiscal Year']) and row['month_num'] < 10 
        else (int(row['Fiscal Year']) - 1 if pd.notna(row['Fiscal Year']) else None),
        axis=1
    )
    
    encounter_df = encounter_df.dropna(subset=['calendar_year'])
    encounter_df['calendar_year'] = encounter_df['calendar_year'].astype(int)
    encounter_df['date'] = pd.to_datetime(
        encounter_df['calendar_year'].astype(str) + '-' + 
        encounter_df['month_num'].astype(str) + '-01'
    )
    
    # Monthly aggregation using lazy frames
    encounter_pl = pl.from_pandas(encounter_df[['date', 'Encounter Count']])
    encounter_monthly = (
        encounter_pl.lazy()
        .group_by('date')
        .agg(pl.col('Encounter Count').sum())
        .sort('date')
        .collect()
        .to_pandas()
    )
    
    logger.info(f"Encounter data prepared: {len(encounter_monthly)} months")
    logger.info(f"Date range: {encounter_monthly['date'].min()} to {encounter_monthly['date'].max()}")
    
    return encounter_df, encounter_monthly


def merge_datasets(visa_df, visa_df_pd, encounter_monthly):
    """Merge visa and encounter datasets"""
    if visa_df is None or encounter_monthly is None:
        return None
    
    # Use lazy frame for visa aggregation
    visa_monthly = (
        visa_df.lazy()
        .group_by('date')
        .agg(pl.col('issuances').sum().alias('visa_issuances'))
        .sort('date')
        .collect()
        .to_pandas()
    )
    
    merged_df = visa_monthly.merge(encounter_monthly, on='date', how='outer').sort_values('date')
    merged_df['visa_issuances'] = merged_df['visa_issuances'].fillna(0)
    merged_df['Encounter Count'] = merged_df['Encounter Count'].fillna(0)
    
    logger.info(f"Merged dataset: {len(merged_df)} months")
    logger.info(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    
    return merged_df


# ==============================================================================
# PHASE 2: ENHANCED CORE VISUALIZATIONS
# ==============================================================================

def create_dual_axis_plot(merged_df, palette):
    """Create enhanced dual-axis plot"""
    if merged_df is None or merged_df.empty:
        logger.warning("Cannot create dual-axis plot: no data")
        return
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    ax1.plot(merged_df['date'], merged_df['visa_issuances'], 
             color=palette['visa_primary'], linewidth=2.5, label='Visa Issuances', alpha=0.85)
    ax1.fill_between(merged_df['date'], merged_df['visa_issuances'], alpha=0.15, color=palette['visa_primary'])
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Monthly Visa Issuances', color=palette['visa_primary'], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=palette['visa_primary'])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    ax2.plot(merged_df['date'], merged_df['Encounter Count'], 
             color=palette['encounter_primary'], linewidth=2.5, label='Border Encounters', alpha=0.85)
    ax2.fill_between(merged_df['date'], merged_df['Encounter Count'], alpha=0.15, color=palette['encounter_primary'])
    ax2.set_ylabel('Monthly Encounters', color=palette['encounter_primary'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=palette['encounter_primary'])
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)
    
    add_title_and_save(fig, 
                       'Visa Issuances vs. Border Encounters Over Time',
                       'Monthly data with trend visualization',
                       '01_visa_encounters_dual_axis', logger=logger)


def create_visa_by_type_plots(visa_df, palette):
    """Create visa trends by type visualizations"""
    if visa_df is None:
        logger.warning("Cannot create visa by type plots: no data")
        return
    
    # Use lazy frame for aggregation
    visa_by_type = (
        visa_df.lazy()
        .group_by(['date', 'visa_type'])
        .agg(pl.col('issuances').sum())
        .sort('date')
        .collect()
        .to_pandas()
    )
    
    # Plot 1: Visa trends by type
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = sns.color_palette('husl', len(visa_by_type['visa_type'].unique()))
    for i, vtype in enumerate(sorted(visa_by_type['visa_type'].unique())):
        subset = visa_by_type[visa_by_type['visa_type'] == vtype]
        ax.plot(subset['date'], subset['issuances'], 
                marker='o', markersize=4, linewidth=2.5, label=vtype, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Monthly Issuances', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.95, ncol=2)
    
    add_title_and_save(fig, 
                       'Visa Issuances Over Time by Type',
                       'Color-coded by visa category',
                       '02_visa_by_type', logger=logger)
    
    # Plot 2: Month-over-month changes
    visa_changes = []
    for vtype in visa_by_type['visa_type'].unique():
        subset = visa_by_type[visa_by_type['visa_type'] == vtype].copy()
        subset = subset.sort_values('date').reset_index(drop=True)
        subset['change'] = subset['issuances'].diff()
        visa_changes.append(subset)
    
    visa_by_type_change = pd.concat(visa_changes, ignore_index=True)
    visa_by_type_change = visa_by_type_change.dropna(subset=['change'])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = sns.color_palette('husl', len(visa_by_type_change['visa_type'].unique()))
    for i, vtype in enumerate(sorted(visa_by_type_change['visa_type'].unique())):
        subset = visa_by_type_change[visa_by_type_change['visa_type'] == vtype]
        ax.plot(subset['date'], subset['change'], 
                marker='o', markersize=4, linewidth=2.5, label=vtype, color=colors[i], alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Monthly Change in Issuances (vs. Previous Month)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.95, ncol=2)
    
    add_title_and_save(fig, 
                       'Month-over-Month Changes in Visa Issuances by Type',
                       'Positive = Growth from previous month, Negative = Decline',
                       '03_visa_by_type_changes', logger=logger)


# ==============================================================================
# PHASE 3: REGIONAL & COUNTRY-LEVEL ANALYSIS
# ==============================================================================

def prepare_regional_data(visa_df, encounter_df):
    """Prepare regional aggregation data"""
    region_map = {
        'Mexico': 'North America', 'Guatemala': 'Central America', 'Honduras': 'Central America',
        'El Salvador': 'Central America', 'Colombia': 'South America', 'Venezuela': 'South America',
        'Ecuador': 'South America', 'Peru': 'South America', 'Brazil': 'South America',
        'Philippines': 'Asia', 'Vietnam': 'Asia', 'China': 'Asia', 'India': 'Asia',
        'Pakistan': 'Asia', 'Nigeria': 'Africa', 'Haiti': 'Caribbean', 'Dominican Republic': 'Caribbean'
    }
    
    # Use lazy frame for visa by country
    visa_by_country = (
        visa_df.lazy()
        .group_by('country')
        .agg(pl.col('issuances').sum().alias('total_visas'))
        .collect()
        .to_pandas()
    )
    visa_by_country['region'] = visa_by_country['country'].map(region_map).fillna('Other')
    
    regional_visa = visa_by_country.groupby('region')['total_visas'].sum().sort_values(ascending=False)
    
    # Regional encounter data using lazy frame
    enc_df = pl.from_pandas(encounter_df[['Citizenship Grouping', 'Encounter Count']])
    enc_by_country = (
        enc_df.lazy()
        .group_by('Citizenship Grouping')
        .agg(pl.col('Encounter Count').sum())
        .collect()
        .to_pandas()
    )
    enc_by_country.columns = ['country', 'total_encounters']
    enc_by_country['region'] = enc_by_country['country'].map(region_map).fillna('Other')
    
    regional_enc = enc_by_country.groupby('region')['total_encounters'].sum().sort_values(ascending=False)
    
    logger.info("Regional data prepared")
    
    return visa_by_country, regional_visa, enc_by_country, regional_enc


def create_regional_comparison(regional_visa, regional_enc, palette):
    """Create regional comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    regions = regional_visa.index
    colors_left = sns.color_palette('Blues_r', len(regions))
    ax1.barh(regions, regional_visa.values, color=colors_left, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Total Visa Issuances', fontsize=11)
    ax1.set_title('Visa Issuances by Region', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(regional_visa.values):
        ax1.text(v, i, f' {v:,.0f}', va='center', fontsize=10)
    
    regions_enc = regional_enc.reindex(regions).fillna(0)
    colors_right = sns.color_palette('Reds_r', len(regions))
    ax2.barh(regions, regional_enc.reindex(regions).fillna(0).values, color=colors_right, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Total Encounters', fontsize=11)
    ax2.set_title('Border Encounters by Region', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(regional_enc.reindex(regions).fillna(0).values):
        ax2.text(v, i, f' {v:,.0f}', va='center', fontsize=10)
    
    fig.suptitle('Regional Comparison: Legal Immigration vs. Border Encounters', fontsize=14)
    save_figure(fig, '04_regional_comparison', logger=logger)


def create_top_10_countries_plots(visa_df, visa_by_country, encounter_df, palette):
    """Create top 10 countries analysis"""
    top_countries = visa_by_country.nlargest(10, 'total_visas')['country'].tolist()
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, country in enumerate(top_countries):
        # Use lazy frame for country visa data
        country_visa = (
            visa_df.filter(pl.col('country') == country)
            .lazy()
            .group_by('date')
            .agg(pl.col('issuances').sum())
            .sort('date')
            .collect()
            .to_pandas()
        )
        
        country_enc = encounter_df[encounter_df['Citizenship Grouping'] == country].copy()
        if len(country_enc) > 0:
            country_enc_monthly = country_enc.groupby('date')['Encounter Count'].sum()
        else:
            country_enc_monthly = pd.Series()
        
        ax = axes[idx]
        
        if len(country_visa) > 0:
            ax.plot(country_visa['date'], country_visa['issuances'], 
                    color=palette['visa_primary'], linewidth=2, label='Visa', alpha=0.8)
        
        if len(country_enc_monthly) > 0:
            ax2 = ax.twinx()
            ax2.plot(country_enc_monthly.index, country_enc_monthly.values, 
                    color=palette['encounter_primary'], linewidth=2, label='Encounters', alpha=0.8)
            ax2.set_ylabel('')
            ax2.tick_params(axis='y', labelcolor=palette['encounter_primary'], labelsize=8)
        
        ax.set_title(country, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=7)
        if idx >= 5:
            ax.set_xlabel('Year', fontsize=9)
    
    fig.suptitle('Top 10 Countries: Visa Issuances (Blue) vs. Encounters (Red)', fontsize=14)
    save_figure(fig, '05_top_10_countries', logger=logger)
    
    return top_countries


# ==============================================================================
# PHASE 4: SEASONAL PATTERN ANALYSIS
# ==============================================================================

def create_seasonal_heatmap(visa_df_pd, palette):
    """Create seasonal patterns heatmap"""
    visa_df_pd['month'] = visa_df_pd['date'].dt.month
    visa_df_pd['year'] = visa_df_pd['date'].dt.year
    
    visa_seasonal = visa_df_pd.pivot_table(
        values='issuances', index='month', columns='year', aggfunc='sum'
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(visa_seasonal, annot=False, cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Total Visa Issuances'}, linewidths=0.5)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Month', fontsize=11)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels(month_labels, rotation=0)
    
    add_title_and_save(fig, 
                       'Seasonal Patterns: Monthly Visa Issuances Heatmap',
                       'Darker colors indicate higher visa issuance volume',
                       '06_seasonal_visa_heatmap_by_year', logger=logger)


def create_seasonal_by_country(visa_df_pd, top_countries, palette):
    """Create seasonal patterns for top countries"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, country in enumerate(top_countries[:6]):
        country_data = visa_df_pd[visa_df_pd['country'] == country].copy()
        country_data['month'] = country_data['date'].dt.month
        
        seasonal = country_data.groupby('month')['issuances'].agg(['mean', 'std'])
        
        ax = axes[idx]
        ax.bar(seasonal.index, seasonal['mean'], color=palette['visa_primary'], 
               alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(seasonal.index, seasonal['mean'], yerr=seasonal['std'], 
                    fmt='none', color='black', capsize=3, alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Month', fontsize=10)
        ax.set_ylabel('Avg Visa Issuances', fontsize=10)
        ax.set_title(country, fontsize=11)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Seasonal Patterns: Top 6 Countries - Average Visa Issuances by Month', 
                 fontsize=14)
    save_figure(fig, '07_seasonal_by_country', logger=logger)


# ==============================================================================
# PHASE 5: DISTRIBUTION & VOLATILITY ANALYSIS
# ==============================================================================

def prepare_distribution_data(visa_df_pd, encounter_df, top_countries):
    """Prepare distribution data for analysis"""
    distribution_data = []
    
    for country in top_countries:
        country_visa_vals = visa_df_pd[visa_df_pd['country'] == country]['issuances'].values
        country_enc_vals = encounter_df[encounter_df['Citizenship Grouping'] == country]['Encounter Count'].values
        
        if len(country_visa_vals) > 0:
            cv = np.std(country_visa_vals) / (np.mean(country_visa_vals) + 1e-6)
            distribution_data.append({
                'country': country,
                'visa_values': country_visa_vals,
                'visa_cv': cv,
                'visa_mean': np.mean(country_visa_vals),
                'encounter_values': country_enc_vals if len(country_enc_vals) > 0 else np.array([0])
            })
    
    return distribution_data


def create_distribution_boxplot(distribution_data, palette):
    """Create box plot of distributions"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    visa_distributions = [d['visa_values'] for d in distribution_data]
    country_names = [d['country'] for d in distribution_data]
    
    bp = ax.boxplot(visa_distributions, labels=country_names, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor(palette['visa_primary'])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Monthly Visa Issuances', fontsize=12)
    ax.set_title('Distribution of Visa Issuances by Country', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    save_figure(fig, '08_visa_distribution_boxplot', logger=logger)


def create_volatility_chart(distribution_data, palette):
    """Create volatility analysis chart"""
    volatility_data = pd.DataFrame(distribution_data)[['country', 'visa_cv', 'visa_mean']]
    volatility_data = volatility_data.sort_values('visa_cv', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors_vol = sns.color_palette('RdYlGn_r', len(volatility_data))
    bars = ax.barh(volatility_data['country'], volatility_data['visa_cv'], color=colors_vol, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Coefficient of Variation (Volatility)', fontsize=11)
    ax.set_title('Visa Flow Volatility by Country', fontsize=13)
    ax.text(0.02, 0.98, 'Lower = More Stable Flows | Higher = More Volatile', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(volatility_data['visa_cv']):
        ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)
    
    save_figure(fig, '10_visa_volatility', logger=logger)
    
    return volatility_data


# ==============================================================================
# PHASE 6: CORRELATION & LAG ANALYSIS
# ==============================================================================

def calculate_correlation_by_country(visa_df_pd, encounter_df, top_countries):
    """Calculate correlation between visa and encounters by country"""
    correlation_data = []
    
    logger.info("Checking data overlap for each country")
    
    for country in top_countries:
        country_visa_ts = visa_df_pd[visa_df_pd['country'] == country].groupby('date')['issuances'].sum().sort_index()
        country_enc_ts = encounter_df[encounter_df['Citizenship Grouping'] == country].groupby('date')['Encounter Count'].sum().sort_index()
        
        common_dates = country_visa_ts.index.intersection(country_enc_ts.index)
        
        if len(common_dates) > 0:
            visa_vals = country_visa_ts.loc[common_dates].values
            enc_vals = country_enc_ts.loc[common_dates].values
            
            if len(visa_vals) != len(enc_vals) or np.any(np.isnan(visa_vals)) or np.any(np.isnan(enc_vals)):
                continue
            
            visa_norm = (visa_vals - np.mean(visa_vals)) / (np.std(visa_vals) + 1e-6)
            enc_norm = (enc_vals - np.mean(enc_vals)) / (np.std(enc_vals) + 1e-6)
            
            corr, pval = pearsonr(visa_norm, enc_norm)
            correlation_data.append({
                'country': country,
                'correlation': corr,
                'p_value': pval,
                'n_common': len(common_dates)
            })
    
    corr_df = pd.DataFrame(correlation_data).sort_values('correlation', ascending=False)
    
    logger.info(f"Correlation analysis: {len(top_countries)} countries analyzed, {len(corr_df)} with sufficient data")
    
    return corr_df


def calculate_correlation_by_region(visa_df_pd, encounter_df):
    """Calculate correlation between visa and encounters by region"""
    visa_to_encounter_map = {
        'Mexico': 'Mexico',
        'Guatemala': 'Guatemala',
        'Honduras': 'Honduras',
        'El Salvador': 'El Salvador',
        'Colombia': 'Other',
        'Venezuela': 'Other',
        'Ecuador': 'Other',
        'Peru': 'Other',
        'Brazil': 'Other',
        'Dominican Republic': 'Other',
        'Philippines': 'Other',
        'Vietnam': 'Other',
        'India': 'Other',
        'China - mainland born': 'Other',
        'Afghanistan': 'Other',
        'Cuba': 'Other',
        'Bangladesh': 'Other',
        'Pakistan': 'Other'
    }
    
    correlation_data = []
    
    logger.info("Checking data overlap - REGION-BASED CORRELATION")
    
    encounter_regions = ['Mexico', 'Guatemala', 'Honduras', 'El Salvador', 'Other']
    
    for region in encounter_regions:
        visa_countries_in_region = [c for c, r in visa_to_encounter_map.items() if r == region]
        
        region_visa_ts = visa_df_pd[visa_df_pd['country'].isin(visa_countries_in_region)].groupby('date')['issuances'].sum().sort_index()
        region_enc_ts = encounter_df[encounter_df['Citizenship Grouping'] == region].groupby('date')['Encounter Count'].sum().sort_index()
        
        common_dates = region_visa_ts.index.intersection(region_enc_ts.index)
        
        if len(common_dates) >= 12:
            visa_vals = region_visa_ts.loc[common_dates].values
            enc_vals = region_enc_ts.loc[common_dates].values
            
            if len(visa_vals) != len(enc_vals) or np.any(np.isnan(visa_vals)) or np.any(np.isnan(enc_vals)):
                continue
            
            visa_norm = (visa_vals - np.mean(visa_vals)) / (np.std(visa_vals) + 1e-6)
            enc_norm = (enc_vals - np.mean(enc_vals)) / (np.std(enc_vals) + 1e-6)
            
            corr, pval = pearsonr(visa_norm, enc_norm)
            correlation_data.append({
                'region': region,
                'correlation': corr,
                'p_value': pval,
                'n_common': len(common_dates),
                'visa_countries': ', '.join(visa_countries_in_region)
            })
    
    corr_df = pd.DataFrame(correlation_data).sort_values('correlation', ascending=False)
    
    logger.info(f"Region-based correlation: {len(encounter_regions)} regions analyzed, {len(corr_df)} with sufficient data")
    
    return corr_df


def create_correlation_chart(corr_df, palette, is_region=True):
    """Create correlation visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(corr_df) > 0:
        colors_corr = [palette['seasonal_green'] if x > 0 else palette['encounter_primary'] for x in corr_df['correlation']]
        corr_df_sorted = corr_df.sort_values('correlation')
        
        col_name = 'region' if is_region else 'country'
        bars = ax.barh(corr_df_sorted[col_name], corr_df_sorted['correlation'], color=colors_corr, alpha=0.75, edgecolor='black', linewidth=0.5)
        
        for i, (col_val, corr, pval) in enumerate(zip(corr_df_sorted[col_name], corr_df_sorted['correlation'], corr_df_sorted['p_value'])):
            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
            offset = 0.02 if corr > 0 else -0.02
            ax.text(corr + offset, i, f' {corr:.2f}{sig}', va='center', fontsize=9)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
        ax.set_title('Correlation: Visa Issuances vs. Border Encounters by Region', fontsize=13)
        ax.text(0.02, 0.98, '*** p<0.001, ** p<0.01, * p<0.05', transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(-1, 1)
    else:
        ax.text(0.5, 0.5, 'No sufficient data for correlation analysis', ha='center', va='center')
    
    save_figure(fig, '09_correlation_visa_encounters', logger=logger)


# ==============================================================================
# PHASE 7: YEAR-OVER-YEAR GROWTH TRENDS
# ==============================================================================

def calculate_growth_rates(visa_df_pd, top_countries):
    """Calculate year-over-year growth rates"""
    growth_data = []
    
    for country in top_countries:
        country_visa_annual = visa_df_pd[visa_df_pd['country'] == country].groupby('year')['issuances'].sum()
        
        if len(country_visa_annual) > 1:
            pct_changes = country_visa_annual.pct_change() * 100
            avg_growth = pct_changes.mean()
        else:
            avg_growth = 0
        
        growth_data.append({
            'country': country,
            'avg_annual_growth': avg_growth,
            'total_years': len(country_visa_annual)
        })
    
    growth_df = pd.DataFrame(growth_data).sort_values('avg_annual_growth', ascending=False)
    
    logger.info("Annual growth rates calculated")
    
    return growth_df


def create_growth_trends_chart(growth_df, palette):
    """Create year-over-year growth chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_growth = [palette['seasonal_green'] if x > 0 else palette['encounter_primary'] for x in growth_df['avg_annual_growth']]
    growth_sorted = growth_df.sort_values('avg_annual_growth')
    
    bars = ax.barh(growth_sorted['country'], growth_sorted['avg_annual_growth'], color=colors_growth, alpha=0.75, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Average Annual Growth Rate (%)', fontsize=11)
    ax.set_title('Annual Visa Growth Trends by Country', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(growth_sorted['avg_annual_growth']):
        offset = 1 if v > 0 else -1
        ax.text(v + offset, i, f' {v:.1f}%', va='center', fontsize=9)
    
    save_figure(fig, '11_annual_growth_trends', logger=logger)


# ==============================================================================
# PHASE 8: COUNTRY CLUSTERING & PATTERNS
# ==============================================================================

def prepare_clustering_features(visa_df_pd, encounter_df, top_countries):
    """Prepare country feature matrix for clustering"""
    clustering_features = []
    
    for country in top_countries:
        country_visa = visa_df_pd[visa_df_pd['country'] == country]['issuances']
        country_enc = encounter_df[encounter_df['Citizenship Grouping'] == country]['Encounter Count']
        
        features = {
            'country': country,
            'total_visa': country_visa.sum(),
            'avg_visa': country_visa.mean(),
            'std_visa': country_visa.std(),
            'total_enc': country_enc.sum() if len(country_enc) > 0 else 0,
            'avg_enc': country_enc.mean() if len(country_enc) > 0 else 0,
            'cv_visa': (country_visa.std() / (country_visa.mean() + 1e-6))
        }
        clustering_features.append(features)
    
    cluster_df = pd.DataFrame(clustering_features)
    
    scaler = StandardScaler()
    X = cluster_df[['total_visa', 'avg_visa', 'total_enc', 'cv_visa']].fillna(0)
    X_scaled = scaler.fit_transform(X)
    
    logger.info("Clustering features prepared")
    
    return cluster_df, X_scaled


def create_clustering_scatter(cluster_df, palette):
    """Create country positioning scatter plot"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scatter = ax.scatter(cluster_df['total_visa'], cluster_df['total_enc'], 
                        s=cluster_df['cv_visa']*500 + 100,
                        c=cluster_df['avg_visa'], cmap='viridis', 
                        alpha=0.6, edgecolor='black', linewidth=1)
    
    for idx, row in cluster_df.iterrows():
        ax.annotate(row['country'], 
                   xy=(row['total_visa'], row['total_enc']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)
    
    ax.set_xlabel('Total Visa Issuances', fontsize=12)
    ax.set_ylabel('Total Border Encounters', fontsize=12)
    ax.set_title('Country Positioning: Legal vs. Illegal Migration Patterns', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Avg Monthly Visa Issuances', fontsize=10)
    
    ax.text(0.02, 0.98, 'Bubble size = Flow volatility (larger = more volatile)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_figure(fig, '12_country_clustering_scatter', logger=logger)


# ==============================================================================
# PHASE 10: KEY INSIGHTS & FINDINGS
# ==============================================================================

def log_key_findings(regional_visa, regional_enc, volatility_data, growth_df, corr_df):
    """Log key findings from analysis"""
    logger.info("="*90)
    logger.info("KEY FINDINGS FROM ENHANCED ANALYSIS")
    logger.info("="*90)
    
    logger.info(f"Highest visa issuances: {regional_visa.index[0]} ({regional_visa.iloc[0]:,.0f} total)")
    logger.info(f"Highest border encounters: {regional_enc.index[0]} ({regional_enc.iloc[0]:,.0f} total)")
    logger.info(f"Most stable visa flows: {volatility_data['country'].iloc[-1]} (CV: {volatility_data['visa_cv'].iloc[-1]:.2f})")
    logger.info(f"Most volatile visa flows: {volatility_data['country'].iloc[0]} (CV: {volatility_data['visa_cv'].iloc[0]:.2f})")
    
    logger.info(f"Fastest growing country: {growth_df['country'].iloc[0]} ({growth_df['avg_annual_growth'].iloc[0]:.1f}% annual growth)")
    logger.info(f"Declining country: {growth_df['country'].iloc[-1]} ({growth_df['avg_annual_growth'].iloc[-1]:.1f}% annual growth)")
    
    positive_corr = corr_df[corr_df['correlation'] > 0]
    negative_corr = corr_df[corr_df['correlation'] < 0]
    logger.info(f"Strong positive correlation regions: {len(positive_corr)}")
    logger.info(f"Negative correlation regions: {len(negative_corr)}")
    
    logger.info("="*90)
    logger.info("CHARTS GENERATED: 12 visualizations saved to data/plots/")
    logger.info("="*90)


# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def main():
    """Main execution function"""
    logger.info("Starting enhanced migration analysis...")
    
    # Setup styling
    palette = setup_styling()
    logger.info("Professional styling configured")
    
    # Define paths
    global PLOTS_DIR
    PLOTS_DIR = Path('./data/plots')
    PLOTS_DIR.mkdir(exist_ok=True)
    
    DATA_PROCESSED = Path('./data/processed')
    DATA_RAW = Path('./data/raw')
    
    # Phase 1: Load and prepare data
    logger.info("Loading data...")
    visa_df, visa_df_pd, encounter_df = load_data(DATA_PROCESSED, DATA_RAW)
    
    if visa_df is None or encounter_df is None:
        logger.error("Failed to load required data")
        return
    
    encounter_df, encounter_monthly = prepare_encounter_data(encounter_df)
    merged_df = merge_datasets(visa_df, visa_df_pd, encounter_monthly)
    
    # Phase 2: Core visualizations
    logger.info("Creating core visualizations...")
    create_dual_axis_plot(merged_df, palette)
    create_visa_by_type_plots(visa_df, palette)
    
    # Phase 3: Regional analysis
    logger.info("Creating regional analysis...")
    visa_by_country, regional_visa, enc_by_country, regional_enc = prepare_regional_data(visa_df, encounter_df)
    create_regional_comparison(regional_visa, regional_enc, palette)
    top_countries = create_top_10_countries_plots(visa_df, visa_by_country, encounter_df, palette)
    
    # Phase 4: Seasonal patterns
    logger.info("Creating seasonal patterns...")
    visa_df_pd['year'] = visa_df_pd['date'].dt.year
    create_seasonal_heatmap(visa_df_pd, palette)
    create_seasonal_by_country(visa_df_pd, top_countries, palette)
    
    # Phase 5: Distribution and volatility
    logger.info("Creating distribution analysis...")
    distribution_data = prepare_distribution_data(visa_df_pd, encounter_df, top_countries)
    create_distribution_boxplot(distribution_data, palette)
    volatility_data = create_volatility_chart(distribution_data, palette)
    
    # Phase 6: Correlation analysis
    logger.info("Creating correlation analysis...")
    corr_df = calculate_correlation_by_region(visa_df_pd, encounter_df)
    create_correlation_chart(corr_df, palette, is_region=True)
    
    # Phase 7: Growth trends
    logger.info("Creating growth trends...")
    growth_df = calculate_growth_rates(visa_df_pd, top_countries)
    create_growth_trends_chart(growth_df, palette)
    
    # Phase 8: Clustering
    logger.info("Creating clustering analysis...")
    cluster_df, X_scaled = prepare_clustering_features(visa_df_pd, encounter_df, top_countries)
    create_clustering_scatter(cluster_df, palette)
    
    # Phase 10: Key findings
    logger.info("Generating key findings...")
    log_key_findings(regional_visa, regional_enc, volatility_data, growth_df, corr_df)
    
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()

