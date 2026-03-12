# Migration Prediction Analysis

A data-driven project that analyzes historical migration data and predicts future migration surges using machine learning techniques.



## Overview

This project investigates human migration patterns by studying push and pull factors that influence people's decisions to move between countries. By leveraging historical migration data and real-time interest indicators, the goal is to build predictive models that can forecast future migration surges — valuable for policy makers, city planners, and public health officials to anticipate a surge in immigrant in advance to prepare.



## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ibitec7/migration.git
cd migration
```

### 2. Install dependencies

> **Recommended: Dev Containers**

[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension on VS Code lets you use a Docker container as a full-featured development environment. It helps with compatibility between different OS and environments. Install the Dev Containers extension for VS Code.

Open the project in VS Code, then open the Command Palette with `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac) and run the **Dev Containers: Reopen in Container** command. VS Code will build the container and reload the project inside it.

> **Running Locally: uv**

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. If you don't have it installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Create a virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

> **Alternative: pip**

A `requirements.txt` file will is available, you can use pip to install dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
# or install in editable mode
pip install -e .
```

### 3. Get started with development

1. Install the devcontainers extension on VS Code 


## Project Structure

```
migration/
├── data/
│   ├── raw/          # Raw, unmodified datasets as downloaded from sources
│   └── processed/    # Cleaned, transformed, and feature-engineered data
├── notebooks/        # Jupyter notebooks for exploratory data analysis (EDA)
├── src/
│   ├── collection/   # Scripts for scraping and fetching data from external sources
│   ├── processing/   # Data cleaning, transformation, and feature engineering pipelines
│   └── models/       # ML model definitions, training scripts, and evaluation utilities
├── pyproject.toml    # Project metadata and dependency definitions (managed by uv)
├── requirements.txt  # Pip-compatible dependency list (coming soon)
└── README.md         # Project documentation
```

| Path | Description |
| -------------------------------- | --------------------------------------------------------------------------- |
| `data/raw/` | Original datasets downloaded from government, financial, and API sources. Never modified directly. |
| `data/processed/` | Cleaned and enriched datasets ready for model training and analysis. |
| `notebooks/` | Interactive Jupyter notebooks used for exploration, visualisation, and prototyping. |
| `src/collection/` | Data collection scripts — web scrapers, API clients (Google Trends, IMF, CBP, Travel.State.Gov). |
| `src/processing/` | Feature engineering and preprocessing pipelines that transform raw data into model-ready inputs. |
| `src/models/` | Model training, hyperparameter tuning, evaluation, and serialisation logic. |
| `pyproject.toml` | Defines project metadata, Python version requirements, and all package dependencies. |
| `requirements.txt` | Flat dependency list for pip users (to be generated from `pyproject.toml`). |
| `README.md` | This file — project overview, setup instructions, and documentation. |



## Objectives

- Analyze historical migration data from government and financial sources
- Identify key **push factors** (inflation, economic instability, conflict) and **pull factors** (job opportunities, currency strength, economic stability)
- Track real-time migration interest using Google Trends
- Build predictive models to forecast future migration flows



## Methodology

### Independent Variables
- **Push Factors**: Inflation, exchange rates, economic uncertainty, conflict, news sentiment
- **Pull Factors**: Currency value, employment opportunities, economic stability
- **Confounding Factors**: Remittances, passport issuances, existing diaspora communities

### Dependent Variables
- Number of legal immigrants
- Number of illegal border crossings
- Number of visas granted

### Models Used
| Model               | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
|   BERT              |  Analyzing the sentiment of news                      |
|   Model2            |  Predicting migration surges                          |



## Data Sources

| Source                                                  | Type       | Usage                                   |
| ------------------------------------------------------- | ---------- | --------------------------------------- |
| [Travel.State.Gov](https://travel.state.gov)                       | Government | Ground truth — legal immigrants         |
| [US Customs and Border Protection](https://www.cbp.gov)            | Government | Ground truth — illegal border crossings |
| [IMF Financial Data](https://www.imf.org/en/Data)                  | Financial  | Push & pull economic factors            |
| [Google Trends (via pytrends)](https://pypi.org/project/pytrends/) | API        | Real-time migration interest tracking   |
| [Google News (via pygooglenews)](https://pypi.org/project/pygooglenews/) | API        | News sentiment analysis   |
| [ETL on Yahoo Finance (via indicators-cli ETL tool)](https://pypi.org/project/indicators-cli/) | ETL        | Financial health and indicators    |
