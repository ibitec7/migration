---
title: Encounter Data
aliases: [Border Encounters, CBP Data, Southwest Border Data]
tags: [data-source, ground-truth, encounter]
created: 2026-04-12
---

# Encounter Data

US Customs and Border Protection (CBP) Southwest Land Border encounter statistics, serving as the **ground truth for illegal border crossings**.

## Source

- **Publisher**: US Customs and Border Protection
- **URL**: [cbp.gov](https://www.cbp.gov)
- **Format**: 8 CSV files covering FY2019 – FY2026 YTDC
- **Coverage**: Southwest border, monthly aggregates
- **Baseline**: ~50,000/month in 2018–2019

## Collection

The [[collection-module]] (`src/collection/encounter.py`) scrapes CSV download links from cbp.gov using async semaphore-controlled downloads with retry logic. Files land in `data/raw/encounter/`.

## Processing

The [[processing-module]] (`src/processing/merge.py`) consolidates 8 CSV files:

1. Scan all CSVs in encounter directory
2. Map columns: Fiscal Year → year, month abbreviation → month number
3. Concatenate using Polars lazy frames for efficiency
4. Output: unified encounter DataFrame with standardized schema

## Key Characteristics

- **Extreme volatility**: May 2023 peak of 880,000 encounters = **18× the 2018–2019 baseline**
- **Dual-channel correlation**: r = 0.438 with legal [[visa-data|visa issuances]], suggesting common push/pull factors drive both legal and illegal migration
- **Fiscal year alignment**: Data reported by federal fiscal year (Oct–Sep), requiring conversion to calendar year for alignment with other sources

## Role in the Pipeline

Encounter data serves as a **secondary target/validation** variable:
- Used in [[cross-correlation]] to validate that Google Trends signals predict both channels
- Appears in EDA notebooks for dual-axis visa-vs-encounter visualizations
- Validates that the system captures both legal and illegal migration dynamics

## See Also

- [[visa-data]] — Primary ground truth (legal immigration)
- [[data-collection]] — Broader ingestion architecture
- [[cross-correlation]] — Encounter-trends analysis
- [[collection-module]] — Source code reference
