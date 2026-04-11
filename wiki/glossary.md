---
title: Glossary
aliases: [Terms, Definitions, Key Terms]
tags: [reference, glossary]
created: 2026-04-12
---

# Glossary

Quick reference for domain terms used throughout this wiki.

---

### ADF Test
**Augmented Dickey-Fuller test** — a statistical test for stationarity in time series. Used in [[cross-correlation]] to validate that series are suitable for correlation analysis.

### BCE
**Binary Cross-Entropy** — a loss function for binary classification. Used in [[surge-joint-loss]] to penalize missed or false-alarm surges.

### Benjamini-Hochberg
A procedure for controlling the **False Discovery Rate (FDR)** when performing many simultaneous hypothesis tests. Applied in [[multiple-comparison-correction]] to filter the 58 significant lead-lag signals.

### CBP
**US Customs and Border Protection** — the agency that publishes Southwest border [[encounter-data]].

### CCF
**Cross-Correlation Function** — measures similarity between two time series at different lag offsets. Core technique in [[cross-correlation]] and [[lead-lag-analysis]].

### cuML
NVIDIA's GPU-accelerated machine learning library. Used for [[random-forest]] training and [[event-clustering]] (HDBSCAN).

### Encounter
An event where CBP apprehends or encounters a noncitizen at or between ports of entry. See [[encounter-data]].

### FDR
**False Discovery Rate** — the expected proportion of false positives among all rejected hypotheses. Controlled via [[multiple-comparison-correction|Benjamini-Hochberg]].

### HDBSCAN
**Hierarchical Density-Based Spatial Clustering of Applications with Noise** — a clustering algorithm that finds variable-density clusters without requiring a pre-set number of clusters. Used in [[event-clustering]] on news embeddings.

### Horizon
The number of months ahead a prediction targets. This system forecasts at horizons of 1–6 months (Lead 1 through Lead 6). See [[horizon-aware-ensemble]].

### Huber Loss
A regression loss that is quadratic for small errors and linear for large ones, making it robust to outliers. Used in [[surge-joint-loss]].

### IMF REER
**International Monetary Fund Real Effective Exchange Rate** — a trade-weighted currency index adjusted for inflation. See [[exchange-rates]].

### Jina v5
A text embedding model producing 768-dimensional vectors. Deployed via [[jina-v5-embeddings|TensorRT INT8]] for news article encoding.

### Lead-Lag
The temporal offset between a predictive signal and a target variable. A "lag of 3" means the signal precedes the target by 3 months. See [[lead-lag-analysis]].

### LED
**Longformer Encoder-Decoder** — a transformer variant handling up to 16,384 input tokens. Used in [[event-clustering]] for generating 2–3 word cluster labels.

### MoM
**Month-over-Month** — the percentage change from one month to the next. Used in [[surge-detection]] (>30% MoM = surge candidate).

### OOT
**Out-of-Time** — a validation strategy where the train/test split is temporal (train ≤ 2022, test 2023+). See [[training-pipeline]].

### Panel Data
A dataset with observations across both time and cross-sectional units (countries × months). See [[panel-construction]].

### Pearson r
**Pearson correlation coefficient** — measures linear association between two variables, ranging from −1 to +1. Core metric in [[lead-lag-analysis]].

### REER
See **IMF REER**.

### Surge
A significant spike in migration volume. Defined as either > 75th percentile or > 30% MoM growth (in [[surge-detection]]), or > 1.5σ above rolling mean (in model training). See [[surge-joint-loss]].

### SurgeJointLoss
A custom dual-objective loss function combining Huber regression with BCE surge classification. See [[surge-joint-loss]].

### TensorRT
NVIDIA's inference optimization toolkit that quantizes and compiles models for faster GPU execution. Used across [[jina-v5-embeddings]], [[flan-t5-summarization]], and [[tensorrt-engines]].

### VAR
**Vector Autoregression** — a multivariate time series model where each variable is predicted from lagged values of itself and all other variables. Used in [[cross-correlation]] for benchmarking Google Trends predictive power.

### Visa Issuance
The number of visas granted by US consulates/embassies, published monthly by the Department of State. Primary ground truth variable. See [[visa-data]].

### Wikilink
An Obsidian-style link using double-bracket syntax (`[[ ]]`) that connects pages in this knowledge graph.

---

## See Also

- [[project-overview]] — Full methodology context
- [[index]] — Map of Content
