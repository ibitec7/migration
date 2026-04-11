# **Migration Surge Prediction: An Early Warning System**

**SDSC 2005 — Introduction to Computational Social Science · Group 10 Final Report**

**Team Members:** Ibrahim (58776310), Haseeb (58791350), Asadohoo (58452930), Sameer (58771398), Ayaan (58970130), Pak Hei Ng (58650183)

---

## **Abstract**

Governments and humanitarian organizations are routinely caught off-guard by sudden spikes in migration, yet the underlying signals — shifting news sentiment, search behaviour, and macroeconomic volatility — are observable months in advance. This study presents a multi-modal early warning system that synthesizes over 170,000 news articles, Google Trends search-intent data, and Real Effective Exchange Rate (REER) indicators across 15 high-volume origin countries over a nine-year observation window (2017–2026) to predict surges in both legal immigration (visa issuances) and unauthorized border encounters at one-to-six-month horizons. We evaluate four architectures — a GPU-accelerated Random Forest baseline, a recurrent LSTM with a novel dual-objective *SurgeJointLoss*, a Transformer encoder, and a Horizon-Aware Ensemble that dynamically re-weights trust across models according to forecast lead time. Evaluated strictly out-of-time on walk-forward splits (training ≤ 2022, testing 2023+), the ensemble achieves an F1 score of 0.96 at the one-month horizon, degrading gracefully to 0.86 at six months, while maintaining recall above 92 % at every lead — meaning fewer than one in ten genuine crises goes undetected. These results demonstrate that fusing heterogeneous digital trace data with purpose-built deep learning architectures can transform migration forecasting from a reactive exercise into a proactive planning tool.

---

## **1. Introduction**

### **1.1 Study Context & Motivation**

Global migration is driven by a deeply intertwined web of push and pull factors — geopolitical instability, economic opportunity, conflict, climate events, and family reunification — making the timing and magnitude of migration flows inherently difficult to anticipate. For the roughly 280 million international migrants worldwide (UNDESA, 2020), the decision to move is rarely sudden; it is incubated by months of deteriorating conditions at home and perceived opportunity abroad. The signals of that incubation period — a spike in "US asylum" searches, a currency devaluation, a surge in conflict-related news coverage — are increasingly observable in real time through digital trace data.

Yet current migration management remains overwhelmingly *reactive*. Governments and city planners confront migration surges only *after* they materialize — scrambling to allocate housing, legal processing capacity, and humanitarian aid when border crossings have already overwhelmed existing infrastructure. The scale of these surges can be extraordinary: in May 2023, U.S. Southwest border encounters peaked at approximately 880,000 in a single month — roughly 18 times the 2018–2019 baseline — creating acute humanitarian bottlenecks across the southern border.

Traditional forecasting models suffer from what we term a **"precision blindspot."** By relying on slow-moving, lagging historical indicators (e.g., prior-year census counts or quarterly economic reports), these models adequately predict the normal baseline of migration flows but systematically fail to forecast the extreme spikes that actually trigger crises. A model that predicts "no surge" 95 % of the time can appear highly accurate in aggregate, yet be operationally useless precisely when it matters most.

This project proposes an **Early Warning System** that addresses this blindspot by synthesizing three families of real-time digital signals — (1) news event sentiment reflecting geopolitical and social conditions in origin countries, (2) Google Trends search-intent data capturing migration-related interest, and (3) macroeconomic exchange-rate volatility — into actionable, multi-horizon surge forecasts spanning one to six months ahead. By combining these heterogeneous data streams with an ensemble of machine learning architectures specifically designed to detect threshold exceedance events, the system aims to give policy-makers a concrete, quantitative lead time to prepare.

### **1.2 Research Questions**

* **RQ1:** How can multi-modal digital trace data (news sentiment, search intent, and macroeconomic indicators) be synthesized to accurately predict sudden surges in migration?
* **RQ2:** How does the predictive accuracy of various machine learning architectures (Random Forest, LSTM, Transformer) change across different forecasting horizons (1 to 6 months)?
* **RQ3:** Which data modality — news sentiment, Google Trends, or exchange rates — serves as the strongest early indicator of a migration spike?

---

## **2. Methods**

### **2.1 Data Collection & Multi-Modal Ingestion**

Our system ingests four distinct data streams, aligned to a monthly cadence and covering the 15 highest-volume origin countries for U.S. migration: Afghanistan, China, Colombia, Cuba, Dominican Republic, El Salvador, Guatemala, Haiti, Honduras, India, Mexico, Pakistan, Philippines, Venezuela, and Vietnam.

**Ground Truth — Visa Issuances.** We collected monthly immigrant visa issuance statistics from the U.S. Department of State (travel.state.gov), spanning January 2017 through September 2025. The raw corpus consists of 108 monthly PDF reports that were programmatically parsed using PyMuPDF table extraction. Each report contains per-country, per-visa-class issuance counts. We mapped 118 distinct visa classification codes into seven interpretable categories (family-sponsored, employment-based, special immigrant, humanitarian, returning resident, diversity lottery, and nonimmigrant), producing a unified `visa_master` panel of approximately 100,000 records. Exploratory analysis revealed that visa issuance volume follows a Zipfian (power-law) distribution: the top 20 % of countries account for 88.3 % of all visas issued, with Cuba, Mexico, and Afghanistan generating the highest excess volume above the 75th percentile.

**Ground Truth — Border Encounters.** Southwest U.S.–Mexico land border encounter data was obtained from U.S. Customs and Border Protection (cbp.gov), comprising eight CSV datasets covering fiscal years 2019 through 2026 (year-to-date). These were merged and normalized to a calendar-month cadence using Polars lazy evaluation for memory efficiency. The combined encounter time series exhibits extreme volatility, with a peak of approximately 880,000 monthly encounters in May 2023 compared to a baseline of ~50,000 in 2018–2019.

*[Figure 1: Dual-axis time series of monthly visa issuances vs. border encounters, 2017–2025]*

**News Articles (NLP Pipeline).** To capture geopolitical and social dynamics in real time, we orchestrated a large-scale automated scraping operation targeting Google News across all 15 countries and 8 migration-relevant topics (e.g., "US visa," "US asylum," "CBP One," "USAID"). This produced 120 unique country-topic query combinations. The pipeline was engineered for throughput and fault tolerance:

* Articles were collected via the Google News RSS API, with encoded URLs resolved through a batched decoding endpoint (20 articles per batch request, reducing API calls by ~95 % compared to individual resolution).
* Full article text was fetched asynchronously with bounded concurrency (semaphore-controlled, maximum 16 simultaneous connections), exponential backoff with jitter to respect rate limits, and a 30-second watchdog timeout per request.
* HTML-to-text extraction was performed via Trafilatura, parallelized across CPU cores using a ProcessPoolExecutor (4 workers) to bypass Python's GIL bottleneck.
* Intermediate results were checkpointed to disk every 25 articles via atomic writes (write to temporary file, then rename), ensuring crash-resilient resumption.

This pipeline yielded **170,754** raw articles, of which **104,333** passed quality filters (HTTP status 200, non-empty body, extractable content) — a pipeline success rate of approximately 61.1 %.

**Google Trends (Search Intent).** For each of the 15 countries, we collected monthly Google Trends time-series data for eight migration-intent keywords: `us_visa`, `us_asylum`, `green_card`, `cbp_one`, `usaid`, `united_states_passport`, `us_jobs`, and `study_us`. These keywords were selected to capture the spectrum of legal migration interest (visa/green card queries), crisis-driven intent (asylum/CBP One), and economic pull signals (jobs/study). Data was sourced via the Hugging Face Hub for reproducibility.

**Macroeconomic Indicators — Exchange Rates.** Real Effective Exchange Rate (REER) data was obtained from the IMF/World Bank for use as a proxy of economic volatility in origin countries. Exchange rate data was available for 6 of the 15 target countries (China, Colombia, Dominican Republic, Mexico, Pakistan, Philippines), representing a known coverage limitation.

### **2.2 NLP Processing & Feature Engineering**

The 104,333 valid news articles underwent a multi-stage NLP pipeline to transform unstructured text into structured, model-ready features.

**Embedding Generation.** Each article was tokenized (maximum sequence length: 8,192 tokens) and encoded into a 768-dimensional semantic embedding using a Jina v5 embedding model accelerated via TensorRT INT8 quantization. Length-aware batching — sorting sequences by token count before inference — minimized padding overhead. Quantization validation confirmed a mean cosine similarity of 0.9999 against the full-precision FP32 baseline, indicating negligible information loss from compression.

**Event Clustering.** To discover coherent thematic event clusters within the news corpus for each country, we applied HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) on the embedding vectors, accelerated via NVIDIA cuML on GPU. Hyperparameters were set to `min_cluster_size=10` and `min_samples=5`, with Euclidean distance as the metric. This density-based approach naturally handles noise (unclustered articles) and does not require pre-specifying the number of clusters — a critical advantage when the underlying event landscape varies dramatically by country. The resulting clusters were then visualized via UMAP and t-SNE dimensionality reduction, with silhouette scores computed per country to assess cluster quality.

**Autonomous Cluster Labeling.** Rather than manually reviewing thousands of clusters, we deployed a TensorRT-optimized Flan-T5-Large model (INT8 W8A16 quantization) to generate concise, human-readable labels for each cluster. For each cluster, five representative articles were sampled and fed into Flan-T5 with a structured prompt requesting a 2–3 word descriptive label. This produced labels such as *"Honduras Evangelical Pastors"*, *"Cuba — American President Trump's Policy"*, and *"China's Maritime Intelligence"* — interpretable thematic signatures that could be directly correlated with migration outcomes.

**Sentiment Scoring.** A lexicon-based sentiment score was computed for each article, using domain-specific positive terms (e.g., "growth," "peace," "jobs," "stability") and negative terms (e.g., "crisis," "conflict," "violence," "inflation," "unemployment"). The score was defined as $\frac{(\text{positive count} - \text{negative count})}{\sqrt{\text{token count}}}$, clipped to $[-1, 1]$, and aggregated to monthly country-cluster means. Sentiment shocks were identified using z-score thresholds: negative shocks at $z \leq -1.0$ and positive shocks at $z \geq +1.0$.

**Panel Construction — Feature Alignment.** All data streams were merged into a unified panel at the monthly × country level. For each country-month observation, we constructed:

* **Lag features** ($t-1$ through $t-6$): Six months of historical values for each of three modalities — visa issuance volume, exchange rate, and news event count — yielding 18 input features per observation.
* **Lead targets** ($t+1$ through $t+6$): Six future months of visa issuance volume, allowing the model to produce simultaneous multi-horizon forecasts.

Missing exchange rates were forward/backward-filled, and rows with insufficient lag/lead coverage were dropped to ensure clean supervision signals.

### **2.3 Lead-Lag Signal Analysis**

Before building predictive models, we conducted a systematic lead-lag correlation analysis across all three signal families to assess their individual predictive value — directly addressing **RQ3**.

**Methodology.** For each country–signal–lag combination (lags 0–6 months), we computed Pearson correlation between the signal and visa issuance volume, requiring a minimum of 12 months of overlapping data. Given the high dimensionality of the testing space (hundreds of country–signal–lag combinations), all p-values were corrected for multiple comparisons using the **Benjamini-Hochberg False Discovery Rate (FDR)** procedure, yielding adjusted q-values. Statistical significance was assessed at $q \leq 0.10$.

**Surge Detection.** Beyond linear correlation, we evaluated whether signal-active months aligned with actual visa surge months using a dual-criterion surge definition: a month was classified as a "surge" if issuance volume exceeded the 75th percentile *or* showed month-over-month growth exceeding 30 %. Signal quality was quantified via **surge precision** (fraction of predicted surges that were true surges) and **surge lift** (improvement over the baseline surge rate).

### **2.4 Predictive Models**

We evaluated four architectures, each offering distinct inductive biases for the surge prediction task.

**cuML Random Forest (Baseline).** A GPU-accelerated Random Forest regressor served as our non-temporal baseline. Six independent models were trained — one per forecast horizon — each with 50 estimators and a maximum depth of 6, using the 18-dimensional flattened lag feature vector as input. By fitting separate models per horizon, each could develop horizon-specific decision boundaries without being constrained to shared parameters.

**LSTM with SurgeJointLoss.** A two-layer Long Short-Term Memory network (hidden dimension 64) was designed to exploit the sequential structure of the six-month input window. A key architectural innovation was the inclusion of **learned country embeddings** (dimension 8), concatenated with the input features at each time step — allowing the model to capture country-specific migration dynamics (e.g., structural differences between Mexico and Pakistan) as part of the recurrent computation. The LSTM was trained with a novel **SurgeJointLoss** objective:

$$\mathcal{L} = \alpha \cdot \text{Huber}(\hat{y}, y) + (1 - \alpha) \cdot \text{BCE}(\hat{y} - \tau, \mathbb{1}[y > 1.5\sigma])$$

where $\alpha = 0.6$ balances volume regression accuracy (Huber loss, robust to outliers) against explicit surge classification (Binary Cross-Entropy against a flag indicating whether the true volume exceeded 1.5 standard deviations above the rolling mean). This dual objective directly penalizes both missed crises and false alarms within a single training pass, rather than relying on post-hoc thresholding of regression outputs.

**Transformer Encoder.** A two-layer Transformer encoder with 4 attention heads ($d_{\text{model}} = 64$, feedforward dimension 256) was trained to model the same six-month input sequences. Unlike the LSTM, the Transformer uses self-attention to weigh all time steps simultaneously, with **learnable positional encodings** (rather than fixed sinusoidal) that allow the model to discover its own optimal temporal representation. The Transformer was trained with standard MSE loss, serving as a pure attention-based baseline without country embeddings or surge-specific objectives.

**Horizon-Aware Ensemble.** The final model dynamically blends predictions from all three architectures with horizon-specific weights that shift trust according to each model's empirical strengths:

| Lead | RF Weight | Transformer Weight | LSTM Weight | Rationale |
|:----:|:---------:|:------------------:|:-----------:|-----------|
| 1 mo | 60 % | 10 % | 30 % | RF and LSTM dominate short-term structural prediction |
| 2 mo | 50 % | 20 % | 30 % | RF still dominant; Transformer gaining |
| 3 mo | 30 % | 40 % | 30 % | Balanced transition |
| 4 mo | 20 % | 60 % | 20 % | Transformer increasingly critical for longer patterns |
| 5 mo | 10 % | 80 % | 10 % | Transformer dominates long-horizon recall |
| 6 mo | 5 % | 85 % | 10 % | Transformer captures slow-moving geopolitical dynamics |

*Table 1: Horizon-Aware Ensemble weighting scheme. Weights are normalized to sum to 1.0 per horizon.*

The intuition is that tree-based models excel at near-term threshold approximation (where the recent past is highly informative), while the Transformer's global attention mechanism maintains superior recall over longer horizons where slow-moving geopolitical trends become the dominant signal. The LSTM contributes high-precision alerts throughout, smoothing ensemble outputs with its conservative, surge-calibrated predictions.

### **2.5 Validation Strategy**

All models were evaluated under a strict **walk-forward, out-of-time (OOT)** protocol: training was performed exclusively on data through December 2022, with evaluation on rolling predictions throughout 2023 and beyond. This prevents any form of look-ahead bias and simulates real-world deployment conditions where the model must predict into a genuinely unseen future.

**Surge Classification Threshold.** A month-country observation was classified as a "surge" if the actual volume exceeded $1.5$ standard deviations above the normative rolling mean. Approximately 103 to 114 surge instances were present in the OOT test set, providing a sufficient sample for robust classification evaluation.

**Metrics.** In the context of an operational Early Warning System, we prioritize:

* **Precision** (minimizing false alarms): Of all months flagged as surges, what fraction were genuine crises? High precision prevents alert fatigue.
* **Recall** (minimizing missed crises): Of all genuine surge months, what fraction did the system correctly detect? High recall ensures no crisis goes unwarned.
* **F1 Score**: The harmonic mean of Precision and Recall, balancing both objectives.

For volume accuracy, we additionally report **Root Mean Squared Error (RMSE)** on the continuous prediction of monthly migrant flow volumes.

---

## **3. Results**

### **3.1 Multi-Signal Analysis: Which Modality Predicts Migration?**

Addressing **RQ3**, we evaluated each data modality's independent predictive power before model training.

**News Event Signals.** Across 15 countries, 392 country–event-label–lag combinations were tested. Of these, **58 (14.8 %)** achieved statistical significance at $q \leq 0.10$ after FDR correction. The mean absolute correlation across all tested combinations was modest ($|\bar{r}| = 0.126$), but country-specific signals were substantially stronger. The top findings include:

| Country | Event Label | Lag | $r$ | $q$-value | Surge Lift |
|---------|-------------|:---:|:---:|:---------:|:----------:|
| Mexico | "Mexico's World Watch" | 1 mo | −0.436 | 0.0009 | — |
| India | "India's economy is growing faster" | 2 mo | −0.424 | 0.0010 | 0.932 |
| Honduras | "Honduras Evangelical Pastors" | 2 mo | +0.327 | 0.0114 | 1.229 |
| Vietnam | "Vietnam's fake news" | 5 mo | −0.403 | 0.0017 | 0.816 |
| El Salvador | "El Salvador reports 11 dead" | 5 mo | −0.361 | 0.0055 | 0.842 |

*Table 2: Top event-to-visa lead-lag signals (ranked by absolute correlation, q ≤ 0.10).*

At the country level, 11 of 15 countries exhibited at least one significant event signal, supporting the use of news events as a practical component in multi-signal forecasting.

**Sentiment Signals.** Sentiment analysis revealed even sharper relationships. The standout finding was Dominican Republic, where a specific event cluster ("Dominican Republic to have its first...") showed a Pearson correlation of **$r = 0.617$** at a 3-month lag ($q = 3.3 \times 10^{-9}$) — the single strongest lead signal identified in the entire study. Cuba's "American President Trump's Cuba policy" cluster showed $r = -0.595$ at lag 0 ($q = 3.4 \times 10^{-9}$), reflecting the immediate impact of U.S. policy shifts on Cuban visa flows.

**Exchange Rate Signals.** Among the 6 countries with REER data, 23 of 42 country-lag rows (54.8 %) achieved significance. The Dominican Republic dominated with $r = 0.498$ at a 2-month lag ($q = 2.6 \times 10^{-6}$), followed by Mexico ($r = 0.289$, lag 6, $q = 0.018$). Overall, exchange-rate shocks provided only slight improvement over baseline surge rates (mean lift: 1.009), confirming their value as a complementary rather than standalone signal.

**Google Trends.** Trend-visa correlations were significant for 66 of 120 tested rows (55.0 %). Notable signals included China's `us_visa` keyword ($r = 0.406$, $q = 1.1 \times 10^{-4}$), Colombia's `study_us` ($r = -0.381$, $q = 2.8 \times 10^{-4}$), and Afghanistan's `us_jobs` ($r = -0.484$, $q = 4.4 \times 10^{-6}$). However, when we benchmarked a trends-enhanced Vector Autoregression (VAR) model against a pure autoregressive baseline, the trends-enhanced model *underperformed* on average (mean RMSE improvement: −5.97 % for visas, −6.02 % for encounters), indicating that while correlations exist, translating them into out-of-sample accuracy gains requires nonlinear modelling.

**Modality Ranking (RQ3 Answer).** News sentiment emerged as the strongest individual predictor, producing the highest single-signal correlations (up to $r = 0.617$) and the most country-specific actionable leads. Exchange rates provided meaningful signal where data was available but were limited to 6 countries. Google Trends showed the broadest coverage (55 % significance rate) but the weakest out-of-sample translation. This finding directly motivated our multi-modal ensemble approach — no single modality is sufficient, but their fusion overcomes individual weaknesses.

### **3.2 Model Performance**

#### **3.2.1 Volume Prediction Accuracy (RMSE)**

| Horizon | cuML Random Forest | Transformer | LSTM |
|:-------:|:------------------:|:-----------:|:----:|
| Lead 1 | 298.79 | 303.43 | **277.40** |
| Lead 2 | 341.79 | 343.61 | **320.77** |
| Lead 3 | 351.55 | 359.20 | **346.03** |
| Lead 4 | **364.01** | 381.20 | 381.00 |
| Lead 5 | 428.35 | 407.67 | **404.23** |
| Lead 6 | **401.38** | 426.48 | 424.59 |

*Table 3: Out-of-time RMSE by model and horizon (lower is better). Bold indicates best per horizon.*

The LSTM's recurrent memory state gives it a decisive edge on near-term volume prediction (Leads 1–3), outperforming the baseline by 10–20 RMSE points. By Lead 4–6, the simpler feature-split assumptions of the Random Forest maintain competitive parity, suggesting that at longer horizons, the additional complexity of deep models yields diminishing returns for point estimation.

#### **3.2.2 Surge Classification Performance**

The operationally critical evaluation: how well does each architecture detect extreme threshold exceedance events ($> 1.5\sigma$) in the OOT test set?

**Random Forest:**

| Horizon | Precision | Recall | F1 |
|:-------:|:---------:|:------:|:--:|
| Lead 1 | 0.96 | 0.97 | **0.97** |
| Lead 2 | 0.93 | 0.95 | **0.94** |
| Lead 3 | 0.88 | 0.96 | **0.92** |
| Lead 4 | 0.85 | 0.94 | **0.89** |
| Lead 5 | 0.82 | 0.93 | **0.88** |
| Lead 6 | 0.78 | 0.91 | **0.84** |

**Transformer:**

| Horizon | Precision | Recall | F1 |
|:-------:|:---------:|:------:|:--:|
| Lead 1 | 0.95 | 0.93 | **0.94** |
| Lead 2 | 0.95 | 0.94 | **0.94** |
| Lead 3 | 0.91 | 0.93 | **0.92** |
| Lead 4 | 0.86 | 0.90 | **0.88** |
| Lead 5 | 0.85 | 0.90 | **0.88** |
| Lead 6 | 0.83 | 0.90 | **0.87** |

**LSTM (SurgeJointLoss):**

| Horizon | Precision | Recall | F1 |
|:-------:|:---------:|:------:|:--:|
| Lead 1 | 0.97 | 0.85 | **0.91** |
| Lead 2 | 0.94 | 0.84 | **0.89** |
| Lead 3 | 0.91 | 0.83 | **0.87** |
| Lead 4 | 0.88 | 0.81 | **0.84** |
| Lead 5 | 0.85 | 0.80 | **0.82** |
| Lead 6 | 0.82 | 0.79 | **0.80** |

**Horizon-Aware Ensemble:**

| Horizon | Precision | Recall | F1 |
|:-------:|:---------:|:------:|:--:|
| Lead 1 | 0.96 | 0.96 | **0.96** |
| Lead 2 | 0.93 | 0.96 | **0.95** |
| Lead 3 | 0.92 | 0.94 | **0.93** |
| Lead 4 | 0.88 | 0.94 | **0.91** |
| Lead 5 | 0.83 | 0.94 | **0.88** |
| Lead 6 | 0.80 | 0.92 | **0.86** |

*Tables 4a–d: Surge classification metrics per model and horizon (OOT test set, ~103–114 surge instances).*

*[Figure 2: F1 score trajectory across Lead 1–6 for all four models]*

**Average F1 across all horizons:**

| Model | Avg F1 |
|-------|:------:|
| cuML Random Forest | 0.907 |
| Transformer Encoder | 0.905 |
| LSTM (SurgeJointLoss) | 0.855 |
| **Horizon-Aware Ensemble** | **0.915** |

*Table 5: Average F1 score (Leads 1–6) by model.*

### **3.3 Architecture Insights (RQ2 Answer)**

The results reveal a clear division of labour among architectures across the forecast horizon — directly answering **RQ2**:

1. **Random Forest excels at near-term threshold approximation.** Its aggressive discrete decision boundaries achieve the highest single-model F1 at Lead 1 (0.97), capturing broad surge envelopes with minimal miss rate (Recall: 0.97). However, precision degrades at longer horizons as the feature space becomes less informative.

2. **The LSTM delivers the highest-precision alerts.** With a Precision of 0.97 at Lead 1, the LSTM — driven by the SurgeJointLoss objective — is the most conservative model: when it fires an alarm, it is the most likely to be a true positive. This makes it invaluable as a "high-confidence confirmation" signal. The tradeoff is lower recall (0.85 at Lead 1, 0.79 at Lead 6), reflecting its tendency toward under-prediction.

3. **The Transformer maintains superior long-horizon recall.** At Lead 6, the Transformer retains 90 % recall (vs. 79 % for LSTM, 91 % for RF) and achieves the highest F1 among individual models at Lead 6 (0.87). Its self-attention mechanism appears to capture slow-moving geopolitical dynamics that recurrent models lose over longer sequences.

4. **The Ensemble synergizes all three.** By dynamically shifting weight from RF/LSTM dominance at Lead 1 toward Transformer dominance at Lead 6, the ensemble achieves the best or near-best F1 at *every* horizon. Critically, it maintains **recall above 92 % across all six months**, meaning it catches more than 9 out of 10 genuine crises regardless of how far ahead it is predicting — while keeping false alarm rates at or below 20 %.

---

## **4. Discussion**

### **4.1 Practical Implications**

The system's ability to produce calibrated surge probabilities at 1-to-6-month horizons opens several concrete applications for policy and humanitarian planning:

* **At Lead 1 (F1: 0.96):** Immigration processing centres can adjust staffing and capacity allocation for the coming month with high confidence, reducing wait times and overcrowding.
* **At Lead 3 (F1: 0.93):** Humanitarian organizations (UNHCR, IOM, NGOs) can pre-position supplies, temporary shelter capacity, and medical resources in anticipated surge corridors.
* **At Lead 6 (F1: 0.86):** National-level policy planning — budgetary allocations, bilateral diplomatic engagement with origin countries, and inter-agency coordination — can be initiated with sufficient lead time for institutional decision-making.

The multi-signal analysis findings also carry direct policy relevance. The discovery that news sentiment signals (particularly around U.S. policy announcements toward Cuba and the Dominican Republic) serve as the strongest leading indicators suggests that **monitoring media coverage in origin countries could serve as a low-cost, high-signal early warning input** — even in the absence of sophisticated modelling infrastructure.

More broadly, this work exemplifies how Computational Social Science methods can transform public policy from reactive to proactive. By treating migration as a measurable phenomenon with observable antecedent signals, rather than an unpredictable shock, we enable evidence-based preparation rather than crisis management.

### **4.2 Limitations**

**The Digital Divide.** Both Google Trends and digital news reflect populations with internet access and engagement with online search. The most vulnerable migrant populations — those displaced by conflict, extreme poverty, or climate events in regions with limited digital infrastructure — may be systematically underrepresented in these data streams. This creates a potential blind spot in which the system is better calibrated for digitally-connected populations (e.g., urban middle-class migrants from India or China) than for the most acutely at-risk groups (e.g., rural Haitian or Honduran families).

**Correlation, Not Causation.** This is a *predictive* model, not a *causal* one. A negative sentiment shock in news coverage of El Salvador may predict a visa surge two months later, but this correlation may reflect a common underlying cause (e.g., a specific policy event) rather than a direct causal pathway from news to migration. The lead-lag framework identifies temporal associations, but establishing causal mechanisms would require quasi-experimental designs or instrument-variable approaches that lie outside the scope of this work.

**Exchange Rate Data Gap.** REER data was available for only 6 of 15 target countries, excluding some of the highest-priority origins (Afghanistan, Cuba, El Salvador, Guatemala, Haiti, Honduras, Venezuela, Vietnam, India). This substantially limits the macro-economic signal for the remaining nine countries and may bias the multi-signal findings toward those with richer data availability.

**Google Trends as a Weak Standalone Predictor.** Despite significant correlations at the keyword level, the trends-enhanced VAR model underperformed a pure autoregressive baseline on average (−5.97 % RMSE). This suggests that naive linear models cannot exploit the search-intent signal effectively, and that the predictive value of trends is best captured through nonlinear architectures (RF, LSTM, Transformer) rather than traditional econometric methods.

**News Coverage Bias.** The 61.1 % pipeline success rate — while robust for a web-scraping operation at scale — means that roughly 39 % of potentially relevant articles were lost to extraction failures, paywalls, or anti-scraping measures. Article coverage is also skewed toward English-language media, potentially underrepresenting local-language reporting that may carry different (and possibly stronger) migration-related signals.

**Feature Interpretability and Selective Data Collection.** The underlying models — particularly the Random Forest — carry exploitable interpretability through feature importance scores, yet this signal is currently unused in guiding data collection. Systematic importance analysis could identify which specific sentiment clusters, exchange rate lags, or search keywords drive the most predictive value per horizon, allowing future collection efforts to be concentrated on high-signal inputs rather than continuing to span all 120 country-topic combinations indiscriminately. A targeted pipeline informed by this analysis could achieve comparable accuracy at substantially lower operational cost.

**Model Staleness and Distributional Drift.** The model's learned assumptions are anchored to the 2017–2022 training window, and over time the structural relationships between signals and outcomes may shift — a phenomenon known as *concept drift*. A concrete risk is that the Zipfian distribution of visa issuances, where the top 20 % of countries account for 88.3 % of issuances, may not remain stable as geopolitical pressures redistribute migration toward currently low-volume origins, pushing the system into an out-of-distribution regime. Without periodic retraining and calibration monitoring, the Early Warning System risks degrading precisely when novel migration patterns emerge.

**Black Swan Events and the Limits of Historical Training.** The model can only learn patterns represented in its training data, meaning genuinely novel shocks — a sudden change in U.S. asylum law, a region-scale natural disaster, or an unprecedented policy event — lie beyond its anticipatory reach. These *black swan* events are disproportionately likely to produce the largest migration surges, precisely the crises an Early Warning System is most needed to detect. This boundary makes it critical that the system be treated as a complement to, not a replacement for, domain expert judgement that can incorporate real-time geopolitical intelligence.

**Survivorship Bias in Encounter Ground Truth.** The border encounter series records only unauthorized crossings that were detected by CBP — migrants who successfully evaded apprehension are entirely absent from the data, creating *survivorship bias*. This means the model implicitly learns to predict enforcement detection rates as much as true migration pressure, and any surge in undetected crossings — which tends to accompany periods of resource strain — will appear as a model underestimate rather than a genuine false negative. Recall figures on encounter targets should therefore be interpreted with this structural limitation in mind.

### **4.3 Future Work**

Several extensions could strengthen and generalize the system:

* **Social media integration:** Real-time signals from platforms such as X (Twitter), Facebook, and WhatsApp community groups could provide finer-grained, less media-filtered insight into migration intent — particularly for populations underrepresented in traditional news.
* **Cross-country transfer learning:** Training on data-rich countries (Mexico, India) and transferring learned representations to data-sparse countries (Haiti, Afghanistan) could improve prediction where local signal is limited.
* **Real-time streaming deployment:** The current batch-monthly pipeline could be adapted for continuous ingestion and prediction, providing rolling daily or weekly surge forecasts.
* **Richer macroeconomic indicators:** Incorporating inflation rates, unemployment figures, remittance flows, and commodity prices could fill the gap left by the limited exchange rate coverage.
* **Causal discovery:** Applying Granger causality tests, structural equation models, or natural-experiment designs to move from predictive correlation to actionable causal understanding.

---

## **5. Contribution Statement**

*Each team member's specific contributions are listed below:*

* **Ibrahim:** Led the multi-modal data ingestion pipeline architecture, including the async news scraping system (170K+ articles), Google Trends collection, Hugging Face data sync infrastructure, and data processing pipelines. Contributed to writing the Methods section of this report.
* **Haseeb:** Designed and tuned the LSTM and Transformer Encoder model architectures, including the SurgeJointLoss implementation. Generated model performance visualizations.
* **Asadohoo:** Developed the automated news scraping and URL decoding pipeline, performed NLP processing (embedding generation, event clustering, sentiment analysis) across the full article corpus.
* **Sameer:** Conducted the Exploratory Data Analysis on encounter and visa ground truth data, including the Zipfian distribution analysis and seasonal decomposition. Drafted the Introduction and Motivation.
* **Ayaan:** Built the Horizon-Aware Ensemble model with dynamic per-horizon weighting, computed the 1–6 month surge classification metrics, and conducted the lead-lag signal analysis across all modalities.
* **Pak Hei Ng:** Managed the cuML Random Forest baseline and GPU-accelerated HDBSCAN clustering, aggregated the final report, and handled formatting and academic citations.

---

## **6. References**

### Data Sources

1. U.S. Department of State, Bureau of Consular Affairs. *Monthly Immigrant Visa Issuances.* travel.state.gov. Accessed 2017–2025.
2. U.S. Customs and Border Protection. *Southwest Land Border Encounters.* cbp.gov. Fiscal Years 2019–2026.
3. International Monetary Fund. *Real Effective Exchange Rate (REER).* IMF Data. imf.org/en/Data.
4. Google Trends. *Search interest data.* trends.google.com. Via pytrends API.
5. Google News. *News article metadata and content.* Via pygooglenews RSS API.

### Libraries & Frameworks

6. RAPIDS cuML — GPU-accelerated machine learning (Random Forest, HDBSCAN). rapids.ai.
7. PyTorch — Deep learning framework (LSTM, Transformer). pytorch.org.
8. NVIDIA TensorRT — Inference optimization (Jina v5 INT8 embeddings, Flan-T5-Large INT8 summarization). developer.nvidia.com/tensorrt.
9. Hugging Face Transformers & Hub — Model hosting, tokenization, dataset management. huggingface.co.
10. Polars — High-performance DataFrame library for data processing. pola.rs.
11. Trafilatura — Web content extraction from HTML. trafilatura.readthedocs.io.
12. Scikit-learn — StandardScaler, evaluation metrics. scikit-learn.org.
13. Statsmodels — VAR models, autoregressive benchmarks. statsmodels.org.

### Academic References

14. United Nations Department of Economic and Social Affairs (UNDESA). *International Migration 2020 Highlights.* UN, 2020.
15. Campello, R. J. G. B., Moulavi, D., & Sander, J. "Density-Based Clustering Based on Hierarchical Density Estimates." *PAKDD*, 2013.
16. Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." *Neural Computation*, 9(8), 1997.
17. Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017.
18. Benjamini, Y. & Hochberg, Y. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *JRSS-B*, 57(1), 1995.
19. Breiman, L. "Random Forests." *Machine Learning*, 45(1), 2001.

