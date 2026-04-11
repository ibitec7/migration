\# US MIGRATION SURGE PREDICTION

An Early Warning System for prediction of migration surges



Group 10

\* Ibrahim (58776310)

\* Haseeb (58791350) 

\* Asadohoo Sameer (58452930)

\* Ayaan (58771398)

\* Pak Hei Ng (58970130, 58650183)



\---



\## PROBLEM STATEMENT

\* \*\*Reactive Crisis Management:\*\* Governments and city planners are consistently forced to react to sudden surges in migration rather than preparing in advance, leading to strained infrastructure and humanitarian bottlenecks.

\* \*\*Complex Catalysts:\*\* The decision to migrate is caused by a combination of geopolitical, economic, and social variables (push and pull factors) that are difficult to monitor and quantify before migration occurs.

\* \*\*A Precision Blindspot:\*\* Traditional forecasting models rely on slow, lagging historical data, successfully predicting 'normal' days. However they fail to forecast sudden extreme spikes that are actually problematic.

\* \*\*The Missing Early Warning System:\*\* There is a need for a predictive system capable of synthesizing real-time intent, economic volatility, and global news sentiment into actionable, multi-horizon forecasts.



\---



\## WORKFLOW



\* \*\*Exploratory Ground Truth Analysis:\*\* Scrape \& analyze Encounters \& Visa issuance ground truth data; understand distributions; guide further work.

\* \*\*NEWS LLM:\*\*

&#x20; \* \*\*Massive Scale:\*\* Scraped 170,754+ news articles. Filtered to 104,333 valid entities (\~61.1% rate), covering 8 years from 2017.

&#x20; \* \*\*Optimizations:\*\* Optimized scraping workflow, TensorRT for LLMs, enables overnight scraping, multiprocessing.

&#x20; \* \*\*Automated Analysis Scripts:\*\* Used automated scripts for multi-modal data analysis.

\* \*\*Multi-Modal Data Ingestion:\*\* Automated pipeline for historical macroeconomic indicators, news, real-time Google Trends.

\* \*\*Advanced Feature Engineering:\*\* Created multi-month lagged features, 1-6 month time horizon.

\* \*\*Predictive Modeling:\*\* Trained varied suite: Random Forests (RF), LSTM, and Transformer architectures.

\* \*\*Horizon-Aware Ensemble Model:\*\* Final ensemble dynamically weights models based on forecast horizon. RF for short term accuracy, transformers for long term surges.



\*\[Image: Shield workflow diagram detailing short term and long term targets]\*



\---



\## DESCRIPTIVE ANALYSIS 



\* We analyzed the seasonality of Visa volume across the timeline.

\* We observed that the volume decreased drastically after 2020, likely because of the pandemic.

\* We further observed that the summer/fall of 2024 has remained the hottest season for visa volume.



\*\[Image: Seasonal Patterns: Monthly Visa Issuances Heatmap from 2017 to 2025]\*



\---



\## DESCRIPTIVE ANALYSIS (CONTINUED)



\* We analyzed the visa volatility to understand which countries are likely to contribute the most to surges.

\* We plotted the most volatile countries that are likely to contribute the most to surges.

\* Based on the data: Cuba, Afghanistan and Pakistan are the most volatile countries that can possibly contribute the most to surges.



\*\[Image: Visa Flow Volatility by Country chart ranking Cuba (2.89), Afghanistan (2.81), Pakistan (2.48), Mexico (2.34), Bangladesh (2.30), India (2.18), China - mainland born (2.12), Vietnam (1.94), Philippines (1.94), Dominican Republic (1.55)]\*



\---



\## DISTRIBUTION ANALYSIS



\* We found that our data closely followed a Zipfian distribution (80-20 rule) for both the Visa volume and surges.

\* We found that the top 20% of countries contributed to 88.3% of the total visa volume.

\* Similarly, the top 20% of countries contributed to 81.7% of the total surges in visas.

\* We used this insight to drive our data collection to focus on these 20% countries.

\* Being able to predict migration surges from these countries would enable us to predict most of the overall surges and volume.



\*\[Image: Concentration of Visa Surges by Country / Distribution of Visa Issuances by Country compared against Theoretical Zipfian (s=1)]\*



\---



\## HIGH-PERFORMANCE DISTRIBUTED NEWS EXTRACTION PIPELINE



\*\*ASYNC TASK QUEUE (Optimized Concurrency)\*\*

\* Config: 4 concurrent workers (Month 06 to Month 12 Active Tasks)



\*\*SYSTEM KEY METRICS \& CAPABILITIES\*\*

Absolute positive:

\* Throughput (600+ articles/minute)

\* Avg. Fetch Time (\~3s)

\* Avg. Decode Time (\~3s)

\* Avg. Extract Time (\~2s)

\* Concurrent Workers (4x)

\* Controlled Rate-Limits (Low 429 errors)

\* Deadlock-free execution (Timeouts everywhere)

\* Stable Peak Memory (200-400MB)



\*\*WORKER PIPELINE FLOW\*\*

1\. \*\*GOOGLE NEWS API FETCH ARTICLES\*\*: Fetch \~1 sec

2\. \*\*BATCHED URL DECODING (Optimize Request Count)\*\*: Process in batches of 20

3\. \*\*FETCH ALL URLS (Controlled)\*\*: Max 12 concurrent requests (Semaphore). Optimized Httpx settings, timeouts, exponential backoff and jitter for rate-limit avoidance.

4\. \[cite\_start]\*\*ACTIVE RATE-LIMIT HANDLING (Intelligent Throttling)\*\*: Throttling with backoff \[5, 10, 20]. Logic to minimize Playwright browser usage (rare fallback).

5\. \*\*PROCESS \& EXTRACT (Parallelized)\*\*: ProcessPoolExecutor. Parallel content extraction, full GIL bypass, 3s extraction timeout, Malformed HTML handling.

6\. \*\*PROCESS \& SAVE MONTHLY JSON\*\*: Save \~2s (async, non-blocking) into Final Artifacts.



\*\*DISTRIBUTED PERFORMANCE TIMELINE (Active System)\*\*

\* Total 12 Month Execution: \~30-40 seconds

\* Per-Month Processing Time: \~10 seconds



\*\*OPTIMIZED SYSTEM MEMORY USAGE (Stable Profile)\*\*

\* Peak Memory (200-400MB)

\* Max 12 responses buffered

\* Parallel trafilatura (4 workers)

\* Efficient memory release



\*\*FINAL DEPLOYMENT CHECKLIST\*\*

(config, retry, code rewrite, syntax, imports, CLI backward compatibility, dir structure, fallback, logging)



\---



\## NEWS ARTICLES



\* Our scrapper extracted the tables, and cleaned the adds and other noise from the article, providing the response in a neat markdown format for the LLMs.

\* We removed all the erroneous responses, which had HTTP codes of 400s or 500s.

\* We used the Jina v5 LLM embedding model to generate embeddings on the news articles to cluster events.

\* We used the HDBSCAN clustering algorithm for generating the clusters on the embeddings.

\* We sampled a few headlines from each cluster and used the Flan T5 Encoder-Decoder model to generate the cluster labels.



\---



\## EVENT CLUSTERS



\* Silhouette Scores for the event clusters by country.

\* Scores >0.5 suggest good quality clusters.

\* Number of unique event clusters by country.



\*\[Image: Silhouette Scores by Country for Event Clusters]\*

\*\[Image: Number of Unique Clusters by Country]\*



\---



\## GOOGLE TRENDS



\* We used Google Trends as an early-intent signal to capture migration interest before official outcomes are published.

\* Our Trends pipeline covered 15 countries and 8 keywords: US Asylum, US Visa, Green Card, US Jobs, Study US, CBP One, USAID, and US Passport.

\* We aligned monthly search signals with two migration outcomes:

&#x20; \* visa issuances

&#x20; \* border encounter counts

\* To quantify predictive structure, we:

&#x20; \* tested same-month and lead-lag relationships (0 to 6 months)

&#x20; \* corrected for multiple comparisons using q-values

&#x20; \* benchmarked out-of-time forecasting performance (baseline AR vs trends-enhanced VAR)



\*\[Image: Cross Correlation Function (CCF) for 'us\_passport', 'us\_visa', and 'cbp\_one' leading encounters in Mexico]\*



\---



\## GOOGLE TRENDS KEY FINDINDS AND PRACTICAL DECISION



\*\[Image: Mexico composite trend vs encounters line chart]\*

\*\[Image: Mexico composite trend vs visas line chart]\*



\---



\## IMF EXCHANGE RATES



\* This time series shows the change in exchange rate and the amount of visas issued, as it can be observed that as the exchange rate spikes, the visa issuances also increase showing a migration pattern to place with a stronger and stable currency.



\---



\## TENSORRT OPTIMIZATIONS FOR LLMS



\* Compiled the LLM models using TensorRT to generate and optimized engine for inference.

\* Overall we achieved 6,000+ Tokens/sec on the Flan T5 Model (sequence length 512) and 20,000+ Tokens/sec on Jina v5 Embedding model (sequence length 8192).

\* \*\*Kernel/Operator Fusion:\*\* TensorRT applied operator fusion, to combine multiple subsequent operations into one kernel.

\* \*\*Layer Fusion:\*\* Combine multiple layer operations into one operations, such as GEMM + Activation or GEMM + Bias + Activation kernel.

\* \*\*CUDA Graphs:\*\* Compiled the compilation graph of the model to avoid kernel launch overhead.

\* \*\*Quantization:\*\* We used int8 quantization for the Flan T5 model and int4 quantization for the Jina v5 embedding model.



\---



\## PREDICTIVE MODELING



\* \*\*Feature Selection:\*\* We used multiple features such as the embeddings on countries to encode geographical information, event/article clusters and embeddings, exchange rates etc. 

\* \*\*Feature Engineering:\*\* We created time time-lagged features to predict surges in advance.

\* \*\*Training Strategy:\*\* We used Out-of-time training strategy, data before 2022 was used to train the models and data after 2022 was used to test the model performance.

\* \*\*Loss Objective (Joint Surge Loss):\*\* Composite alpha balanced loss: Huber Loss (for Visa Volume) + BCE on surge (whether the surge is more than 1 standard deviation).

\* \*\*Model Architectures:\*\* We used a mix of machine learning models: Random Forests, LSTMs, and Transformer Encoder models.

\* \*\*Horizon-Aware Ensemble:\*\* Leveraging the strengths of each model we create an ensemble model that depending on the time horizon being predicted assigns different weightages to different models to get the most optimal result.



\---



\## MODEL METRICS FOR VOLUME PREDICTION (RMSE SCORES)



| Horizon | cuML Random Forest | PyTorch Transformer | PyTorch LSTM |

| :--- | :--- | :--- | :--- |

| Lead 1 | 298.79 | 303.43 | 277.40 |

| Lead 2 | 341.79 | 343.61 | 320.77 |

| Lead 3 | 351.55 | 359.20 | 346.03 |

| Lead 4 | 364.01 | 381.20 | 381.00 |

| Lead 5 | 428.35 | 407.67 | 404.23 |

| Lead 6 | 401.38 | 426.48 | 424.59 |



\---



\## MODEL METRICS (AVERAGE ACROSS 1-6 MONTHS)



| Algorithm | Precision (No Miss Rate) | Recall (No False Alarm Rate) | F1 Score |

| :--- | :--- | :--- | :--- |

| cuML Random Forest | 0.87 | 0.943 | 0.907 |

| LSTM | 0.895 | 0.82 | 0.855 |

| Transformer Encoder | 0.892 | 0.917 | 0.905 |

| Horizon-Aware Ensemble | 0.887 | 0.943 | 0.915 |



\---



\## HORIZON-AWARE ENSEMBLE MODEL



| Horizon | Precision (No False Alarm) | Recall (No Miss Rate) | F1 Score |

| :--- | :--- | :--- | :--- |

| Lead 1 | 0.96 | 0.96 | 0.96 |

| Lead 2 | 0.93 | 0.96 | 0.95 |

| Lead 3 | 0.92 | 0.94 | 0.93 |

| Lead 4 | 0.88 | 0.94 | 0.91 |

| Lead 5 | 0.83 | 0.94 | 0.88 |

| Lead 6 | 0.80 | 0.92 | 0.86 |



\---



\## MODEL BENCHMARKS COMPARISON



\*\[Image: Graph depicting Model Benchmarks Comparison]\*



\---



\## LIMITATIONS AND IMPROVEMENTS



\* The model’s interpretability can be used to understand which features contribute the most to the model’s predictive power, focusing on collecting those features would improve the model performance.

\* Over time, the model’s assumptions may be challenged and make it obsolete easily, factors such as data drift (where our data may no longer follow a zipfian distribution) etc. 

\* Certain unexpected events may occur that can not be predicted by the model, that can lead to a surge in immigration.

\* The ground truth for the illegal immigrants has a survivorship bias (we only know the illegal immigrants who got caught, not the ones who actually made it through).



\## THE END

\* Our code is available at: `https://github.com/ibitec7/migration`

\* Our data and model weights are available at: `https://huggingface.co/sdsc2005-migration`

