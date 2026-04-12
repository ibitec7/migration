# Sentiment Analysis

Rule-based lexicon scoring that classifies news articles on a migration-relevant sentiment scale from −1 (crisis) to +1 (stability).

## Methodology

The `sentiment_score()` function in `src/analysis/events.py`:

1. **Tokenize** article text (headline + body)
2. **Count** matches against positive and negative word lists
3. **Compute** score:

$$
\text{sentiment} = \frac{\text{positive\_count} - \text{negative\_count}}{\text{positive\_count} + \text{negative\_count}}
$$

If both counts are zero, the score is 0 (neutral).

## Lexicons

### Positive Words
`improve`, `growth`, `peace`, `jobs`, `stability` — indicators of improving conditions in origin countries.

### Negative Words
`crisis`, `violence`, `conflict`, `inflation`, `poverty`, `trafficking` — indicators of deteriorating conditions that could trigger migration.

## Design Rationale

A rule-based approach was chosen over ML-based sentiment (e.g., VADER, transformers) because:

- **Domain specificity**: General-purpose sentiment models don't capture migration-relevant nuance
- **Transparency**: Clear word lists are auditable and debuggable
- **Speed**: No model inference overhead — scales to 100K+ articles trivially
- **Interpretability**: Each score can be traced to specific word matches

## Role in the Pipeline

Sentiment scores are computed during [NLP Enrichment](pipeline/nlp-enrichment) and used in:

1. **Monthly aggregation**: Average sentiment per country-cluster-month
2. **Lead-lag analysis**: Sentiment as a predictor of [visa issuances](data-sources/visa-data) at 0–6 month lags
3. **Panel features**: Sentiment enters [Panel Construction](pipeline/panel-construction) alongside event counts and exchange rates

## Key Findings

Sentiment-based signals showed some of the strongest correlations:
- Cuba Trump policy coverage: r = −0.595 (negative sentiment → increased migration)
- See [Event-Visa Findings](findings/event-visa-findings) for full results

## Limitations

- **Bag-of-words**: Ignores context, negation, sarcasm
- **English-only**: Lexicon designed for English-language news
- **Small lexicon**: Only ~11 words — may miss nuanced signals
- **No weighting**: All positive/negative words treated equally

## See Also

- [NLP Enrichment](pipeline/nlp-enrichment) — Pipeline context
- [Event Clustering](analysis/event-clustering) — How articles are grouped before sentiment scoring
- [Lead-Lag Analysis](analysis/lead-lag-analysis) — How sentiment feeds into correlation analysis
- [Event-Visa Findings](findings/event-visa-findings) — What sentiment analysis revealed
- [Glossary](glossary) — Term definitions
