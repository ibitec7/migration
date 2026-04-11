---
title: Multiple Comparison Correction
aliases: [Benjamini-Hochberg, FDR, False Discovery Rate]
tags: [analysis, statistics, fdr, correction]
created: 2026-04-12
---

# Multiple Comparison Correction

**Benjamini-Hochberg (BH) procedure** for controlling the False Discovery Rate (FDR) when performing many simultaneous hypothesis tests across country-signal-lag combinations.

## The Problem

The [[lead-lag-analysis]] tests hundreds of (country, signal, lag) combinations:
- 15 countries × multiple event clusters × 7 lag offsets = hundreds of tests
- At α = 0.05, we'd expect ~5% false positives by chance
- Without correction, many "significant" results would be spurious

## The Solution: Benjamini-Hochberg

The BH procedure controls the **expected proportion of false discoveries** among all rejected hypotheses:

1. Sort all p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}$
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \cdot \alpha$
3. Reject all hypotheses with $p_{(i)} \leq p_{(k)}$

Where $m$ = total number of tests and $\alpha$ = desired FDR level (typically 0.05).

## Application in This Project

### Event-Visa Analysis
- **Tests performed**: Hundreds (15 countries × many clusters × 7 lags)
- **Significant signals after BH correction**: 58
- **Key survivors**: Dominican Republic events (r = 0.617), Cuba policy sentiment (r = −0.595)

### Exchange Rate Analysis
- **Tests performed**: 6 countries × 7 lags = 42
- **Significant after correction**: Dominican Republic (p = 2.6e-06), Mexico (p = 0.0175)

### Google Trends Analysis
- 55% of keyword-country pairs showed pre-correction significance
- Weaker results after correction, motivating the "complementary signal" conclusion

## Implementation

`benjamini_hochberg()` function in `src/analysis/events.py` and `src/analysis/utils.py`:

```python
def benjamini_hochberg(p_values, alpha=0.05):
    # Returns boolean mask of which hypotheses survive correction
```

## See Also

- [[lead-lag-analysis]] — Where the corrections are applied
- [[event-visa-findings]] — Results after correction
- [[exchange-rate-findings]] — Exchange rate results
- [[cross-correlation]] — Trends analysis results
- [[glossary]] — FDR definition
