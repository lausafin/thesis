# Pass 1 vs Pass 2 Comparison Summary

## Screening funnel (full run)

- Total corpus: **28,807**
- Pass-1 any category flagged: **16,021** (55.61%)
- Pass-2 evaluated: **16,021** (55.61% of corpus)
- Pass-2 any severity > 0: **14,372** (89.71% of Pass-2 rows)
- Pass-2 all severities = 0: **1,649** (10.29% of Pass-2 rows)

## Highest category→dimension concordance

| subset   | pass1_category        | pass2_dimension          |   n_pass1_flagged |   pct_nonzero |   pct_severity_ge3 |   mean_severity |
|:---------|:----------------------|:-------------------------|------------------:|--------------:|-------------------:|----------------:|
| full     | behavioral_resistance | steering_asymmetry       |              8983 |         77.77 |              51.42 |          2.4221 |
| full     | factual_integrity     | factual_reversal         |               873 |         75.14 |              70.45 |          3.3036 |
| full     | tonal_change          | tonal_rigidity           |              8851 |         47.7  |              16.43 |          1.2074 |
| full     | factual_integrity     | factual_drift            |               873 |         46.51 |              38.83 |          1.6529 |
| full     | factual_integrity     | hallucinated_specificity |               873 |         46.39 |              36.88 |          1.7892 |
| full     | output_shape          | token_inflation          |             11076 |         44.48 |              25.52 |          1.3975 |
| full     | tonal_change          | persona_amplification    |              8851 |         42.51 |              22.69 |          1.2411 |
| full     | behavioral_resistance | inverse_logit_polarity   |              8983 |         41.12 |              17.56 |          1.1385 |

## Lowest concordance (among nested pairs)

| subset   | pass1_category        | pass2_dimension               |   n_pass1_flagged |   pct_nonzero |   pct_severity_ge3 |   mean_severity |
|:---------|:----------------------|:------------------------------|------------------:|--------------:|-------------------:|----------------:|
| full     | behavioral_resistance | logit_text_decoupling         |              8983 |          3.25 |               3.25 |          0.1624 |
| full     | output_shape          | implicit_assumption_injection |             11076 |          3.37 |               2    |          0.1145 |
| full     | behavioral_resistance | safety_override_resistance    |              8983 |          6.21 |               6.16 |          0.3069 |
| full     | behavioral_resistance | sensitivity_refusal           |              8983 |         10.19 |               6.07 |          0.3326 |
| full     | factual_integrity     | monotonic_factual_scaling     |               873 |         10.88 |               8.02 |          0.3597 |

## Category co-occurrence vs Pass-2 severity

| subset   |   n_categories_flagged |   n_samples |   pct_of_pass2 |   pct_any_pass2_nonzero |   mean_max_severity |   mean_avg_severity |
|:---------|-----------------------:|------------:|---------------:|------------------------:|--------------------:|--------------------:|
| full     |                      1 |        5971 |          37.27 |                   77.07 |               2.38  |               0.225 |
| full     |                      2 |        4489 |          28.02 |                   94.54 |               3.167 |               0.441 |
| full     |                      3 |        3585 |          22.38 |                   99.11 |               3.861 |               0.699 |
| full     |                      4 |        1848 |          11.53 |                   99.89 |               4.365 |               0.886 |
| full     |                      5 |         128 |           0.8  |                   99.22 |               4.648 |               1.467 |

## Caveats

- Pass 2 runs only on Pass-1-flagged samples; false-negative rate of screening is unknown.
- Pass 1 category flags are injected into the Pass 2 prompt (`flagged_categories`).