# Practical Significance Summary

- Total tests: 684
- BH-significant: 503 (73.5%)
- Primary claims (BH-sig + mean≥0.5 + pct≥10.0% + IAA≥0.65 or logit audit): **141**
- Exploratory (BH-sig but not primary): **362**

## Sensitivity grid (BH-significant tests only)

|   mean_threshold |   pct_nonzero_threshold |   n_bh_significant |   n_meeting_thresholds |   pct_of_bh_significant |
|-----------------:|------------------------:|-------------------:|-----------------------:|------------------------:|
|              0.2 |                       1 |                503 |                    322 |                    64   |
|              0.2 |                       5 |                503 |                    321 |                    63.8 |
|              0.2 |                      10 |                503 |                    276 |                    54.9 |
|              0.5 |                       1 |                503 |                    212 |                    42.1 |
|              0.5 |                       5 |                503 |                    212 |                    42.1 |
|              0.5 |                      10 |                503 |                    212 |                    42.1 |
|              1   |                       1 |                503 |                    107 |                    21.3 |
|              1   |                       5 |                503 |                    107 |                    21.3 |
|              1   |                      10 |                503 |                    107 |                    21.3 |

## Dimension tier summary

| dimension                     | dimension_raw                 |   n_tests |   n_sig_bh |   n_primary |   n_exploratory |   mean_severity_avg |   pct_nonzero_avg |   iaa_alpha | confidence   | logit_audit_validated   |
|:------------------------------|:------------------------------|----------:|-----------:|------------:|----------------:|--------------------:|------------------:|------------:|:-------------|:------------------------|
| steering asymmetry            | steering_asymmetry            |        36 |         36 |          36 |               0 |              2.294  |             76.05 |    0.685077 | high         | False                   |
| token inflation               | token_inflation               |        36 |         36 |          32 |               4 |              1.0309 |             32.62 |    0.710016 | high         | False                   |
| inverse logit polarity        | inverse_logit_polarity        |        36 |         36 |          29 |               7 |              1.0403 |             38.02 |    0.869636 | high         | True                    |
| factual reversal              | factual_reversal              |        36 |         31 |          16 |              15 |              1.018  |             23.57 |    0.765504 | high         | False                   |
| option label mismatch         | option_label_mismatch         |        36 |         14 |          10 |               4 |              0.2618 |              9.51 |    0.721782 | high         | False                   |
| factual drift                 | factual_drift                 |        36 |         11 |           4 |               7 |              0.1852 |              5.56 |    0.689142 | high         | False                   |
| repetition looping            | repetition_looping            |        36 |         25 |           4 |              21 |              0.357  |              8.26 |    0.840066 | high         | False                   |
| label content contradiction   | label_content_contradiction   |        36 |         13 |           4 |               9 |              0.1754 |              5.06 |    0.677186 | high         | False                   |
| sensitivity refusal           | sensitivity_refusal           |        36 |         35 |           4 |              31 |              0.1932 |              5.86 |    0.660182 | high         | False                   |
| hallucinated specificity      | hallucinated_specificity      |        36 |         25 |           2 |              23 |              0.1638 |              4.61 |    0.665008 | high         | False                   |
| token compression             | token_compression             |        36 |         35 |           0 |              35 |              0.2593 |              9.46 |    0.509126 | low          | False                   |
| safety override resistance    | safety_override_resistance    |        36 |         15 |           0 |              15 |              0.1537 |              3.12 |    0.525652 | low          | False                   |
| neutralization rephrasing     | neutralization_rephrasing     |        36 |         33 |           0 |              33 |              0.3032 |             10.93 |    0.456717 | low          | False                   |
| persona amplification         | persona_amplification         |        36 |         35 |           0 |              35 |              0.6877 |             24.48 |    0.604557 | moderate     | False                   |
| monotonic factual scaling     | monotonic_factual_scaling     |        36 |          5 |           0 |               5 |              0.0288 |              0.86 |    0.495727 | low          | False                   |
| logit text decoupling         | logit_text_decoupling         |        36 |         19 |           0 |              19 |              0.1165 |              2.33 |    0.389418 | low          | False                   |
| implicit assumption injection | implicit_assumption_injection |        36 |         29 |           0 |              29 |              0.1163 |              3.47 |    0.542048 | low          | False                   |
| hedging escalation            | hedging_escalation            |        36 |         34 |           0 |              34 |              0.3485 |             12.07 |    0.497386 | low          | False                   |
| tonal rigidity                | tonal_rigidity                |        36 |         36 |           0 |              36 |              0.6444 |             25.99 |    0.4931   | low          | False                   |