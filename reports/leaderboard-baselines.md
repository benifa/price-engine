# Leaderboard — `items_lite_baselines`

| Contestant | n | MAE | Median APE | Hit rate | RMSLE |
|---|---:|---:|---:|---:|---:|
| category-median | 1000 | $83.87  (95% CI $76.07–$91.63) | 52.8% | 46.1% | 0.899 |
| global-median | 1000 | $97.19  (95% CI $88.95–$105.30) | 66.7% | 31.7% | 1.028 |

## Protocol

- Hit = absolute error < $40 **or** relative error < 20% (Ed's definition).
- Fine-tuned contestants are graded with their **training-native** prompt format.
- MAE CI is a 95% bootstrap over items; paired ΔMAE CI is a paired bootstrap.
