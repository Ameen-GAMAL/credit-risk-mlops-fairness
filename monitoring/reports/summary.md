## Drift & Fairness-Drift Monitoring

Model run: `0258d4b3d2b7443798c2d1eb38cdccb9` | fairness-drift threshold: 0.15

| Batch | Data drift | Drifted cols | Prediction drift | Fairness drift | ALARM |
|---|---|---|---|---|---|
| batch_01 | False | 0 | False | False | ✅ |
| batch_02 | False | 0 | False | False | ✅ |
| batch_03 | False | 1 | False | False | ✅ |
| batch_04 | True | 3 | False | False | 🚨 |
| batch_05 | True | 3 | False | True | 🚨 |
