# Modelo 3 — Filtrado colaborativo (ALS)

## val — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | repeat    |   5 |           2033 |      0.0670 |   0.0944 |     0.2632 | 0.0991 |
| val          | repeat    |  10 |           2033 |      0.0576 |   0.1556 |     0.3856 | 0.1173 |
| val          | repeat    |  20 |           2033 |      0.0470 |   0.2299 |     0.5121 | 0.1407 |

## val — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | explore   |   5 |           2088 |      0.0106 |   0.0102 |     0.0508 | 0.0134 |
| val          | explore   |  10 |           2088 |      0.0085 |   0.0153 |     0.0747 | 0.0140 |
| val          | explore   |  20 |           2088 |      0.0073 |   0.0251 |     0.1183 | 0.0166 |

## test — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | repeat    |   5 |           2039 |      0.0656 |   0.0918 |     0.2590 | 0.0960 |
| test         | repeat    |  10 |           2039 |      0.0552 |   0.1466 |     0.3713 | 0.1119 |
| test         | repeat    |  20 |           2039 |      0.0460 |   0.2204 |     0.5061 | 0.1358 |

## test — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | explore   |   5 |           2096 |      0.0092 |   0.0097 |     0.0429 | 0.0119 |
| test         | explore   |  10 |           2096 |      0.0085 |   0.0166 |     0.0735 | 0.0142 |
| test         | explore   |  20 |           2096 |      0.0072 |   0.0253 |     0.1174 | 0.0169 |
