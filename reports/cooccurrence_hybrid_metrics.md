# Modelo 2b — Co-ocurrencia híbrida (piso de popularidad + lift) para Explore

## val — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | repeat    |   5 |           2033 |      0.1425 |   0.1638 |     0.4471 | 0.2103 |
| val          | repeat    |  10 |           2033 |      0.1058 |   0.2325 |     0.5470 | 0.2165 |
| val          | repeat    |  20 |           2033 |      0.0778 |   0.3263 |     0.6567 | 0.2383 |

## val — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | explore   |   5 |           2088 |      0.0107 |   0.0120 |     0.0517 | 0.0160 |
| val          | explore   |  10 |           2088 |      0.0093 |   0.0161 |     0.0833 | 0.0164 |
| val          | explore   |  20 |           2088 |      0.0074 |   0.0231 |     0.1193 | 0.0180 |

## test — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | repeat    |   5 |           2039 |      0.1543 |   0.1734 |     0.4625 | 0.2287 |
| test         | repeat    |  10 |           2039 |      0.1140 |   0.2345 |     0.5478 | 0.2297 |
| test         | repeat    |  20 |           2039 |      0.0829 |   0.3246 |     0.6542 | 0.2498 |

## test — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | explore   |   5 |           2096 |      0.0090 |   0.0112 |     0.0429 | 0.0136 |
| test         | explore   |  10 |           2096 |      0.0083 |   0.0159 |     0.0744 | 0.0150 |
| test         | explore   |  20 |           2096 |      0.0071 |   0.0236 |     0.1155 | 0.0173 |
