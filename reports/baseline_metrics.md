# Modelo 1 — Baseline (frecuencia+recencia / popularidad global)

Métricas calculadas por separado para Repeat y Explore. `n_households` indica cuántos hogares tenían al menos un producto verdadero de ese segmento en su canasta objetivo (los que no, quedan fuera del promedio).

## val — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | repeat    |   5 |           2033 |      0.1425 |   0.1638 |     0.4471 | 0.2103 |
| val          | repeat    |  10 |           2033 |      0.1058 |   0.2325 |     0.5470 | 0.2165 |
| val          | repeat    |  20 |           2033 |      0.0778 |   0.3263 |     0.6567 | 0.2383 |

## val — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | explore   |   5 |           2088 |      0.0102 |   0.0112 |     0.0484 | 0.0151 |
| val          | explore   |  10 |           2088 |      0.0083 |   0.0151 |     0.0757 | 0.0152 |
| val          | explore   |  20 |           2088 |      0.0070 |   0.0223 |     0.1149 | 0.0172 |

## test — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | repeat    |   5 |           2039 |      0.1543 |   0.1734 |     0.4625 | 0.2287 |
| test         | repeat    |  10 |           2039 |      0.1140 |   0.2345 |     0.5478 | 0.2297 |
| test         | repeat    |  20 |           2039 |      0.0829 |   0.3246 |     0.6542 | 0.2498 |

## test — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | explore   |   5 |           2096 |      0.0092 |   0.0111 |     0.0434 | 0.0140 |
| test         | explore   |  10 |           2096 |      0.0074 |   0.0150 |     0.0668 | 0.0146 |
| test         | explore   |  20 |           2096 |      0.0063 |   0.0224 |     0.1069 | 0.0167 |
