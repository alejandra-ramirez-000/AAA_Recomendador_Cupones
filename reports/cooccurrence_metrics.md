# Modelo 2 — Co-ocurrencia (lift) para Explore

Repeat sin cambios respecto al Modelo 1 (frecuencia+recencia). Explore ahora usa lift entre productos en vez de popularidad global. `n_households` = hogares con al menos un producto verdadero de ese segmento en su canasta objetivo.

## val — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | repeat    |   5 |           2033 |      0.1425 |   0.1638 |     0.4471 | 0.2103 |
| val          | repeat    |  10 |           2033 |      0.1058 |   0.2325 |     0.5470 | 0.2165 |
| val          | repeat    |  20 |           2033 |      0.0778 |   0.3263 |     0.6567 | 0.2383 |

## val — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| val          | explore   |   5 |           2088 |      0.0020 |   0.0022 |     0.0096 | 0.0027 |
| val          | explore   |  10 |           2088 |      0.0020 |   0.0038 |     0.0172 | 0.0032 |
| val          | explore   |  20 |           2088 |      0.0018 |   0.0061 |     0.0278 | 0.0041 |

## test — repeat

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | repeat    |   5 |           2039 |      0.1543 |   0.1734 |     0.4625 | 0.2287 |
| test         | repeat    |  10 |           2039 |      0.1140 |   0.2345 |     0.5478 | 0.2297 |
| test         | repeat    |  20 |           2039 |      0.0829 |   0.3246 |     0.6542 | 0.2498 |

## test — explore

| eval_split   | segment   |   k |   n_households |   precision |   recall |   hit_rate |   ndcg |
|:-------------|:----------|----:|---------------:|------------:|---------:|-----------:|-------:|
| test         | explore   |   5 |           2096 |      0.0024 |   0.0012 |     0.0095 | 0.0025 |
| test         | explore   |  10 |           2096 |      0.0025 |   0.0033 |     0.0200 | 0.0031 |
| test         | explore   |  20 |           2096 |      0.0021 |   0.0059 |     0.0344 | 0.0039 |
