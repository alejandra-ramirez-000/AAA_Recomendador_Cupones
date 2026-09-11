"""
Modelo 3 — filtrado colaborativo con feedback implícito, vía ALS
(Alternating Least Squares, Hu-Koren-Volinsky 2008).

Aprende un embedding latente de k dimensiones para cada cliente y cada producto
a partir de la matriz cliente x producto de CONFIANZA implícita (no compra/no-compra
binaria: cuánto más veces compró un cliente un producto, más confianza tiene esa
observación). A diferencia del Modelo 2 (que mide relaciones explícitas entre PARES
de productos vía co-ocurrencia/lift), ALS aprende patrones latentes compartidos entre
CLIENTES parecidos — es la implementación más directa de "filtrado colaborativo" tal
como se define en el proyecto (patrones compartidos entre clientes).

confianza(cliente, producto) = 1 + ALS_ALPHA * nº_canastas_del_historial_con_ese_producto
    (formulación estándar de Hu, Koren & Volinsky 2008 para feedback implícito)

Se ajusta el modelo UNA sola vez y se generan dos rankings distintos con el mismo
embedding:
    - Explore: model.recommend(..., filter_already_liked_items=True)
               (excluye productos que el cliente ya compró; usa la API nativa de
               la librería, optimizada en C)
    - Repeat:  score de ALS restringido SOLO a los productos que el cliente ya
               compró (dot product user_factor · item_factor), para ver si el
               embedding latente ordena mejor la recompra que frecuencia+recencia.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # evita el warning/degradación de perf de implicit

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

# Valores elegidos por grid search en VAL (ver 14_tune_als.py / reports/als_tuning.md):
# menos factores y menos alpha consistentemente ganan — señal de que con ~2.500
# clientes y un catálogo de ~91k productos (densidad de interacción ~0.6%), un
# embedding más chico generaliza mejor que uno grande (overfitting con más factores).
ALS_FACTORS = 20
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS = 15
ALS_ALPHA = 15.0
ALS_RANDOM_STATE = 42
N_CANDIDATES = 20
_INVALID_SCORE_THRESHOLD = -1e30  # implicit rellena slots sin candidatos con un score centinela ~ -3.4e38


def _build_confidence_matrix(history_long: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.Series, pd.Series]:
    household_ids = history_long["household_key"].unique()
    household_index = pd.Series(np.arange(len(household_ids)), index=household_ids)

    product_ids = history_long["product_id"].unique()
    item_index = pd.Series(np.arange(len(product_ids)), index=product_ids)

    counts = (
        history_long.groupby(["household_key", "product_id"])["basket_id"]
        .nunique()
        .rename("n_baskets_with_product")
        .reset_index()
    )

    rows = household_index.loc[counts["household_key"]].to_numpy()
    cols = item_index.loc[counts["product_id"]].to_numpy()
    confidence = (1.0 + ALS_ALPHA * counts["n_baskets_with_product"]).to_numpy(dtype=np.float32)

    matrix = sparse.csr_matrix(
        (confidence, (rows, cols)), shape=(len(household_ids), len(product_ids))
    )
    return matrix, household_index, item_index


def fit_als(confidence_matrix: sparse.csr_matrix) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=ALS_RANDOM_STATE,
    )
    model.fit(confidence_matrix)
    return model


def compute_als_scores(history_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta ALS una sola vez y devuelve (repeat_df, explore_df), cada una con
    columnas household_key, product_id, score, rank (top N_CANDIDATES por hogar)."""
    confidence_matrix, household_index, item_index = _build_confidence_matrix(history_long)
    print(
        f"  Ajustando ALS: {confidence_matrix.shape[0]:,} hogares x {confidence_matrix.shape[1]:,} "
        f"productos, {confidence_matrix.nnz:,} interacciones no nulas..."
    )
    model = fit_als(confidence_matrix)

    inv_item = np.empty(len(item_index), dtype=object)
    inv_item[item_index.to_numpy()] = item_index.index.to_numpy()
    inv_household = np.empty(len(household_index), dtype=object)
    inv_household[household_index.to_numpy()] = household_index.index.to_numpy()

    # --- Explore: recomendaciones excluyendo lo ya comprado (API nativa de implicit) ---
    user_ids = np.arange(confidence_matrix.shape[0])
    explore_ids, explore_scores = model.recommend(
        user_ids, confidence_matrix[user_ids], N=N_CANDIDATES, filter_already_liked_items=True
    )

    explore_rows = []
    for i in range(len(user_ids)):
        household_key = inv_household[i]
        for rank in range(explore_ids.shape[1]):
            score_val = explore_scores[i, rank]
            if score_val <= _INVALID_SCORE_THRESHOLD:
                continue
            col = explore_ids[i, rank]
            explore_rows.append((household_key, inv_item[col], float(score_val), rank + 1))
    explore_df = pd.DataFrame(explore_rows, columns=["household_key", "product_id", "score", "rank"])

    # --- Repeat: rankear SOLO los productos que el cliente ya compró, por score de ALS ---
    repeat_rows = []
    for row_i in range(confidence_matrix.shape[0]):
        household_key = inv_household[row_i]
        start, end = confidence_matrix.indptr[row_i], confidence_matrix.indptr[row_i + 1]
        owned_cols = confidence_matrix.indices[start:end]
        if len(owned_cols) == 0:
            continue
        scores = model.user_factors[row_i] @ model.item_factors[owned_cols].T
        order = np.argsort(-scores)[:N_CANDIDATES]
        for rank, idx in enumerate(order, start=1):
            col = owned_cols[idx]
            repeat_rows.append((household_key, inv_item[col], float(scores[idx]), rank))
    repeat_df = pd.DataFrame(repeat_rows, columns=["household_key", "product_id", "score", "rank"])

    return repeat_df, explore_df
