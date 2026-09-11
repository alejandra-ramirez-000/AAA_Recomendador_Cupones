"""
Modelo 2 — scoring Explore vía co-ocurrencia (lift) entre productos.

lift(A,B) = P(A y B en la misma canasta) / (P(A) * P(B))
          = C(A,B) * n_canastas_totales / (n_A * n_B)

Se descartan pares con pocas co-ocurrencias (ruido estadístico) o con lift <= 1
(asociación nula o negativa). Para cada cliente:

    score(cliente, candidato) = sum_{p en historial del cliente} freq_norm(cliente, p) * lift(p, candidato)

es decir, una suma de las asociaciones de cada candidato con los productos que el
cliente ya compra, ponderada por qué tan seguido compra cada uno de esos productos.
Los candidatos ya comprados por el cliente se excluyen. Si a algún cliente no le
alcanzan N_CANDIDATES por co-ocurrencia (ej. historial muy chico o aislado), se
completa con el mismo fallback de popularidad global que usa el Modelo 1 baseline,
para poder evaluar siempre a K=20.

El Repeat de este modelo NO cambia respecto al Modelo 1 (sigue siendo
frecuencia+recencia) — acá solo mejoramos el scoring de Explore.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from baseline_scoring import EXPLORE_POOL_SIZE, N_CANDIDATES, global_popularity_pool

MIN_COOCCURRENCE = 5  # mínimo nº de canastas donde A y B aparecen juntos, para confiar en el lift
MIN_LIFT = 1.0  # solo asociaciones positivas (lift > 1)


def _build_item_index(history_long: pd.DataFrame) -> pd.Series:
    """product_id -> índice de columna (posición fija, arbitraria)."""
    products = history_long["product_id"].unique()
    return pd.Series(np.arange(len(products)), index=products)


def _build_basket_item_matrix(history_long: pd.DataFrame, item_index: pd.Series) -> sparse.csr_matrix:
    basket_ids = history_long["basket_id"].unique()
    basket_index = pd.Series(np.arange(len(basket_ids)), index=basket_ids)

    rows = basket_index.loc[history_long["basket_id"]].to_numpy()
    cols = item_index.loc[history_long["product_id"]].to_numpy()
    data = np.ones(len(history_long), dtype=np.float32)

    mat = sparse.coo_matrix((data, (rows, cols)), shape=(len(basket_ids), len(item_index))).tocsr()
    mat.data[:] = 1.0  # por si un producto aparece más de una vez en la misma canasta (líneas duplicadas)
    return mat


def compute_lift_matrix(
    basket_item_matrix: sparse.csr_matrix, candidate_top_fraction: float | None = None
) -> sparse.csr_matrix:
    """Matriz sparse item x item con el lift de cada par que pasa los filtros de ruido.

    Si `candidate_top_fraction` se especifica (ej. 0.30), el producto CANDIDATO (columna)
    de cada par además tiene que estar entre el top-`candidate_top_fraction` más populares
    globalmente — variante híbrida para no perder el piso que da la popularidad pura, ya
    que el lift por sí solo tiende a des-priorizar justo a los productos más comprados
    (su P(B) alto en el denominador aplana su lift hacia 1 aunque en términos absolutos
    sean los que más probablemente aparezcan en cualquier canasta futura).
    """
    n_baskets = basket_item_matrix.shape[0]
    item_counts = np.asarray(basket_item_matrix.sum(axis=0)).ravel()  # n_A por item

    co_counts = (basket_item_matrix.T @ basket_item_matrix).tocoo()
    off_diag = co_counts.row != co_counts.col
    rows, cols, counts = co_counts.row[off_diag], co_counts.col[off_diag], co_counts.data[off_diag]

    lift_vals = n_baskets * counts / (item_counts[rows] * item_counts[cols])

    keep = (counts >= MIN_COOCCURRENCE) & (lift_vals > MIN_LIFT)
    if candidate_top_fraction is not None:
        popularity_threshold = np.percentile(item_counts, (1 - candidate_top_fraction) * 100)
        keep &= item_counts[cols] >= popularity_threshold

    n_items = len(item_counts)
    return sparse.csr_matrix((lift_vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items))


def _build_affinity_matrix(history_long: pd.DataFrame, item_index: pd.Series) -> tuple[sparse.csr_matrix, pd.Series]:
    """Matriz sparse cliente x item con freq_norm(cliente, producto) en cada producto que compró."""
    household_ids = history_long["household_key"].unique()
    household_index = pd.Series(np.arange(len(household_ids)), index=household_ids)

    n_baskets_total = history_long.groupby("household_key")["basket_id"].nunique()
    per_product = (
        history_long.groupby(["household_key", "product_id"])["basket_id"]
        .nunique()
        .rename("n_baskets_with_product")
        .reset_index()
    )
    per_product["freq_norm"] = per_product["n_baskets_with_product"] / per_product["household_key"].map(
        n_baskets_total
    )

    rows = household_index.loc[per_product["household_key"]].to_numpy()
    cols = item_index.loc[per_product["product_id"]].to_numpy()
    data = per_product["freq_norm"].to_numpy(dtype=np.float32)

    W = sparse.csr_matrix((data, (rows, cols)), shape=(len(household_ids), len(item_index)))
    return W, household_index


def compute_explore_scores_cooccurrence(
    history_long: pd.DataFrame, candidate_top_fraction: float | None = None
) -> pd.DataFrame:
    """candidate_top_fraction=None -> lift puro (Modelo 2). Un valor ej. 0.30 -> variante
    híbrida que solo deja competir por lift a candidatos entre los más populares globalmente."""
    item_index = _build_item_index(history_long)
    inv_item = np.empty(len(item_index), dtype=object)
    inv_item[item_index.to_numpy()] = item_index.index.to_numpy()

    basket_item_matrix = _build_basket_item_matrix(history_long, item_index)
    lift = compute_lift_matrix(basket_item_matrix, candidate_top_fraction=candidate_top_fraction)

    W, household_index = _build_affinity_matrix(history_long, item_index)
    inv_household = np.empty(len(household_index), dtype=object)
    inv_household[household_index.to_numpy()] = household_index.index.to_numpy()

    scores = (W @ lift).tocsr()  # n_hogares x n_items, score de co-ocurrencia agregado

    owned_sets = history_long.groupby("household_key")["product_id"].apply(set)
    pop_pool = global_popularity_pool(history_long, EXPLORE_POOL_SIZE)
    pop_pool_list = list(zip(pop_pool["product_id"], pop_pool["n_baskets_global"]))

    rows_out = []
    n_used_fallback = 0
    for row_i in range(scores.shape[0]):
        household_key = inv_household[row_i]
        owned = owned_sets.get(household_key, set())

        start, end = scores.indptr[row_i], scores.indptr[row_i + 1]
        cols = scores.indices[start:end]
        vals = scores.data[start:end]
        order = np.argsort(-vals)

        ranked: list[tuple] = []
        seen = set()
        for idx in order:
            product_id = inv_item[cols[idx]]
            if product_id in owned or product_id in seen:
                continue
            ranked.append((product_id, float(vals[idx])))
            seen.add(product_id)
            if len(ranked) == N_CANDIDATES:
                break

        if len(ranked) < N_CANDIDATES:
            n_used_fallback += 1
            for product_id, pop_count in pop_pool_list:
                if product_id in owned or product_id in seen:
                    continue
                ranked.append((product_id, float(pop_count)))
                seen.add(product_id)
                if len(ranked) == N_CANDIDATES:
                    break

        for rank, (product_id, score_val) in enumerate(ranked, start=1):
            rows_out.append((household_key, product_id, score_val, rank))

    if n_used_fallback:
        print(
            f"[info] {n_used_fallback} hogares completaron su top-{N_CANDIDATES} de Explore con el "
            f"fallback de popularidad global (co-ocurrencia insuficiente)."
        )

    return pd.DataFrame(rows_out, columns=["household_key", "product_id", "score", "rank"])


def compute_explore_scores_cooccurrence_blended(
    history_long: pd.DataFrame,
    w_lift: float = 0.5,
    w_pop: float = 0.5,
    pop_pool_size: int = EXPLORE_POOL_SIZE,
) -> pd.DataFrame:
    """
    Modelo 2b (corregido) — combina lift normalizado y popularidad normalizada:

        combined(cliente, candidato) = w_lift * lift_norm(cliente, candidato)
                                      + w_pop  * pop_norm(candidato)

    A diferencia de filtrar candidatos por popularidad (que NO alcanza — ver docstring
    de compute_lift_matrix), acá sumamos explícitamente el componente de popularidad
    incluso cuando el lift es 0/ausente, que es justamente lo que pasa con los
    productos más populares (su lift con la mayoría de los productos puntuales de un
    cliente es ≈1, así que MIN_LIFT los descarta del todo del lado de co-ocurrencia).

    lift_norm  = lift_score(cliente,candidato) / max(lift_score(cliente, *)) — normalizado
                 por fila (cada cliente tiene su propia escala de scores).
    pop_norm   = log1p(popularidad_global(candidato)) / log1p(popularidad_global(max))
                 — log para no dejar que el ítem más vendido opaque a todo el resto.
    """
    item_index = _build_item_index(history_long)
    inv_item = np.empty(len(item_index), dtype=object)
    inv_item[item_index.to_numpy()] = item_index.index.to_numpy()

    basket_item_matrix = _build_basket_item_matrix(history_long, item_index)
    lift = compute_lift_matrix(basket_item_matrix)  # matriz completa, sin filtro de popularidad

    item_counts = np.asarray(basket_item_matrix.sum(axis=0)).ravel()
    pop_norm_by_col = np.log1p(item_counts) / np.log1p(item_counts.max())

    W, household_index = _build_affinity_matrix(history_long, item_index)
    inv_household = np.empty(len(household_index), dtype=object)
    inv_household[household_index.to_numpy()] = household_index.index.to_numpy()

    scores = (W @ lift).tocsr()

    owned_sets = history_long.groupby("household_key")["product_id"].apply(set)
    pop_pool = global_popularity_pool(history_long, pop_pool_size)
    pool_cols = item_index.reindex(pop_pool["product_id"]).dropna().astype(int).to_numpy()

    rows_out = []
    for row_i in range(scores.shape[0]):
        household_key = inv_household[row_i]
        owned = owned_sets.get(household_key, set())

        start, end = scores.indptr[row_i], scores.indptr[row_i + 1]
        lift_cols = scores.indices[start:end]
        lift_vals = scores.data[start:end]
        max_lift = lift_vals.max() if len(lift_vals) else 0.0
        lift_norm_map = (
            dict(zip(lift_cols.tolist(), (lift_vals / max_lift).tolist())) if max_lift > 0 else {}
        )

        candidate_cols = np.union1d(lift_cols, pool_cols)

        combined = []
        for col in candidate_cols:
            product_id = inv_item[col]
            if product_id in owned:
                continue
            combined_score = w_lift * lift_norm_map.get(col, 0.0) + w_pop * pop_norm_by_col[col]
            combined.append((product_id, combined_score))

        combined.sort(key=lambda x: -x[1])

        for rank, (product_id, score_val) in enumerate(combined[:N_CANDIDATES], start=1):
            rows_out.append((household_key, product_id, score_val, rank))

    return pd.DataFrame(rows_out, columns=["household_key", "product_id", "score", "rank"])


def prepare_blend_components(history_long: pd.DataFrame) -> list[tuple]:
    """
    Precalcula, por hogar, los candidatos (ya excluyendo lo comprado) con su
    lift_norm y pop_norm alineados como arrays numpy — la parte cara del cómputo
    (matriz de lift, W @ lift) se hace UNA sola vez acá, independiente del peso.
    Devuelve una lista de tuplas (household_key, candidate_products, lift_arr, pop_arr)
    lista para combinar con distintos (w_lift, w_pop) vía rank_from_blend_components,
    sin repetir el trabajo pesado — pensado para un grid search de pesos.
    """
    item_index = _build_item_index(history_long)
    inv_item = np.empty(len(item_index), dtype=object)
    inv_item[item_index.to_numpy()] = item_index.index.to_numpy()

    basket_item_matrix = _build_basket_item_matrix(history_long, item_index)
    lift = compute_lift_matrix(basket_item_matrix)

    item_counts = np.asarray(basket_item_matrix.sum(axis=0)).ravel()
    pop_norm_by_col = np.log1p(item_counts) / np.log1p(item_counts.max())

    W, household_index = _build_affinity_matrix(history_long, item_index)
    inv_household = np.empty(len(household_index), dtype=object)
    inv_household[household_index.to_numpy()] = household_index.index.to_numpy()

    scores = (W @ lift).tocsr()

    owned_sets = history_long.groupby("household_key")["product_id"].apply(set)
    pop_pool = global_popularity_pool(history_long, EXPLORE_POOL_SIZE)
    pool_cols = item_index.reindex(pop_pool["product_id"]).dropna().astype(int).to_numpy()

    per_household = []
    for row_i in range(scores.shape[0]):
        household_key = inv_household[row_i]
        owned = owned_sets.get(household_key, set())

        start, end = scores.indptr[row_i], scores.indptr[row_i + 1]
        lift_cols = scores.indices[start:end]
        lift_vals = scores.data[start:end]
        max_lift = lift_vals.max() if len(lift_vals) else 0.0
        lift_norm_map = (
            dict(zip(lift_cols.tolist(), (lift_vals / max_lift).tolist())) if max_lift > 0 else {}
        )

        candidate_cols = np.union1d(lift_cols, pool_cols)
        candidate_products = inv_item[candidate_cols]
        owned_mask = np.array([p in owned for p in candidate_products])
        candidate_cols = candidate_cols[~owned_mask]
        candidate_products = candidate_products[~owned_mask]

        lift_arr = np.array([lift_norm_map.get(c, 0.0) for c in candidate_cols])
        pop_arr = pop_norm_by_col[candidate_cols]

        per_household.append((household_key, candidate_products, lift_arr, pop_arr))

    return per_household


def rank_from_blend_components(
    per_household: list[tuple], w_lift: float, w_pop: float, n_candidates: int = N_CANDIDATES
) -> pd.DataFrame:
    """Combina los componentes precalculados por prepare_blend_components con un peso dado."""
    rows_out = []
    for household_key, candidate_products, lift_arr, pop_arr in per_household:
        combined = w_lift * lift_arr + w_pop * pop_arr
        order = np.argsort(-combined)[:n_candidates]
        for rank, idx in enumerate(order, start=1):
            rows_out.append((household_key, candidate_products[idx], float(combined[idx]), rank))
    return pd.DataFrame(rows_out, columns=["household_key", "product_id", "score", "rank"])
