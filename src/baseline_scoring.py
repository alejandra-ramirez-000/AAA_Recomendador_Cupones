"""
Funciones de scoring del Modelo 1 (baseline): frecuencia + recencia para Repeat,
popularidad global para Explore.

score_repeat(cliente, producto) = ALPHA * freq_norm + (1 - ALPHA) * recency_score
    freq_norm      = nº de canastas del historial del cliente que incluyen el producto
                      / nº total de canastas del historial del cliente
    recency_score  = exp(-LAMBDA * días_desde_última_compra_del_producto)
                      (días medidos desde el último día del historial permitido, no desde "hoy")

score_explore(producto) = nº de canastas GLOBALES (todos los hogares) que incluyen el producto,
    calculado solo con las canastas del historial permitido (sin usar datos futuros),
    excluyendo para cada cliente los productos que ya compró.

LAMBDA por defecto da una vida media de 30 días (exp(-lambda*30) = 0.5).
"""
import math

import numpy as np
import pandas as pd

ALPHA = 0.5
HALF_LIFE_DAYS = 30
LAMBDA = math.log(2) / HALF_LIFE_DAYS
N_CANDIDATES = 20  # cubre el máximo K que vamos a evaluar (K=20)
EXPLORE_POOL_SIZE = 3000  # candidatos globales más populares antes de filtrar lo ya comprado


def compute_repeat_scores(history_long: pd.DataFrame, alpha: float = ALPHA, lam: float = LAMBDA) -> pd.DataFrame:
    """
    history_long: household_key, basket_id, day, product_id (solo canastas del historial permitido).
    Devuelve top-N_CANDIDATES productos por hogar, ordenados por score descendente.
    """
    n_baskets_total = history_long.groupby("household_key")["basket_id"].nunique().rename("n_baskets_total")
    reference_day = history_long.groupby("household_key")["day"].max().rename("reference_day")

    per_product = (
        history_long.groupby(["household_key", "product_id"])
        .agg(n_baskets_with_product=("basket_id", "nunique"), last_purchase_day=("day", "max"))
        .reset_index()
    )
    per_product = per_product.merge(n_baskets_total, on="household_key")
    per_product = per_product.merge(reference_day, on="household_key")

    per_product["freq_norm"] = per_product["n_baskets_with_product"] / per_product["n_baskets_total"]
    per_product["days_since_last"] = per_product["reference_day"] - per_product["last_purchase_day"]
    per_product["recency_score"] = np.exp(-lam * per_product["days_since_last"])
    per_product["score"] = alpha * per_product["freq_norm"] + (1 - alpha) * per_product["recency_score"]

    per_product["rank"] = per_product.groupby("household_key")["score"].rank(
        method="first", ascending=False
    )
    top = per_product[per_product["rank"] <= N_CANDIDATES].copy()
    top["rank"] = top["rank"].astype(int)
    return top[["household_key", "product_id", "score", "rank"]].sort_values(["household_key", "rank"])


def global_popularity_pool(history_long: pd.DataFrame, pool_size: int = EXPLORE_POOL_SIZE) -> pd.DataFrame:
    """Top-`pool_size` productos más populares (nº de canastas globales distintas), calculado
    SOLO con el historial permitido (sin usar datos futuros). Se reutiliza como fallback en
    otros modelos cuando no alcanzan candidatos propios (ej. co-ocurrencia)."""
    global_pop = (
        history_long.groupby("product_id")["basket_id"]
        .nunique()
        .rename("n_baskets_global")
        .reset_index()
        .sort_values("n_baskets_global", ascending=False)
        .reset_index(drop=True)
    )
    return global_pop.head(pool_size)


def compute_explore_scores(history_long: pd.DataFrame) -> pd.DataFrame:
    """
    Popularidad global (no personalizada) calculada SOLO con el historial permitido,
    excluyendo por hogar los productos que ese hogar ya compró.
    """
    pool = global_popularity_pool(history_long)
    pool_products = pool["product_id"].tolist()
    pool_scores = dict(zip(pool["product_id"], pool["n_baskets_global"]))

    owned_sets = history_long.groupby("household_key")["product_id"].apply(set)

    rows = []
    n_insufficient = 0
    for household_key, owned in owned_sets.items():
        candidates = [p for p in pool_products if p not in owned][:N_CANDIDATES]
        if len(candidates) < N_CANDIDATES:
            n_insufficient += 1
        for rank, product_id in enumerate(candidates, start=1):
            rows.append((household_key, product_id, pool_scores[product_id], rank))

    if n_insufficient:
        print(
            f"[warn] {n_insufficient} hogares se quedaron con menos de {N_CANDIDATES} candidatos Explore "
            f"tras excluir lo ya comprado (subí EXPLORE_POOL_SIZE si esto es frecuente)."
        )

    return pd.DataFrame(rows, columns=["household_key", "product_id", "score", "rank"])
