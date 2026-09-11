"""
Grid search chico de hiperparámetros de ALS (factors, regularization, alpha),
evaluado SOLO en 'val'. Cada combinación se ajusta desde cero (el fit de ALS es
rápido, ~5s), así que un grid moderado es barato.

Corré desde la raíz del proyecto (después de 03_build_targets.py):
    python src/14_tune_als.py

Genera:
    reports/als_tuning.md
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools

import pandas as pd
from implicit.als import AlternatingLeastSquares

import als_scoring
from als_scoring import _build_confidence_matrix
from basket_utils import exploded_long
from evaluation import evaluate
from paths import DATA_PROCESSED_DIR, REPORTS_DIR

FACTORS_GRID = [20, 50, 100]
REGULARIZATION_GRID = [0.01, 0.1]
ALPHA_GRID = [15, 40]
ITERATIONS = 15
RANDOM_STATE = 42
K_FOR_SELECTION = 20


def score_with_params(history_long: pd.DataFrame, factors: int, regularization: float, alpha: float):
    # Reusa la construcción de la matriz de confianza tal cual está en als_scoring,
    # pero con ALPHA del grid en vez de la constante del módulo.
    als_scoring.ALS_ALPHA = alpha
    confidence_matrix, household_index, item_index = _build_confidence_matrix(history_long)

    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, iterations=ITERATIONS, random_state=RANDOM_STATE
    )
    model.fit(confidence_matrix)

    import numpy as np

    inv_item = np.empty(len(item_index), dtype=object)
    inv_item[item_index.to_numpy()] = item_index.index.to_numpy()
    inv_household = np.empty(len(household_index), dtype=object)
    inv_household[household_index.to_numpy()] = household_index.index.to_numpy()

    user_ids = np.arange(confidence_matrix.shape[0])
    explore_ids, explore_scores = model.recommend(
        user_ids, confidence_matrix[user_ids], N=als_scoring.N_CANDIDATES, filter_already_liked_items=True
    )
    explore_rows = []
    for i in range(len(user_ids)):
        household_key = inv_household[i]
        for rank in range(explore_ids.shape[1]):
            score_val = explore_scores[i, rank]
            if score_val <= als_scoring._INVALID_SCORE_THRESHOLD:
                continue
            explore_rows.append((household_key, inv_item[explore_ids[i, rank]], float(score_val), rank + 1))
    explore_df = pd.DataFrame(explore_rows, columns=["household_key", "product_id", "score", "rank"])

    repeat_rows = []
    for row_i in range(confidence_matrix.shape[0]):
        household_key = inv_household[row_i]
        start, end = confidence_matrix.indptr[row_i], confidence_matrix.indptr[row_i + 1]
        owned_cols = confidence_matrix.indices[start:end]
        if len(owned_cols) == 0:
            continue
        scores = model.user_factors[row_i] @ model.item_factors[owned_cols].T
        order = np.argsort(-scores)[: als_scoring.N_CANDIDATES]
        for rank, idx in enumerate(order, start=1):
            repeat_rows.append((household_key, inv_item[owned_cols[idx]], float(scores[idx]), rank))
    repeat_df = pd.DataFrame(repeat_rows, columns=["household_key", "product_id", "score", "rank"])

    return repeat_df, explore_df


def main() -> None:
    baskets = pd.read_parquet(DATA_PROCESSED_DIR / "baskets.parquet")
    baskets = baskets[baskets["split"] != "excluded"]
    long = exploded_long(baskets)
    history_long = long[long["split"] == "train"]

    targets = pd.read_parquet(DATA_PROCESSED_DIR / "targets.parquet")

    rows = []
    for factors, regularization, alpha in itertools.product(FACTORS_GRID, REGULARIZATION_GRID, ALPHA_GRID):
        repeat_df, explore_df = score_with_params(history_long, factors, regularization, alpha)
        repeat_df["eval_split"] = "val"
        repeat_df["segment"] = "repeat"
        explore_df["eval_split"] = "val"
        explore_df["segment"] = "explore"
        preds = pd.concat([repeat_df, explore_df], ignore_index=True)

        repeat_result = evaluate(preds, targets, "val", "repeat", ks=(K_FOR_SELECTION,))
        explore_result = evaluate(preds, targets, "val", "explore", ks=(K_FOR_SELECTION,))

        repeat_ndcg = repeat_result.iloc[0]["ndcg"]
        explore_ndcg = explore_result.iloc[0]["ndcg"]
        avg_ndcg = (repeat_ndcg + explore_ndcg) / 2

        rows.append(
            {
                "factors": factors,
                "regularization": regularization,
                "alpha": alpha,
                "repeat_hit_rate@20": repeat_result.iloc[0]["hit_rate"],
                "repeat_ndcg@20": repeat_ndcg,
                "explore_hit_rate@20": explore_result.iloc[0]["hit_rate"],
                "explore_ndcg@20": explore_ndcg,
                "avg_ndcg@20": avg_ndcg,
            }
        )
        print(
            f"factors={factors:>3} reg={regularization:<5} alpha={alpha:<3} -> "
            f"repeat_ndcg={repeat_ndcg:.4f} (hit={repeat_result.iloc[0]['hit_rate']:.4f})  "
            f"explore_ndcg={explore_ndcg:.4f} (hit={explore_result.iloc[0]['hit_rate']:.4f})  "
            f"avg_ndcg={avg_ndcg:.4f}"
        )

    grid_df = pd.DataFrame(rows).sort_values("avg_ndcg@20", ascending=False)
    best = grid_df.iloc[0]
    print(
        f"\nMejor combinación por avg_ndcg@20 en val: factors={int(best['factors'])}, "
        f"regularization={best['regularization']}, alpha={best['alpha']}"
    )

    lines = [
        "# Grid search de hiperparámetros ALS — evaluado en val\n",
        f"Mejor combinación: **factors={int(best['factors'])}, regularization={best['regularization']}, "
        f"alpha={best['alpha']}** (avg_ndcg@20={best['avg_ndcg@20']:.4f})\n",
        grid_df.to_markdown(index=False, floatfmt=".4f"),
    ]
    out_path = REPORTS_DIR / "als_tuning.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
