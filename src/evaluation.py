"""
Métricas de recomendación: Precision@K, Recall@K, Hit Rate@K, NDCG@K.

Se calculan por hogar y se promedian. Un hogar sin ningún producto verdadero
en el segmento evaluado (ej. sin productos Explore en su canasta objetivo)
se EXCLUYE del promedio de ese segmento (no aporta 0 ni indefine la métrica) —
es la convención estándar en recsys, y se reporta cuántos hogares entraron
en el promedio para que quede explícito.
"""
import numpy as np
import pandas as pd


def precision_at_k(pred_list: list, truth_set: set, k: int) -> float:
    return len(set(pred_list[:k]) & truth_set) / k


def recall_at_k(pred_list: list, truth_set: set, k: int) -> float:
    return len(set(pred_list[:k]) & truth_set) / len(truth_set)


def hit_rate_at_k(pred_list: list, truth_set: set, k: int) -> float:
    return 1.0 if (set(pred_list[:k]) & truth_set) else 0.0


def ndcg_at_k(pred_list: list, truth_set: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 1) for i, p in enumerate(pred_list[:k], start=1) if p in truth_set
    )
    ideal_hits = min(len(truth_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _truth_sets(targets: pd.DataFrame, eval_split: str, segment: str) -> dict:
    subset = targets[(targets["eval_split"] == eval_split) & (targets["label"] == segment)]
    return subset.groupby("household_key")["product_id"].apply(set).to_dict()


def _pred_lists(predictions: pd.DataFrame, eval_split: str, segment: str) -> dict:
    subset = predictions[
        (predictions["eval_split"] == eval_split) & (predictions["segment"] == segment)
    ].sort_values(["household_key", "rank"])
    return subset.groupby("household_key")["product_id"].apply(list).to_dict()


def evaluate(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    eval_split: str,
    segment: str,
    ks: tuple = (5, 10, 20),
) -> pd.DataFrame:
    """Devuelve un DataFrame con 1 fila por K: n_households, precision, recall, hit_rate, ndcg."""
    truth = _truth_sets(targets, eval_split, segment)
    pred_lists = _pred_lists(predictions, eval_split, segment)

    rows = []
    for k in ks:
        precisions, recalls, hits, ndcgs = [], [], [], []
        for household_key, truth_set in truth.items():
            pred_list = pred_lists.get(household_key, [])
            precisions.append(precision_at_k(pred_list, truth_set, k))
            recalls.append(recall_at_k(pred_list, truth_set, k))
            hits.append(hit_rate_at_k(pred_list, truth_set, k))
            ndcgs.append(ndcg_at_k(pred_list, truth_set, k))
        rows.append(
            {
                "k": k,
                "n_households": len(precisions),
                "precision": np.mean(precisions) if precisions else float("nan"),
                "recall": np.mean(recalls) if recalls else float("nan"),
                "hit_rate": np.mean(hits) if hits else float("nan"),
                "ndcg": np.mean(ndcgs) if ndcgs else float("nan"),
            }
        )
    return pd.DataFrame(rows)
