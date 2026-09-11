"""
Modelo 2b — variante híbrida: Explore = combinación ponderada de lift normalizado
+ popularidad normalizada (ver docstring de compute_explore_scores_cooccurrence_blended
en cooccurrence_scoring.py para por qué un filtro simple de candidatos por percentil
de popularidad NO alcanza). Repeat sin cambios (igual que Modelo 1 y 2).

Corré desde la raíz del proyecto (después de 02_build_baskets_and_split.py):
    python src/09_cooccurrence_hybrid_model.py

Genera:
    data/processed/cooccurrence_hybrid_predictions.parquet
"""
import pandas as pd

from baseline_scoring import compute_repeat_scores
from basket_utils import exploded_long
from cooccurrence_scoring import compute_explore_scores_cooccurrence_blended
from paths import DATA_PROCESSED_DIR

# w_lift=0.3 elegido por grid search en VAL (ver 11_tune_hybrid_weight.py /
# reports/hybrid_weight_tuning.md) — el punto donde el híbrido supera a la
# popularidad pura en recall/hit_rate/ndcg@20 sin perder demasiado por el lado
# de lift (que domina y empeora todo a partir de w_lift > 0.5).
W_LIFT = 0.3
W_POP = 0.7


def build_predictions_for_split(long: pd.DataFrame, eval_split: str, history_splits: list[str]) -> pd.DataFrame:
    history_long = long[long["split"].isin(history_splits)]

    print(f"[{eval_split}] historial: splits {history_splits}, {history_long['household_key'].nunique():,} hogares")

    repeat = compute_repeat_scores(history_long)
    repeat["segment"] = "repeat"

    explore = compute_explore_scores_cooccurrence_blended(history_long, w_lift=W_LIFT, w_pop=W_POP)
    explore["segment"] = "explore"

    preds = pd.concat([repeat, explore], ignore_index=True)
    preds["eval_split"] = eval_split
    return preds[["household_key", "eval_split", "segment", "product_id", "score", "rank"]]


def main() -> None:
    baskets = pd.read_parquet(DATA_PROCESSED_DIR / "baskets.parquet")
    baskets = baskets[baskets["split"] != "excluded"]
    long = exploded_long(baskets)

    preds_val = build_predictions_for_split(long, "val", ["train"])
    preds_test = build_predictions_for_split(long, "test", ["train", "val"])

    predictions = pd.concat([preds_val, preds_test], ignore_index=True)

    print("\nCandidatos generados por split y segmento:")
    print(
        predictions.groupby(["eval_split", "segment"])["household_key"]
        .nunique()
        .rename("n_hogares_con_candidatos")
        .to_string()
    )

    out_path = DATA_PROCESSED_DIR / "cooccurrence_hybrid_predictions.parquet"
    predictions.to_parquet(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
