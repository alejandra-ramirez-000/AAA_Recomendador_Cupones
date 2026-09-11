"""
Modelo 1 — Baseline: frecuencia + recencia (Repeat) y popularidad global (Explore).

Genera el top-20 de candidatos por hogar, por segmento (repeat/explore) y por
split de evaluación (val/test), usando solo el historial permitido en cada caso:
    - val  -> historial = canastas 'train'
    - test -> historial = canastas 'train' + 'val'

Corré desde la raíz del proyecto (después de 02_build_baskets_and_split.py):
    python src/04_baseline_model.py

Genera:
    data/processed/baseline_predictions.parquet
        columnas: household_key, eval_split, segment, product_id, score, rank
"""
import pandas as pd

from baseline_scoring import compute_explore_scores, compute_repeat_scores
from basket_utils import exploded_long
from paths import DATA_PROCESSED_DIR


def build_predictions_for_split(long: pd.DataFrame, eval_split: str, history_splits: list[str]) -> pd.DataFrame:
    history_long = long[long["split"].isin(history_splits)]

    print(f"[{eval_split}] historial: splits {history_splits}, {history_long['household_key'].nunique():,} hogares")

    repeat = compute_repeat_scores(history_long)
    repeat["segment"] = "repeat"

    explore = compute_explore_scores(history_long)
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

    out_path = DATA_PROCESSED_DIR / "baseline_predictions.parquet"
    predictions.to_parquet(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
