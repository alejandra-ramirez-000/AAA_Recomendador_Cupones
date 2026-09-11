"""
Construye el ground-truth Repeat/Explore para las canastas objetivo (val y test).

Para cada hogar (los marcados 'excluded' en el split quedan afuera):
    - target 'val'  -> su canasta con split == 'val'; historial permitido = canastas 'train'
    - target 'test' -> su canasta con split == 'test'; historial permitido = canastas 'train' + 'val'

Cada producto de la canasta objetivo se etiqueta:
    - 'repeat'  si ya aparece en el historial permitido de ESE hogar
    - 'explore' si no aparece

Corré desde la raíz del proyecto:
    python src/03_build_targets.py

Genera:
    data/processed/targets.parquet
        columnas: household_key, eval_split ('val'/'test'), product_id, label ('repeat'/'explore')
"""
import pandas as pd

from basket_utils import exploded_long
from paths import DATA_PROCESSED_DIR


def _labeled_targets(long: pd.DataFrame, eval_split: str, history_splits: list[str]) -> pd.DataFrame:
    history = (
        long[long["split"].isin(history_splits)][["household_key", "product_id"]]
        .drop_duplicates()
    )
    history["_in_history"] = True

    target = (
        long[long["split"] == eval_split][["household_key", "product_id"]]
        .drop_duplicates()
    )
    merged = target.merge(history, on=["household_key", "product_id"], how="left")
    merged["label"] = merged["_in_history"].map({True: "repeat"})
    merged["label"] = merged["label"].fillna("explore")
    merged["eval_split"] = eval_split
    return merged[["household_key", "eval_split", "product_id", "label"]]


def build_targets(baskets: pd.DataFrame) -> pd.DataFrame:
    long = exploded_long(baskets)
    targets_val = _labeled_targets(long, "val", ["train"])
    targets_test = _labeled_targets(long, "test", ["train", "val"])
    return pd.concat([targets_val, targets_test], ignore_index=True)


def main() -> None:
    baskets = pd.read_parquet(DATA_PROCESSED_DIR / "baskets.parquet")
    baskets = baskets[baskets["split"] != "excluded"]

    print("Construyendo etiquetas Repeat/Explore para val y test...")
    targets = build_targets(baskets)

    summary = (
        targets.groupby(["eval_split", "label"])["product_id"]
        .count()
        .rename("n_productos")
        .reset_index()
    )
    print("\nProductos objetivo por split y etiqueta:")
    print(summary.to_string(index=False))

    # Nº de hogares que sí tienen al menos un producto Explore en su target
    # (los que no, quedan afuera del promedio de métricas Explore para ese split).
    for eval_split in ("val", "test"):
        n_hh_total = targets.loc[targets["eval_split"] == eval_split, "household_key"].nunique()
        n_hh_explore = targets.loc[
            (targets["eval_split"] == eval_split) & (targets["label"] == "explore"), "household_key"
        ].nunique()
        print(
            f"\n[{eval_split}] hogares con >=1 producto Explore en el target: "
            f"{n_hh_explore:,} / {n_hh_total:,} ({n_hh_explore / n_hh_total:.1%})"
        )

    out_path = DATA_PROCESSED_DIR / "targets.parquet"
    targets.to_parquet(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
