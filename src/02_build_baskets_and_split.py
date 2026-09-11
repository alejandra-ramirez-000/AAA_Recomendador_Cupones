"""
Reconstruye la secuencia de canastas por cliente y arma el split temporal
train/val/test con estrategia LEAVE-LAST-OUT POR CLIENTE:

    Para un hogar con M canastas (M >= 3), ordenadas cronológicamente:
        canastas 1..M-2  -> "train"  (historial base)
        canasta  M-1      -> "val"    (target de validación)
        canasta  M         -> "test"   (target de evaluación final)

    Hogares con M < 3 no tienen suficiente historial para esta estructura
    (no se puede tener historial + target de val + target de test) y se
    EXCLUYEN del modelado, quedando registrados como tal.

IMPORTANTE (para el próximo paso de feature engineering):
    - Para predecir la canasta de VALIDACIÓN de un hogar, el historial
      permitido es únicamente sus canastas "train" (1..M-2).
    - Para predecir la canasta de TEST (evaluación final), el historial
      permitido es "train" + "val" (1..M-1), es decir, toda la información
      disponible hasta justo antes del target real.
    - Nunca se usa información de una canasta posterior al target que se
      está prediciendo (así evitamos leakage temporal).

Corré desde la raíz del proyecto:
    python src/02_build_baskets_and_split.py

Genera:
    data/processed/baskets.parquet             -> 1 fila por (household_key, basket_id)
    data/processed/household_split_summary.parquet -> 1 fila por household_key
"""
import pandas as pd

from data_loading import load_transactions
from paths import DATA_PROCESSED_DIR


def build_basket_table(txns: pd.DataFrame) -> pd.DataFrame:
    """Colapsa transaction_data (1 fila por línea de ticket) a 1 fila por canasta."""
    grouped = txns.groupby(["household_key", "basket_id"], as_index=False).agg(
        day=("day", "first"),
        week_no=("week_no", "first"),
        trans_time_min=("trans_time", "min"),
        n_items=("product_id", "nunique"),
        total_qty=("quantity", "sum"),
        total_sales_value=("sales_value", "sum"),
    )
    # Lista de productos distintos por canasta (necesaria para el target Repeat/Explore
    # y para las features de co-ocurrencia en el próximo paso).
    products_per_basket = (
        txns.groupby(["household_key", "basket_id"])["product_id"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .rename("products")
    )
    grouped = grouped.merge(products_per_basket, on=["household_key", "basket_id"])
    return grouped


def assign_sequence_and_split(baskets: pd.DataFrame) -> pd.DataFrame:
    """Ordena las canastas de cada hogar cronológicamente y asigna train/val/test."""
    baskets = baskets.sort_values(
        ["household_key", "day", "trans_time_min", "basket_id"]
    ).copy()

    baskets["basket_seq"] = baskets.groupby("household_key").cumcount() + 1
    n_baskets_hh = baskets.groupby("household_key")["basket_id"].transform("nunique")
    baskets["n_baskets_household"] = n_baskets_hh
    baskets["basket_seq_from_end"] = baskets["n_baskets_household"] - baskets["basket_seq"] + 1

    def label_split(row) -> str:
        if row["n_baskets_household"] < 3:
            return "excluded"
        if row["basket_seq_from_end"] == 1:
            return "test"
        if row["basket_seq_from_end"] == 2:
            return "val"
        return "train"

    baskets["split"] = baskets.apply(label_split, axis=1)
    return baskets


def build_household_summary(baskets: pd.DataFrame) -> pd.DataFrame:
    summary = (
        baskets.groupby("household_key")
        .agg(
            n_baskets=("basket_id", "nunique"),
            first_day=("day", "min"),
            last_day=("day", "max"),
        )
        .reset_index()
    )
    summary["excluded"] = summary["n_baskets"] < 3
    summary["n_train_baskets"] = (summary["n_baskets"] - 2).clip(lower=0)
    return summary


def main() -> None:
    txns = load_transactions()

    print("Colapsando transacciones a nivel de canasta...")
    baskets = build_basket_table(txns)

    print("Asignando secuencia cronológica y split leave-last-out por hogar...")
    baskets = assign_sequence_and_split(baskets)

    summary = build_household_summary(baskets)

    n_households = summary["household_key"].nunique()
    n_excluded = summary["excluded"].sum()
    split_counts = baskets["split"].value_counts()

    print(f"\nHogares totales: {n_households:,}")
    print(f"Hogares excluidos (< 3 canastas): {n_excluded:,} ({n_excluded / n_households:.1%})")
    print("\nCanastas por split:")
    print(split_counts.to_string())

    baskets_path = DATA_PROCESSED_DIR / "baskets.parquet"
    summary_path = DATA_PROCESSED_DIR / "household_split_summary.parquet"
    baskets.to_parquet(baskets_path, index=False)
    summary.to_parquet(summary_path, index=False)

    print(f"\nGuardado: {baskets_path}")
    print(f"Guardado: {summary_path}")
    print(
        "\nSiguiente paso: feature engineering (frecuencia, recencia, afinidad de "
        "categoría, co-ocurrencia, popularidad) calculado por hogar SOLO con las "
        "canastas 'train' (para predecir 'val') y 'train'+'val' (para predecir 'test')."
    )


if __name__ == "__main__":
    main()
