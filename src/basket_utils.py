"""Utilidades compartidas sobre la tabla de canastas (data/processed/baskets.parquet)."""
import pandas as pd


def exploded_long(baskets: pd.DataFrame) -> pd.DataFrame:
    """1 fila por (household_key, basket_id, product_id) a partir de la columna 'products' (lista)."""
    long = baskets[["household_key", "basket_id", "day", "split", "products"]].explode("products")
    long = long.rename(columns={"products": "product_id"}).dropna(subset=["product_id"])
    return long
