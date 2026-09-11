"""
Carga de las tablas crudas de dunnhumby - The Complete Journey.

Normaliza todos los nombres de columna a snake_case minúscula para no
depender de si el CSV vino de Kaggle o del portal oficial (varían un poco
en mayúsculas/minúsculas entre mirrors).
"""
import pandas as pd

from paths import raw_path


def _read_csv_normalized(key: str, **kwargs) -> pd.DataFrame:
    path = raw_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró '{path.name}' en {path.parent}.\n"
            f"Revisá las instrucciones de descarga en data/raw/README.md."
        )
    df = pd.read_csv(path, **kwargs)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_transactions() -> pd.DataFrame:
    """household_key, basket_id, day, product_id, quantity, sales_value,
    store_id, retail_disc, trans_time, week_no, coupon_disc, coupon_match_disc"""
    df = _read_csv_normalized("transactions")
    # trans_time viene como HHMM en entero/str (ej 1954 = 19:54) -> lo dejamos crudo,
    # se interpreta como clave de orden dentro del mismo día, no como hora real.
    return df


def load_products() -> pd.DataFrame:
    """product_id, manufacturer, department, brand, commodity_desc,
    sub_commodity_desc, curr_size_of_product"""
    return _read_csv_normalized("products")


def load_hh_demographic() -> pd.DataFrame:
    """NO USAR COMO INPUT DEL MODELO (restricción del proyecto). Solo para
    análisis descriptivo exploratorio si hace falta."""
    return _read_csv_normalized("hh_demographic")


def load_campaign_desc() -> pd.DataFrame:
    return _read_csv_normalized("campaign_desc")


def load_campaign_table() -> pd.DataFrame:
    return _read_csv_normalized("campaign_table")


def load_coupon() -> pd.DataFrame:
    return _read_csv_normalized("coupon")


def load_coupon_redempt() -> pd.DataFrame:
    return _read_csv_normalized("coupon_redempt")


def load_causal_data() -> pd.DataFrame:
    return _read_csv_normalized("causal_data")
