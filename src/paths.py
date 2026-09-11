"""Rutas centrales del proyecto. Todo script importa de acá para no hardcodear paths."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "transactions": "transaction_data.csv",
    "products": "product.csv",
    "hh_demographic": "hh_demographic.csv",
    "campaign_desc": "campaign_desc.csv",
    "campaign_table": "campaign_table.csv",
    "coupon": "coupon.csv",
    "coupon_redempt": "coupon_redempt.csv",
    "causal_data": "causal_data.csv",
}


def raw_path(key: str) -> Path:
    """Devuelve el Path a un CSV crudo por nombre lógico (ver RAW_FILES)."""
    if key not in RAW_FILES:
        raise KeyError(f"'{key}' no es una tabla conocida. Opciones: {list(RAW_FILES)}")
    return DATA_RAW_DIR / RAW_FILES[key]
