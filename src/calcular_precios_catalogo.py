"""Precio de referencia por producto, para el catálogo de la app (prototipo).

No forma parte del pipeline congelado (no se registra en `pipeline.py`): es un
cálculo auxiliar, aparte, solo para que la app pueda mostrar un precio y armar
un carrito de compra simulado. Usa el precio efectivamente pagado (SALES_VALUE
/ QUANTITY) de cada transacción y se queda con la mediana por producto, para
no depender de outliers de ofertas o devoluciones.

Ejecutar desde la raíz del proyecto:
    python -m src.calcular_precios_catalogo
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_TRANSACCIONES = RAIZ / "data" / "transaction_data.csv"
RUTA_SALIDA = RAIZ / "outputs" / "demo" / "precios_estimados.csv"


def calcular_precios() -> pd.DataFrame:
    columnas = ["PRODUCT_ID", "QUANTITY", "SALES_VALUE"]
    transacciones = pd.read_csv(RUTA_TRANSACCIONES, usecols=columnas)
    transacciones = transacciones.loc[(transacciones["QUANTITY"] > 0) & (transacciones["SALES_VALUE"] > 0)]
    precio_unitario = transacciones["SALES_VALUE"] / transacciones["QUANTITY"]
    precios = (
        precio_unitario.groupby(transacciones["PRODUCT_ID"])
        .median()
        .round(2)
        .rename("precio_referencia")
        .reset_index()
    )
    return precios


def main() -> None:
    precios = calcular_precios()
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    precios.to_csv(RUTA_SALIDA, index=False)
    print(f"Precios de referencia para {len(precios):,} productos -> {RUTA_SALIDA}".replace(",", "."))


if __name__ == "__main__":
    main()
