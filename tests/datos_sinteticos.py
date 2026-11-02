"""Genera un dataset sintético pequeño con el mismo formato que dunnhumby.

Sirve para pruebas de integración: ejecuta todo el pipeline en segundos y
contiene los casos que las reglas y métricas deben manejar (dos grupos de
gustos, combustible, tabaco, cantidades cero y un catálogo con variantes).
"""

# Path y numpy/pandas construyen y guardan los CSV.
from pathlib import Path

import numpy as np
import pandas as pd


def generar_datos_sinteticos(carpeta: Path, hogares: int = 40, semilla: int = 7) -> None:
    """Escribe transaction_data.csv y product.csv en `carpeta`."""

    generador = np.random.default_rng(semilla)
    carpeta.mkdir(parents=True, exist_ok=True)

    # Catálogo: 1 = nafta, 2 = cigarrillos, 3-4 = dos variantes de la misma leche, resto almacén y verdulería.
    productos = [
        (1, "KIOSK-GAS", "COUPON/MISC ITEMS", "GASOLINE-REG UNLEADED", "Private", ""),
        (2, "DRUG GM", "CIGARETTES", "CIGARETTES", "National", "1 PK"),
        (3, "GROCERY", "FLUID MILK PRODUCTS", "FLUID MILK WHITE ONLY", "Private", "1 GA"),
        (4, "GROCERY", "FLUID MILK PRODUCTS", "FLUID MILK WHITE ONLY", "Private", "1 GA"),
    ]
    for producto_id in range(5, 121):
        departamento = "GROCERY" if producto_id % 3 else "PRODUCE"
        productos.append(
            (producto_id, departamento, f"COMMODITY {producto_id // 10}", f"SUB {producto_id}", "National", f"{producto_id} OZ")
        )
    catalogo = pd.DataFrame(
        productos, columns=["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT"]
    )
    catalogo.insert(1, "MANUFACTURER", 1)
    catalogo.to_csv(carpeta / "product.csv", index=False)

    # Dos grupos de hogares con productos preferidos distintos; todos compran leche, nafta o tabaco a veces.
    preferidos = {0: np.arange(5, 60), 1: np.arange(60, 121)}
    filas = []
    canasta_id = 1000
    for hogar in range(1, hogares + 1):
        grupo = hogar % 2
        favoritos = generador.choice(preferidos[grupo], size=12, replace=False)
        dia = int(generador.integers(1, 20))
        for _ in range(int(generador.integers(8, 16))):
            dia += int(generador.integers(1, 10))
            canasta_id += 1
            cantidad_items = int(generador.integers(4, 9))
            items = list(generador.choice(favoritos, size=min(cantidad_items, 6), replace=False))
            items += list(generador.choice(preferidos[grupo], size=2, replace=False))
            if generador.random() < 0.5:
                items.append(int(generador.choice([1, 2, 3, 4])))
            for producto_id in dict.fromkeys(int(item) for item in items):
                cantidad = 0 if generador.random() < 0.02 else int(generador.integers(1, 4))
                filas.append(
                    {
                        "household_key": hogar, "BASKET_ID": canasta_id, "DAY": dia, "PRODUCT_ID": producto_id,
                        "QUANTITY": cantidad, "SALES_VALUE": round(cantidad * float(generador.uniform(0.5, 5)), 2),
                        "STORE_ID": 1, "RETAIL_DISC": 0.0, "TRANS_TIME": 1000 + producto_id, "WEEK_NO": dia // 7 + 1,
                        "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0,
                    }
                )
    pd.DataFrame(filas).to_csv(carpeta / "transaction_data.csv", index=False)
