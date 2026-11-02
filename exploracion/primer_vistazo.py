"""Primer vistazo a transaction_data.csv (exploración inicial, conservada como registro).

Estaba en data/prueba.py; se movió para que data/ contenga solo datos originales.
El análisis reproducible está en src/analizar_datos.py.

Ejecutar desde la raíz del proyecto:
    python exploracion/primer_vistazo.py
"""

import pandas as pd

ruta = "data/transaction_data.csv"

df = pd.read_csv(ruta)

print(df.shape)
print(df.columns)
print(df.info())
print(df.head())