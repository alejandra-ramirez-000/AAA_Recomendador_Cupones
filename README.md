# Recomendador de cupones personalizados — Proyecto Integrador

Sistema de recomendación de cupones personalizados para supermercados basado en
historial de compras (dunnhumby — The Complete Journey), distinguiendo dos tipos
de recomendación:

- **Repeat**: productos que el cliente ya compra habitualmente (afinidad/recompra).
- **Explore**: productos que no compra, pero relacionados con sus intereses o con
  patrones de clientes parecidos.

Materia: Aplicaciones de Aprendizaje Automático — UTEC, Licenciatura en Ingeniería
de Datos e Inteligencia Artificial.

## Restricciones del proyecto

- No se usan variables demográficas (edad, ingreso, composición familiar) como
  input del modelo (existen en `hh_demographic.csv` pero el loader nunca se llama
  desde ningún script de features/modelo).
- No se usan identificadores personales (nombre, cédula, teléfono, email) — el
  dataset ya viene anonimizado por `household_key`.
- Separación **temporal** (no aleatoria) por cliente, para evitar leakage.

## Setup

```bash
pip install -r requirements.txt
```

Después descargá el dataset siguiendo [data/raw/README.md](data/raw/README.md) —
8 CSV que van en `data/raw/` (no se suben a git, ~808MB, son públicos y cada uno
los baja directo del dataset original).

## Cómo correr el pipeline completo

Desde `src/`, en este orden (cada script imprime lo que genera y dónde):

| # | Script | Qué hace |
|---|---|---|
| 01 | `01_eda.py` | EDA inicial → `reports/eda_summary.md` + gráficos |
| 02 | `02_build_baskets_and_split.py` | Reconstruye canastas y arma el split train/val/test (leave-last-out por cliente) → `data/processed/baskets.parquet` |
| 03 | `03_build_targets.py` | Etiqueta ground-truth Repeat/Explore de las canastas objetivo → `data/processed/targets.parquet` |
| 04 | `04_baseline_model.py` | Modelo 1: frecuencia+recencia (Repeat) / popularidad global (Explore) |
| 05 | `05_evaluate_baseline.py` | Evalúa Modelo 1 |
| 06 | `06_cooccurrence_model.py` | Modelo 2: co-ocurrencia (lift) para Explore |
| 07 | `07_evaluate_cooccurrence.py` | Evalúa Modelo 2 |
| 08 | `08_compare_models.py` | Tabla comparativa de todos los modelos evaluados hasta el momento → `reports/model_comparison.md` |
| 09 | `09_cooccurrence_hybrid_model.py` | Modelo 2b: híbrido lift + popularidad (peso tuneado en val) |
| 10 | `10_evaluate_cooccurrence_hybrid.py` | Evalúa Modelo 2b |
| 11 | `11_tune_hybrid_weight.py` | Grid search del peso del híbrido, solo en val → `reports/hybrid_weight_tuning.md` |
| 12 | `12_als_model.py` | Modelo 3: filtrado colaborativo (ALS), Repeat y Explore |
| 13 | `13_evaluate_als.py` | Evalúa Modelo 3 |
| 14 | `14_tune_als.py` | Grid search de hiperparámetros de ALS, solo en val → `reports/als_tuning.md` |

`08_compare_models.py` se puede re-correr en cualquier momento (después de tener
al menos un `*_metrics.csv` en `data/processed/`) para refrescar la comparación.

## Decisiones de diseño clave

- **Canasta** = productos distintos bajo un mismo `BASKET_ID` (un ticket real).
- **Split**: leave-last-out por hogar — última canasta = test, penúltima = val,
  resto = historial de train. Hogares con <3 canastas quedan excluidos (0.3% del total).
- **Target**: para cada canasta objetivo, cada producto se etiqueta Repeat (ya
  estaba en el historial permitido de ese hogar) o Explore (no estaba).
- **Métricas**: Precision/Recall/Hit Rate/NDCG@{5,10,20}, siempre calculadas por
  separado para Repeat y Explore.
- **Granularidad**: Repeat a nivel `PRODUCT_ID` (SKU exacto), Explore a nivel
  `SUB_COMMODITY_DESC` en el diseño conceptual (el código actual evalúa Explore
  a nivel producto; ver notas en el informe).

## Resultado de la comparación de modelos (test, K=20)

| Modelo | Repeat Hit Rate | Repeat NDCG | Explore Hit Rate | Explore NDCG |
|---|---|---|---|---|
| 1. Baseline (frecuencia+recencia / popularidad) | **0.654** | **0.250** | 0.107 | 0.017 |
| 2. Co-ocurrencia (lift puro) | 0.654 | 0.250 | 0.034 | 0.004 |
| 2b. Híbrido (lift+popularidad, tuneado) | 0.654 | 0.250 | 0.116 | **0.017** |
| 3. ALS (filtrado colaborativo, tuneado) | 0.506 | 0.136 | **0.117** | 0.017 |

Detalle completo en [reports/model_comparison.md](reports/model_comparison.md).
Recomendación actual: **Repeat = Modelo 1**, **Explore = Modelo 2b** (empata con
ALS pero es mucho más simple).

## Estructura

```
data/raw/          CSV originales de dunnhumby (no versionados en git)
data/processed/    Parquet/CSV intermedios generados por el pipeline
reports/           EDA, métricas y comparaciones en markdown + gráficos
src/               Scripts numerados en orden de ejecución + módulos compartidos
```
