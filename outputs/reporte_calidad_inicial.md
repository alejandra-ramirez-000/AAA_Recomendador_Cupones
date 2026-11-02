# Reporte inicial de calidad de datos

Fecha de ejecución: 2026-10-01.

Este reporte se obtuvo ejecutando:

```powershell
python -m src.cargar_datos --validar-completo
```

## Fuentes revisadas

| Fuente | Filas | Resultado |
| --- | ---: | --- |
| `transaction_data.csv` | 2.595.732 | Cargada correctamente. |
| `product.csv` | 92.353 | Cargada correctamente. |

## Controles de transacciones

| Indicador | Resultado |
| --- | ---: |
| Hogares únicos | 2.500 |
| Canastas únicas | 276.484 |
| Productos comprados únicos | 92.339 |
| Filas duplicadas exactas | 0 |
| Nulos en las 12 columnas | 0 |
| Productos vendidos sin catálogo | 0 |
| Días negativos | 0 |
| Cantidades no positivas | 14.466 |

Las 14.466 cantidades no positivas son exactamente `QUANTITY = 0`; no hay
cantidades negativas. Además se identificaron 18.850 filas con
`SALES_VALUE = 0` y ninguna venta negativa.

## Controles de catálogo

No se detectaron valores nulos ni filas duplicadas exactas en el catálogo. Hay
92.353 identificadores de producto en el catálogo y 92.339 de ellos aparecen en
las transacciones.

## Interpretación inicial

Las dos fuentes centrales son suficientes para iniciar el EDA y la construcción
del historial temporal. Las filas con cantidad cero no se eliminarán aún: su
tratamiento se decidirá con evidencia durante el análisis exploratorio y quedará
registrado en `BITACORA_DECISIONES.md`.
