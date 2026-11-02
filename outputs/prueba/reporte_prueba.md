# Evaluación final sobre prueba

- **Fecha:** 2026-10-15T09:26:47
- **Configuración:** `config/modelo_final.json`, congelada el 2026-10-15
- **Lista:** K = 10, 7 repeat + 3 explore (repeat primero, luego explore; si un tipo no alcanza su cupo se completa con el otro)
- **Historial usado:** entrenamiento + validación; evaluación sobre la última canasta de cada hogar
- **Entorno:** Python 3.14.3, numpy 2.4.3, pandas 3.0.1, scipy 1.18.1
- **Tiempos (s):** {'entrenamiento': 83.092, 'recomendacion_total': 5.816, 'ms_recomendacion_por_hogar': 2.335}

Evaluación única: ningún número de este reporte se usó para elegir modelos,
parámetros, K ni mezcla.

## Resultados (K = 10)

| Lista | Hogares evaluables | Precision | Recall | Hit Rate | NDCG | Hit Rate validación | NDCG validación |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Sistema final (7 + 3)** | 2491 | 0.094179 | 0.140151 | 0.507427 | 0.173970 | 0.501405 | 0.167833 |
| Aciertos explore dentro de la lista final | 2094 | 0.004489 | 0.008912 | 0.043935 | 0.011296 | - | - |
| Repeat: frecuencia + recencia | 2031 | 0.134958 | 0.294475 | 0.645002 | 0.273717 | 0.645352 | 0.265452 |
| Explore: ALS | 2094 | 0.009790 | 0.017211 | 0.086437 | 0.017029 | 0.094964 | 0.017432 |
| Referencia: explore por popularidad | 2094 | 0.007259 | 0.014765 | 0.065425 | 0.014378 | 0.071942 | 0.014842 |
| Referencia: lista solo repeat | 2491 | 0.110116 | 0.156075 | 0.526295 | 0.186649 | - | - |

## Métricas de sistema de la lista final

- Cobertura de hogares: 1.0000
- Productos distintos recomendados: 5827
- Participación de los 10 productos más recomendados: 0.2854
- Departamentos distintos por lista: 3.368527

La columna de validación corresponde a los mismos modelos entrenados solo con
entrenamiento. La diferencia entre ambas mezcla dos efectos: el optimismo por
haber elegido sobre validación y la dificultad propia de cada canasta. La
referencia de popularidad, que no fue seleccionada, ayuda a separarlos.
