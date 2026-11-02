# Análisis exploratorio para la defensa

Datos: entrenamiento + validación (la prueba no se usa). Generado por `python -m src.analizar_repeticion`.

## 1. Recompra dentro de cada canasta

- Sin contar la primera canasta de cada hogar, en promedio el **50.1 %** de los productos de una canasta ya habían sido comprados antes por el hogar (46.5 % si se pondera por línea).
- A nivel de variante (misma subcategoría, marca y tamaño): **60.7 %**.

| tramo | canastas | proporcion_repetida | proporcion_repetida_variante |
| --- | --- | --- | --- |
| 1 | 2491 | 0.0 | 0.0 |
| 2-5 | 9917 | 0.1222 | 0.1644 |
| 6-10 | 12097 | 0.235 | 0.307 |
| 11-25 | 33776 | 0.3412 | 0.4346 |
| 26-50 | 46897 | 0.4372 | 0.5461 |
| 51-100 | 64220 | 0.5245 | 0.6392 |
| 101+ | 103985 | 0.6355 | 0.7477 |

## 2. Objetivos de validación por hogar

| hogares | con_recompra | con_producto_nuevo | con_ambos | solo_recompra | solo_producto_nuevo | ninguno |
| --- | --- | --- | --- | --- | --- | --- |
| 2491 | 2033 | 2085 | 1627 | 406 | 458 | 0 |

## 3. Rendimiento según la longitud del historial (K = 10, validación)

| lista | segmento | hogares_evaluables | hit_rate | ndcg |
| --- | --- | --- | --- | --- |
| repeat: frecuencia + recencia | corto (1-48 canastas) | 586 | 0.668942 | 0.297875 |
| repeat: frecuencia + recencia | medio (49-116 canastas) | 694 | 0.631124 | 0.237819 |
| repeat: frecuencia + recencia | largo (117-1296 canastas) | 753 | 0.640106 | 0.265688 |
| explore: popularidad | corto (1-48 canastas) | 771 | 0.123217 | 0.025447 |
| explore: popularidad | medio (49-116 canastas) | 713 | 0.058906 | 0.0108 |
| explore: popularidad | largo (117-1296 canastas) | 601 | 0.021631 | 0.006034 |
| explore: ALS | corto (1-48 canastas) | 771 | 0.155642 | 0.028327 |
| explore: ALS | medio (49-116 canastas) | 713 | 0.081346 | 0.013503 |
| explore: ALS | largo (117-1296 canastas) | 601 | 0.033278 | 0.008116 |

## 4. Concentración de la popularidad

- El 1 % de los productos más comprados reúne el **34.8 %** de las apariciones en canastas; el 10 %, el **76.8 %**.
- 33.092 de 90.927 productos aparecen en una sola canasta. Gini: 0.844.

### Productos distintos por departamento

| DEPARTMENT | productos_catalogo | productos_comprados | recomendados_repeat: frecuencia + recencia | recomendados_explore: popularidad | recomendados_explore: ALS |
| --- | --- | --- | --- | --- | --- |
| GROCERY | 39021 | 38529 | 5278 | 45 | 657 |
| DRUG GM | 31529 | 31015 | 1144 | 3 | 73 |
| PRODUCE | 3118 | 3051 | 385 | 24 | 140 |
| COSMETICS | 3011 | 2931 | 30 | 0 | 0 |
| NUTRITION | 2914 | 2885 | 215 | 0 | 3 |
| MEAT | 2544 | 2524 | 169 | 3 | 65 |
| MEAT-PCKGD | 2427 | 2395 | 351 | 1 | 56 |
| DELI | 2354 | 2320 | 220 | 2 | 26 |
| PASTRY | 2149 | 2120 | 150 | 1 | 15 |
| FLORAL | 938 | 897 | 29 | 0 | 0 |
| SEAFOOD-PCKGD | 563 | 553 | 37 | 0 | 1 |
| MISC. TRANS. | 490 | 460 | 8 | 0 | 0 |

## 5. Combustible y tabaco

| grupo | productos_en_catalogo | porcentaje_de_lineas | porcentaje_de_hogares_compradores | porcentaje_compradores_que_recompran | canastas_medias_por_comprador | posiciones_en_listas_sin_reglas | porcentaje_hogares_con_el_grupo_en_su_lista |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Combustible (KIOSK-GAS) | 16 | 0.86 | 54.4 | 82.7 | 16.0 | 1160 | 44.7 |
| Tabaco | 513 | 0.52 | 45.4 | 70.7 | 10.3 | 235 | 8.2 |
