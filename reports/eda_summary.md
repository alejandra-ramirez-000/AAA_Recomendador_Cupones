# EDA inicial — dunnhumby The Complete Journey

## 1. Tamaño general

- Líneas de transacción (filas): **2,595,732**
- Hogares (household_key) distintos: **2,500**
- Canastas (basket_id) distintas: **276,484**
- Productos distintos comprados: **92,339** (catálogo completo: 92,353)
- Tiendas distintas: **582**
- Promedio de canastas por hogar: **110.59**
- Promedio de líneas por canasta: **9.39**

- Chequeo de integridad: canastas con más de 1 household_key asociado: **0** (debería ser 0)

## 2. Cobertura temporal

- Rango de `day`: 1 a 711 (710 días ≈ 101 semanas)
- Rango de `week_no`: 1 a 102

## 3. Tamaño de canasta (productos distintos por ticket)

```
count    276484.000000
mean          9.388362
std          12.283792
min           1.000000
25%           2.000000
50%           5.000000
75%          12.000000
max         168.000000
Name: product_id, dtype: float64
```

## 4. Canastas por hogar

```
count    2500.000000
mean      110.593600
std       115.669368
min         1.000000
25%        39.000000
50%        79.000000
75%       142.250000
max      1300.000000
Name: basket_id, dtype: float64
```

- Hogares con **≥ 3 canastas** (usables para split train/val/test leave-last-out): **2,492** (99.7%)
- Hogares con **< 3 canastas** (se excluyen del split): **8** (0.3%)

## 5. Concentración de popularidad

- Los 10 productos más comprados aparecen en el **43.9%** de todas las canastas (referencia de cuán larga es la cola de productos poco frecuentes).
- Productos comprados en una sola canasta en todo el dataset: **33,614** de 92,339
