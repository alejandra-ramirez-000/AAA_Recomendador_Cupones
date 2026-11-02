# División temporal por hogar

Los archivos de esta carpeta fueron generados por
`python -m src.dividir_temporalmente`.

Cada hogar incluido tiene al menos tres canastas con productos de cantidad
positiva. Las canastas antiguas están en entrenamiento, la penúltima en
validación y la última en prueba. Esta separación evita que un modelo conozca
compras futuras al generar recomendaciones.

## Resultado de la primera ejecución

| Partición | Hogares | Canastas | Líneas de producto |
| --- | ---: | ---: | ---: |
| Entrenamiento | 2.491 | 270.892 | 2.525.115 |
| Validación | 2.491 | 2.491 | 27.716 |
| Prueba | 2.491 | 2.491 | 28.280 |

El EDA contaba 2.492 hogares con al menos tres canastas sin filtrar. Un hogar
queda fuera en esta etapa porque, después de excluir las líneas con
`QUANTITY = 0`, ya no conserva tres canastas de compras positivas.
