# Resultados del análisis exploratorio

Estos archivos fueron generados por `python -m src.analizar_datos` a partir de
las fuentes originales. Se pueden regenerar; no deben editarse manualmente.

## Hallazgos iniciales

- El conjunto contiene 2.500 hogares, 276.484 canastas, 92.339 productos
  comprados y 102 semanas observadas.
- 2.492 hogares (99,68 %) tienen al menos tres canastas. Por lo tanto, la
  separación temporal por hogar prevista en el informe es factible para casi
  toda la población.
- La serie `canastas_por_semana.png` muestra una rampa en las primeras semanas,
  lo que respalda evitar una única división temporal global.
- El departamento `KIOSK-GAS` tiene un volumen de unidades extremadamente alto.
  El gráfico de departamentos utiliza escala logarítmica para hacerlo visible
  junto con los demás departamentos. Antes de entrenar, se analizará si esa
  característica representa unidades físicas comparables con el resto.
- Hay 14.466 filas con `QUANTITY = 0`. No se incluyeron en los rankings de
  unidades positivas, pero no se borraron de las fuentes originales.

La interpretación y futuras decisiones se registran en
`BITACORA_DECISIONES.md`.
