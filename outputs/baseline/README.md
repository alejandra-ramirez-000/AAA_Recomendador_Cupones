# Baseline de recomendaciones - validación

Este resultado fue generado mediante `python -m src.recomendar_baseline`.
Entrenamiento y popularidad usan solamente
`outputs/division_temporal/interacciones_entrenamiento.csv`. La prueba permanece
sin usar para comparar el modelo seleccionado al final del proyecto.

## Estrategias

- **Repeat:** 70 % frecuencia relativa de compra y 30 % recencia. Sugiere solo
  productos comprados antes por el mismo hogar.
- **Explore:** popularidad global logarítmica de productos en entrenamiento.
  Excluye los productos ya comprados por el hogar.

## Resultados de validación

| Tipo | K | Precision | Recall | Hit Rate | NDCG | Hogares evaluables |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat | 5 | 0,172946 | 0,209622 | 0,539597 | 0,255418 | 2.033 |
| Repeat | 10 | 0,128677 | 0,294779 | 0,645352 | 0,265452 | 2.033 |
| Repeat | 20 | 0,092941 | 0,400395 | 0,742253 | 0,290043 | 2.033 |
| Explore | 5 | 0,009784 | 0,010942 | 0,046523 | 0,014802 | 2.085 |
| Explore | 10 | 0,007866 | 0,014831 | 0,071942 | 0,014842 | 2.085 |
| Explore | 20 | 0,006715 | 0,021683 | 0,109832 | 0,016715 | 2.085 |

El baseline repeat confirma que frecuencia y recencia recuperan productos que
vuelven a comprarse. Explore basado solo en popularidad ofrece una referencia
deliberadamente simple y baja; la fase de coocurrencia deberá demostrar si logra
mejorar este resultado sin usar información futura.
