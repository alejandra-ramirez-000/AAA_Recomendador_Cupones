# Coocurrencia e híbrido - validación

Los resultados se generaron con `python -m src.recomendar_coocurrencia` usando
solo entrenamiento. La validación conserva el mismo criterio explore del
baseline: productos comprados en validación que el hogar no había comprado
antes.

## Configuración inicial

- Producto candidato a asociación: al menos 500 canastas de entrenamiento.
- Par candidato: al menos 5 canastas compartidas.
- Asociación: `lift > 1`, penalizada por evidencia baja con `log(1 + soporte)`.
- Híbrido: 80 % coocurrencia normalizada y 20 % popularidad.

## Métricas de validación

| Modelo | K | Precision | Recall | Hit Rate | NDCG |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coocurrencia | 5 | 0,006811 | 0,004970 | 0,030695 | 0,007786 |
| Coocurrencia | 10 | 0,005947 | 0,008312 | 0,052278 | 0,008274 |
| Coocurrencia | 20 | 0,005659 | 0,014722 | 0,089688 | 0,010526 |
| Híbrido | 5 | 0,007578 | 0,005631 | 0,034053 | 0,008522 |
| Híbrido | 10 | 0,006715 | 0,009655 | 0,058513 | 0,009294 |
| Híbrido | 20 | 0,006235 | 0,015242 | 0,100240 | 0,011384 |

El híbrido mejora a coocurrencia pura, pero no supera la popularidad baseline
en Hit Rate@10 (0,071942). Es un resultado válido: señala que esta configuración
no será el modelo explore seleccionado sin ajustes o una alternativa mejor.
