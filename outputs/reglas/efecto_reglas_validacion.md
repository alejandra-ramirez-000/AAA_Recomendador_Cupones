# Efecto de las reglas de negocio (validación)

Las reglas se aplican sobre las listas candidatas (20 repeat y 20 explore por hogar)
antes de armar la lista final de 10. No cambian el modelo; miden el costo de
volver elegibles las recomendaciones. Configuración: `config/reglas_negocio.json`.

La tercera columna mide contra las compras elegibles: se quitan del objetivo
las compras que las reglas nunca permitirían ofrecer (p. ej. nafta). Separa el
costo de la regla de la calidad del modelo sobre productos ofrecibles.

| Métrica (K = 10) | Sin reglas | Con reglas | Con reglas, objetivo elegible |
| --- | ---: | ---: | ---: |
| hit_rate | 0.501405 | 0.45283 | 0.480384 |
| ndcg | 0.167833 | 0.134772 | 0.143501 |
| precision | 0.089081 | 0.083782 | 0.087402 |
| recall | 0.14227 | 0.091261 | 0.101603 |
| hit_rate_explore | 0.040288 | 0.041247 | 0.042054 |
| cobertura_hogares | 1.0 | 1.0 | 1.0 |
| productos_distintos | 5925 | 6116 | 6116 |
| departamentos_por_lista | 3.356483 | 3.002007 | 3.002007 |
| posiciones_no_elegibles | 1483 | 0 | 0 |
| hogares_con_lista_incompleta | 0 | 0 | 0 |
| posiciones_explore | 7492 | 7497 | 7497 |

## Candidatos filtrados por regla

| Tipo | Regla | Motivo | Candidatos |
| --- | --- | --- | ---: |
| explore_als | evitar_variantes_en_explore | Variante de un producto que el hogar ya compra (misma subcategoría, marca y tamaño). | 11885 |
| repeat | evitar_variantes_en_lista | Otra variante del mismo producto ya está más arriba en la lista. | 4217 |
| explore_als | evitar_variantes_en_lista | Otra variante del mismo producto ya está más arriba en la lista. | 3591 |
| explore_als | departamentos_excluidos | Combustible: precio regulado y no es objetivo de cupones de supermercado. | 1109 |
| repeat | departamentos_excluidos | Combustible: precio regulado y no es objetivo de cupones de supermercado. | 748 |
| repeat | categorias_excluidas | Tabaco: su promoción está prohibida (en Uruguay, Ley 18.256). | 428 |
| repeat | departamentos_excluidos | Transacción administrativa, no es un producto. | 115 |
| explore_als | departamentos_excluidos | Transacción administrativa, no es un producto. | 92 |
| explore_als | categorias_excluidas | Tabaco: su promoción está prohibida (en Uruguay, Ley 18.256). | 23 |
| repeat | departamentos_excluidos | Registro de cupones o descuentos, no es un producto. | 8 |
| repeat | departamentos_excluidos | Servicio postal, no es un producto. | 1 |
