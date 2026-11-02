# Contexto completo, estado actual y plan de cierre

**Proyecto:** Sistema de recomendación de productos candidatos a cupones personalizados  
**Curso:** Aplicaciones de Aprendizaje Automático - Proyecto Integrador AIAI  
**Fecha de actualización:** 19 de octubre de 2026
**Estado general:** plan de trabajo completo. Hay modelos seleccionados en
validación, una evaluación única sobre prueba, reglas de negocio, un prototipo
Streamlit, un pipeline reproducible, el informe final (`INFORME_FINAL.md`) y el
material de presentación y defensa (`PRESENTACION_Y_DEFENSA.md`). Falta el
ensayo del equipo y el commit final.

---

## 1. Propósito de este documento

Este documento concentra el contexto necesario para continuar, explicar y
defender el proyecto sin depender de conversaciones anteriores. Resume:

- qué problema se busca resolver;
- qué exige la consigna académica;
- qué datos están disponibles y cuáles se utilizaron;
- qué código y artefactos se construyeron;
- cómo se evitó la fuga de información temporal;
- qué modelos se implementaron;
- qué resultados se obtuvieron y cómo deben interpretarse;
- qué decisiones se tomaron y con qué evidencia;
- qué limitaciones e incidentes técnicos aparecieron;
- qué se completó del plan de cierre y qué queda pendiente (revisión, commit y ensayo);
- qué afirmaciones pueden hacerse y cuáles no.

El documento describe el estado real del repositorio a la fecha indicada. No
presenta como terminado aquello que todavía es una propuesta o un trabajo
pendiente.

---

## 2. Contexto académico y entregables

La consigna del Proyecto Integrador AIAI evalúa la capacidad de transformar una
necesidad real o verosímil en una solución de aprendizaje automático defendible.
La nota no depende de una métrica aislada ni de la apariencia de la demo. Se
valoran especialmente:

- definición y relevancia del problema;
- datos y factibilidad;
- arquitectura e integración;
- justificación de los modelos;
- métricas y validación;
- análisis honesto de resultados y limitaciones;
- existencia de un prototipo funcional;
- demostración del flujo completo;
- defensa técnica individual.

### 2.1 Entregable 1

El Entregable 1 es `Informe CUPON AAA.md`. Ese documento contiene la propuesta
del proyecto: problema, objetivos, usuarios, datos, preparación, arquitectura,
modelos candidatos, definición de repeat/explore, métricas, riesgos y plan de
trabajo.

El Entregable 1 ya funciona como la definición conceptual del trabajo. No debe
confundirse con el presente documento, que registra el desarrollo realizado
después de la propuesta.

### 2.2 Entregable 2

La versión final debe incluir:

- un documento técnico breve de aproximadamente 10 a 15 páginas;
- una aplicación o prueba de concepto funcional;
- una demostración del recorrido entrada -> procesamiento -> modelo -> salida
  -> interpretación;
- resultados medidos y analizados;
- limitaciones y mejoras;
- una defensa en la que cada integrante pueda explicar el proyecto completo.

### 2.3 Evaluación continua

Los scripts, pruebas, reportes, bitácora y resultados guardados en el
repositorio sirven como evidencia de avance. Aun así, los Colabs, cuestionarios
y otras actividades de Moodle pertenecen a la evaluación del curso y deben ser
gestionados por el equipo fuera de este repositorio cuando corresponda.

---

## 3. Problema y pregunta principal

Los supermercados ofrecen promociones con frecuencia, pero una promoción
general puede ser relevante para algunos clientes e irrelevante para otros. Un
programa de fidelización permite asociar compras a un hogar anonimizado y
reconstruir su historial de consumo.

La pregunta principal es:

> ¿Cómo utilizar el historial de compras de un cliente para recomendar
> productos sobre los cuales ofrecer cupones personalizados?

La propuesta no intenta decidir por sí sola si un cupón debe emitirse. El
modelo genera **productos candidatos**. En una implementación real, esos
candidatos deberían pasar por reglas comerciales relacionadas con stock,
margen, campañas activas, presupuesto y restricciones del negocio.

---

## 4. Usuario y beneficiarios

### Usuario directo

El usuario directo sería el área comercial, de marketing o fidelización de un
supermercado. El sistema funcionaría como apoyo para seleccionar productos
candidatos a promociones personalizadas.

### Beneficiarios

- **Cliente:** recibiría promociones más relacionadas con su historial e
  intereses.
- **Supermercado:** podría hacer un uso más relevante de su información de
  fidelización y, potencialmente, mejorar canje, venta o retención.

Es importante distinguir una expectativa de negocio de un resultado medido. El
dataset permite evaluar si las recomendaciones coinciden con compras
posteriores, pero no permite demostrar por sí solo que un cupón cause una
compra. Esa afirmación requeriría un experimento controlado, por ejemplo una
prueba A/B con clientes reales.

---

## 5. Definición de repeat y explore

El sistema separa dos objetivos distintos:

### Repeat

Productos que el hogar ya compró y podría volver a necesitar. Las señales
actuales son:

- frecuencia de aparición en canastas anteriores;
- recencia de la última compra;
- pertenencia al historial del mismo hogar.

### Explore

Productos que el hogar no había comprado previamente, pero podrían ser
relevantes por:

- popularidad global;
- coocurrencia con productos del historial;
- patrones compartidos con otros hogares mediante filtrado colaborativo;
- afinidad de categoría u otras reglas futuras.

Esta separación evita que el sistema recomiende únicamente artículos ya
conocidos. También exige evaluar ambos tipos de forma independiente, porque
predecir una recompra suele ser considerablemente más fácil que anticipar una
compra nueva.

---

## 6. Datos disponibles

El proyecto utiliza el dataset dunnhumby - The Complete Journey. En `data/`
están disponibles las siguientes fuentes:

| Archivo | Contenido | Uso actual |
| --- | --- | --- |
| `transaction_data.csv` | Historial de compras por hogar, canasta, día y producto | Fuente principal del modelado. |
| `product.csv` | Catálogo con departamento, marca, categoría, subcategoría y tamaño | Descripción e interpretación de recomendaciones. |
| `hh_demographic.csv` | Datos demográficos de una parte de los hogares | No se usa como entrada del modelo inicial. |
| `coupon.csv` | Relación entre cupones, productos y campañas | Reservado para reglas o análisis posterior. |
| `coupon_redempt.csv` | Canjes históricos de cupones | Reservado para análisis posterior; no se usa para atribuir causalidad. |
| `campaign_table.csv` | Hogares asociados a campañas | Reservado para una etapa posterior. |
| `campaign_desc.csv` | Tipo y periodo de campañas | Reservado para una etapa posterior. |
| `causal_data.csv` | Display y mailer por producto, tienda y semana | No se carga en el flujo inicial por tamaño y alcance. |

### 6.1 Variables principales de transacciones

- `household_key`: identificador anonimizado del hogar.
- `BASKET_ID`: identificador de la canasta o unidad de compra.
- `DAY`: secuencia temporal en días.
- `WEEK_NO`: semana del dataset.
- `TRANS_TIME`: hora de la transacción.
- `PRODUCT_ID`: identificador del producto.
- `QUANTITY`: cantidad registrada.
- `SALES_VALUE`: importe de venta.
- `STORE_ID`: tienda.
- `RETAIL_DISC`: descuento minorista.
- `COUPON_DISC`: descuento de cupón.
- `COUPON_MATCH_DISC`: descuento complementario asociado al cupón.

### 6.2 Variables principales del catálogo

- `PRODUCT_ID`;
- `MANUFACTURER`;
- `DEPARTMENT`;
- `BRAND`;
- `COMMODITY_DESC`;
- `SUB_COMMODITY_DESC`;
- `CURR_SIZE_OF_PRODUCT`.

### 6.3 Datos deliberadamente no utilizados como entrada inicial

Los datos demográficos no se incorporaron al modelo. Esta decisión mantiene el
foco en el comportamiento de compra, reduce riesgos de privacidad y evita usar
atributos sensibles sin una justificación clara. Los identificadores son
anonimizados y no se dispone de nombre, cédula, teléfono o correo electrónico.

---

## 7. Calidad de datos y hallazgos iniciales

La validación completa de `transaction_data.csv` y `product.csv` produjo los
siguientes resultados:

| Indicador | Resultado |
| --- | ---: |
| Filas de transacciones | 2.595.732 |
| Filas del catálogo | 92.353 |
| Hogares únicos | 2.500 |
| Canastas únicas | 276.484 |
| Productos comprados únicos | 92.339 |
| Nulos en las 12 columnas de transacciones | 0 |
| Duplicados exactos de transacciones | 0 |
| Productos vendidos sin catálogo | 0 |
| Días negativos | 0 |
| Cantidades no positivas | 14.466 |

Las 14.466 cantidades no positivas son exactamente cero; no existen cantidades
negativas. De esas filas:

- 14.399 tienen `SALES_VALUE = 0`;
- 67 tienen venta positiva;
- 4.955 registran algún descuento de cupón.

También se encontraron 18.850 filas con `SALES_VALUE = 0`, considerando todas
las cantidades, y ninguna venta negativa.

### Interpretación

Las fuentes centrales tienen claves completas y se pueden unir sin productos
huérfanos. Sin embargo, `QUANTITY = 0` no representa una señal implícita clara
de compra. Por eso esas filas se conservan en los datos originales, pero se
excluyen de entrenamiento y evaluación del recomendador.

---

## 8. Análisis exploratorio realizado

El EDA descriptivo se implementó como un script reproducible en
`src/analizar_datos.py`, y el EDA para la defensa en `src/analizar_repeticion.py`
(ver 8.5). Las evidencias son CSV, JSON y PNG en `outputs/eda/`. El notebook
`notebooks/01_eda_defensa.ipynb` solo presenta esos resultados.

### 8.1 Dimensiones temporales y de comportamiento

| Indicador | Resultado |
| --- | ---: |
| Hogares | 2.500 |
| Canastas | 276.484 |
| Productos comprados | 92.339 |
| Días observados | 711 |
| Semanas observadas | 102 |
| Hogares con al menos tres canastas antes del filtro | 2.492 |
| Porcentaje con al menos tres canastas | 99,68 % |

### 8.2 Rampa temporal

La cantidad de canastas por semana presenta una rampa durante las primeras
semanas. Esto indica una incorporación gradual de hogares al dataset. Un corte
temporal global podría comparar hogares con distinta antigüedad de forma poco
equilibrada. La división temporal por hogar es, por lo tanto, coherente con la
estructura observada.

### 8.3 Distribución de canastas

La mayoría de los hogares tiene historial suficiente para entrenamiento,
validación y prueba. La distribución presenta cola larga: algunos hogares
tienen muchas más canastas que otros. Por ese motivo, las métricas finales se
promedian por hogar, evitando que hogares con más actividad dominen por número
de filas.

### 8.4 Volumen por departamento

`KIOSK-GAS` presenta un volumen de unidades extremadamente superior al resto.
El gráfico utiliza escala logarítmica para no ocultar otros departamentos.
Este comportamiento tiene dos consecuencias:

1. las cantidades físicas no son necesariamente comparables entre todos los
   departamentos;
2. gasolina aparece como candidato frecuente en el baseline explore.

Los modelos se evaluaron sin excluir `KIOSK-GAS` (D-003). La exclusión se
implementó después como una regla configurable y separada del modelo, con su
efecto medido aparte (D-008, sección 19).

### 8.5 EDA para la defensa (completado, D-011)

El análisis de evidencia se implementó en `src/analizar_repeticion.py` (etapa
`eda_defensa` del pipeline) y se presenta en `notebooks/01_eda_defensa.ipynb`.
Usa solo entrenamiento + validación.

| Hallazgo | Resultado |
| --- | --- |
| Productos de una canasta ya comprados antes (sin la primera canasta) | 50,1 % (46,5 % ponderado por línea) |
| Ídem, contando cualquier variante del mismo producto | 60,7 % |
| Recompra en canastas 2 a 5 / a partir de la canasta 101 | 12 % / 64 % |
| Hogares con recompra / con producto nuevo / con ambos en validación | 2.033 / 2.085 / 1.627 (de 2.491) |
| Participación del 1 % / 10 % de productos más comprados | 34,8 % / 76,8 % (Gini 0,844) |
| Productos que aparecen en una sola canasta | 33.092 de 90.927 |
| Combustible: hogares compradores / que lo recompran / con combustible en su lista sin reglas | 54,4 % / 82,7 % / 44,7 % |
| Tabaco: hogares compradores / que lo recompran / con tabaco en su lista sin reglas | 45,4 % / 70,7 % / 8,2 % |

**Rendimiento según la longitud del historial (Hit Rate@10, validación):**

| Lista | Corto (1 a 48 canastas) | Medio (49 a 116) | Largo (117 o más) |
| --- | ---: | ---: | ---: |
| Repeat: frecuencia + recencia | 0,669 | 0,631 | 0,640 |
| Explore: popularidad | 0,123 | 0,059 | 0,022 |
| Explore: ALS | 0,156 | 0,081 | 0,033 |

Repeat rinde parejo. Explore rinde mucho mejor con clientes nuevos, porque
todavía no compraron lo habitual. Esto queda como mejora futura: hacer depender
la mezcla de la antigüedad del hogar.

El notebook previsto en el plan original existe como presentación
(`notebooks/01_eda_defensa.ipynb`). La fuente reproducible sigue siendo el
script.

---

## 9. Arquitectura implementada

```text
CSV originales (data/, nunca se modifican; el pipeline lo verifica con SHA-256)
    |
    v
Carga y validación de calidad ................. src/cargar_datos.py
    |
    v
EDA descriptivo y EDA para la defensa ......... src/analizar_datos.py, src/analizar_repeticion.py
    |
    v
Filtro QUANTITY > 0 + división temporal por hogar   src/dividir_temporalmente.py
    +--> Entrenamiento: canastas antiguas
    +--> Validación: penúltima canasta
    +--> Prueba: última canasta (usada una sola vez, D-007)
    |
    v
Candidatos (20 por tipo y hogar)
    +--> Repeat: frecuencia + recencia ......... src/recomendar_baseline.py  [SELECCIONADO]
    +--> Explore: popularidad .................. src/recomendar_baseline.py
    +--> Explore: coocurrencia e híbrido ....... src/recomendar_coocurrencia.py
    +--> Repeat y explore: ALS implícito ....... src/recomendar_als.py       [SELECCIONADO para explore]
    |
    v
Evaluación común (ranking y sistema) .......... src/evaluar_modelos.py
    |
    v
Selección por tipo + mezcla 7 + 3 ............. src/seleccionar_modelo.py -> config/modelo_final.json
    |
    v
Evaluación única sobre prueba ................. src/evaluar_prueba.py -> outputs/prueba/
    |
    v
Reglas de negocio configurables ............... src/reglas_negocio.py + config/reglas_negocio.json
    |
    v
Lista final explicada (todo el historial) ..... src/preparar_demo.py -> outputs/demo/
    |
    v
Prototipo Streamlit ........................... app/app.py
    |
    v
Feedback simulado (útil / no útil) -> outputs/demo/feedback_simulado.csv
```

Todo el flujo se ejecuta con `python -m src.pipeline` (D-010). Las etapas
`seleccion` y `prueba` están congeladas y solo corren con `--recongelar` e
`--incluir-prueba`.

**Fuera del prototipo, como integración futura:** stock, margen y campañas
reales; feedback real; reentrenamiento periódico; monitoreo; prueba A/B.

---

## 10. División temporal y prevención de fuga de información

La división se realiza por hogar después de excluir cantidades cero:

1. se agrupan las líneas por `household_key` y `BASKET_ID`;
2. las canastas se ordenan por `DAY`, `TRANS_TIME` y `BASKET_ID`;
3. se conservan hogares con al menos tres canastas positivas;
4. las canastas antiguas forman entrenamiento;
5. la penúltima canasta forma validación;
6. la última canasta forma prueba.

### 10.1 Resultado de la división

| Partición | Hogares | Canastas | Líneas de producto |
| --- | ---: | ---: | ---: |
| Entrenamiento | 2.491 | 270.892 | 2.525.115 |
| Validación | 2.491 | 2.491 | 27.716 |
| Prueba | 2.491 | 2.491 | 28.280 |

El EDA encontraba 2.492 hogares con al menos tres canastas antes de aplicar el
filtro de cantidades. Un hogar deja de ser elegible porque, después de excluir
`QUANTITY = 0`, queda con menos de tres canastas de compra positiva.

### 10.2 Estado del conjunto de prueba

El conjunto de prueba se usó **una sola vez**, el 15/10/2026, después de congelar
modelos, parámetros, K y mezcla en `config/modelo_final.json` (D-006, D-007). El
script `src/evaluar_prueba.py` se niega a sobrescribir su reporte sin
`--forzar`, y el pipeline no ejecuta esa etapa por defecto. Después de la
prueba solo se agregaron reglas de elegibilidad, que no cambian el modelo y cuyo
efecto se midió en validación. La lista evaluada en prueba se regeneró sin abrir
la canasta de prueba y resultó idéntica a la guardada.

### 10.3 Validaciones automáticas

El código comprueba que:

- existan las tres particiones;
- todos los hogares incluidos aparezcan en entrenamiento, validación y prueba;
- el orden de las etiquetas no retroceda dentro del historial;
- explore no recomiende productos ya observados cuando se prueba esa regla;
- las métricas reconozcan correctamente un acierto en primera posición.

---

## 11. Modelos implementados

### 11.1 Baseline repeat: frecuencia y recencia

Para cada par hogar-producto se calcula:

- cantidad de canastas de entrenamiento que contienen el producto;
- última canasta en que se compró;
- frecuencia normalizada respecto al producto más frecuente del hogar;
- distancia desde la última compra;
- recencia como `1 / (1 + distancia)`.

El puntaje utilizado es:

```text
puntaje_repeat = 0,70 * frecuencia_normalizada
               + 0,30 * recencia
```

El modelo solo recomienda productos ya presentes en el historial del hogar.
La salida incluye frecuencia, última compra, distancia, posición, categoría y
subcategoría, lo que permite explicar cada recomendación.

### 11.2 Baseline explore: popularidad

La popularidad se calcula únicamente con entrenamiento:

- número de canastas que contienen el producto;
- número de hogares que lo compraron;
- transformación logarítmica `log(1 + canastas)`;
- normalización entre cero y uno.

Para cada hogar se recorre el ranking global y se excluyen productos ya
comprados. Es un baseline intencionalmente simple y no personalizado más allá
del filtro de historial.

### 11.3 Coocurrencia

El modelo identifica pares de productos que aparecen en la misma canasta. Para
reducir asociaciones espurias y controlar memoria:

- cada producto debe aparecer en al menos 500 canastas;
- cada par debe aparecer en al menos cinco canastas;
- se conserva `lift > 1`;
- el puntaje es `lift * log(1 + soporte_del_par)`.

Para generar candidatos por hogar:

- se consideran hasta 50 productos recientes del hogar;
- se buscan destinos asociados;
- se excluyen todos los productos ya comprados;
- si varios productos históricos apoyan el mismo candidato, sus señales se
  acumulan.

### 11.4 Híbrido

El híbrido utiliza los mismos candidatos de coocurrencia y combina:

```text
puntaje_hibrido = 0,80 * coocurrencia_normalizada
                 + 0,20 * popularidad_normalizada
```

La popularidad ayuda a priorizar candidatos con mayor evidencia general, pero
no agrega candidatos fuera de los generados por coocurrencia.

### 11.5 ALS implícito (implementación propia, D-005)

`src/recomendar_als.py` implementa ALS para feedback implícito (Hu, Koren y
Volinsky, 2008) **solo con numpy y scipy.sparse**. Cada actualización se resuelve
con 3 pasos de gradiente conjugado, el mismo esquema que la librería `implicit`.

- **Por qué no se usa `implicit`:** se instala, pero el Control de aplicaciones
  de Windows bloquea sus DLL ("Una directiva de Control de aplicaciones bloqueó
  este archivo"). Lo mismo pasa con scikit-learn, así que tampoco sirvió
  TruncatedSVD como alternativa. No se tocó la configuración de seguridad.
- **Señal:** canastas en que aparece cada producto del hogar, con
  `confianza = 1 + alpha · log(1 + frecuencia)`.
- **Grilla fijada antes de ver resultados:** factores {32, 64}, regularización
  {0,05; 0,10}, alpha {2; 10} e iteraciones {15, 30}; semilla 42; criterio: NDCG@10
  explore en validación.
- **Configuración elegida:** 32 factores, regularización 0,10, alpha 2 y 30
  iteraciones.
- **Salidas:** listas repeat (dentro del historial) y explore (fuera del
  historial). Cada sugerencia explore incluye el producto del historial más
  similar en el espacio latente, que sirve como explicación.
- **Costo:** unos 2,5 s por iteración con 32 factores; unos 77 a 83 s por
  entrenamiento completo; unos 2,3 ms de recomendación por hogar.
- **Verificación:** una prueba compara el gradiente conjugado con la solución
  exacta del sistema lineal.

### 11.6 Lista final y mezcla

La lista que recibe cada hogar tiene **10 productos: 7 repeat (frecuencia +
recencia) seguidos de 3 explore (ALS)**. Si un tipo no alcanza su cupo, se
completa con el otro y se descartan duplicados (`combinar_listas` en
`src/seleccionar_modelo.py`).

---

## 12. Metodología de evaluación

Los resultados actuales se calculan sobre la canasta de validación de cada
hogar, usando rankings de K = 5, 10 y 20.

### 12.1 Productos relevantes para repeat

Son productos comprados en validación que ya estaban en el historial de
entrenamiento del hogar.

### 12.2 Productos relevantes para explore

Son productos comprados en validación que no estaban en el historial de
entrenamiento del hogar.

### 12.3 Métricas

- **Precision@K:** proporción de posiciones recomendadas que coinciden con
  productos relevantes.
- **Recall@K:** proporción de productos relevantes recuperados dentro de las
  primeras K posiciones.
- **Hit Rate@K:** proporción de hogares con al menos un acierto dentro de K.
- **NDCG@K:** calidad del orden, asignando más valor a aciertos ubicados en las
  primeras posiciones.

Las métricas se promedian por hogar. Solo se incluyen como evaluables los
hogares que tienen al menos un producto relevante del tipo analizado:

- 2.033 hogares evaluables para repeat;
- 2.085 hogares evaluables para explore.

Esta diferencia es correcta: no todos los hogares presentan una recompra y una
compra nueva en la misma canasta de validación.

### 12.4 Evaluación común y métricas de sistema

Desde el Bloque 1, todas las métricas se calculan en `src/evaluar_modelos.py`.
Al refactorizar se verificó que dieran exactamente los mismos valores que
antes. Además de las métricas de ranking, el módulo mide:

- cobertura de hogares y de catálogo;
- concentración: participación de los 10 productos más recomendados y Gini de
  exposición;
- diversidad: departamentos distintos por lista;
- porcentaje de recomendaciones con explicación;
- tiempos de entrenamiento y de recomendación por hogar.

El tipo de objetivo `todos` (cualquier compra cuenta como acierto) se usa para
evaluar la lista combinada repeat + explore.

---

## 13. Resultados del baseline

### 13.1 Repeat

| K | Precision | Recall | Hit Rate | NDCG | Hogares evaluables |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0,172946 | 0,209622 | 0,539597 | 0,255418 | 2.033 |
| 10 | 0,128677 | 0,294779 | 0,645352 | 0,265452 | 2.033 |
| 20 | 0,092941 | 0,400395 | 0,742253 | 0,290043 | 2.033 |

### Análisis de repeat

- Precision disminuye al aumentar K porque se agregan posiciones menos seguras.
- Recall y Hit Rate aumentan porque una lista mayor tiene más oportunidad de
  recuperar productos comprados después.
- Hit Rate@10 de 0,645352 significa que aproximadamente 64,5 % de los hogares
  con al menos una recompra relevante recibe al menos un acierto entre las diez
  primeras recomendaciones repeat.
- Recall@20 de 0,400395 muestra que una lista de veinte productos recupera en
  promedio cerca de 40 % de los productos de recompra relevantes.
- NDCG aumenta moderadamente con K. Hay aciertos útiles, pero todavía existe
  margen para mejorar el orden dentro de la lista.

El baseline repeat es una referencia sólida, interpretable y barata. Antes de
reemplazarlo debe demostrarse una mejora que justifique mayor complejidad.

### 13.2 Explore por popularidad

| K | Precision | Recall | Hit Rate | NDCG | Hogares evaluables |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0,009784 | 0,010942 | 0,046523 | 0,014802 | 2.085 |
| 10 | 0,007866 | 0,014831 | 0,071942 | 0,014842 | 2.085 |
| 20 | 0,006715 | 0,021683 | 0,109832 | 0,016715 | 2.085 |

### Análisis de explore por popularidad

- El problema explore es mucho más difícil que repeat: intenta anticipar
  productos que el hogar nunca había comprado.
- Hit Rate@10 de 0,071942 significa que aproximadamente 7,2 % de los hogares
  evaluables recibe al menos un acierto explore entre diez sugerencias.
- El resultado bajo no implica un error del código. Refleja que popularidad
  global tiene poca personalización y que el universo de productos es amplio.
- El baseline es útil precisamente porque establece una referencia sencilla
  que modelos más complejos deben superar.
- La presencia de productos de `KIOSK-GAS` entre los más populares confirma que
  luego habrá que separar calidad predictiva de elegibilidad comercial.

---

## 14. Resultados de coocurrencia e híbrido

| Modelo | K | Precision | Recall | Hit Rate | NDCG |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coocurrencia | 5 | 0,006811 | 0,004970 | 0,030695 | 0,007786 |
| Coocurrencia | 10 | 0,005947 | 0,008312 | 0,052278 | 0,008274 |
| Coocurrencia | 20 | 0,005659 | 0,014722 | 0,089688 | 0,010526 |
| Híbrido | 5 | 0,007578 | 0,005631 | 0,034053 | 0,008522 |
| Híbrido | 10 | 0,006715 | 0,009655 | 0,058513 | 0,009294 |
| Híbrido | 20 | 0,006235 | 0,015242 | 0,100240 | 0,011384 |

### 14.1 Comparación en K = 10

| Modelo explore | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularidad baseline | **0,007866** | **0,014831** | **0,071942** | **0,014842** |
| Coocurrencia | 0,005947 | 0,008312 | 0,052278 | 0,008274 |
| Híbrido | 0,006715 | 0,009655 | 0,058513 | 0,009294 |

### 14.2 Interpretación

- El híbrido supera a coocurrencia pura en todos los K evaluados.
- Ninguno supera al baseline de popularidad en la configuración actual.
- En K = 20, el híbrido se aproxima más al baseline en Hit Rate
  (0,100240 frente a 0,109832), pero continúa por debajo.
- La coocurrencia trabaja solo con productos muy frecuentes debido al umbral de
  500 canastas. Esto mejora estabilidad y costo, pero reduce cobertura de
  productos de nicho.
- El score basado en lift puede favorecer relaciones específicas con buen lift
  aunque no sean los productos nuevos más probables para el hogar.
- Usar solo los 50 productos recientes controla costo, pero puede ignorar
  preferencias históricas estables.
- El híbrido no agrega candidatos de popularidad; solo reordena candidatos de
  coocurrencia. Una mejora futura podría incorporar una unión explícita de
  candidatos y no únicamente una mezcla de puntajes.

### 14.3 Conclusión (superada por la selección final)

Coocurrencia e híbrido quedaron como experimentos comparativos y no fueron
seleccionados. La comparación final, que incluye ALS, está en las secciones 16
y 17.

---

## 15. Incidente de escala en coocurrencia

La primera configuración consideraba productos presentes en al menos 20
canastas. Esa configuración dejaba 18.402 productos y aproximadamente 25,5
millones de pares potenciales. El proceso llegó a utilizar gran cantidad de
memoria y terminó antes de producir resultados.

Se analizaron alternativas:

| Soporte mínimo de producto | Productos elegibles | Pares potenciales aproximados |
| ---: | ---: | ---: |
| 20 | 18.402 | 25.522.538 |
| 50 | 10.179 | 20.260.609 |
| 100 | 5.300 | 13.833.052 |
| 200 | 2.197 | 7.367.030 |
| 500 | 573 | 2.504.931 |
| 1.000 | 193 | 1.001.747 |

Se seleccionó 500 como umbral inicial. La ejecución final produjo 99.263
asociaciones con `lift > 1`.

Este incidente no debe ocultarse. Es evidencia de una restricción real de
arquitectura y costo computacional. La configuración elegida permite una
comparación reproducible, pero limita la cobertura. Para ampliar coocurrencia se
podría evaluar una matriz dispersa, conteo por bloques, top-N por producto o un
motor distribuido, siempre que la complejidad adicional sea justificable.

---

## 16. Resultados de ALS en validación

Grilla de 16 configuraciones (`outputs/als/grilla_validacion.csv`). La
regularización casi no cambia los resultados, y pasar de 15 a 30 iteraciones
aporta muy poco. Con alpha 10 y con 64 factores empeoran tanto repeat como
explore.

| Modelo | K | Precision | Recall | Hit Rate | NDCG |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALS repeat | 10 | 0,071471 | 0,175022 | 0,461879 | 0,141019 |
| ALS explore | 5 | 0,012950 | 0,012069 | 0,060911 | 0,016487 |
| ALS explore | 10 | 0,010935 | 0,019651 | 0,094964 | 0,017432 |
| ALS explore | 20 | 0,008921 | 0,029192 | 0,142926 | 0,019756 |

### 16.1 Comparación completa en validación (K = 10)

| Modelo | Tipo | Hit Rate | NDCG | Productos distintos | Los 10 más recomendados concentran | Entrenamiento | ms por hogar |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frecuencia + recencia | repeat | **0,645** | **0,265** | 8.111 | 16 % | 1,4 s | 0,6 |
| ALS | repeat | 0,462 | 0,141 | 1.273 | 35 % | 77 s | 2,4 |
| Popularidad | explore | 0,072 | 0,0148 | 81 | 51 % | 1,4 s | 0,6 |
| Coocurrencia | explore | 0,052 | 0,0083 | 490 | 17 % | 5,3 s | 4,6 |
| Híbrido | explore | 0,059 | 0,0093 | 479 | 18 % | 5,3 s | 4,6 |
| **ALS** | explore | **0,095** | **0,0174** | **1.051** | 30 % | 77 s | 2,4 |

Todos los modelos cubren el 100 % de los hogares y todas sus recomendaciones
tienen explicación. Tabla completa: `outputs/comparacion/comparacion_validacion.md`.

---

## 17. Selección del modelo y mezcla (D-006)

- **Repeat:** se conserva **frecuencia + recencia**. ALS queda muy por debajo
  y es más costoso.
- **Explore:** se elige **ALS**: +17 % de NDCG y +32 % de Hit Rate sobre
  popularidad, y 13 veces más productos distintos.
- **Mezcla:** el Hit Rate total crece siempre con más repeat
  (`outputs/seleccion/curva_mezcla_validacion.csv`), así que la métrica sola
  eliminaría explore. Por eso se aplicó una **regla de producto**: el mejor
  NDCG@10 con al menos 3 posiciones explore. Resultado: **7 repeat + 3
  explore, K = 10**.

| Repeat + explore | Hit Rate total | Hogares que aciertan algo nuevo |
| --- | ---: | ---: |
| 10 + 0 | 0,527 | 0 % |
| **7 + 3** | **0,501** | **4,0 %** |
| 6 + 4 | 0,485 | 5,2 % |
| 5 + 5 | 0,464 | 6,1 % |

El equipo confirmó 7 + 3 antes de abrir prueba. La configuración quedó
congelada en `config/modelo_final.json`, que se genera con
`python -m src.seleccionar_modelo`.

---

## 18. Evaluación final sobre prueba (D-007)

`python -m src.evaluar_prueba` reentrenó la configuración congelada con
entrenamiento + validación y evaluó **una sola vez** contra la última canasta.
Reporte: `outputs/prueba/reporte_prueba.md`.

| Lista (K = 10) | Hogares | Precision | Recall | Hit Rate | NDCG | Hit Rate en validación |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Sistema final (7 + 3)** | 2.491 | 0,094179 | 0,140151 | **0,507427** | 0,173970 | 0,501405 |
| Repeat: frecuencia + recencia | 2.031 | 0,134958 | 0,294475 | 0,645002 | 0,273717 | 0,645352 |
| Explore: ALS | 2.094 | 0,009790 | 0,017211 | 0,086437 | 0,017029 | 0,094964 |
| Referencia: popularidad | 2.094 | 0,007259 | 0,014765 | 0,065425 | 0,014378 | 0,071942 |
| Referencia: lista solo repeat | 2.491 | 0,110116 | 0,156075 | 0,526295 | 0,186649 | — |

**Lectura:**

- El sistema es estable.
- Las dos alternativas explore bajan en prueba, también popularidad, que no fue
  seleccionada. Por eso la caída se atribuye sobre todo a la canasta de prueba.
- ALS mantiene la ventaja: +32 % de Hit Rate y +18 % de NDCG.
- El 4,4 % de los hogares acierta un producto nuevo.
- Métricas de sistema: 100 % de hogares cubiertos, 5.827 productos distintos y
  3,4 departamentos por lista. Reentrenar lleva unos 83 s y recomendar, unos
  2,3 ms por hogar.

Estas cifras son **sin reglas de negocio**, tal como se definió antes de abrir
la prueba.

---

## 19. Reglas de negocio (D-008)

Se aplican después del ranking (`src/reglas_negocio.py`) y se configuran en
`config/reglas_negocio.json`. Cada candidato descartado queda registrado con su
regla y su motivo.

| Regla | Motivo |
| --- | --- |
| Excluir `KIOSK-GAS` | Combustible: precio regulado; no es objetivo de cupones. |
| Excluir `CIGARETTES`, `CIGARS` y `TOBACCO OTHER` | Tabaco: su promoción está prohibida (en Uruguay, Ley 18.256). `SMOKING CESSATIONS` queda permitido. |
| Excluir departamentos que no son productos | Transacciones, registros de cupones, donaciones, correo y `RX`. |
| Evitar variantes en explore | Mismo producto (subcategoría, marca y tamaño) con otro código no es "nuevo". |
| Evitar variantes dentro de una lista | Es la misma oferta; queda la mejor ubicada. |
| Exigir catálogo y duplicados | Explicar y no repetir. |
| Soporte mínimo explore (10 canastas) | Evidencia mínima; en la práctica no filtró nada. |
| Stock y margen | Puntos de integración vacíos. Configurar un margen sin datos da error a propósito. |
| Cupón histórico (`coupon.csv`) | Opción disponible, desactivada. |

**Supuestos:** las exclusiones son decisiones del equipo, no políticas reales
de un supermercado. El alcohol (`SPIRITS`) quedó permitido; se puede revisar.

**Efecto en validación** (`outputs/reglas/efecto_reglas_validacion.md`):

| K = 10 | Sin reglas | Con reglas | Con reglas, solo compras elegibles |
| --- | ---: | ---: | ---: |
| Hit Rate | 0,501 | 0,453 | 0,480 |
| Hogares que aciertan algo nuevo | 4,0 % | 4,1 % | 4,2 % |
| Posiciones no elegibles | 1.483 | 0 | 0 |
| Productos distintos recomendados | 5.925 | 6.116 | 6.116 |

La caída del Hit Rate no indica un peor modelo:

- Combustible y tabaco eran aciertos fáciles pero no ofrecibles.
- El resto de la diferencia viene de quitar variantes repetidas, porque la
  métrica cuenta aciertos por código de producto y no por oferta.
- 11.885 candidatos explore eran variantes de productos ya comprados, y 7.808
  eran variantes repetidas dentro de una lista.

**Origen de las reglas de tabaco y variantes:** surgieron al probar el
prototipo con datos reales:

- el hogar 20 recibía cigarrillos y la misma leche con otro código;
- el hogar 186 tenía el mismo snack dos veces en la lista.

Las métricas no lo mostraban.

---

## 20. Prototipo (D-009)

`app/app.py` es una app Streamlit que se ejecuta con
`streamlit run app/app.py` y queda en http://localhost:8501, accesible solo desde
el propio equipo. Solo lee archivos precalculados por `python -m src.preparar_demo`,
que reentrena el sistema congelado con **todo** el historial, aplica las reglas
y redacta una explicación por producto.

- **Entrada:** 5 hogares de ejemplo deterministas (221 típico, 1510 historial
  más largo, 186 historial más corto, 956 con muchos productos filtrados, 20 con
  una sugerencia explore bien explicada), o cualquier `household_key`.
- **Pestañas:**
  - Recomendaciones: 7 repeat y 3 explore con puntaje y explicación.
  - Historial: productos frecuentes y última canasta.
  - Filtrados por reglas: con su motivo.
  - Feedback: útil o no útil, registrado en `outputs/demo/feedback_simulado.csv`.
  - Vista "Sobre el modelo": resultados de prueba, aclarando que no son efecto
    causal.
- **Exportación CSV** de la lista.
- **Casos límite:**
  - Hogar inexistente: lista de respaldo con productos populares elegibles,
    uno por subcategoría.
  - Historial corto: aviso de poca evidencia.
- **Ejemplos de explicación:**
  - *"Lo compró en 86 canastas; la última vez en su penúltima compra."*
  - *"Nuevo para el hogar. Los hogares que compran «Hamburger buns», como este,
    también lo eligen."*

Todas las listas de la demo tienen 10 productos, sin repetidos, sin variantes
repetidas, sin productos no elegibles y con explicación. Las listas de la demo
no tienen métricas propias, porque no existe una canasta futura contra la cual
medirlas.

---

## 21. Pipeline, pruebas y reproducibilidad (D-010)

- `python -m src.pipeline` ejecuta 12 etapas en orden: calidad, eda, division,
  baseline, coocurrencia, als, comparacion, [seleccion], eda_defensa, [prueba],
  reglas y demo.
- Registra versiones, commit y fecha en `outputs/entorno.txt`, y comando,
  duración y resultado de cada etapa en `outputs/registro_pipeline.json`.
- Calcula SHA-256 de los CSV de `data/` antes y después, y falla si alguno
  cambió.
- `seleccion` y `prueba` solo corren con `--recongelar` e `--incluir-prueba`,
  y la prueba no se repite sin `--forzar-prueba`.
- Opciones útiles: `--desde`, `--hasta`, `--solo`, `--als-rapido` y `--listar`.
- Una corrida completa con datos reales tarda unos 20 minutos, sobre todo por
  la grilla de ALS.

**Pruebas:** 49 automatizadas (`python -m unittest discover -s tests`). Incluyen
una de integración (`tests/test_pipeline.py`) que corre las 12 etapas sobre un
dataset sintético (`tests/datos_sinteticos.py`) en una carpeta temporal, en
unos 15 s. Esa prueba encontró un error real: la evaluación final y la demo
solo sabían reentrenar ALS. Se generalizó la generación de candidatos
(`generar_candidatos` en `src/evaluar_prueba.py`) para aceptar ALS o
popularidad.

**Verificación con los datos reales tras ese cambio:**

- la lista final de prueba regenerada es idéntica a la guardada (se comprobó
  sin abrir la canasta de prueba);
- la demo tiene el mismo contenido;
- los datos de entrada siguen intactos.

---

## 22. Decisiones tomadas

El detalle de cada decisión, con su evidencia y sus alternativas, está en
`BITACORA_DECISIONES.md`.

| ID | Decisión |
| --- | --- |
| D-001 | Transacciones + catálogo como fuentes; demografía, campañas y datos causales fuera del modelo. |
| D-002 | Excluir `QUANTITY = 0` de las interacciones sin borrar los originales. |
| D-003 | Evaluar el modelo sin filtros comerciales; medir las reglas aparte. |
| D-004 | Coocurrencia con soporte mínimo de 500 canastas por producto y 5 por par, por memoria. |
| D-005 | ALS implementado con numpy y scipy (Windows bloquea `implicit`); grilla y resultados. |
| D-006 | Repeat = frecuencia + recencia; explore = ALS; lista 7 + 3 con K = 10; configuración congelada. |
| D-007 | Evaluación única sobre prueba: Hit Rate@10 de 0,507. |
| D-008 | Reglas de negocio configurables (combustible, tabaco, no productos, variantes). |
| D-009 | Prototipo Streamlit con datos precalculados; solo accesible desde `localhost`. |
| D-010 | Pipeline reproducible, protecciones y prueba de integración. |
| D-011 | EDA para la defensa: recompra, cold start, concentración, combustible y tabaco. |

---

## 23. Código implementado

| Archivo | Responsabilidad | Comando |
| --- | --- | --- |
| `src/cargar_datos.py` | Carga y validación de calidad. | `python -m src.cargar_datos --validar-completo` |
| `src/analizar_datos.py` | EDA descriptivo. | `python -m src.analizar_datos` |
| `src/analizar_repeticion.py` | EDA para la defensa. | `python -m src.analizar_repeticion` |
| `src/dividir_temporalmente.py` | División temporal por hogar. | `python -m src.dividir_temporalmente` |
| `src/recomendar_baseline.py` | Repeat (frecuencia + recencia) y explore (popularidad). | `python -m src.recomendar_baseline` |
| `src/recomendar_coocurrencia.py` | Coocurrencia e híbrido. | `python -m src.recomendar_coocurrencia` |
| `src/recomendar_als.py` | ALS implícito propio y su grilla. | `python -m src.recomendar_als [--rapido]` |
| `src/evaluar_modelos.py` | Métricas comunes y tabla comparativa. | `python -m src.evaluar_modelos` |
| `src/seleccionar_modelo.py` | Elección por tipo, curva de mezcla y configuración congelada. | `python -m src.seleccionar_modelo` |
| `src/evaluar_prueba.py` | Generación de candidatos y evaluación única. | `python -m src.evaluar_prueba` |
| `src/reglas_negocio.py` | Reglas de elegibilidad y su efecto. | `python -m src.reglas_negocio` |
| `src/preparar_demo.py` | Datos y explicaciones del prototipo. | `python -m src.preparar_demo` |
| `src/pipeline.py` | Ejecución de punta a punta. | `python -m src.pipeline` |
| `app/app.py` | Prototipo Streamlit. | `streamlit run app/app.py` |
| `notebooks/01_eda_defensa.ipynb` | Presentación del EDA (solo lee resultados). | Jupyter |
| `exploracion/primer_vistazo.py` | Exploración inicial; antes estaba en `data/prueba.py`. | Fuera del flujo |

**Pruebas** (`tests/`): carga, división, baseline y coocurrencia
(`test_cargar_datos.py`), evaluación, ALS, selección, evaluación final, reglas,
demo, EDA de defensa y pipeline de integración.

---

## 24. Artefactos generados

Todos son derivados y se regeneran con el pipeline. Los CSV de `data/` no se
modifican.

| Carpeta | Contenido principal |
| --- | --- |
| `outputs/reporte_calidad_inicial.md` | Validación completa de calidad. |
| `outputs/eda/` | EDA descriptivo, archivos `defensa_*` (tablas y 4 gráficos) y `analisis_defensa.md`. |
| `outputs/division_temporal/` | Tres particiones y resumen. |
| `outputs/baseline/` | Perfil repeat, popularidad, listas y métricas de validación. |
| `outputs/coocurrencia/` | Asociaciones, listas y métricas de validación. |
| `outputs/als/` | Grilla, listas y métricas de validación. |
| `outputs/comparacion/` | Tabla comparativa de modelos (CSV y Markdown). |
| `outputs/seleccion/` | Curva de mezcla repeat/explore. |
| `outputs/prueba/` | Reporte, métricas y lista final explicada de la evaluación única. |
| `outputs/reglas/` | Efecto de las reglas y candidatos filtrados con su motivo. |
| `outputs/demo/` | Datos del prototipo (listas, filtrados, historial, respaldo, ejemplos). |
| `outputs/entorno.txt`, `outputs/registro_pipeline.json` | Registro de la última corrida del pipeline. |
| `config/modelo_final.json` | Configuración congelada (D-006). |
| `config/reglas_negocio.json` | Reglas de negocio (D-008). |

**Documentos:**

| Archivo | Contenido |
| --- | --- |
| `INFORME_FINAL.md` | Informe del Entregable 2. |
| `PRESENTACION_Y_DEFENSA.md` | Diapositivas, guion de demo y guía de preguntas. |
| `BITACORA_DECISIONES.md` | Decisiones de la D-001 a la D-011. |
| `GUIA_DE_TRABAJO.md` | Plan por bloques que guio el cierre. |
| `README.md` | Instalación, comandos, estructura y resultados. |

---

## 25. Dependencias y entorno

`requirements.txt`: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy y
streamlit. `implicit` se retiró (D-005).

- **Entorno verificado:** Python 3.14.3, numpy 2.4.3, pandas 3.0.1, scipy
  1.18.1, matplotlib 3.11.1, seaborn 0.13.2 y streamlit 1.57.0. Las métricas
  guardadas se reprodujeron idénticas con pandas 3.0.1.
- **Advertencia de Windows:** el Control de aplicaciones o Smart App Control
  bloqueó DLL de numpy, scipy, scikit-learn e `implicit` dentro de un entorno
  virtual nuevo, y de scikit-learn e `implicit` incluso en el Python del
  sistema. El proyecto funciona con el Python del sistema y no depende de
  scikit-learn en tiempo de ejecución.
- `.streamlit/config.toml` limita el prototipo a `localhost`, porque por defecto
  quedaba expuesto a la red.

---

## 26. Limitaciones

**Datos y generalización**

- El dataset es de un minorista estadounidense y no representa necesariamente
  hábitos actuales ni de supermercados uruguayos.
- No hay stock, margen ni campañas reales. Las reglas son supuestos del equipo.

**Evaluación offline**

- Una compra posterior no prueba que el cupón la cause: no se mide efecto
  incremental.
- Una recomendación podría ser útil aunque no se compre en esa canasta puntual.
- Validación y prueba son una sola canasta por hogar.
- La métrica cuenta aciertos por código de producto y no por oferta.
- La configuración de ALS se eligió en validación. La estimación sin sesgo es
  la de prueba.

**Cold start**

- Los hogares con menos de 3 canastas no se modelan; un hogar nuevo recibe una
  lista genérica.
- Explore rinde mucho mejor con historial corto (0,156 contra 0,033), pero la
  mezcla fija 7 + 3 no aprovecha esa diferencia.
- La coocurrencia excluyó gran parte del catálogo por memoria.

**Implementación**

- ALS propio es más lento que una librería optimizada.
- El feedback es simulado.
- No hay monitoreo ni reentrenamiento automático.

---

## 27. Qué se puede afirmar con evidencia

- El flujo completo es reproducible con un comando, tiene 49 pruebas en verde y
  verifica que los datos originales no cambian.
- La división temporal por hogar evita usar compras futuras, y la prueba se usó
  una sola vez con la configuración congelada.
- Repeat por frecuencia + recencia logra Hit Rate@10 de 0,645 tanto en
  validación como en prueba, y supera a ALS para recompras.
- ALS supera a popularidad, coocurrencia e híbrido en explore. En prueba logra
  0,086 contra 0,065 de popularidad (+32 %), y recomienda unas 13 veces más
  productos distintos.
- El sistema final 7 + 3 logra Hit Rate@10 de 0,507 en prueba (0,501 en
  validación), sin reglas.
- Con reglas, ninguna lista contiene combustible, tabaco, productos no
  ofrecibles ni variantes repetidas. Sobre compras elegibles, el Hit Rate en
  validación es 0,480.
- En promedio, la mitad de cada canasta es recompra, y explore rinde mejor con
  clientes nuevos.
- Cada recomendación mostrada tiene una explicación y cada exclusión, su motivo.

---

## 28. Qué no se debe afirmar

- Que el sistema aumente ventas, canjes o retención, ni que una recomendación
  cause una compra.
- Que esté listo para producción.
- Que existan filtros reales de stock o margen, o que las reglas sean políticas
  reales de un supermercado.
- Que las métricas representen a clientes uruguayos actuales.
- Que las cifras con reglas sean resultados de prueba: se midieron en
  validación.
- Que ALS se haya entrenado con la librería `implicit`: es una implementación
  propia equivalente.
- Que el feedback del prototipo sea real.

---

## 29. Estado por fase

| Fase | Estado | Evidencia |
| --- | --- | --- |
| 0. Entorno y calidad | Completada | Cargador, reporte de calidad y entorno registrado |
| 1. EDA | Completada | EDA descriptivo, EDA de defensa y notebook (D-011) |
| 2. División temporal | Completada | Tres particiones y validaciones automáticas |
| 3. Baseline | Completada | Repeat seleccionado |
| 4. Coocurrencia e híbrido | Completada | No seleccionados |
| 5. ALS | Completada | Implementación propia y grilla (D-005) |
| 6. Evaluación comparativa y selección | Completada | Tabla comparativa y configuración congelada (D-006) |
| 7. Evaluación final sobre prueba | Completada, una sola vez | `outputs/prueba/` (D-007) |
| 8. Reglas de negocio | Completada | Configuración y efecto medido (D-008) |
| 9. Prototipo | Completada | App Streamlit probada en el navegador (D-009) |
| 10. Pipeline y pruebas | Completada | 49 pruebas y prueba de integración (D-010) |
| 11. Informe final | Completada | `INFORME_FINAL.md` |
| 12. Presentación y defensa | Material listo | `PRESENTACION_Y_DEFENSA.md`; falta el ensayo |

---

## 30. Pendientes

1. **Revisión del equipo** de todo el trabajo. Conviene decidir en particular:
   - si se mantienen las exclusiones de `config/reglas_negocio.json`, por
     ejemplo el alcohol;
   - si el informe necesita ajustes de redacción o formato para Moodle.
2. **Commit** de todos los cambios. Están sin commitear a pedido del equipo;
   conviene hacer commits ordenados por bloque.
3. **Ensayo de la presentación** con preguntas cruzadas y la checklist de
   `PRESENTACION_Y_DEFENSA.md`.
4. **Opcional:** exportar el informe a PDF o Word si la entrega lo exige.

**Mejoras futuras, que no se hacen en este proyecto:**

- prueba A/B;
- mezcla según la antigüedad del hogar;
- evaluación por oferta en lugar de por código;
- integración real de stock, margen y campañas;
- feedback real;
- coocurrencia escalable;
- más canastas de evaluación por hogar.

---

## 31. Resumen ejecutivo final

El proyecto cumplió el plan. Sobre 2,6 millones de compras de 2.491 hogares
elegibles, se construyó un sistema que propone 10 candidatos a cupón por hogar:

- 7 repeat, con frecuencia + recencia;
- 3 explore, con ALS implícito implementado por el equipo;
- reglas de negocio explicadas, que excluyen combustible, tabaco y variantes.

Los modelos y la mezcla se eligieron en validación, y la prueba se usó una sola
vez. En prueba, **el 50,7 % de los hogares compró al menos uno de los 10
productos sugeridos**, con resultados estables respecto de validación. ALS
supera a popularidad en explore, y el baseline simple sigue siendo la mejor
opción para repeat.

El trabajo quedó documentado y es reproducible:

- un pipeline con un solo comando y 49 pruebas;
- un prototipo funcional con explicaciones;
- un informe final y material de defensa.

Lo que falta es humano: la revisión del equipo, el commit y el ensayo. El
límite principal no cambia: los resultados miden coincidencia con compras
futuras, no el efecto causal de un cupón, y eso solo lo mediría una prueba A/B.
