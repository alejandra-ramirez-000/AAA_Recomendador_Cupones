# Plan de trabajo — Cupones personalizados AAA

## 1. Propósito y alcance

Construir una prueba de concepto que, dado un `household_key` y su historial de
compras anterior a una fecha de corte, recomiende productos candidatos a cupón.
El resultado deberá distinguir recomendaciones **repeat** (productos ya
comprados) de recomendaciones **explore** (productos no comprados previamente
por el hogar).

El alcance inicial se centra en el comportamiento de compra. No se emplearán
las variables demográficas como entrada del modelo. Los datos de campañas y
cupones se conservarán para análisis posterior y posibles reglas de negocio.

## 2. Hitos académicos y evidencia requerida

El plan se organiza para cumplir los dos entregables y el seguimiento continuo
solicitados para AIAI. Las fechas exactas se completarán cuando se publiquen en
Moodle.

| Hito | Evidencia a preparar | Criterio de finalización |
| --- | --- | --- |
| Entregable 1 - Propuesta | Informe actual revisado, plan de trabajo, diagrama de arquitectura y presentación de 8 minutos | El equipo puede justificar problema, datos, arquitectura, modelos, métricas y límites; deja 5 minutos para preguntas. |
| Seguimiento continuo | Bitácora de avances, notebooks, Colabs y entregas parciales con hallazgos reales | Cada avance puede ejecutarse, explicarse y relacionarse con conceptos de la materia. |
| Entregable 2 - Versión final | Documento de 10 a 15 páginas, prototipo funcional, resultados y demo | En noviembre se puede recorrer entrada -> procesamiento -> modelo -> salida -> interpretación en 10 minutos. |
| Defensa técnica | Guion de preguntas, fichas de decisiones y ensayo grupal | Cada integrante explica el flujo completo y responde 6 minutos de preguntas, incluso sobre partes que no implementó. |

### Diagrama obligatorio de arquitectura

Antes del Entregable 1 se actualizará el diagrama del informe y se verificará
que muestre explícitamente esta cadena:

```text
Transacciones + catálogo -> validación/preparación -> historial temporal
-> generación de candidatos -> ranking/modelo -> reglas de negocio
-> prototipo para el área comercial -> recomendaciones interpretables
-> registro de feedback/uso para una futura actualización del modelo
```

El diagrama debe indicar qué datos entran en cada etapa, qué se almacena o
exporta y qué componentes son parte del prototipo actual frente a los que son
una ampliación futura. De esta manera no se presenta como implementado algo que
solo es una hipótesis de producción.

## 3. Regla de legibilidad del código

Todo código nuevo seguirá estas reglas:

1. Usar nombres descriptivos en español o inglés, pero no abreviaturas crípticas.
2. Agregar un encabezado a cada archivo que explique su responsabilidad, datos
   de entrada, datos de salida y cómo ejecutarlo.
3. Documentar cada función con una descripción de parámetros, resultado y
   supuestos.
4. Comentar casi cada línea que transforme datos, calcule una métrica o aplique
   una regla de recomendación. Las líneas evidentes, como un `import`, también
   tendrán un comentario breve cuando ayude a quien estudia el código.
5. Separar pasos complejos en variables intermedias nombradas y comentadas;
   evitar expresiones largas encadenadas.
6. Incluir mensajes de validación claros y una pequeña muestra impresa de los
   resultados en cada etapa.

Ejemplo de estilo esperado:

```python
# Conservamos solo las compras realizadas hasta la fecha de corte para no usar
# información que el modelo aún no conocería.
compras_historicas = transacciones[transacciones["DAY"] <= dia_de_corte].copy()

# Contamos cuántas veces cada hogar compró cada producto dentro de su historial.
frecuencia_producto = (
    compras_historicas.groupby(["household_key", "PRODUCT_ID"])
    .size()
    .reset_index(name="cantidad_compras")
)
```

## 4. Estructura propuesta

```text
Proyecto_AAA/
├── data/                         # Datos originales: no se modifican
├── notebooks/                    # Análisis exploratorio y resultados visuales
├── src/                          # Funciones reutilizables, comentadas
│   ├── cargar_datos.py
│   ├── preparar_datos.py
│   ├── dividir_temporalmente.py
│   ├── recomendar_baseline.py
│   ├── recomendar_coocurrencia.py
│   ├── recomendar_als.py
│   ├── evaluar_modelos.py
│   └── reglas_negocio.py
├── tests/                        # Validaciones sencillas de funciones clave
├── outputs/                      # Tablas, métricas y recomendaciones generadas
├── requirements.txt              # Dependencias reproducibles
└── README.md                     # Instrucciones de ejecución del prototipo
```

Los datos de gran tamaño se leerán solo con las columnas necesarias y, cuando
corresponda, por bloques (`chunksize`). `causal_data.csv` no se cargará al
inicio: se integrará únicamente si aporta una regla o evaluación concreta.

## 5. Fases y entregables

### Fase 0 — Preparación del entorno y contrato de datos

**Objetivo:** asegurar que todo el equipo puede ejecutar el proyecto de forma
reproducible y conoce cada fuente.

- Crear `requirements.txt` con versiones mínimas de `pandas`, `numpy`,
  `scikit-learn`, `scipy`, `matplotlib`, `seaborn` y, si ALS se implementa,
  `implicit`.
- Crear `src/cargar_datos.py` para cargar y validar columnas, tipos y valores
  faltantes de `transaction_data.csv` y `product.csv`.
- Documentar el diccionario de variables relevantes:
  `household_key`, `BASKET_ID`, `DAY`, `PRODUCT_ID`, `QUANTITY`,
  `SALES_VALUE`, descuentos, categoría y subcategoría.
- Verificar filas duplicadas, productos sin catálogo y fechas/días inválidos.
- Abrir una bitácora de decisiones: para cada decisión se registrará la fecha,
  evidencia disponible, alternativa descartada y motivo. Será una fuente de
  estudio para la defensa y evita justificar decisiones retrospectivamente.

**Entregables:** entorno documentado, función de carga y reporte de calidad
inicial.

### Fase 1 — Análisis exploratorio (EDA)

**Objetivo:** entender el patrón de compras antes de decidir los parámetros.

- Cuantificar hogares, canastas, productos y periodo temporal.
- Calcular compras y canastas por hogar, productos más vendidos y distribución
  de cantidad/precio/descuento.
- Unir transacciones con catálogo para analizar departamentos, categorías y
  subcategorías.
- Medir la proporción de productos repetidos entre canastas sucesivas de cada
  hogar; esto valida la utilidad esperada del componente repeat.
- Crear gráficos y tablas en un notebook claramente explicado.

**Entregables:** `notebooks/01_eda.ipynb` y un resumen de hallazgos que defina
filtros razonables (por ejemplo, productos extremadamente raros).

### Fase 2 — Construcción del historial y división temporal

**Objetivo:** preparar datos sin fuga de información futura.

- Ordenar las canastas de cada hogar por `DAY` y `BASKET_ID`.
- Conservar hogares con al menos tres canastas.
- Separar por hogar: canastas antiguas para entrenamiento, la penúltima para
  validación y la última para prueba. Si se requiere más estabilidad, reservar
  las últimas dos para prueba y la anterior para validación.
- Crear tablas de interacción hogar-producto y hogar-canasta para cada división.
- Confirmar con verificaciones automáticas que ningún evento de validación o
  prueba aparece en el historial usado para generar sus recomendaciones.

**Entregables:** `src/dividir_temporalmente.py`, tablas procesadas y un reporte
de tamaños por partición.

### Fase 3 — Modelo base interpretable

**Objetivo:** disponer de una referencia sólida antes de modelos complejos.

- **Repeat:** puntuar productos ya comprados combinando frecuencia, recencia y,
  opcionalmente, cantidad adquirida.
- **Explore:** recomendar productos populares que el hogar no haya comprado;
  calcular popularidad solo con entrenamiento.
- Definir un parámetro de mezcla, por ejemplo 60 % repeat y 40 % explore, y
  ajustarlo con validación.
- Mostrar para algunos hogares el producto, su categoría, frecuencia, última
  compra y motivo de recomendación.
- Redactar una ficha breve del baseline: qué necesidad resuelve, por qué se
  ajusta a datos implícitos y qué limitaciones tiene frente a coocurrencia y
  ALS. Cada modelo candidato tendrá una ficha equivalente.

**Entregables:** `src/recomendar_baseline.py`, recomendaciones ejemplo y
métricas de validación.

### Fase 4 — Recomendaciones por coocurrencia e híbrido

**Objetivo:** mejorar el componente explore con relaciones entre productos.

- Construir pares de productos que aparecen en una misma canasta de
  entrenamiento.
- Calcular una medida explícita y documentada, inicialmente `lift` o similitud
  de coseno; descartar asociaciones con poco soporte.
- Para cada hogar, usar sus productos históricos para proponer productos
  relacionados que aún no compró.
- Crear una variante híbrida que combine el puntaje de coocurrencia con
  popularidad normalizada.

**Entregables:** `src/recomendar_coocurrencia.py`, tabla de asociaciones y
comparativa contra el baseline.

### Fase 5 — Filtrado colaborativo ALS

**Objetivo:** evaluar personalización basada en patrones compartidos entre
hogares.

- Construir una matriz dispersa hogar-producto con una señal implícita
  explicada (por ejemplo, número de canastas que contienen el producto).
- Entrenar ALS únicamente con entrenamiento.
- Ajustar factores latentes, regularización e iteraciones con validación.
- Generar listas repeat y explore filtrando o etiquetando según el historial
  del hogar, sin ocultar cómo se tomó esa decisión.
- Si la instalación o el costo computacional impiden ALS, documentar el motivo
  y conservar los modelos anteriores como alternativas válidas.

**Entregables:** `src/recomendar_als.py`, parámetros elegidos, tiempo de
ejecución y resultados de validación.

### Fase 6 — Evaluación rigurosa

**Objetivo:** seleccionar un enfoque con evidencia reproducible.

- Calcular Precision@5, @10 y @20; Recall@5, @10 y @20; Hit Rate@K y NDCG@K.
- Evaluar en validación para elegir parámetros y una única vez en prueba para
  informar el resultado final.
- Reportar métricas globales y por tipo repeat/explore.
- Comparar calidad, cobertura, diversidad, interpretabilidad, tiempo de
  entrenamiento y costo de implementación.
- Separar en el informe las métricas de ML de las métricas de aplicación. Las
  primeras serán Precision, Recall, Hit Rate y NDCG. Para el prototipo se
  medirán tiempo de respuesta por consulta, porcentaje de recomendaciones con
  explicación visible y cobertura de hogares/productos que pueden recibir una
  recomendación. El canje de cupones, ventas incrementales y retención se
  documentarán como métricas de negocio futuras, no como resultados medidos por
  este dataset sin un experimento causal.
- Inspeccionar ejemplos buenos y malos para comprobar que las métricas reflejan
  recomendaciones razonables.

**Entregables:** `src/evaluar_modelos.py`, tabla comparativa y decisión del
modelo final justificada.

### Fase 7 — Reglas de negocio y prototipo

**Objetivo:** convertir recomendaciones técnicas en candidatos a cupón claros.

- Aplicar filtros configurables: excluir productos sin disponibilidad conocida,
  duplicados, productos de baja evidencia o artículos ya promocionados, cuando
  los datos lo permitan.
- Dejar preparado un punto de integración para campañas, cupones canjeados y
  `causal_data.csv`; no atribuir causalidad a un cupón sin un diseño
  experimental.
- Crear un prototipo sencillo donde se ingrese o seleccione un hogar y se vean
  dos listas: repeat y explore, con puntaje y explicación breve.
- Exportar los resultados a CSV para uso del área comercial.
- Registrar de forma local y explícitamente las acciones de demostración, por
  ejemplo hogar consultado, recomendación vista y respuesta simulada. El
  registro será evidencia del componente de feedback de la arquitectura, pero
  no se confundirá con feedback real de clientes.

**Entregables:** prototipo ejecutable, archivo de recomendaciones y guía de
uso.

### Fase 8 — Cierre, prueba y presentación

**Objetivo:** dejar una entrega defendible y fácil de retomar.

- Añadir pruebas para la división temporal, exclusión de productos ya comprados
  en explore y cálculo de métricas.
- Ejecutar el flujo completo desde datos originales y registrar versiones,
  parámetros y fecha de ejecución.
- Documentar limitaciones: datos históricos, cold start, stock/margen ausentes
  y diferencia entre predicción de compra e impacto causal de un cupón.
- Preparar una demostración con hogares anonimizados y recomendaciones
  explicables.
- Preparar el documento final de 10 a 15 páginas y mantener un anexo técnico
  con tablas completas, código y resultados extendidos.
- Realizar al menos un ensayo de la defensa: cada integrante presenta una parte
  breve y responde preguntas sobre datos, arquitectura, modelos, métricas,
  resultados y limitaciones de todo el sistema.

**Entregables:** README final, resultados reproducibles, presentación y demo.

## 6. Orden sugerido de implementación

1. Fase 0 y Fase 1.
2. Fase 2.
3. Fase 3 (primer resultado completo y demostrable).
4. Fase 6 para medir el baseline correctamente.
5. Fase 4 y nuevamente Fase 6.
6. Fase 5 y nuevamente Fase 6.
7. Fases 7 y 8.

La primera meta funcional será llegar al final de la Fase 3: generar para un
hogar una lista repeat y una explore usando exclusivamente compras anteriores a
su fecha de corte. A partir de allí cada modelo nuevo deberá superar o justificar
frente a ese baseline antes de incorporarse al prototipo.

## 7. Criterios de aceptación finales

- Ninguna recomendación usa información posterior al momento que pretende
  predecir.
- Cada recomendación se identifica como repeat o explore y tiene un puntaje o
  explicación comprensible.
- El modelo final se compara con un baseline mediante las cuatro métricas
  definidas en el informe.
- El proyecto se ejecuta siguiendo el README y genera los mismos resultados
  con los mismos datos y parámetros.
- El código tiene comentarios didácticos, variables descriptivas y validaciones
  suficientes para que otra persona pueda seguirlo paso a paso.
- La documentación final diferencia resultados medidos, hipótesis de negocio y
  mejoras futuras; no oculta resultados limitados o negativos.
- El diagrama y la demo muestran de manera coherente datos, modelo, aplicación,
  usuario y el mecanismo de feedback registrado.
- Cada integrante puede explicar y justificar decisiones de cualquier fase del
  proyecto, no solo el módulo que implementó.
