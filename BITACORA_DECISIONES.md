# Bitácora de decisiones

Este documento registra decisiones tomadas durante el proyecto. Su objetivo es
que el equipo pueda explicar en la defensa qué se decidió, con qué evidencia y
qué alternativas se consideraron.

## D-001 - Datos de entrada para la primera versión

- **Fecha:** 2026-10-01.
- **Decisión:** la primera versión del recomendador utilizará
  `transaction_data.csv` como fuente de comportamiento y `product.csv` para
  describir los productos recomendados.
- **Evidencia:** ambas fuentes contienen las claves necesarias para reconstruir
  hogares, canastas, orden temporal y productos; el catálogo permite explicar
  departamento, categoría y marca.
- **Alternativas consideradas:** datos demográficos, campañas, canjes y datos
  causales.
- **Motivo:** el informe define una primera versión basada en historial de
  compras. Las otras fuentes se conservarán para reglas de negocio o análisis
  posterior, sin mezclarlas aún en el entrenamiento.
- **Consecuencia:** las primeras métricas evaluarán capacidad de predecir compras
  posteriores; no probarán impacto causal de un cupón.

## D-002 - Cantidades iguales a cero

- **Fecha:** 2026-10-01.
- **Decisión:** excluir las filas con `QUANTITY = 0` de las interacciones de
  compra usadas para entrenar y evaluar el recomendador. Las fuentes originales
  se preservan sin cambios.
- **Evidencia:** la validación completa encontró 14.466 filas con cantidad cero
  y ninguna negativa. De esas filas, 14.399 tienen `SALES_VALUE = 0`; solo 67
  tienen una venta positiva. Además, 4.955 registran algún descuento de cupón.
- **Alternativas consideradas:** excluirlas antes de cualquier análisis o
  tratarlas como compras válidas.
- **Motivo:** una señal implícita de compra requiere unidades adquiridas
  positivas. Las filas de cantidad cero pueden conservarse para un futuro
  análisis de campañas o cupones, pero tratarlas como compra distorsionaría los
  rankings de productos.
- **Consecuencia:** una canasta que solo tenga cantidades cero no formará parte
  del historial de interacciones. Esta regla se aplicará antes de la separación
  temporal y se documentará en los resultados.

## D-003 - Baseline sin filtros comerciales

- **Fecha:** 2026-10-01.
- **Decisión:** el baseline evalúa todos los productos que aparecen en el
  entrenamiento, incluidos departamentos que podrían no ser candidatos a cupón
  en un contexto comercial real.
- **Evidencia:** `KIOSK-GAS` domina la popularidad global y aparece primero en
  algunas recomendaciones explore. El dataset no proporciona reglas completas
  de stock, margen ni campañas vigentes para decidir elegibilidad.
- **Motivo:** aplicar un filtro inventado alteraría la comparación inicial entre
  modelos. El baseline debe ser una referencia reproducible del comportamiento
  histórico sin reglas no verificadas.
- **Próximo paso:** la Fase 7 agregará filtros configurables y reportará su
  efecto por separado, sin presentar esos resultados como evaluación pura del
  modelo.

## D-004 - Umbral para asociaciones de coocurrencia

- **Fecha:** 2026-10-01.
- **Decisión:** la primera ejecución de coocurrencia usa productos presentes en
  al menos 500 canastas y pares presentes en al menos cinco canastas.
- **Evidencia:** un umbral de 20 canastas generaba aproximadamente 25,5 millones
  de pares potenciales y agotó la memoria. Con 500 canastas quedan 573 productos
  y aproximadamente 2,5 millones de pares potenciales; la ejecución produjo
  99.263 asociaciones con lift mayor a uno.
- **Motivo:** mantener un experimento reproducible y ejecutable en el equipo
  disponible es preferible a ocultar un problema de escala.
- **Limitación:** el modelo deja fuera asociaciones de productos poco frecuentes;
  se comparará con ALS u otras variantes antes de considerarlo final.

## D-005 - ALS implementado sin la librería `implicit`

- **Fecha:** 2026-10-15.
- **Decisión:** ALS implícito se implementa en `src/recomendar_als.py` usando
  solo numpy y scipy.sparse, en lugar de la librería `implicit`.
- **Evidencia:** `implicit` 0.7.3 se instala, pero al importarla Windows
  responde "Una directiva de Control de aplicaciones bloqueó este archivo"
  (DLL `topk`). Lo mismo ocurre con scikit-learn, así que tampoco sirve
  TruncatedSVD como alternativa. numpy, pandas y scipy.sparse sí cargan. No
  se modificó la configuración de seguridad del equipo.
- **Método:** formulación de Hu, Koren y Volinsky (2008) con gradiente
  conjugado de 3 pasos por actualización (Takács et al., 2011), el mismo
  esquema que `implicit` usa por defecto. Una prueba automática verifica que
  el gradiente conjugado coincide con la solución exacta del sistema lineal.
- **Señal y confianza:** frecuencia en canastas del par hogar-producto (no
  unidades), con `confianza = 1 + alpha * log(1 + frecuencia)`.
- **Grilla fijada antes de ver resultados:** factores {32, 64} x
  regularización {0,05; 0,10} x alpha {2; 10} x iteraciones {15, 30}, semilla 42.
  Las iteraciones 15 y 30 se evalúan dentro de la misma corrida.
- **Criterio de selección:** mayor NDCG@10 explore en validación (desempate
  por NDCG@10 repeat). El conjunto de prueba no se usa.
- **Explicabilidad:** cada sugerencia explore informa el producto del historial
  más similar en el espacio latente (similitud coseno de factores).
- **Limitación:** más lenta que `implicit` (unos 2,5 s por iteración con 32
  factores), lo que obliga a una grilla pequeña.
- **Resultado de la grilla (validación, K = 10):** la mejor configuración es
  32 factores, regularización 0,10, alpha 2 y 30 iteraciones. La
  regularización casi no cambia los resultados, y pasar de 15 a 30
  iteraciones aporta muy poco. Con alpha 10 y con 64 factores empeoran tanto
  repeat como explore.
- **Comparación:**
  - Explore ALS: Hit Rate@10 0,094964 y NDCG@10 0,017432. Popularidad:
    0,071942 y 0,014842. ALS supera a popularidad en las cuatro métricas y
    recomienda 1.051 productos distintos (popularidad, 81).
  - Repeat ALS: Hit Rate@10 0,461879. Frecuencia + recencia: 0,645352. Para
    repeat, el baseline sigue siendo claramente mejor.
- **Advertencia:** la configuración se eligió y se reporta sobre la misma
  validación, así que la cifra de ALS puede ser algo optimista. La
  estimación sin sesgo saldrá del conjunto de prueba (Etapa E).
- **Observación:** los mejores valores quedaron en el borde inferior de la
  grilla (alpha 2, 32 factores). Ampliarla hacia valores menores es opcional;
  si se hace, se debe registrar como una nueva decisión y no como un ajuste
  oculto.

## D-006 - Selección de modelos y mezcla repeat/explore

- **Fecha:** 2026-10-15.
- **Decisión:**
  - Repeat: frecuencia + recencia (baseline).
  - Explore: ALS (32 factores, regularización 0,10, alpha 2, 30 iteraciones).
  - Lista final de K = 10 con 7 repeat y 3 explore, con los repeat primero.
  - Configuración congelada en `config/modelo_final.json`, generada por
    `python -m src.seleccionar_modelo`.
- **Evidencia (validación, K = 10):**
  - Repeat: el baseline tiene NDCG 0,265 y ALS 0,141. Se conserva el baseline,
    que además es más simple y explicable.
  - Explore: ALS tiene NDCG 0,0174 y popularidad 0,0148 (+17 %). En Hit Rate,
    0,095 contra 0,072. ALS también recomienda 1.051 productos distintos
    contra 81.
  - Curva de mezcla (`outputs/seleccion/curva_mezcla_validacion.csv`): el
    Hit Rate total crece en forma monótona con la proporción repeat (0,079
    con 0 repeat; 0,501 con 7; 0,527 con 10). Medir solo aciertos llevaría
    siempre a una lista casi solo repeat.
- **Regla de mezcla:** mayor NDCG@10 total con al menos 3 posiciones explore.
  Ese mínimo es un requisito de producto, no un resultado del modelo: el
  sistema existe para proponer también productos nuevos. Con 7 + 3 se pierden
  2,6 puntos de Hit Rate total frente a una lista solo repeat (0,501 contra
  0,527), y el 4,0 % de los hogares acierta al menos un producto nuevo.
- **Alternativas:**
  - 6 + 4: Hit Rate 0,485 y descubrimiento 5,2 %.
  - 5 + 5: Hit Rate 0,464 y descubrimiento 6,1 %.
  - Se puede cambiar el mínimo con `--minimo-explore`, pero solo antes de
    abrir el conjunto de prueba.
- **Hogares con poco historial:** si un tipo no alcanza su cupo, la lista se
  completa con el otro. Así todo hogar recibe K productos.
- **Consecuencia:** desde aquí no se ajustan modelos, parámetros, K ni
  mezcla. La próxima etapa reentrena con entrenamiento + validación y evalúa
  una única vez sobre prueba.

## D-007 - Evaluación final única sobre prueba

- **Fecha:** 2026-10-15.
- **Procedimiento:** `python -m src.evaluar_prueba` leyó
  `config/modelo_final.json` sin modificarlo. Reentrenó repeat y ALS sobre
  entrenamiento + validación y evaluó una sola vez sobre la última canasta de
  cada hogar. Reporte: `outputs/prueba/reporte_prueba.md`.
- **Resultado principal (K = 10, 7 repeat + 3 explore):** Hit Rate 0,507,
  NDCG 0,174 y Recall 0,140 sobre 2.491 hogares. En validación: Hit Rate 0,501
  y NDCG 0,168.
- **Componentes:**
  - Repeat: Hit Rate 0,645, igual que en validación.
  - ALS explore: Hit Rate 0,086 (validación 0,095).
  - Popularidad (referencia): Hit Rate 0,065 (validación 0,072).
  - ALS mantiene la ventaja: +32 % en Hit Rate y +18 % en NDCG sobre
    popularidad.
- **Lectura:** las dos alternativas explore bajan algo en prueba, también
  popularidad, que no fue elegida. Por eso la caída se explica en buena parte
  por la canasta de prueba y no solo por haber elegido sobre validación. El
  sistema completo se mantiene estable.
- **Descubrimiento:** el 4,4 % de los hogares acierta al menos un producto
  nuevo con las 3 posiciones explore.
- **Sistema:** 100 % de hogares cubiertos, 5.827 productos distintos
  recomendados y 3,4 departamentos por lista. Reentrenar lleva unos 83 s y
  recomendar, unos 2,3 ms por hogar.
- **Nota:** después de la ejecución se corrigió solo la frase de interpretación
  del reporte, que atribuía toda la diferencia a la selección de modelos. No se
  modificó ningún número ni se volvió a evaluar.
- **Regla desde aquí:** el conjunto de prueba queda consumido. Cualquier
  cambio de modelo posterior se presenta como mejora futura, sin nuevas cifras
  de prueba.

## D-008 - Reglas de negocio configurables

- **Fecha:** 2026-10-15.
- **Decisión:** las reglas de elegibilidad se aplican después del ranking, en
  `src/reglas_negocio.py`, y se configuran en `config/reglas_negocio.json`.
  Cada candidato descartado queda registrado con su regla y su motivo
  (`outputs/reglas/filtradas_validacion.csv`). Completa lo previsto en D-003:
  el modelo se evalúa sin filtros y el efecto de las reglas se mide aparte.
- **Reglas activas:**
  - exigir catálogo con departamento;
  - evitar duplicados;
  - excluir `KIOSK-GAS` (combustible);
  - excluir departamentos que no son productos: transacciones, registros de
    cupones, donaciones, servicio postal y medicamentos con receta (`RX`);
  - excluir tabaco: categorías `CIGARETTES`, `CIGARS` y `TOBACCO OTHER`, cuya
    promoción prohíbe la Ley 18.256 en Uruguay. `SMOKING CESSATIONS` queda
    permitido;
  - evitar variantes dentro de una lista: dos códigos con la misma
    subcategoría, marca y tamaño son la misma oferta; queda el mejor ubicado;
  - evitar variantes en explore: un producto nuevo por código, pero con la
    misma subcategoría, marca y tamaño que algo ya comprado, no se ofrece como
    explore;
  - soporte mínimo de 10 canastas para sugerencias explore.
- **Cómo se detectaron tabaco y variantes:** al probar el prototipo con
  datos reales. En el hogar 20 aparecieron cigarrillos como repeat y la misma
  leche con otro código como explore. En el hogar 186, el mismo snack dos veces
  en la lista. Las métricas no lo mostraban.
- **Puntos de integración inactivos:**
  - Stock: lista vacía.
  - Margen: si se configura sin una fuente de márgenes, el sistema da error a
    propósito en lugar de ignorarlo.
  - Cupón histórico: se puede activar usando `coupon.csv`.
- **Supuestos:** las exclusiones son decisiones del equipo, no reglas reales
  del supermercado. Cada una lleva su motivo escrito en la configuración. El
  alcohol (`SPIRITS`) queda permitido, pero se puede excluir si el equipo lo
  decide.
- **Efecto en validación (K = 10, lista 7 + 3):**
  - Sin reglas, 1.483 posiciones de la lista final eran no elegibles
    (combustible, tabaco, etc.); con reglas, 0.
  - Variantes: 11.885 candidatos explore eran variantes de productos ya
    comprados, y 7.808 eran variantes repetidas dentro de una misma lista
    (4.217 repeat y 3.591 explore).
  - Hit Rate total: 0,501 sin reglas y 0,453 con reglas. Si se quitan del
    objetivo las compras no elegibles, es 0,480. La mayor parte de la caída
    viene de la nafta y el tabaco, que eran aciertos fáciles pero no
    ofrecibles. El resto, de quitar variantes repetidas: antes, dos variantes
    daban dos oportunidades de acertar el código exacto comprado. Las métricas
    cuentan aciertos por código de producto, no por oferta, así que esa parte
    de la caída es un efecto de la métrica más que una pérdida real.
  - Descubrimiento explore: 4,0 % sin reglas y 4,2 % con reglas (objetivo
    elegible).
  - Todos los hogares siguen recibiendo 10 productos elegibles. Los productos
    distintos recomendados suben de 5.925 a 6.116. Hay 3,00 departamentos por
    lista, contra 3,36 sin reglas (cuando se incluía el combustible).
  - El soporte mínimo no filtró ningún candidato; queda como protección.
- **Alcance:** el efecto se midió en validación. El conjunto de prueba no se
  vuelve a usar (D-007). El prototipo muestra las listas con reglas aplicadas
  y los productos filtrados con su motivo.

## D-009 - Prototipo para el área comercial

- **Fecha:** 2026-10-15.
- **Decisión:** el prototipo es una app Streamlit (`app/app.py`) que solo lee
  archivos precalculados por `python -m src.preparar_demo`. No entrena en
  vivo, así que la demo es determinista y rápida.
- **Datos de la demo:** el sistema congelado (D-006) con las reglas (D-008) se
  reentrena con todo el historial disponible (entrenamiento + validación +
  prueba) y recomienda para la próxima compra. Estas listas no tienen métricas
  propias: la app muestra las métricas ya reportadas de prueba (D-007) y
  aclara que miden coincidencia con compras, no efecto causal.
- **Flujo mostrado:**
  1. Elegir un hogar: 5 casos de ejemplo elegidos de forma determinista, o
     cualquier `household_key`.
  2. Ver el resumen del historial.
  3. Ver 7 repeat y 3 explore, cada uno con su explicación en texto.
  4. Revisar los candidatos descartados por las reglas, con su motivo.
  5. Exportar la lista a CSV.
  6. Dar feedback simulado (👍/👎), que se guarda en
     `outputs/demo/feedback_simulado.csv`.
- **Casos de error:**
  - Hogar inexistente: aviso y respaldo con los productos populares
    elegibles, uno por subcategoría.
  - Historial corto: aviso de poca evidencia.
- **Hallazgos de probar la demo:** aparecieron cigarrillos y variantes del
  mismo producto, que se resolvieron con reglas (ver D-008). Es evidencia de
  que la revisión cualitativa encuentra problemas que las métricas no
  muestran.
- **Seguridad:** la app escucha solo en `localhost` (`.streamlit/config.toml`).
  Por defecto Streamlit se exponía en la red.

## D-010 - Pipeline reproducible y pruebas de integración

- **Fecha:** 2026-10-19.
- **Decisión:** `python -m src.pipeline` ejecuta las etapas en orden, usando
  los mismos comandos del README:
  calidad → EDA → división → baseline → coocurrencia → ALS → comparación →
  [selección] → [prueba] → reglas → demo.
- **Registro de cada corrida:**
  - versiones de Python y librerías, y commit de git, en `outputs/entorno.txt`;
  - comando, duración y código de salida de cada etapa, en
    `outputs/registro_pipeline.json`.
- **Protecciones:**
  - Selección y prueba no se ejecutan sin `--recongelar` e
    `--incluir-prueba`, para no reescribir la configuración congelada (D-006)
    ni reabrir la prueba (D-007).
  - La prueba no se repite sin `--forzar-prueba`.
  - El pipeline calcula SHA-256 de los CSV de `data/` antes y después, y
    falla si alguno cambió.
- **Prueba de integración:** `tests/test_pipeline.py` corre todas las etapas (12 desde D-011)
  sobre un dataset sintético (`tests/datos_sinteticos.py`) en una carpeta
  temporal, en unos 15 s. Ese dataset incluye dos grupos de gustos, nafta,
  tabaco, variantes y cantidades cero. La prueba verifica:
  - que todas las etapas terminen bien y los datos queden intactos;
  - que la división y la configuración congelada sean correctas;
  - que la prueba no pueda repetirse;
  - que la demo cumpla las reglas: 10 productos, sin nafta, sin tabaco, sin
    variantes repetidas y con explicaciones.
- **Hallazgo:** con datos sintéticos, la selección elige popularidad para
  explore, y la evaluación final y la demo solo sabían reentrenar ALS. Se
  generalizó la generación de candidatos (`generar_candidatos`) para aceptar
  ALS o popularidad.
- **Verificación con los datos reales:**
  - La lista final de prueba regenerada es idéntica a la guardada (se
    comprobó sin abrir la canasta de prueba).
  - La demo regenerada tiene el mismo contenido; solo cambió el nombre de
    `puntaje_als` a `puntaje_explore`.
  - Los datos de entrada siguen intactos.
- **Orden:** `data/prueba.py`, la exploración inicial, se movió a
  `exploracion/primer_vistazo.py`, para que `data/` contenga solo datos
  originales.

## D-011 - EDA para la defensa: evidencia de las decisiones

- **Fecha:** 2026-10-19.
- **Qué se hizo:**
  - `src/analizar_repeticion.py`, que también corre como etapa `eda_defensa`
    del pipeline.
  - `notebooks/01_eda_defensa.ipynb`, solo de presentación.
  - Se usan entrenamiento + validación y las listas de validación guardadas.
    No se usa prueba y no se cambian modelos.
- **Hallazgos:**
  1. **Recompra:** sin contar la primera canasta, el 50,1 % de los productos
     de una canasta ya habían sido comprados por el hogar (46,5 % ponderado
     por línea). Sube con la antigüedad del cliente: 12 % en las canastas 2 a
     5 y 64 % desde la 101. A nivel de variante, la recompra media es 60,7 %.
     Respalda separar repeat y explore, y la regla de variantes (D-008).
  2. **Objetivos en validación:** de 2.491 hogares, 2.033 recompran, 2.085
     compran algo nuevo, 1.627 hacen ambas cosas y ninguno se queda sin
     objetivo.
  3. **Historial y cold start:** repeat rinde parejo en los tres segmentos
     (Hit Rate@10 entre 0,63 y 0,67). Explore depende mucho del historial:
     - ALS: 0,156 con historial corto (1 a 48 canastas), 0,081 con historial
       medio y 0,033 con historial largo (más de 116).
     - Popularidad: 0,123, 0,059 y 0,022.

     ALS gana en los tres segmentos.
  4. **Concentración:** el 1 % de los productos reúne el 34,8 % de las
     apariciones en canastas y el 10 %, el 76,8 % (Gini 0,844). De 90.927
     productos, 33.092 aparecen en una sola canasta. Las listas explore de
     popularidad recomiendan 45 productos distintos de GROCERY; las de ALS, 657.
  5. **Combustible:** lo compra el 54,4 % de los hogares y el 82,7 % de ellos
     lo recompra. Sin reglas aparecía en la lista del 44,7 % de los hogares.
  6. **Tabaco:** lo compra el 45,4 % de los hogares y el 70,7 % de ellos lo
     recompra. Sin reglas aparecía en la lista del 8,2 % de los hogares.
- **Consecuencia:** no se cambia nada del sistema congelado. El hallazgo 3
  queda como mejora futura: la proporción repeat/explore podría depender de
  la antigüedad del hogar, con más explore para clientes nuevos. Debería
  validarse en un experimento nuevo, porque la prueba ya fue usada (D-007).
