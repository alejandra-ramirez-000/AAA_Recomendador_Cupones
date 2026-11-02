# Guía de trabajo para continuar el proyecto

**Proyecto:** Recomendador de productos candidatos a cupones personalizados  
**Base:** `CONTEXTO_COMPLETO_PROYECTO.md` (estado al 15/10/2026)  
**Objetivo de esta guía:** convertir el plan de cierre (secciones 24–27 del
contexto) en bloques de trabajo ordenados, con archivos, comandos y criterios
de terminado concretos.

---

## 0. Punto de partida en una mirada

| Ya está hecho | Falta |
| --- | --- |
| Carga y calidad (`src/cargar_datos.py`) | Módulo común de evaluación |
| EDA reproducible (`src/analizar_datos.py`) | EDA de repetición entre canastas y cold start |
| División temporal por hogar (`src/dividir_temporalmente.py`) | ALS (o su inviabilidad documentada) |
| Baseline repeat + popularidad (`src/recomendar_baseline.py`) | Tabla comparativa y selección de modelo |
| Coocurrencia + híbrido (`src/recomendar_coocurrencia.py`) | Evaluación única sobre prueba |
| 6 pruebas unitarias en verde | Reglas de negocio, prototipo, pipeline, README, informe |

**Resultados a superar (validación, K = 10):**

- Repeat (frecuencia + recencia): Hit Rate@10 = **0,645**
- Explore (popularidad): Hit Rate@10 = **0,072**

**Regla de oro durante todo el trabajo:** `outputs/division_temporal/interacciones_prueba.csv`
**no se lee** hasta el Bloque 5. Todo ajuste y comparación se hace en validación.

---

## 1. Higiene del repositorio (antes de empezar, ~1 hora)

Pequeños arreglos que evitan problemas más adelante:

- [ ] Crear `.gitignore` con `__pycache__/`, `*.pyc`, `.venv/`, `tmp/` y
      sacar del índice lo ya versionado:
      `git rm -r --cached src/__pycache__ tests/__pycache__ tmp`
- [ ] Decidir qué hacer con `data/prueba.py`: borrarlo o moverlo a
      `notebooks/`/`scratch/` con un comentario explicando que es exploratorio.
- [ ] Commitear `CONTEXTO_COMPLETO_PROYECTO.md` y el cambio pendiente en
      `README.md`.
- [ ] Verificar el entorno: `python -m unittest discover -s tests -v` → 6 OK.

---

## 2. Bloque 1 — Evaluación común y reproducibilidad (Etapa A) · **crítico**

**Meta:** un único comando produce una tabla comparable de todos los modelos
sobre validación.

### Tareas

1. **Crear `src/evaluar_modelos.py`** y mover allí desde
   `src/recomendar_baseline.py`:
   - `obtener_productos_relevantes` (separación repeat/explore);
   - `calcular_metricas_en_k` (Precision, Recall, Hit Rate, NDCG).
   - En `recomendar_baseline.py` y `recomendar_coocurrencia.py` reemplazar por
     `from src.evaluar_modelos import ...`. Las métricas resultantes deben ser
     **idénticas** a los JSON actuales (comparar antes/después).
2. **Agregar métricas de sistema** (funciones nuevas en el mismo módulo):
   - cobertura de hogares (hogares con ≥ 1 recomendación / hogares elegibles);
   - cobertura de catálogo (productos distintos recomendados / productos en entrenamiento);
   - concentración de exposición (p. ej. % de recomendaciones que acaparan los
     10 productos más recomendados, o índice de Gini);
   - diversidad intra-lista (departamentos/categorías distintos por lista);
   - tiempo de entrenamiento y tiempo medio de recomendación por hogar;
   - % de recomendaciones con explicación.
3. **Tabla comparativa automática:** que el módulo lea los
   `metricas_validacion.json` (y los CSV de recomendaciones) de cada carpeta de
   `outputs/` y escriba `outputs/comparacion/comparacion_validacion.csv` (+ `.md`).
   Comando objetivo: `python -m src.evaluar_modelos`.
4. **Pruebas de borde** en `tests/`: hogar sin relevantes, lista menor que K,
   empates de puntaje, producto ausente del catálogo, métricas de cobertura.
5. **Actualizar `README.md`** con todas las fases ya completadas y sus comandos.

### Terminado cuando
- `python -m src.evaluar_modelos` genera la tabla comparativa sin tocar prueba.
- Las métricas de baseline y coocurrencia no cambiaron tras el refactor.
- Pruebas en verde.

---

## 3. Bloque 2 — EDA para la defensa (Etapa B) · alta

**Meta:** justificar con datos la separación repeat/explore.

### Tareas (agregar a `src/analizar_datos.py` o a un nuevo `src/analizar_repeticion.py`)

1. % de productos de cada canasta que ya estaban en canastas anteriores del
   hogar (repetidos vs. nuevos), y su evolución en el tiempo.
2. % de hogares con objetivos repeat y explore en validación (ya se sabe:
   2.033 y 2.085 de 2.491 — dejarlo exportado).
3. Distribución de productos únicos por hogar y longitud de historial.
4. Desempeño del baseline por segmento de historial (p. ej. terciles de
   cantidad de canastas) → evidencia de cold start.
5. Concentración de popularidad (curva de Lorenz / top‑N acumulado) y
   cobertura del catálogo por departamento.
6. Tabla/figura específica sobre `KIOSK-GAS`.
7. Decidir: ¿notebook de presentación (`notebooks/01_eda.ipynb`) que solo
   lea los outputs? Recomendado: sí, pero sin lógica propia.

### Terminado cuando
- Existen tablas/gráficos en `outputs/eda/` que el informe puede citar para
  "por qué repeat y explore se evalúan por separado".

---

## 4. Bloque 3 — ALS (Etapa C) · **crítico**

**Meta:** comparar filtrado colaborativo contra los baselines con la misma
metodología.

### Tareas

1. **Validar `implicit` en Windows primero** (puede ser el principal riesgo):
   `pip install implicit` y un ejemplo mínimo. Si falla, pasar al plan B
   (punto 8) sin perder tiempo.
2. Crear `src/recomendar_als.py`:
   - matriz dispersa hogar × producto (`scipy.sparse.csr_matrix`) **solo con
     entrenamiento**;
   - señal = número de canastas que contienen el producto (no cantidades);
   - confianza `1 + alpha * log(1 + frecuencia_en_canastas)`;
   - mapeo de índices ↔ `household_key`/`PRODUCT_ID` guardado junto al modelo.
3. Grilla pequeña y fija (documentar en la bitácora antes de correr):
   factores {32, 64} × regularización {0,05; 0,10} × iteraciones {15, 30} ×
   alpha {pocos valores}. Fijar `random_state`.
4. Separar la salida de ALS en **repeat** (productos ya comprados) y
   **explore** (`filter_already_liked_items=True`).
5. Evaluar con `src.evaluar_modelos` (mismas funciones, mismos K = 5/10/20).
6. Registrar tiempo, memoria y cobertura de cada configuración.
7. Exportar a `outputs/als/` (métricas JSON, recomendaciones, grilla CSV).
8. **Plan B:** si `implicit` no funciona o es inviable, `TruncatedSVD` de
   scikit-learn sobre la misma matriz, y documentar la limitación (D-005).

### Terminado cuando
- ALS aparece en la tabla comparativa junto a repeat, popularidad,
  coocurrencia e híbrido.
- Nueva entrada en `BITACORA_DECISIONES.md` (señal de confianza y grilla).

---

## 5. Bloque 4 — Selección del modelo en validación (Etapa D) · **crítico**

**Meta:** decisión explícita y defendible **antes** de abrir prueba.

1. Para repeat y explore por separado, comparar: métricas de ranking,
   cobertura, diversidad, interpretabilidad, tiempo/memoria, complejidad.
2. Puede elegirse un modelo distinto por objetivo (p. ej. repeat = baseline,
   explore = popularidad o ALS). **Si nada supera claramente al baseline, elegir
   el baseline** y decirlo.
3. Definir la mezcla final de la lista (punto de partida 60 % repeat / 40 %
   explore) y el K principal de la demo; validar la mezcla con métricas de la
   lista combinada.
4. Registrar como **D-006 — Selección del modelo** (resultados, costo,
   limitaciones) y congelar la configuración en un archivo, p. ej.
   `config/modelo_final.json`.

### Terminado cuando
- La bitácora contiene la decisión y existe la configuración congelada.

---

## 6. Bloque 5 — Evaluación final sobre prueba (Etapa E) · **crítico**

**Meta:** estimación final sin sesgo de selección. Se ejecuta **una sola vez**.

1. Script `src/evaluar_prueba.py` que lee `config/modelo_final.json`.
2. Reentrenar con entrenamiento + validación.
3. Redefinir repeat/explore respecto del historial ampliado.
4. Evaluar sobre la última canasta.
5. Guardar en `outputs/prueba/`: métricas, configuración, fecha, versiones de
   librerías y tiempos.
6. **No** ajustar nada después. Si algo sale mal, documentarlo como
   limitación.

### Terminado cuando
- Existe `outputs/prueba/reporte_prueba.md` separado de los experimentos.

---

## 7. Bloque 6 — Reglas de negocio configurables (Etapa F) · alta

1. `src/reglas_negocio.py` + `config/reglas_negocio.json`, aplicado **después**
   del ranking (no altera el entrenamiento).
2. Reglas: departamentos excluidos (`KIOSK-GAS` como opción visible), producto
   en catálogo, sin duplicados, soporte mínimo, ganchos vacíos para stock,
   margen y campaña.
3. Cada recomendación filtrada lleva el motivo (`motivo_filtro`).
4. Reportar métricas y cobertura **antes y después** del filtro como resultado
   aparte, no como métrica del modelo.

---

## 8. Bloque 7 — Prototipo Streamlit (Etapa G) · **crítico**

1. `app/app.py` (agregar `streamlit` a `requirements.txt`).
2. La app **solo lee** recomendaciones precalculadas (sin entrenar en vivo).
3. Contenido mínimo:
   - selector de `household_key` + 3–5 hogares de demo preseleccionados;
   - resumen del historial del hogar;
   - pestañas Repeat / Explore con producto, categoría, posición, puntaje;
   - explicación textual por recomendación (p. ej. "comprado en 12 canastas,
     última hace 3" / "popular entre 1.800 hogares");
   - indicador de reglas comerciales aplicadas y productos filtrados;
   - exportar CSV;
   - botón de "feedback simulado" que escribe a un log local.
4. Manejar hogar inexistente y hogar con poco historial (mensaje claro +
   fallback a popularidad).
5. Comando: `streamlit run app/app.py`.

### Terminado cuando
- En < 10 min se recorre entrada → procesamiento → modelo → salida →
  interpretación, incluidos casos de error.

---

## 9. Bloque 8 — Pipeline, pruebas y documentación (Etapa H) · alta

1. `src/pipeline.py` (o `run_all.ps1`) que ejecute en orden: calidad → EDA →
   división → baseline → coocurrencia → ALS → comparación (y, con bandera
   explícita, prueba).
2. Registrar parámetros y semillas en cada salida.
3. Prueba de integración con un mini‑dataset sintético en `tests/fixtures/`.
4. Verificar que ningún script escriba en `data/`.
5. Generar `outputs/entorno.txt` (`pip freeze`, versión de Python).
6. README final: instalación, comandos, estructura, resultados.

---

## 10. Bloque 9 — Informe, presentación y defensa (Etapa I) · **crítico**

1. Informe de 10–15 páginas siguiendo los 12 puntos de la sección 24 del
   contexto. Reutilizar texto de `CONTEXTO_COMPLETO_PROYECTO.md`, respetando
   las secciones 21 ("se puede afirmar") y 22 ("no se debe afirmar").
2. Incluir ejemplos cualitativos de 2–3 hogares.
3. Presentación de ~10 minutos + demo.
4. Ensayo con preguntas cruzadas: cada integrante explica datos, división
   temporal, puntajes, métricas, selección del modelo, límites causales y
   cómo se integraría feedback real.

---

## 11. Orden sugerido y calendario orientativo

Ajustar a la fecha real de entrega publicada en Moodle (noviembre).

| Semana | Bloques | Hito de control |
| --- | --- | --- |
| 1 | Higiene + Bloque 1 + inicio Bloque 3 (validar `implicit`) | Tabla comparativa automática funcionando |
| 2 | Bloque 3 (ALS) + Bloque 2 (EDA) en paralelo | ALS en la tabla comparativa |
| 3 | Bloque 4 + Bloque 5 + Bloque 6 | Modelo congelado y reporte de prueba |
| 4 | Bloque 7 + Bloque 8 | Demo completa y pipeline reproducible |
| 5 | Bloque 9 | Informe entregado y ensayo de defensa |

Paralelización posible en equipo: Bloque 2 (EDA) y Bloque 7 (esqueleto de la
app con datos del baseline) pueden avanzar mientras otra persona hace ALS.

---

## 12. Próxima sesión de trabajo (concreta)

1. Higiene del repositorio (sección 1).
2. Crear `src/evaluar_modelos.py` y mover las funciones de métricas.
3. Reejecutar `python -m src.recomendar_baseline` y
   `python -m src.recomendar_coocurrencia` y comprobar que los JSON no cambian.
4. Probar `import implicit` en el entorno.
5. Actualizar README.

---

## 13. Lista de control final (Definición de terminado)

- [ ] Datos originales intactos
- [ ] Ejecución completa documentada
- [ ] Sin fuga temporal
- [ ] Todos los modelos comparados con la misma metodología
- [ ] Modelo final elegido en validación
- [ ] Prueba usada una sola vez, tras congelar decisiones
- [ ] Repeat y explore definidos y evaluados por separado
- [ ] Cada recomendación mostrada con explicación
- [ ] Reglas de negocio separadas del modelo
- [ ] Prototipo con casos preparados y casos de error
- [ ] Cobertura, latencia, calidad y limitaciones reportadas
- [ ] Informe distingue resultados, hipótesis y mejoras futuras
- [ ] Pruebas automatizadas en verde
- [ ] Cada integrante puede defender el flujo completo
