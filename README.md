# Proyecto AAA - Cupones personalizados

Prueba de concepto para recomendar productos candidatos a cupones
personalizados a partir del historial de compras anonimizado de cada hogar
(dataset dunnhumby - The Complete Journey). Las recomendaciones se separan en:

- **repeat:** productos que el hogar ya compró y podría volver a comprar;
- **explore:** productos que el hogar todavía no compró.

**Informe final (Entregable 2): [`INFORME_FINAL.md`](INFORME_FINAL.md).**
Diapositivas, guion de demo y guía de defensa: [`PRESENTACION_Y_DEFENSA.md`](PRESENTACION_Y_DEFENSA.md).

El contexto consolidado, los resultados, las decisiones y el plan de cierre se
encuentran en `CONTEXTO_COMPLETO_PROYECTO.md`. La hoja de ruta operativa está en
`GUIA_DE_TRABAJO.md` y las decisiones en `BITACORA_DECISIONES.md`.

## Estado actual

| Fase | Estado |
| --- | --- |
| 0. Carga y calidad de datos | Completada |
| 1. EDA | Completado: descriptivo y evidencia para la defensa (D-011) |
| 2. División temporal por hogar | Completada |
| 3. Baseline repeat + popularidad | Completado en validación |
| 4. Coocurrencia e híbrido | Completado en validación |
| 5. ALS | Implementado con numpy/scipy (D-005); grilla evaluada en validación |
| 6. Evaluación comparativa y selección | Completada: repeat = frecuencia + recencia, explore = ALS, lista 7 + 3 (D-006) |
| 7. Evaluación final sobre prueba | Completada una sola vez (D-007) |
| 8. Reglas de negocio configurables | Completada (D-008) |
| 9. Prototipo Streamlit | Completado (D-009) |
| 10. Pipeline reproducible y pruebas de integración | Completado (D-010) |
| 11. Informe final | Completado (`INFORME_FINAL.md`) |
| 12. Presentación y defensa | Material listo (`PRESENTACION_Y_DEFENSA.md`); falta el ensayo |

Los datos originales en `data/` nunca se modifican. El conjunto de prueba
(`outputs/division_temporal/interacciones_prueba.csv`) no se utiliza hasta
congelar el modelo final; ya fue usado una única vez (D-007).

## Preparación

Requisitos: Python 3.12 o superior (verificado con 3.14.3). Las versiones
exactas de cada corrida quedan en `outputs/entorno.txt`.

Crear un entorno virtual e instalar las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Los CSV del dataset no se versionan en git (~808MB, públicos): descargalos
siguiendo [`data/README.md`](data/README.md) y colocalos en `data/`.

> **Windows con Control de aplicaciones o Smart App Control:** puede bloquear
> las DLL de algunas librerías recién instaladas, sobre todo dentro de un
> entorno virtual. Si aparece "Una directiva de Control de aplicaciones
> bloqueó este archivo", usá el Python del sistema. Por esta razón ALS está
> implementado sin la librería `implicit` (ver D-005).

## Ejecución completa (pipeline)

```powershell
python -m src.pipeline                 # todas las etapas, salvo selección y prueba (congeladas)
python -m src.pipeline --listar        # muestra las etapas
python -m src.pipeline --desde als     # retoma desde una etapa
python -m src.pipeline --solo reglas demo
```

Las etapas `seleccion` y `prueba` solo corren con `--recongelar` e
`--incluir-prueba`. La prueba no se repite sin `--forzar-prueba`. Cada corrida
registra el entorno y su duración, y verifica con hashes que `data/` no cambió.
Una corrida completa sobre los datos reales tarda unos 20 minutos, sobre todo
por la grilla de ALS (`--als-rapido` usa una sola configuración).

## Ejecución por etapas

Desde la raíz del proyecto, en este orden:

```powershell
# 0. Control de calidad (muestra o completo)
python -m src.cargar_datos --filas-muestra 100000
python -m src.cargar_datos --validar-completo

# 1. Análisis exploratorio -> outputs/eda/
python -m src.analizar_datos

# 2. División temporal entrenamiento/validación/prueba -> outputs/division_temporal/
python -m src.dividir_temporalmente

# 3. Baseline repeat (frecuencia + recencia) y explore (popularidad) -> outputs/baseline/
python -m src.recomendar_baseline

# 4. Coocurrencia e híbrido -> outputs/coocurrencia/
python -m src.recomendar_coocurrencia

# 5. ALS implícito: grilla en validación y mejor configuración -> outputs/als/
#    (--rapido corre una sola configuración pequeña)
python -m src.recomendar_als

# 6. Tabla comparativa de todos los modelos en validación -> outputs/comparacion/
python -m src.evaluar_modelos

# 7. Selección de modelos y mezcla repeat/explore -> outputs/seleccion/ y config/modelo_final.json
python -m src.seleccionar_modelo

# 7b. EDA para la defensa (recompra, cold start, concentración, combustible y tabaco)
#     -> outputs/eda/defensa_* y outputs/eda/analisis_defensa.md
python -m src.analizar_repeticion

# 8. Evaluación final ÚNICA sobre prueba -> outputs/prueba/ (no sobrescribe sin --forzar)
python -m src.evaluar_prueba

# 9. Efecto de las reglas de negocio (config/reglas_negocio.json) en validación -> outputs/reglas/
python -m src.reglas_negocio

# 10. Datos del prototipo (entrena con todo el historial y aplica reglas) -> outputs/demo/
python -m src.preparar_demo
```

## Prototipo

Después de `python -m src.preparar_demo`:

```powershell
streamlit run app/app.py
```

Abre http://localhost:8501 (solo accesible desde el propio equipo). Permite
elegir un hogar de ejemplo o cualquier `household_key` y muestra:

- el historial;
- 7 recomendaciones repeat y 3 explore, cada una con su explicación;
- los candidatos descartados por las reglas, con su motivo;
- exportación a CSV y feedback simulado.

Un hogar inexistente recibe una lista de respaldo con productos populares
elegibles.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `src/cargar_datos.py` | Carga y validación de esquema y calidad. |
| `src/analizar_datos.py` | EDA reproducible (tablas y gráficos). |
| `src/analizar_repeticion.py` | EDA para la defensa: recompra, cold start, concentración, combustible y tabaco. |
| `notebooks/01_eda_defensa.ipynb` | Presentación del EDA de defensa (solo lee `outputs/eda/`). |
| `src/dividir_temporalmente.py` | División temporal por hogar. |
| `src/recomendar_baseline.py` | Baseline repeat y explore. |
| `src/recomendar_coocurrencia.py` | Coocurrencia con lift e híbrido con popularidad. |
| `src/recomendar_als.py` | ALS implícito propio (numpy + scipy.sparse, gradiente conjugado). |
| `src/seleccionar_modelo.py` | Elección por tipo, curva de mezcla y configuración congelada. |
| `src/evaluar_prueba.py` | Reentrenamiento con entrenamiento + validación y evaluación única sobre prueba. |
| `src/reglas_negocio.py` | Reglas de elegibilidad para cupón, con motivo por exclusión. |
| `src/preparar_demo.py` | Precalcula listas, explicaciones e historial para el prototipo. |
| `app/app.py` | Prototipo Streamlit para el área comercial. |
| `src/pipeline.py` | Ejecución de punta a punta con registro de entorno y control de datos. |
| `src/evaluar_modelos.py` | Métricas comunes (Precision, Recall, Hit Rate, NDCG, cobertura, concentración, diversidad, explicabilidad) y tabla comparativa. |
| `tests/` | Pruebas unitarias y de integración (`test_pipeline.py`, con datos sintéticos de `datos_sinteticos.py`). |
| `exploracion/` | Exploración inicial conservada como registro; no forma parte del flujo. |
| `config/reglas_negocio.json` | Departamentos excluidos (KIOSK-GAS, etc.), soporte mínimo e integraciones de stock y margen. |
| `config/modelo_final.json` | Configuración final congelada antes de evaluar prueba. |
| `outputs/` | Artefactos derivados, regenerables. |

## Resultados en validación (K = 10)

| Modelo | Tipo | Precision | Recall | Hit Rate | NDCG |
| --- | --- | ---: | ---: | ---: | ---: |
| Frecuencia + recencia | repeat | 0,128677 | 0,294779 | 0,645352 | 0,265452 |
| Popularidad | explore | 0,007866 | 0,014831 | 0,071942 | 0,014842 |
| Coocurrencia | explore | 0,005947 | 0,008312 | 0,052278 | 0,008274 |
| Híbrido | explore | 0,006715 | 0,009655 | 0,058513 | 0,009294 |
| ALS (32 factores, alpha 2) | repeat | 0,071471 | 0,175022 | 0,461879 | 0,141019 |
| ALS (32 factores, alpha 2) | explore | 0,010935 | 0,019651 | 0,094964 | 0,017432 |

### Resultado final sobre prueba (K = 10, 7 repeat + 3 explore)

| Lista | Hit Rate | NDCG | Recall |
| --- | ---: | ---: | ---: |
| Sistema final | 0,507427 | 0,173970 | 0,140151 |
| Repeat (frecuencia + recencia) | 0,645002 | 0,273717 | 0,294475 |
| Explore (ALS) | 0,086437 | 0,017029 | 0,017211 |
| Referencia: popularidad | 0,065425 | 0,014378 | 0,014765 |

Detalle en `outputs/prueba/reporte_prueba.md`.

La tabla completa de validación, con cobertura, concentración, diversidad y tiempos, está en
`outputs/comparacion/comparacion_validacion.md`.

## Ejecutar las pruebas

```powershell
python -m unittest discover -s tests -v
```

Son 49 pruebas, incluida la de integración, que corre todo el pipeline sobre
datos sintéticos en una carpeta temporal (unos 15 s).
