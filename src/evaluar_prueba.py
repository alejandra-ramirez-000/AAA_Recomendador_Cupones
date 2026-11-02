"""Evaluación final, única, del sistema congelado sobre el conjunto de prueba.

Lee `config/modelo_final.json` (D-006) y no ajusta nada:

1. une entrenamiento + validación como historial ampliado de cada hogar;
2. reentrena repeat (frecuencia + recencia) y explore (ALS con los parámetros
   congelados) sobre ese historial;
3. arma la lista final (7 repeat + 3 explore, K = 10) para cada hogar;
4. redefine repeat/explore respecto del historial ampliado y evalúa contra la
   última canasta de cada hogar (prueba).

Como referencia, que no sirve para decidir nada, también se evalúan popularidad
(explore) y una lista solo repeat con el mismo procedimiento.

Para proteger el carácter único de la evaluación, el script se niega a
sobrescribir un reporte existente salvo con `--forzar`, y cualquier repetición
debe registrarse en la bitácora.

Salidas en outputs/prueba/:
    reporte_prueba.md
    metricas_prueba.json
    recomendaciones_finales_prueba.csv

Ejemplo de uso:
    python -m src.evaluar_prueba
"""

# argparse permite cambiar rutas sin modificar el código.
import argparse

# json lee la configuración congelada y guarda métricas.
import json

# platform y datetime registran el entorno y la fecha de la evaluación.
import platform
from datetime import datetime

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# perf_counter mide tiempos de entrenamiento y recomendación.
from time import perf_counter

# Librerías numéricas, también para registrar sus versiones.
import numpy as np
import pandas as pd
import scipy

# Piezas ya validadas de cada etapa.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos
from src.evaluar_modelos import (
    DIRECTORIO_DE_DIVISION_POR_DEFECTO,
    RAIZ_DEL_PROYECTO,
    VALORES_DE_K,
    calcular_metricas_de_sistema,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)
from src.recomendar_als import (
    construir_matriz_de_confianza,
    entrenar_als,
    generar_recomendaciones_als,
)
from src.recomendar_baseline import (
    construir_perfil_repeat,
    construir_popularidad,
    enriquecer_con_catalogo,
    generar_recomendaciones_explore,
    generar_recomendaciones_repeat,
)
from src.seleccionar_modelo import RUTA_CONFIGURACION_POR_DEFECTO, combinar_listas


# Carpeta de resultados de la evaluación final.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "prueba"

# Modelos que este script sabe reentrenar; otra configuración exigiría ampliarlo.
MODELOS_REPEAT_SOPORTADOS = {"baseline_frecuencia_recencia"}
MODELOS_EXPLORE_SOPORTADOS = {"als_explore", "baseline_popularidad"}

# Largo de las listas parciales; mayor que K para poder completar huecos.
LARGO_DE_LISTAS_PARCIALES = 20


def construir_historial_ampliado(entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> pd.DataFrame:
    """Une entrenamiento y validación: todo lo observado antes de la canasta de prueba."""

    # Ambas particiones conservan orden_de_canasta, que sigue siendo creciente por hogar.
    return pd.concat([entrenamiento, validacion], ignore_index=True)


def generar_candidatos(
    historial: pd.DataFrame, configuracion: dict, largo: int = LARGO_DE_LISTAS_PARCIALES
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reentrena los modelos congelados y devuelve listas candidatas repeat y explore.

    La lista explore siempre trae las mismas columnas explicativas, sea cual sea
    el modelo: `puntaje_explore`, `producto_que_lo_explica`, `similitud_explicativa`
    (ALS) y `canastas_con_producto` (popularidad). Devuelve (repeat, explore, tiempos).
    """

    # Verificamos que la configuración corresponda a modelos que sabemos reentrenar.
    modelo_repeat = configuracion["repeat"]["modelo"]
    modelo_explore = configuracion["explore"]["modelo"]
    if modelo_repeat not in MODELOS_REPEAT_SOPORTADOS:
        raise NotImplementedError(f"Repeat no soportado: {modelo_repeat}")
    if modelo_explore not in MODELOS_EXPLORE_SOPORTADOS:
        raise NotImplementedError(f"Explore no soportado: {modelo_explore}")

    # Entrenamiento: perfil repeat y el modelo explore elegido.
    inicio = perf_counter()
    perfil_repeat = construir_perfil_repeat(historial)
    popularidad = construir_popularidad(historial)
    if modelo_explore == "als_explore":
        parametros = configuracion["parametros_als"]
        matriz, hogares, productos, _ = construir_matriz_de_confianza(historial, parametros["alpha"])
        factores_hogar, factores_producto = entrenar_als(
            matriz, parametros["factores"], parametros["regularizacion"], parametros["iteraciones"], parametros["semilla"]
        )
    segundos_entrenamiento = perf_counter() - inicio

    # Recomendación: listas candidatas de cada tipo.
    inicio = perf_counter()
    repeat = generar_recomendaciones_repeat(perfil_repeat, largo)
    if modelo_explore == "als_explore":
        _, explore = generar_recomendaciones_als(
            matriz, hogares, productos, factores_hogar, factores_producto, largo, con_explicacion=True
        )
        explore = explore.rename(columns={"puntaje_als": "puntaje_explore"})
    else:
        explore = generar_recomendaciones_explore(historial, popularidad, largo)
        explore = explore.rename(columns={"puntaje_popularidad": "puntaje_explore"})
    segundos_recomendacion = perf_counter() - inicio

    # Columnas explicativas comunes; la evidencia de popularidad sirve para cualquier modelo.
    for columna in ("producto_que_lo_explica", "similitud_explicativa"):
        if columna not in explore.columns:
            explore[columna] = pd.NA
    if "canastas_con_producto" not in explore.columns:
        explore = explore.merge(popularidad[["PRODUCT_ID", "canastas_con_producto"]], on="PRODUCT_ID", how="left")

    cantidad_hogares = historial["household_key"].nunique()
    tiempos = {
        "entrenamiento": round(segundos_entrenamiento, 3),
        "recomendacion_total": round(segundos_recomendacion, 3),
        "ms_recomendacion_por_hogar": round(1000 * segundos_recomendacion / cantidad_hogares, 3),
    }
    return repeat, explore, tiempos


def generar_sistema_final(
    historial: pd.DataFrame, configuracion: dict, largo: int = LARGO_DE_LISTAS_PARCIALES
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Reentrena repeat y explore congelados y arma la lista final.

    Devuelve (lista_final, repeat, explore, tiempos).
    """

    repeat, explore, tiempos = generar_candidatos(historial, configuracion, largo)
    lista_final = combinar_listas(
        repeat, explore, configuracion["k"], configuracion["mezcla"]["proporcion_repeat"]
    )
    return lista_final, repeat, explore, tiempos


def agregar_explicaciones(
    lista_final: pd.DataFrame, repeat: pd.DataFrame, explore: pd.DataFrame
) -> pd.DataFrame:
    """Adjunta a cada posición la señal que la justifica."""

    # Repeat: frecuencia y distancia desde la última compra.
    explicacion_repeat = repeat[["household_key", "PRODUCT_ID", "frecuencia_en_canastas", "distancia_desde_ultima_compra"]]

    # Explore: producto del historial más similar en el espacio latente.
    explicacion_explore = explore[
        ["household_key", "PRODUCT_ID", "producto_que_lo_explica", "similitud_explicativa", "canastas_con_producto"]
    ]

    resultado = lista_final.merge(explicacion_repeat, on=["household_key", "PRODUCT_ID"], how="left")
    return resultado.merge(explicacion_explore, on=["household_key", "PRODUCT_ID"], how="left")


def evaluar_listas(
    lista_final: pd.DataFrame,
    repeat: pd.DataFrame,
    explore: pd.DataFrame,
    popularidad_explore: pd.DataFrame,
    solo_repeat: pd.DataFrame,
    historial: pd.DataFrame,
    prueba: pd.DataFrame,
    k: int,
) -> dict:
    """Calcula todas las métricas de prueba con el módulo común."""

    relevantes = {
        tipo: obtener_productos_relevantes(prueba, historial, tipo) for tipo in ("todos", "repeat", "explore")
    }
    explore_en_lista = lista_final.loc[lista_final["origen"] == "explore"]

    return {
        "sistema_final": {
            "lista_combinada": calcular_metricas_en_k(lista_final, relevantes["todos"], k),
            "aciertos_explore_en_lista": calcular_metricas_en_k(explore_en_lista, relevantes["explore"], k),
            "sistema": calcular_metricas_de_sistema(
                lista_final, k, set(prueba["household_key"]), set(historial["PRODUCT_ID"])
            ),
        },
        "componentes": {
            "repeat_frecuencia_recencia": [calcular_metricas_en_k(repeat, relevantes["repeat"], kk) for kk in VALORES_DE_K],
            "explore_als": [calcular_metricas_en_k(explore, relevantes["explore"], kk) for kk in VALORES_DE_K],
        },
        "referencias_no_seleccionadas": {
            "explore_popularidad": [
                calcular_metricas_en_k(popularidad_explore, relevantes["explore"], kk) for kk in VALORES_DE_K
            ],
            "lista_solo_repeat": calcular_metricas_en_k(solo_repeat, relevantes["todos"], k),
        },
    }


def escribir_reporte(ruta: Path, resultado: dict, metricas_validacion: dict) -> None:
    """Escribe un reporte Markdown legible de la evaluación final."""

    final = resultado["metricas"]["sistema_final"]
    combinada = final["lista_combinada"]
    descubrimiento = final["aciertos_explore_en_lista"]
    sistema = final["sistema"]
    repeat10 = resultado["metricas"]["componentes"]["repeat_frecuencia_recencia"][1]
    als10 = resultado["metricas"]["componentes"]["explore_als"][1]
    pop10 = resultado["metricas"]["referencias_no_seleccionadas"]["explore_popularidad"][1]
    solo_repeat = resultado["metricas"]["referencias_no_seleccionadas"]["lista_solo_repeat"]
    mezcla = resultado["configuracion"]["mezcla"]

    def fila(nombre: str, m: dict, validacion: dict | None = None) -> str:
        val = f"{validacion['hit_rate']:.6f} | {validacion['ndcg']:.6f}" if validacion else "- | -"
        return (
            f"| {nombre} | {m['hogares_evaluables']} | {m['precision']:.6f} | {m['recall']:.6f} | "
            f"{m['hit_rate']:.6f} | {m['ndcg']:.6f} | {val} |"
        )

    lineas = [
        "# Evaluación final sobre prueba",
        "",
        f"- **Fecha:** {resultado['fecha']}",
        f"- **Configuración:** `config/modelo_final.json`, congelada el {resultado['configuracion']['fecha_de_congelamiento']}",
        f"- **Lista:** K = {resultado['configuracion']['k']}, {mezcla['posiciones_repeat']} repeat + "
        f"{mezcla['posiciones_explore']} explore ({mezcla['orden']})",
        f"- **Historial usado:** entrenamiento + validación; evaluación sobre la última canasta de cada hogar",
        f"- **Entorno:** {resultado['entorno']}",
        f"- **Tiempos (s):** {resultado['tiempos_segundos']}",
        "",
        "Evaluación única: ningún número de este reporte se usó para elegir modelos,",
        "parámetros, K ni mezcla.",
        "",
        "## Resultados (K = 10)",
        "",
        "| Lista | Hogares evaluables | Precision | Recall | Hit Rate | NDCG | Hit Rate validación | NDCG validación |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        fila("**Sistema final (7 + 3)**", combinada, metricas_validacion.get("lista_combinada")),
        fila("Aciertos explore dentro de la lista final", descubrimiento),
        fila("Repeat: frecuencia + recencia", repeat10, metricas_validacion.get("repeat")),
        fila(f"Explore: {resultado['configuracion']['explore']['modelo']}", als10, metricas_validacion.get("explore_als")),
        fila("Referencia: explore por popularidad", pop10, metricas_validacion.get("explore_popularidad")),
        fila("Referencia: lista solo repeat", solo_repeat),
        "",
        "## Métricas de sistema de la lista final",
        "",
        f"- Cobertura de hogares: {sistema['cobertura_hogares']:.4f}",
        f"- Productos distintos recomendados: {sistema['productos_distintos_recomendados']}",
        f"- Participación de los 10 productos más recomendados: {sistema['participacion_top_10_productos']:.4f}",
        f"- Departamentos distintos por lista: {sistema['departamentos_distintos_por_lista']}",
        "",
        "La columna de validación corresponde a los mismos modelos entrenados solo con",
        "entrenamiento. La diferencia entre ambas mezcla dos efectos: el optimismo por",
        "haber elegido sobre validación y la dificultad propia de cada canasta. La",
        "referencia de popularidad, que no fue seleccionada, ayuda a separarlos.",
        "",
    ]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def leer_metricas_de_validacion(directorio_resultados: Path) -> dict:
    """Recupera las métricas de validación K = 10 para comparar con prueba."""

    def k10(lista: list[dict]) -> dict:
        return next(m for m in lista if m["k"] == 10)

    resultado: dict = {}
    with (directorio_resultados / "baseline" / "metricas_validacion.json").open(encoding="utf-8") as archivo:
        baseline = json.load(archivo)
    resultado["repeat"] = k10(baseline["repeat"])
    resultado["explore_popularidad"] = k10(baseline["explore"])
    with (directorio_resultados / "als" / "metricas_validacion.json").open(encoding="utf-8") as archivo:
        resultado["explore_als"] = k10(json.load(archivo)["explore"])
    return resultado


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de la evaluación final."""

    analizador = argparse.ArgumentParser(description="Evaluación final única sobre el conjunto de prueba.")
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)
    analizador.add_argument("--configuracion", type=Path, default=RUTA_CONFIGURACION_POR_DEFECTO)
    analizador.add_argument("--resultados-dir", type=Path, default=RAIZ_DEL_PROYECTO / "outputs")
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)
    analizador.add_argument("--forzar", action="store_true", help="Sobrescribe un reporte existente.")
    return analizador.parse_args()


def main() -> None:
    """Ejecuta la evaluación final y guarda el reporte."""

    argumentos = interpretar_argumentos()

    # Protegemos el uso único del conjunto de prueba.
    ruta_reporte = argumentos.output_dir / "reporte_prueba.md"
    if ruta_reporte.exists() and not argumentos.forzar:
        raise SystemExit(
            f"Ya existe {ruta_reporte}. La evaluación sobre prueba es única; "
            "use --forzar solo si se registra el motivo en la bitácora."
        )

    # Configuración congelada.
    with argumentos.configuracion.open(encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    # Datos: historial ampliado y canasta de prueba.
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")
    prueba = pd.read_csv(argumentos.division_dir / "interacciones_prueba.csv")
    historial = construir_historial_ampliado(entrenamiento, validacion)

    # Sistema final reentrenado sobre el historial ampliado.
    lista_final, repeat, explore, tiempos = generar_sistema_final(historial, configuracion)

    # Referencias con el mismo historial (no influyen en ninguna decisión).
    popularidad_explore = generar_recomendaciones_explore(
        historial, construir_popularidad(historial), LARGO_DE_LISTAS_PARCIALES
    )
    solo_repeat = combinar_listas(repeat, explore, configuracion["k"], 1.0)

    # Lista final explicada y con catálogo (el departamento se usa para medir diversidad).
    lista_explicada = enriquecer_con_catalogo(
        agregar_explicaciones(lista_final, repeat, explore), cargar_productos(argumentos.data_dir)
    )

    # Métricas con el módulo común.
    metricas = evaluar_listas(
        lista_explicada, repeat, explore, popularidad_explore, solo_repeat, historial, prueba, configuracion["k"]
    )

    # Validación de la lista combinada, guardada al congelar la mezcla.
    metricas_validacion = leer_metricas_de_validacion(argumentos.resultados_dir)
    metricas_validacion["lista_combinada"] = configuracion["mezcla"]["metricas_validacion"]

    resultado = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "configuracion": configuracion,
        "entorno": (
            f"Python {platform.python_version()}, numpy {np.__version__}, "
            f"pandas {pd.__version__}, scipy {scipy.__version__}"
        ),
        "tiempos_segundos": tiempos,
        "metricas": metricas,
    }

    # Guardamos la lista final, las métricas y el reporte.
    argumentos.output_dir.mkdir(parents=True, exist_ok=True)
    lista_explicada.to_csv(argumentos.output_dir / "recomendaciones_finales_prueba.csv", index=False)
    with (argumentos.output_dir / "metricas_prueba.json").open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, indent=2, ensure_ascii=False, default=str)
    escribir_reporte(ruta_reporte, resultado, metricas_validacion)

    print(ruta_reporte.read_text(encoding="utf-8"))


# Ejecutamos la evaluación solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
