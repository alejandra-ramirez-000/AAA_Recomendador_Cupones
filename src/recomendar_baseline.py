"""Genera y evalúa el baseline interpretable de recomendaciones.

Repeat combina frecuencia de compra y recencia dentro del historial de
entrenamiento. Explore ordena productos populares del entrenamiento y descarta
los que el hogar ya compró. Ningún puntaje utiliza validación ni prueba.

Ejemplo de uso:
    python -m src.recomendar_baseline
"""

# argparse permite elegir carpetas sin modificar el código.
import argparse

# json guarda las métricas de validación en un formato sencillo de consultar.
import json

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# perf_counter mide tiempos de entrenamiento y recomendación con buena resolución.
from time import perf_counter

# numpy aporta logaritmos y valores numéricos eficientes para los puntajes.
import numpy as np

# pandas construye perfiles, rankings y tablas de evaluación.
import pandas as pd

# Reutilizamos el cargador del catálogo para mostrar nombres de productos.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos

# Todas las métricas se calculan con el módulo común de evaluación.
from src.evaluar_modelos import (
    VALORES_DE_K,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)


# Esta es la carpeta producida por la división temporal de la Fase 2.
DIRECTORIO_DE_DIVISION_POR_DEFECTO = Path(__file__).resolve().parents[1] / "outputs" / "division_temporal"

# Esta carpeta guarda rankings y métricas derivados del baseline.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "outputs" / "baseline"


def construir_perfil_repeat(entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """Calcula frecuencia, recencia y puntaje repeat por hogar y producto."""

    # Contamos en cuántas canastas de entrenamiento aparece cada producto.
    perfil = (
        entrenamiento.groupby(["household_key", "PRODUCT_ID"], as_index=False)
        .agg(
            frecuencia_en_canastas=("BASKET_ID", "nunique"),
            ultima_canasta_comprada=("orden_de_canasta", "max"),
        )
    )

    # Calculamos la mayor frecuencia de cada hogar para normalizar entre cero y uno.
    perfil["frecuencia_maxima_del_hogar"] = perfil.groupby("household_key")[
        "frecuencia_en_canastas"
    ].transform("max")

    # Calculamos la última canasta del historial de cada hogar antes de validar.
    perfil["ultima_canasta_del_hogar"] = perfil.groupby("household_key")[
        "ultima_canasta_comprada"
    ].transform("max")

    # Normalizamos frecuencia para que hogares con historiales distintos sean comparables.
    perfil["puntaje_frecuencia"] = (
        perfil["frecuencia_en_canastas"] / perfil["frecuencia_maxima_del_hogar"]
    )

    # Medimos cuántas canastas pasaron desde la última compra de cada producto.
    perfil["distancia_desde_ultima_compra"] = (
        perfil["ultima_canasta_del_hogar"] - perfil["ultima_canasta_comprada"]
    )

    # Transformamos la distancia en recencia: 1 es la compra más reciente.
    perfil["puntaje_recencia"] = 1 / (
        1 + perfil["distancia_desde_ultima_compra"]
    )

    # Combinamos una señal estable de frecuencia con una señal de recencia reciente.
    perfil["puntaje_repeat"] = (
        0.7 * perfil["puntaje_frecuencia"] + 0.3 * perfil["puntaje_recencia"]
    )

    # Devolvemos el perfil para que cada recomendación sea explicable después.
    return perfil


def construir_popularidad(entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """Calcula popularidad global usando exclusivamente canastas de entrenamiento."""

    # Contamos canastas y hogares distintos que adquirieron cada producto.
    popularidad = (
        entrenamiento.groupby("PRODUCT_ID", as_index=False)
        .agg(
            canastas_con_producto=("BASKET_ID", "nunique"),
            hogares_que_compraron=("household_key", "nunique"),
        )
    )

    # Aplicamos logaritmo para que unos pocos productos masivos no dominen por completo.
    popularidad["popularidad_logaritmica"] = np.log1p(
        popularidad["canastas_con_producto"]
    )

    # Normalizamos la popularidad para mostrar un puntaje entre cero y uno.
    popularidad["puntaje_popularidad"] = (
        popularidad["popularidad_logaritmica"]
        / popularidad["popularidad_logaritmica"].max()
    )

    # Ordenamos una sola vez para reutilizar la lista global con todos los hogares.
    return popularidad.sort_values(
        ["puntaje_popularidad", "PRODUCT_ID"], ascending=[False, True]
    ).reset_index(drop=True)


def generar_recomendaciones_repeat(
    perfil_repeat: pd.DataFrame, cantidad_recomendaciones: int
) -> pd.DataFrame:
    """Selecciona los productos repetibles mejor puntuados de cada hogar."""

    # Ordenamos primero por hogar y después por puntaje para construir el ranking.
    recomendaciones = perfil_repeat.sort_values(
        ["household_key", "puntaje_repeat", "PRODUCT_ID"],
        ascending=[True, False, True],
    ).copy()

    # Numeramos posiciones desde uno dentro de cada hogar.
    recomendaciones["posicion"] = recomendaciones.groupby("household_key").cumcount() + 1

    # Conservamos solo las primeras posiciones solicitadas para cada hogar.
    recomendaciones = recomendaciones.loc[
        recomendaciones["posicion"] <= cantidad_recomendaciones
    ].copy()

    # Identificamos la estrategia para facilitar evaluación y presentación conjunta.
    recomendaciones["tipo_recomendacion"] = "repeat"

    # Devolvemos las columnas que explican el motivo del ranking repeat.
    return recomendaciones[
        [
            "household_key",
            "PRODUCT_ID",
            "posicion",
            "puntaje_repeat",
            "frecuencia_en_canastas",
            "ultima_canasta_comprada",
            "distancia_desde_ultima_compra",
            "tipo_recomendacion",
        ]
    ]


def generar_recomendaciones_explore(
    entrenamiento: pd.DataFrame,
    popularidad: pd.DataFrame,
    cantidad_recomendaciones: int,
) -> pd.DataFrame:
    """Sugiere productos populares que cada hogar no compró en entrenamiento."""

    # Construimos el conjunto de productos históricos de cada hogar una sola vez.
    historial_por_hogar = entrenamiento.groupby("household_key")["PRODUCT_ID"].agg(set)

    # Convertimos la popularidad ordenada en pares para recorrerla eficientemente.
    ranking_global = list(
        popularidad[["PRODUCT_ID", "puntaje_popularidad", "canastas_con_producto"]].itertuples(
            index=False, name=None
        )
    )

    # Acumulamos filas simples para crear el DataFrame de salida al terminar.
    filas_recomendadas: list[dict[str, int | float | str]] = []

    # Recorremos cada hogar que tiene un historial de entrenamiento disponible.
    for household_key, productos_historicos in historial_por_hogar.items():
        # Empezamos el ranking explore desde la primera posición para este hogar.
        posicion = 1

        # Revisamos productos globalmente populares hasta completar el cupo solicitado.
        for producto_id, puntaje, canastas_con_producto in ranking_global:
            # Saltamos productos ya comprados porque explore debe proponer novedades.
            if producto_id in productos_historicos:
                continue

            # Guardamos una recomendación con el motivo observable de su popularidad.
            filas_recomendadas.append(
                {
                    "household_key": household_key,
                    "PRODUCT_ID": producto_id,
                    "posicion": posicion,
                    "puntaje_popularidad": puntaje,
                    "canastas_con_producto": canastas_con_producto,
                    "tipo_recomendacion": "explore",
                }
            )

            # Avanzamos una posición solo cuando se aceptó una recomendación nueva.
            posicion += 1

            # Interrumpimos el recorrido cuando el hogar ya tiene su lista completa.
            if posicion > cantidad_recomendaciones:
                break

    # Convertimos las filas acumuladas en una tabla de recomendaciones.
    return pd.DataFrame(filas_recomendadas)


def enriquecer_con_catalogo(
    recomendaciones: pd.DataFrame, productos: pd.DataFrame
) -> pd.DataFrame:
    """Agrega atributos de catálogo para que la salida sea interpretable."""

    # Elegimos atributos suficientes para identificar sin sobrecargar el archivo de salida.
    columnas_catalogo = [
        "PRODUCT_ID",
        "DEPARTMENT",
        "BRAND",
        "COMMODITY_DESC",
        "SUB_COMMODITY_DESC",
    ]

    # Unimos cada recomendación con una única fila del catálogo de productos.
    return recomendaciones.merge(
        productos[columnas_catalogo], on="PRODUCT_ID", how="left", validate="many_to_one"
    )


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución del baseline."""

    # Creamos el analizador de argumentos mostrado al ejecutar --help.
    analizador = argparse.ArgumentParser(description="Genera el baseline repeat y explore.")

    # Permitimos indicar otra carpeta de resultados de la división temporal.
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)

    # Permitimos guardar resultados en otra carpeta para comparar experimentos.
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)

    # Definimos el máximo de productos solicitado por cada tipo de recomendación.
    analizador.add_argument("--cantidad-recomendaciones", type=int, default=20)

    # Permitimos elegir otra carpeta de datos para ubicar product.csv si fuera necesario.
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)

    # Devolvemos argumentos que argparse ya convirtió a los tipos requeridos.
    return analizador.parse_args()


def main() -> None:
    """Entrena el baseline con entrenamiento y lo evalúa sobre validación."""

    # Leemos rutas y parámetros de la ejecución solicitada.
    argumentos = interpretar_argumentos()

    # Detenemos el proceso temprano si K no es un entero positivo.
    if argumentos.cantidad_recomendaciones <= 0:
        raise ValueError("La cantidad de recomendaciones debe ser mayor que cero.")

    # Cargamos únicamente entrenamiento y validación; prueba queda reservada.
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")

    # Iniciamos el cronómetro del entrenamiento (perfil repeat y popularidad).
    inicio_entrenamiento = perf_counter()

    # Construimos el perfil interpretable de productos que el hogar ya compró.
    perfil_repeat = construir_perfil_repeat(entrenamiento)

    # Construimos popularidad exclusivamente a partir del conjunto de entrenamiento.
    popularidad = construir_popularidad(entrenamiento)

    # Cerramos el tramo de entrenamiento e iniciamos el de recomendación.
    segundos_entrenamiento = perf_counter() - inicio_entrenamiento
    inicio_recomendacion = perf_counter()

    # Generamos listas repeat ordenadas por frecuencia y recencia.
    recomendaciones_repeat = generar_recomendaciones_repeat(
        perfil_repeat, argumentos.cantidad_recomendaciones
    )

    # Generamos listas explore de productos populares que cada hogar aún no compró.
    recomendaciones_explore = generar_recomendaciones_explore(
        entrenamiento, popularidad, argumentos.cantidad_recomendaciones
    )

    # Medimos el tiempo total de recomendación y su promedio por hogar.
    segundos_recomendacion = perf_counter() - inicio_recomendacion
    cantidad_hogares = entrenamiento["household_key"].nunique()

    # Cargamos catálogo para transformar IDs técnicos en productos comprensibles.
    productos = cargar_productos(argumentos.data_dir)

    # Agregamos categorías y marcas a las dos salidas de recomendaciones.
    recomendaciones_repeat = enriquecer_con_catalogo(recomendaciones_repeat, productos)
    recomendaciones_explore = enriquecer_con_catalogo(recomendaciones_explore, productos)

    # Creamos los objetivos reales de validación para repeat y explore por separado.
    relevantes_repeat = obtener_productos_relevantes(validacion, entrenamiento, "repeat")
    relevantes_explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")

    # Calculamos las cuatro métricas para los valores de K definidos en el informe.
    metricas = {
        "repeat": [calcular_metricas_en_k(recomendaciones_repeat, relevantes_repeat, k) for k in VALORES_DE_K],
        "explore": [calcular_metricas_en_k(recomendaciones_explore, relevantes_explore, k) for k in VALORES_DE_K],
        # Los tiempos dependen del equipo; sirven para comparar costo entre modelos.
        "tiempos_segundos": {
            "entrenamiento": round(segundos_entrenamiento, 3),
            "recomendacion_total": round(segundos_recomendacion, 3),
            "ms_recomendacion_por_hogar": round(1000 * segundos_recomendacion / cantidad_hogares, 3),
        },
    }

    # Creamos la carpeta de resultados antes de escribir los rankings y métricas.
    argumentos.output_dir.mkdir(parents=True, exist_ok=True)

    # Guardamos rankings de validación que servirán para inspección y demostración.
    recomendaciones_repeat.to_csv(argumentos.output_dir / "recomendaciones_repeat_validacion.csv", index=False)
    recomendaciones_explore.to_csv(argumentos.output_dir / "recomendaciones_explore_validacion.csv", index=False)

    # Guardamos el perfil y popularidad para revisar de dónde provienen los puntajes.
    perfil_repeat.to_csv(argumentos.output_dir / "perfil_repeat_entrenamiento.csv", index=False)
    popularidad.to_csv(argumentos.output_dir / "popularidad_entrenamiento.csv", index=False)

    # Guardamos métricas con codificación UTF-8 y sangría para lectura humana.
    with (argumentos.output_dir / "metricas_validacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)

    # Mostramos el resultado compactamente en terminal para registrar la ejecución.
    print("\nMétricas del baseline en validación")
    for tipo in ("repeat", "explore"):
        for resultado in metricas[tipo]:
            print(f"- {tipo}@{resultado['k']}: {resultado}")
    print(f"- tiempos: {metricas['tiempos_segundos']}")


# Ejecutamos el entrenamiento solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
