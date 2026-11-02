"""Genera recomendaciones explore con coocurrencias e híbrido de popularidad.

Una coocurrencia indica que dos productos aparecen juntos en una misma canasta
de entrenamiento. El modelo usa lift para evitar recomendar solo productos
globalmente masivos. Después combina esa señal con popularidad en una variante
híbrida. Ninguna asociación utiliza validación ni prueba.

Ejemplo de uso:
    python -m src.recomendar_coocurrencia
"""

# argparse permite cambiar parámetros sin editar el módulo.
import argparse

# Counter acumula pares de productos sin construir una tabla enorme intermedia.
from collections import Counter, defaultdict

# combinations crea pares no dirigidos dentro de cada canasta.
from itertools import combinations

# json guarda métricas y parámetros de ejecución en archivos legibles.
import json

# Path facilita trabajar con rutas derivadas del proyecto.
from pathlib import Path

# perf_counter mide tiempos de entrenamiento y recomendación con buena resolución.
from time import perf_counter

# numpy se usa para normalizar puntajes numéricos.
import numpy as np

# pandas agrupa canastas, maneja rankings y exporta resultados.
import pandas as pd

# Reutilizamos catálogo, popularidad del baseline y el módulo común de evaluación.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos
from src.evaluar_modelos import (
    VALORES_DE_K,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)
from src.recomendar_baseline import (
    DIRECTORIO_DE_DIVISION_POR_DEFECTO,
    construir_popularidad,
    enriquecer_con_catalogo,
)


# Los derivados del modelo de coocurrencia se guardan separados del baseline.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "outputs" / "coocurrencia"


def construir_asociaciones(
    entrenamiento: pd.DataFrame,
    soporte_minimo_producto: int,
    soporte_minimo_par: int,
) -> pd.DataFrame:
    """Calcula pares frecuentes y su lift a partir de canastas de entrenamiento.

    Se filtran productos raros antes de formar pares para que el cálculo sea
    viable y las asociaciones tengan evidencia mínima.
    """

    # Contamos en cuántas canastas aparece cada producto, no cuántas líneas tiene.
    soporte_por_producto = entrenamiento.groupby("PRODUCT_ID")["BASKET_ID"].nunique()

    # Conservamos productos con evidencia suficiente para construir asociaciones estables.
    productos_elegibles = set(
        soporte_por_producto[soporte_por_producto >= soporte_minimo_producto].index
    )

    # Filtramos líneas a esos productos antes de agrupar cada canasta.
    lineas_elegibles = entrenamiento.loc[
        entrenamiento["PRODUCT_ID"].isin(productos_elegibles),
        ["BASKET_ID", "PRODUCT_ID"],
    ]

    # Agrupamos los productos distintos de cada canasta para no contar un producto dos veces.
    productos_por_canasta = lineas_elegibles.groupby("BASKET_ID")["PRODUCT_ID"].agg(
        lambda productos: sorted(set(productos))
    )

    # Creamos un contador para acumular el soporte de cada par no dirigido.
    conteo_de_pares: Counter[tuple[int, int]] = Counter()

    # Recorremos una canasta a la vez para construir combinaciones de dos productos.
    for productos_de_la_canasta in productos_por_canasta:
        # Una canasta con menos de dos productos no aporta una asociación.
        if len(productos_de_la_canasta) < 2:
            continue

        # Sumamos uno a cada par que apareció junto en esa canasta.
        conteo_de_pares.update(combinations(productos_de_la_canasta, 2))

    # Convertimos el contador en filas para calcular métricas de asociación con pandas.
    asociaciones = pd.DataFrame(
        [
            {
                "PRODUCT_ID_ORIGEN": origen,
                "PRODUCT_ID_DESTINO": destino,
                "canastas_con_el_par": soporte,
            }
            for (origen, destino), soporte in conteo_de_pares.items()
            if soporte >= soporte_minimo_par
        ]
    )

    # Devolvemos una tabla vacía con columnas claras si ningún par supera el umbral.
    if asociaciones.empty:
        return pd.DataFrame(
            columns=[
                "PRODUCT_ID_ORIGEN",
                "PRODUCT_ID_DESTINO",
                "canastas_con_el_par",
                "lift",
                "puntaje_asociacion",
            ]
        )

    # Convertimos el soporte de producto a columnas para calcular el lift del par.
    asociaciones["soporte_origen"] = asociaciones["PRODUCT_ID_ORIGEN"].map(
        soporte_por_producto
    )
    asociaciones["soporte_destino"] = asociaciones["PRODUCT_ID_DESTINO"].map(
        soporte_por_producto
    )

    # Usamos todas las canastas de entrenamiento como universo del cálculo de lift.
    cantidad_canastas = entrenamiento["BASKET_ID"].nunique()

    # Lift mayor a uno indica que el par ocurre más de lo esperado por popularidad.
    asociaciones["lift"] = (
        asociaciones["canastas_con_el_par"]
        * cantidad_canastas
        / (asociaciones["soporte_origen"] * asociaciones["soporte_destino"])
    )

    # Penalizamos asociaciones muy poco frecuentes al multiplicar por logaritmo de soporte.
    asociaciones["puntaje_asociacion"] = asociaciones["lift"] * np.log1p(
        asociaciones["canastas_con_el_par"]
    )

    # Devolvemos únicamente señales positivas y ordenadas para inspección posterior.
    return asociaciones.loc[asociaciones["lift"] > 1].sort_values(
        ["puntaje_asociacion", "canastas_con_el_par"], ascending=[False, False]
    ).reset_index(drop=True)


def crear_indice_de_asociaciones(
    asociaciones: pd.DataFrame,
) -> dict[int, list[tuple[int, float, int]]]:
    """Convierte pares no dirigidos en candidatos dirigidos por producto origen."""

    # defaultdict crea automáticamente una lista vacía para cada producto origen.
    indice: dict[int, list[tuple[int, float, int]]] = defaultdict(list)

    # Cada par no dirigido se agrega en ambas direcciones para recomendar reciprocidad.
    for fila in asociaciones.itertuples(index=False):
        # Guardamos destino, puntaje y soporte cuando el primer producto es origen.
        indice[fila.PRODUCT_ID_ORIGEN].append(
            (fila.PRODUCT_ID_DESTINO, fila.puntaje_asociacion, fila.canastas_con_el_par)
        )

        # Guardamos también la dirección inversa del mismo vínculo observado.
        indice[fila.PRODUCT_ID_DESTINO].append(
            (fila.PRODUCT_ID_ORIGEN, fila.puntaje_asociacion, fila.canastas_con_el_par)
        )

    # Ordenamos las listas para que resultados empatados sean siempre reproducibles.
    for producto_origen in indice:
        indice[producto_origen].sort(key=lambda fila: (-fila[1], fila[0]))

    # Devolvemos el índice compacto que usará la generación de candidatos.
    return dict(indice)


def obtener_historial_reciente(
    entrenamiento: pd.DataFrame, maximo_productos: int
) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    """Obtiene historial completo para filtrar y productos recientes para asociar."""

    # Construimos todo el historial para asegurar que explore nunca repita un producto.
    historial_completo = entrenamiento.groupby("household_key")["PRODUCT_ID"].agg(set).to_dict()

    # Ordenamos productos por última canasta para priorizar intereses recientes del hogar.
    ultima_compra = (
        entrenamiento.groupby(["household_key", "PRODUCT_ID"], as_index=False)
        .agg(ultima_canasta=("orden_de_canasta", "max"))
        .sort_values(["household_key", "ultima_canasta", "PRODUCT_ID"], ascending=[True, False, True])
    )

    # Conservamos un máximo de productos recientes para evitar candidatos excesivos.
    productos_recientes = (
        ultima_compra.groupby("household_key").head(maximo_productos)
        .groupby("household_key")["PRODUCT_ID"]
        .agg(list)
        .to_dict()
    )

    # Devolvemos ambos conceptos: historial para excluir y reciente para generar afinidad.
    return historial_completo, productos_recientes


def generar_recomendaciones_coocurrencia(
    entrenamiento: pd.DataFrame,
    asociaciones: pd.DataFrame,
    popularidad: pd.DataFrame,
    cantidad_recomendaciones: int,
    maximo_productos_recientes: int,
    peso_coocurrencia: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera listas coocurrencia pura e híbrida para cada hogar.

    El modelo puro ordena por evidencia de asociación. El híbrido normaliza esa
    evidencia y la mezcla con popularidad global, sin agregar candidatos nuevos.
    """

    # Construimos vínculos dirigidos para consultar rápidamente destinos por producto.
    indice_asociaciones = crear_indice_de_asociaciones(asociaciones)

    # Obtenemos productos conocidos y un subconjunto reciente para cada hogar.
    historial_completo, productos_recientes = obtener_historial_reciente(
        entrenamiento, maximo_productos_recientes
    )

    # Creamos un diccionario de popularidad normalizada por producto para el híbrido.
    popularidad_por_producto = popularidad.set_index("PRODUCT_ID")["puntaje_popularidad"].to_dict()

    # Preparamos listas de filas para construir los dos DataFrames al terminar.
    filas_coocurrencia: list[dict[str, int | float | str]] = []
    filas_hibridas: list[dict[str, int | float | str]] = []

    # Recorremos cada hogar que cuenta con un historial de entrenamiento.
    for household_key, productos_historicos in historial_completo.items():
        # Acumulamos evidencia cuando varios productos del hogar apuntan al mismo destino.
        puntajes_por_candidato: dict[int, float] = defaultdict(float)

        # Conservamos el soporte máximo como explicación simple del vínculo más fuerte.
        soporte_maximo_por_candidato: dict[int, int] = defaultdict(int)

        # Usamos solamente los intereses recientes para generar asociaciones pertinentes.
        for producto_origen in productos_recientes.get(household_key, []):
            # Recorremos asociaciones conocidas desde ese producto; si no hay, lista vacía.
            for producto_destino, puntaje, soporte in indice_asociaciones.get(producto_origen, []):
                # Explore excluye estrictamente cualquier producto histórico del hogar.
                if producto_destino in productos_historicos:
                    continue

                # Sumamos señales cuando más de un producto reciente apoya al candidato.
                puntajes_por_candidato[producto_destino] += puntaje

                # Conservamos el máximo soporte para explicarlo en la salida final.
                soporte_maximo_por_candidato[producto_destino] = max(
                    soporte_maximo_por_candidato[producto_destino], soporte
                )

        # Ordenamos candidatos por evidencia acumulada y luego por identificador estable.
        candidatos = sorted(puntajes_por_candidato.items(), key=lambda fila: (-fila[1], fila[0]))

        # Normalizamos el mayor puntaje del hogar antes de combinarlo con popularidad.
        puntaje_maximo = candidatos[0][1] if candidatos else 0.0

        # Construimos primero la lista de coocurrencia pura hasta el límite solicitado.
        for posicion, (producto_id, puntaje) in enumerate(candidatos[:cantidad_recomendaciones], start=1):
            filas_coocurrencia.append(
                {
                    "household_key": household_key,
                    "PRODUCT_ID": producto_id,
                    "posicion": posicion,
                    "puntaje_coocurrencia": puntaje,
                    "soporte_maximo_del_par": soporte_maximo_por_candidato[producto_id],
                    "tipo_recomendacion": "explore_coocurrencia",
                }
            )

        # Calculamos puntajes híbridos con el mismo conjunto de candidatos asociado.
        candidatos_hibridos = [
            (
                producto_id,
                peso_coocurrencia * (puntaje / puntaje_maximo)
                + (1 - peso_coocurrencia) * popularidad_por_producto.get(producto_id, 0.0),
                puntaje,
            )
            for producto_id, puntaje in candidatos
        ] if puntaje_maximo else []

        # Ordenamos el híbrido para que popularidad pueda desempatar asociaciones parecidas.
        candidatos_hibridos.sort(key=lambda fila: (-fila[1], fila[0]))

        # Construimos la lista híbrida final con sus dos componentes de puntaje visibles.
        for posicion, (producto_id, puntaje_hibrido, puntaje_coocurrencia) in enumerate(
            candidatos_hibridos[:cantidad_recomendaciones], start=1
        ):
            filas_hibridas.append(
                {
                    "household_key": household_key,
                    "PRODUCT_ID": producto_id,
                    "posicion": posicion,
                    "puntaje_hibrido": puntaje_hibrido,
                    "puntaje_coocurrencia": puntaje_coocurrencia,
                    "puntaje_popularidad": popularidad_por_producto.get(producto_id, 0.0),
                    "soporte_maximo_del_par": soporte_maximo_por_candidato[producto_id],
                    "tipo_recomendacion": "explore_hibrido",
                }
            )

    # Convertimos las listas de filas en resultados tabulares para evaluación y exportación.
    return pd.DataFrame(filas_coocurrencia), pd.DataFrame(filas_hibridas)


def interpretar_argumentos() -> argparse.Namespace:
    """Define parámetros explícitos para reproducir la ejecución del modelo."""

    # Creamos el analizador que también documenta cada opción en --help.
    analizador = argparse.ArgumentParser(description="Genera explore por coocurrencia e híbrido.")

    # Permitimos elegir la fuente temporal derivada de la Fase 2.
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)

    # Permitimos separar experimentos en carpetas de salida distintas.
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)

    # Definimos el tamaño de cada ranking evaluado y exportado.
    analizador.add_argument("--cantidad-recomendaciones", type=int, default=20)

    # Exigimos soporte de producto para evitar asociar artículos extremadamente raros.
    analizador.add_argument("--soporte-minimo-producto", type=int, default=500)

    # Exigimos soporte de par para evitar asociaciones observadas solo una vez.
    analizador.add_argument("--soporte-minimo-par", type=int, default=5)

    # Limitamos intereses recientes para que el modelo sea computacionalmente razonable.
    analizador.add_argument("--maximo-productos-recientes", type=int, default=50)

    # Asignamos mayor peso a afinidad que a popularidad dentro del modelo híbrido.
    analizador.add_argument("--peso-coocurrencia", type=float, default=0.8)

    # Permitimos cargar el catálogo desde una ubicación alternativa en pruebas.
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)

    # Devolvemos argumentos interpretados con los tipos definidos arriba.
    return analizador.parse_args()


def main() -> None:
    """Entrena asociaciones con entrenamiento y las evalúa sobre validación."""

    # Leemos parámetros y rutas seleccionadas para la ejecución actual.
    argumentos = interpretar_argumentos()

    # Validamos valores de configuración que afectarían silenciosamente al modelo.
    if argumentos.cantidad_recomendaciones <= 0:
        raise ValueError("La cantidad de recomendaciones debe ser positiva.")
    if not 0 <= argumentos.peso_coocurrencia <= 1:
        raise ValueError("El peso de coocurrencia debe estar entre 0 y 1.")

    # Cargamos entrenamiento y validación; prueba permanece reservada para el final.
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")

    # Iniciamos el cronómetro del entrenamiento (asociaciones y popularidad).
    inicio_entrenamiento = perf_counter()

    # Construimos asociaciones usando exclusivamente canastas de entrenamiento.
    asociaciones = construir_asociaciones(
        entrenamiento,
        argumentos.soporte_minimo_producto,
        argumentos.soporte_minimo_par,
    )

    # Calculamos popularidad de entrenamiento para la variante híbrida.
    popularidad = construir_popularidad(entrenamiento)

    # Cerramos el tramo de entrenamiento e iniciamos el de recomendación.
    segundos_entrenamiento = perf_counter() - inicio_entrenamiento
    inicio_recomendacion = perf_counter()

    # Generamos recomendaciones explore por afinidad y por combinación híbrida.
    recomendaciones_coocurrencia, recomendaciones_hibridas = generar_recomendaciones_coocurrencia(
        entrenamiento,
        asociaciones,
        popularidad,
        argumentos.cantidad_recomendaciones,
        argumentos.maximo_productos_recientes,
        argumentos.peso_coocurrencia,
    )

    # Medimos el tiempo de recomendación; ambas variantes se generan en la misma pasada.
    segundos_recomendacion = perf_counter() - inicio_recomendacion
    cantidad_hogares = entrenamiento["household_key"].nunique()

    # Obtenemos productos nuevos comprados realmente en validación para la evaluación explore.
    relevantes_explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")

    # Calculamos métricas comparables con el explore de popularidad del baseline.
    metricas = {
        "coocurrencia": [calcular_metricas_en_k(recomendaciones_coocurrencia, relevantes_explore, k) for k in VALORES_DE_K],
        "hibrido": [calcular_metricas_en_k(recomendaciones_hibridas, relevantes_explore, k) for k in VALORES_DE_K],
        "parametros": vars(argumentos),
        "cantidad_asociaciones": len(asociaciones),
        # Los tiempos dependen del equipo; sirven para comparar costo entre modelos.
        "tiempos_segundos": {
            "entrenamiento": round(segundos_entrenamiento, 3),
            "recomendacion_total": round(segundos_recomendacion, 3),
            "ms_recomendacion_por_hogar": round(1000 * segundos_recomendacion / cantidad_hogares, 3),
        },
    }

    # Cargamos catálogo para mostrar descripciones legibles en la inspección de ejemplos.
    productos = cargar_productos(argumentos.data_dir)

    # Agregamos atributos de producto sin afectar los puntajes ni la evaluación ya calculada.
    recomendaciones_coocurrencia = enriquecer_con_catalogo(recomendaciones_coocurrencia, productos)
    recomendaciones_hibridas = enriquecer_con_catalogo(recomendaciones_hibridas, productos)

    # Creamos la carpeta de resultados antes de exportar los archivos derivados.
    argumentos.output_dir.mkdir(parents=True, exist_ok=True)

    # Guardamos los rankings de exploración de ambos modelos para análisis cualitativo.
    recomendaciones_coocurrencia.to_csv(argumentos.output_dir / "recomendaciones_coocurrencia_validacion.csv", index=False)
    recomendaciones_hibridas.to_csv(argumentos.output_dir / "recomendaciones_hibridas_validacion.csv", index=False)

    # Guardamos solo las primeras asociaciones por puntaje para que la inspección sea manejable.
    asociaciones.head(10000).to_csv(argumentos.output_dir / "asociaciones_principales.csv", index=False)

    # Guardamos el resumen de métricas y parámetros que permite repetir el experimento.
    with (argumentos.output_dir / "metricas_validacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False, default=str)

    # Mostramos las métricas compactas para registrar la ejecución en terminal.
    print(f"\nAsociaciones con lift mayor a uno: {len(asociaciones)}")
    for nombre_modelo in ("coocurrencia", "hibrido"):
        for resultado in metricas[nombre_modelo]:
            print(f"- {nombre_modelo}@{resultado['k']}: {resultado}")


# Ejecutamos el flujo solo cuando el módulo se invoca desde la terminal.
if __name__ == "__main__":
    main()
