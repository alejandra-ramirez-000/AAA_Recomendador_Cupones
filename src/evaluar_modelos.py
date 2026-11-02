"""Centraliza la evaluación de todos los recomendadores sobre validación.

Este módulo reúne en un solo lugar:

1. la separación de productos relevantes en repeat y explore;
2. las métricas de ranking Precision, Recall, Hit Rate y NDCG en K;
3. las métricas de sistema: cobertura, concentración, diversidad, explicación;
4. la tabla comparativa de todos los modelos ya ejecutados.

Los recomendadores importan las funciones 1 y 2 desde aquí, de modo que todos se
evalúan con exactamente la misma lógica. La tabla comparativa solo lee
entrenamiento, validación y las salidas guardadas en `outputs/`; el conjunto de
prueba nunca se abre.

Entradas:
    outputs/division_temporal/interacciones_entrenamiento.csv
    outputs/division_temporal/interacciones_validacion.csv
    outputs/<modelo>/recomendaciones_*_validacion.csv
    outputs/<modelo>/metricas_validacion.json

Salidas:
    outputs/comparacion/comparacion_validacion.csv
    outputs/comparacion/comparacion_validacion.md

Ejemplo de uso (después de ejecutar baseline y coocurrencia):
    python -m src.evaluar_modelos
"""

# argparse permite elegir carpetas y valores de K sin modificar el código.
import argparse

# json lee los tiempos guardados por cada recomendador.
import json

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# numpy aporta logaritmos y operaciones numéricas para NDCG y Gini.
import numpy as np

# pandas organiza recomendaciones, objetivos y la tabla comparativa.
import pandas as pd


# Raíz del repositorio, calculada a partir de la ubicación de este archivo.
RAIZ_DEL_PROYECTO = Path(__file__).resolve().parents[1]

# Carpeta producida por la división temporal de la Fase 2.
DIRECTORIO_DE_DIVISION_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "division_temporal"

# Carpeta donde se guardan los resultados de todos los modelos.
DIRECTORIO_DE_RESULTADOS_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs"

# Carpeta donde se escribe la tabla comparativa.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "comparacion"

# Valores de K definidos en el informe para todas las evaluaciones.
VALORES_DE_K = (5, 10, 20)

# Registro de modelos conocidos. Cada entrada indica:
# - nombre: cómo aparece el modelo en la tabla;
# - tipo: objetivo evaluado ("repeat" o "explore");
# - carpeta: subcarpeta de outputs/ con sus resultados;
# - archivo: CSV de recomendaciones de validación;
# - columna_explicacion: columna que justifica cada recomendación;
# - clave_tiempos: entrada del JSON de métricas donde el modelo guardó sus tiempos.
# Para incorporar ALS u otro modelo basta con agregar una entrada aquí.
MODELOS_REGISTRADOS = [
    {
        "nombre": "baseline_frecuencia_recencia",
        "tipo": "repeat",
        "carpeta": "baseline",
        "archivo": "recomendaciones_repeat_validacion.csv",
        "columna_explicacion": "frecuencia_en_canastas",
        "clave_tiempos": "tiempos_segundos",
    },
    {
        "nombre": "baseline_popularidad",
        "tipo": "explore",
        "carpeta": "baseline",
        "archivo": "recomendaciones_explore_validacion.csv",
        "columna_explicacion": "canastas_con_producto",
        "clave_tiempos": "tiempos_segundos",
    },
    {
        "nombre": "coocurrencia",
        "tipo": "explore",
        "carpeta": "coocurrencia",
        "archivo": "recomendaciones_coocurrencia_validacion.csv",
        "columna_explicacion": "soporte_maximo_del_par",
        "clave_tiempos": "tiempos_segundos",
    },
    {
        "nombre": "hibrido_coocurrencia_popularidad",
        "tipo": "explore",
        "carpeta": "coocurrencia",
        "archivo": "recomendaciones_hibridas_validacion.csv",
        "columna_explicacion": "soporte_maximo_del_par",
        "clave_tiempos": "tiempos_segundos",
    },
    {
        "nombre": "als_repeat",
        "tipo": "repeat",
        "carpeta": "als",
        "archivo": "recomendaciones_repeat_validacion.csv",
        "columna_explicacion": "frecuencia_en_canastas",
        "clave_tiempos": "tiempos_segundos",
    },
    {
        "nombre": "als_explore",
        "tipo": "explore",
        "carpeta": "als",
        "archivo": "recomendaciones_explore_validacion.csv",
        "columna_explicacion": "similitud_explicativa",
        "clave_tiempos": "tiempos_segundos",
    },
]


def obtener_productos_relevantes(
    validacion: pd.DataFrame, entrenamiento: pd.DataFrame, tipo: str
) -> dict[int, set[int]]:
    """Obtiene productos comprados en validación separados en repeat o explore."""

    # Construimos el conjunto de productos conocidos antes de la validación por hogar.
    historiales = entrenamiento.groupby("household_key")["PRODUCT_ID"].agg(set)

    # Construimos el conjunto de productos comprados en la canasta de validación.
    compras_validacion = validacion.groupby("household_key")["PRODUCT_ID"].agg(set)

    # Creamos la estructura que será usada para comparar recomendaciones y realidad.
    relevantes: dict[int, set[int]] = {}

    # Recorremos cada hogar de validación para separar sus compras por tipo.
    for household_key, productos_validacion in compras_validacion.items():
        # Recuperamos el historial previo del hogar para distinguir novedad de repetición.
        productos_historicos = historiales[household_key]

        # Para repeat conservamos compras de validación ya presentes en el historial.
        if tipo == "repeat":
            relevantes[household_key] = productos_validacion & productos_historicos

        # Para explore conservamos compras de validación que eran nuevas para el hogar.
        elif tipo == "explore":
            relevantes[household_key] = productos_validacion - productos_historicos

        # Para listas combinadas cualquier compra de validación es un acierto.
        elif tipo == "todos":
            relevantes[household_key] = set(productos_validacion)

        # Evitamos aceptar nombres de tipo incorrectos que ocultarían un error de evaluación.
        else:
            raise ValueError("El tipo debe ser 'repeat', 'explore' o 'todos'.")

    # Devolvemos productos reales relevantes para cada hogar, incluso conjuntos vacíos.
    return relevantes


def calcular_metricas_en_k(
    recomendaciones: pd.DataFrame,
    relevantes_por_hogar: dict[int, set[int]],
    k: int,
) -> dict[str, float | int]:
    """Calcula Precision, Recall, Hit Rate y NDCG binarios para un K dado."""

    # Solo evaluamos hogares que realmente compraron al menos un producto del tipo analizado.
    hogares_evaluables = {
        hogar: relevantes
        for hogar, relevantes in relevantes_por_hogar.items()
        if relevantes
    }

    # Si no existe un objetivo relevante, devolvemos métricas explícitamente no disponibles.
    if not hogares_evaluables:
        return {"k": k, "hogares_evaluables": 0, "precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "ndcg": 0.0}

    # Indexamos recomendaciones por hogar y conservamos solo las primeras K posiciones.
    recomendaciones_por_hogar = (
        recomendaciones.loc[recomendaciones["posicion"] <= k]
        .sort_values(["household_key", "posicion"])
        .groupby("household_key")["PRODUCT_ID"]
        .agg(list)
        .to_dict()
    )

    # Inicializamos acumuladores para calcular promedios al final del recorrido.
    suma_precision = 0.0
    suma_recall = 0.0
    suma_hit_rate = 0.0
    suma_ndcg = 0.0

    # Evaluamos cada hogar contra sus productos realmente comprados en validación.
    for household_key, relevantes in hogares_evaluables.items():
        # Usamos una lista vacía si por algún motivo no se generó recomendación para el hogar.
        recomendados = recomendaciones_por_hogar.get(household_key, [])

        # Marcamos si cada posición recomendada coincide con una compra relevante.
        aciertos_por_posicion = [producto in relevantes for producto in recomendados]

        # Contamos aciertos para Precision, Recall y Hit Rate.
        cantidad_aciertos = sum(aciertos_por_posicion)

        # Precision@K usa K como denominador para penalizar posiciones sin acierto.
        suma_precision += cantidad_aciertos / k

        # Recall@K considera cuántos productos relevantes recuperó la lista.
        suma_recall += cantidad_aciertos / len(relevantes)

        # Hit Rate vale uno cuando existe por lo menos un acierto dentro de K.
        suma_hit_rate += float(cantidad_aciertos > 0)

        # DCG asigna más valor a aciertos ubicados al inicio de la lista.
        dcg = sum(
            1 / np.log2(posicion + 1)
            for posicion, es_acierto in enumerate(aciertos_por_posicion, start=1)
            if es_acierto
        )

        # IDCG representa el mejor orden posible para los aciertos disponibles.
        cantidad_ideal = min(len(relevantes), k)
        idcg = sum(1 / np.log2(posicion + 1) for posicion in range(1, cantidad_ideal + 1))

        # NDCG normaliza el ranking observado respecto al mejor ranking alcanzable.
        suma_ndcg += dcg / idcg if idcg else 0.0

    # Calculamos el promedio entre hogares para no favorecer a hogares con más productos.
    cantidad_hogares = len(hogares_evaluables)

    # Devolvemos un diccionario serializable para guardar las métricas en JSON.
    return {
        "k": k,
        "hogares_evaluables": cantidad_hogares,
        "precision": round(suma_precision / cantidad_hogares, 6),
        "recall": round(suma_recall / cantidad_hogares, 6),
        "hit_rate": round(suma_hit_rate / cantidad_hogares, 6),
        "ndcg": round(float(suma_ndcg / cantidad_hogares), 6),
    }


def calcular_indice_de_gini(exposiciones: pd.Series) -> float:
    """Mide la desigualdad de exposición entre productos (0 = pareja, 1 = concentrada).

    Recibe cuántas veces fue recomendado cada producto del universo considerado,
    incluidos los productos con cero exposiciones.
    """

    # Ordenamos las exposiciones de menor a mayor, como exige la fórmula de Gini.
    valores = np.sort(exposiciones.to_numpy(dtype=float))

    # Sin productos o sin exposiciones no existe desigualdad que medir.
    if valores.size == 0 or valores.sum() == 0:
        return 0.0

    # Numeramos posiciones desde uno para aplicar la fórmula de Gini ordenada.
    posiciones = np.arange(1, valores.size + 1)

    # Aplicamos G = sum((2i - n - 1) * x_i) / (n * sum(x)).
    gini = np.sum((2 * posiciones - valores.size - 1) * valores) / (valores.size * valores.sum())

    # Devolvemos un número de Python redondeado para facilitar su lectura.
    return round(float(gini), 6)


def calcular_metricas_de_sistema(
    recomendaciones: pd.DataFrame,
    k: int,
    hogares_elegibles: set[int],
    productos_de_entrenamiento: set[int],
    columna_explicacion: str | None = None,
) -> dict[str, float | int]:
    """Calcula cobertura, concentración, diversidad y explicabilidad en K.

    Parámetros:
        recomendaciones: tabla con household_key, PRODUCT_ID, posicion y,
            opcionalmente, DEPARTMENT y la columna de explicación.
        k: cantidad de posiciones consideradas por hogar.
        hogares_elegibles: hogares que deberían recibir recomendaciones.
        productos_de_entrenamiento: universo de productos disponibles para recomendar.
        columna_explicacion: columna cuyo valor positivo justifica la recomendación.
    """

    # Conservamos solo las primeras K posiciones de cada hogar.
    primeras_k = recomendaciones.loc[recomendaciones["posicion"] <= k]

    # Cobertura de hogares: proporción de hogares elegibles con al menos una recomendación.
    hogares_con_recomendacion = set(primeras_k["household_key"]) & hogares_elegibles
    cobertura_hogares = len(hogares_con_recomendacion) / len(hogares_elegibles) if hogares_elegibles else 0.0

    # Cobertura de catálogo: proporción del universo de entrenamiento que alguna vez se recomienda.
    productos_recomendados = set(primeras_k["PRODUCT_ID"])
    cobertura_productos = (
        len(productos_recomendados & productos_de_entrenamiento) / len(productos_de_entrenamiento)
        if productos_de_entrenamiento
        else 0.0
    )

    # Contamos cuántas veces aparece cada producto recomendado.
    exposiciones = primeras_k["PRODUCT_ID"].value_counts()

    # Concentración top 10: qué parte de todas las recomendaciones acaparan los 10 productos más expuestos.
    total_recomendaciones = int(exposiciones.sum())
    participacion_top_10 = exposiciones.head(10).sum() / total_recomendaciones if total_recomendaciones else 0.0

    # Agregamos con cero exposiciones los productos nunca recomendados para un Gini sobre todo el universo.
    productos_sin_exposicion = len(productos_de_entrenamiento - productos_recomendados)
    exposiciones_completas = pd.concat(
        [exposiciones, pd.Series(np.zeros(productos_sin_exposicion))], ignore_index=True
    )
    gini_exposicion = calcular_indice_de_gini(exposiciones_completas)

    # Diversidad: promedio de departamentos distintos dentro de la lista de cada hogar.
    if "DEPARTMENT" in primeras_k.columns and not primeras_k.empty:
        departamentos_por_hogar = primeras_k.groupby("household_key")["DEPARTMENT"].nunique()
        diversidad_departamentos = float(departamentos_por_hogar.mean())
    else:
        diversidad_departamentos = float("nan")

    # Explicabilidad: proporción de recomendaciones con una señal observable que las justifique.
    if columna_explicacion and columna_explicacion in primeras_k.columns and not primeras_k.empty:
        senal = pd.to_numeric(primeras_k[columna_explicacion], errors="coerce")
        porcentaje_con_explicacion = float((senal.fillna(0) > 0).mean())
    else:
        porcentaje_con_explicacion = 0.0

    # Devolvemos valores redondeados y serializables.
    return {
        "k": k,
        "cobertura_hogares": round(cobertura_hogares, 6),
        "cobertura_productos": round(cobertura_productos, 6),
        "productos_distintos_recomendados": len(productos_recomendados),
        "participacion_top_10_productos": round(float(participacion_top_10), 6),
        "gini_exposicion": gini_exposicion,
        "departamentos_distintos_por_lista": round(diversidad_departamentos, 6),
        "porcentaje_con_explicacion": round(porcentaje_con_explicacion, 6),
    }


def leer_tiempos(directorio_del_modelo: Path, clave_tiempos: str) -> dict[str, float]:
    """Lee los tiempos guardados por un recomendador, si existen."""

    # Ubicamos el archivo de métricas que escribe cada recomendador.
    ruta_metricas = directorio_del_modelo / "metricas_validacion.json"

    # Si el modelo no guardó métricas devolvemos un diccionario vacío.
    if not ruta_metricas.exists():
        return {}

    # Leemos el JSON con codificación UTF-8.
    with ruta_metricas.open(encoding="utf-8") as archivo:
        metricas_guardadas = json.load(archivo)

    # Devolvemos los tiempos o un diccionario vacío si la ejecución era anterior a su registro.
    return metricas_guardadas.get(clave_tiempos, {})


def construir_tabla_comparativa(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    directorio_resultados: Path,
    modelos: list[dict[str, str]],
    valores_de_k: tuple[int, ...] = VALORES_DE_K,
) -> pd.DataFrame:
    """Evalúa todos los modelos registrados con la misma metodología."""

    # Calculamos una sola vez los objetivos reales de cada tipo.
    relevantes_por_tipo = {
        "repeat": obtener_productos_relevantes(validacion, entrenamiento, "repeat"),
        "explore": obtener_productos_relevantes(validacion, entrenamiento, "explore"),
    }

    # Los hogares elegibles son los que tienen canasta de validación.
    hogares_elegibles = set(validacion["household_key"])

    # El universo de productos recomendables es todo lo observado en entrenamiento.
    productos_de_entrenamiento = set(entrenamiento["PRODUCT_ID"])

    # Acumulamos una fila por modelo y valor de K.
    filas: list[dict[str, object]] = []

    # Recorremos cada modelo registrado.
    for modelo in modelos:
        # Construimos la ruta del CSV de recomendaciones del modelo.
        directorio_del_modelo = directorio_resultados / modelo["carpeta"]
        ruta_recomendaciones = directorio_del_modelo / modelo["archivo"]

        # Saltamos modelos todavía no ejecutados, avisando en terminal.
        if not ruta_recomendaciones.exists():
            print(f"Aviso: no se encontró {ruta_recomendaciones}; se omite {modelo['nombre']}.")
            continue

        # Leemos las recomendaciones de validación ya generadas.
        recomendaciones = pd.read_csv(ruta_recomendaciones)

        # Leemos los tiempos registrados por el recomendador, si existen.
        tiempos = leer_tiempos(directorio_del_modelo, modelo["clave_tiempos"])

        # Evaluamos el modelo en cada K con las mismas funciones.
        for k in valores_de_k:
            metricas_ranking = calcular_metricas_en_k(
                recomendaciones, relevantes_por_tipo[modelo["tipo"]], k
            )
            metricas_sistema = calcular_metricas_de_sistema(
                recomendaciones,
                k,
                hogares_elegibles,
                productos_de_entrenamiento,
                modelo["columna_explicacion"],
            )

            # Combinamos identificación, métricas de ranking, de sistema y tiempos.
            filas.append(
                {
                    "modelo": modelo["nombre"],
                    "tipo": modelo["tipo"],
                    **metricas_ranking,
                    **{clave: valor for clave, valor in metricas_sistema.items() if clave != "k"},
                    "segundos_entrenamiento": tiempos.get("entrenamiento"),
                    "ms_recomendacion_por_hogar": tiempos.get("ms_recomendacion_por_hogar"),
                }
            )

    # Ordenamos para leer primero repeat y luego explore, cada uno por K.
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla = tabla.sort_values(["tipo", "k", "modelo"], ascending=[False, True, True]).reset_index(drop=True)

    # Devolvemos la tabla lista para exportar.
    return tabla


def tabla_a_markdown(tabla: pd.DataFrame) -> str:
    """Convierte la tabla comparativa en Markdown sin depender de librerías extra."""

    # Formateamos números con seis decimales y dejamos vacíos los valores ausentes.
    def formatear(valor: object) -> str:
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return ""
        if isinstance(valor, float):
            return f"{valor:.6f}"
        return str(valor)

    # Construimos encabezado, separador y filas.
    columnas = list(tabla.columns)
    lineas = [
        "| " + " | ".join(columnas) + " |",
        "| " + " | ".join("---" for _ in columnas) + " |",
    ]
    for fila in tabla.itertuples(index=False):
        lineas.append("| " + " | ".join(formatear(valor) for valor in fila) + " |")

    # Devolvemos el texto completo con un salto de línea final.
    return "\n".join(lineas) + "\n"


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución de la comparación."""

    # Creamos el analizador de argumentos mostrado al ejecutar --help.
    analizador = argparse.ArgumentParser(
        description="Compara todos los recomendadores sobre validación con la misma metodología."
    )

    # Permitimos indicar otra carpeta de división temporal.
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)

    # Permitimos indicar otra carpeta raíz de resultados de modelos.
    analizador.add_argument("--resultados-dir", type=Path, default=DIRECTORIO_DE_RESULTADOS_POR_DEFECTO)

    # Permitimos escribir la comparación en otra carpeta.
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)

    # Devolvemos argumentos ya convertidos a sus tipos.
    return analizador.parse_args()


def main() -> None:
    """Genera la tabla comparativa de modelos sobre validación."""

    # Leemos rutas de la ejecución solicitada.
    argumentos = interpretar_argumentos()

    # Cargamos únicamente entrenamiento y validación; prueba queda reservada.
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")

    # Evaluamos todos los modelos registrados.
    tabla = construir_tabla_comparativa(
        entrenamiento, validacion, argumentos.resultados_dir, MODELOS_REGISTRADOS
    )

    # Creamos la carpeta de salida antes de escribir.
    argumentos.output_dir.mkdir(parents=True, exist_ok=True)

    # Guardamos la tabla en CSV para análisis y en Markdown para el informe.
    tabla.to_csv(argumentos.output_dir / "comparacion_validacion.csv", index=False)
    (argumentos.output_dir / "comparacion_validacion.md").write_text(
        tabla_a_markdown(tabla), encoding="utf-8"
    )

    # Mostramos un resumen compacto en K = 10 para registrar la ejecución.
    columnas_resumen = [
        "modelo", "tipo", "precision", "recall", "hit_rate", "ndcg",
        "cobertura_productos", "gini_exposicion", "porcentaje_con_explicacion",
    ]
    print("\nComparación en validación (K = 10)")
    print(tabla.loc[tabla["k"] == 10, columnas_resumen].to_string(index=False))


# Ejecutamos la comparación solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
