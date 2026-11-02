"""Análisis exploratorio para la defensa: evidencia de las decisiones de diseño.

Responde con datos:
1. ¿Qué parte de cada canasta es recompra y qué parte es nueva? (justifica repeat/explore)
2. ¿Cuántos hogares tienen objetivos repeat, explore o ambos en validación?
3. ¿Cómo rinde el sistema según la longitud del historial? (cold start)
4. ¿Qué tan concentrada está la popularidad y qué parte del catálogo se recomienda?
5. ¿Cuánto pesan combustible y tabaco, y cuántas posiciones ocupaban sin reglas?

Usa solo entrenamiento y validación, y las listas de validación ya guardadas.
El conjunto de prueba no se abre. No cambia modelos ni parámetros: es evidencia
descriptiva de decisiones ya tomadas (D-006, D-008).

Salidas en outputs/eda/ (prefijo `defensa_`) y un resumen en
`outputs/eda/analisis_defensa.md`.

Ejemplo de uso:
    python -m src.analizar_repeticion
"""

# argparse permite cambiar rutas sin modificar el código.
import argparse

# json guarda el resumen numérico y lee las reglas.
import json

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# matplotlib genera los gráficos sin abrir ventanas.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy y pandas calculan las tablas.
import numpy as np
import pandas as pd

# Piezas ya validadas.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos
from src.evaluar_modelos import (
    DIRECTORIO_DE_DIVISION_POR_DEFECTO,
    RAIZ_DEL_PROYECTO,
    calcular_indice_de_gini,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)
from src.reglas_negocio import RUTA_REGLAS_POR_DEFECTO
from src.seleccionar_modelo import RUTA_CONFIGURACION_POR_DEFECTO, combinar_listas


# Carpeta de salida del EDA.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "eda"

# Tramos de posición de la canasta dentro del historial del hogar.
TRAMOS_DE_CANASTA = [1, 2, 6, 11, 26, 51, 101, np.inf]
ETIQUETAS_DE_TRAMO = ["1", "2-5", "6-10", "11-25", "26-50", "51-100", "101+"]

# Columnas que definen una "variante": misma oferta para el cliente aunque cambie el código.
COLUMNAS_VARIANTE = ["SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT"]

# Listas de validación que se analizan: (nombre, tipo, ruta relativa a outputs/).
LISTAS_DE_VALIDACION = [
    ("repeat: frecuencia + recencia", "repeat", "baseline/recomendaciones_repeat_validacion.csv"),
    ("explore: popularidad", "explore", "baseline/recomendaciones_explore_validacion.csv"),
    ("explore: ALS", "explore", "als/recomendaciones_explore_validacion.csv"),
]


def calcular_repeticion(interacciones: pd.DataFrame, catalogo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Proporción de productos de cada canasta que el hogar ya había comprado antes.

    Devuelve una fila por canasta con `productos`, `nuevos` y
    `proporcion_repetida`. Con catálogo, también `proporcion_repetida_variante`:
    se considera repetido si el hogar ya había comprado otra variante (misma
    subcategoría, marca y tamaño).
    """

    # Una fila por producto distinto dentro de cada canasta.
    lineas = interacciones[["household_key", "BASKET_ID", "orden_de_canasta", "PRODUCT_ID"]].drop_duplicates(
        ["household_key", "BASKET_ID", "PRODUCT_ID"]
    )

    # Un producto es nuevo si esta canasta es la primera en que el hogar lo compra.
    primera_compra = lineas.groupby(["household_key", "PRODUCT_ID"])["orden_de_canasta"].transform("min")
    lineas = lineas.assign(es_nuevo=lineas["orden_de_canasta"] == primera_compra)

    agregaciones = {"productos": ("PRODUCT_ID", "size"), "nuevos": ("es_nuevo", "sum")}

    # Lo mismo, a nivel de variante.
    if catalogo is not None:
        clave = catalogo.set_index("PRODUCT_ID")[COLUMNAS_VARIANTE].fillna("").astype(str).agg("|".join, axis=1)
        lineas = lineas.assign(variante=lineas["PRODUCT_ID"].map(clave).fillna(lineas["PRODUCT_ID"].astype(str)))
        primera_variante = lineas.groupby(["household_key", "variante"])["orden_de_canasta"].transform("min")
        lineas = lineas.assign(es_nueva_variante=lineas["orden_de_canasta"] == primera_variante)
        agregaciones["nuevas_variantes"] = ("es_nueva_variante", "sum")

    canastas = lineas.groupby(["household_key", "BASKET_ID", "orden_de_canasta"], as_index=False).agg(**agregaciones)
    canastas["proporcion_repetida"] = 1 - canastas["nuevos"] / canastas["productos"]
    if catalogo is not None:
        canastas["proporcion_repetida_variante"] = 1 - canastas["nuevas_variantes"] / canastas["productos"]
    return canastas


def resumir_repeticion_por_tramo(canastas: pd.DataFrame) -> pd.DataFrame:
    """Promedia la proporción repetida según la posición de la canasta en el historial."""

    tramo = pd.cut(canastas["orden_de_canasta"], TRAMOS_DE_CANASTA, right=False, labels=ETIQUETAS_DE_TRAMO)
    columnas = [c for c in ("proporcion_repetida", "proporcion_repetida_variante") if c in canastas.columns]
    resumen = canastas.assign(tramo=tramo).groupby("tramo", observed=True).agg(
        canastas=("BASKET_ID", "size"), **{c: (c, "mean") for c in columnas}
    )
    return resumen.round(4).reset_index()


def contar_objetivos(entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> dict[str, int]:
    """Hogares con recompras, con productos nuevos, con ambos o con ninguno en validación."""

    repeat = obtener_productos_relevantes(validacion, entrenamiento, "repeat")
    explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")
    hogares = set(repeat) | set(explore)
    con_repeat = {h for h in hogares if repeat.get(h)}
    con_explore = {h for h in hogares if explore.get(h)}
    return {
        "hogares": len(hogares),
        "con_recompra": len(con_repeat),
        "con_producto_nuevo": len(con_explore),
        "con_ambos": len(con_repeat & con_explore),
        "solo_recompra": len(con_repeat - con_explore),
        "solo_producto_nuevo": len(con_explore - con_repeat),
        "ninguno": len(hogares - con_repeat - con_explore),
    }


def segmentar_por_historial(entrenamiento: pd.DataFrame, cantidad_segmentos: int = 3) -> pd.Series:
    """Asigna cada hogar a un segmento según cuántas canastas tiene en entrenamiento."""

    canastas = entrenamiento.groupby("household_key")["BASKET_ID"].nunique()
    segmentos = pd.qcut(canastas, cantidad_segmentos, duplicates="drop")
    etiquetas = {
        intervalo: f"{nombre} ({int(np.floor(intervalo.left)) + 1}-{int(intervalo.right)} canastas)"
        for intervalo, nombre in zip(segmentos.cat.categories, ["corto", "medio", "largo", "muy largo"])
    }
    orden = [etiquetas[intervalo] for intervalo in segmentos.cat.categories]
    return pd.Series(pd.Categorical(segmentos.map(etiquetas).astype(str), categories=orden, ordered=True), index=canastas.index)


def rendimiento_por_historial(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    listas: dict[str, tuple[str, pd.DataFrame]],
    k: int = 10,
) -> pd.DataFrame:
    """Métricas en K por segmento de historial para cada lista de validación."""

    segmento = segmentar_por_historial(entrenamiento)
    relevantes = {tipo: obtener_productos_relevantes(validacion, entrenamiento, tipo) for tipo in ("repeat", "explore")}

    filas = []
    for nombre, (tipo, lista) in listas.items():
        for etiqueta in segmento.cat.categories:
            hogares = set(segmento.index[segmento == etiqueta])
            objetivo = {h: productos for h, productos in relevantes[tipo].items() if h in hogares}
            metricas = calcular_metricas_en_k(lista, objetivo, k)
            filas.append({"lista": nombre, "segmento": etiqueta, "hogares_del_segmento": len(hogares), **metricas})
    return pd.DataFrame(filas)


def calcular_concentracion(entrenamiento: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Concentración de compras por producto y curva de Lorenz."""

    canastas_por_producto = (
        entrenamiento.drop_duplicates(["BASKET_ID", "PRODUCT_ID"]).groupby("PRODUCT_ID").size().sort_values(ascending=False)
    )
    total = canastas_por_producto.sum()
    acumulado = canastas_por_producto.cumsum() / total
    cantidad = len(canastas_por_producto)

    def participacion(fraccion: float) -> float:
        return round(float(acumulado.iloc[max(int(np.ceil(cantidad * fraccion)) - 1, 0)]), 4)

    resumen = {
        "productos_comprados": int(cantidad),
        "participacion_top_1_pct": participacion(0.01),
        "participacion_top_10_pct": participacion(0.10),
        "productos_comprados_en_una_sola_canasta": int((canastas_por_producto == 1).sum()),
        "gini": calcular_indice_de_gini(canastas_por_producto),
    }

    # Curva de Lorenz submuestreada para graficar.
    posiciones = np.unique(np.linspace(0, cantidad - 1, 200).astype(int))
    curva = pd.DataFrame(
        {"fraccion_productos": (posiciones + 1) / cantidad, "fraccion_compras": acumulado.to_numpy()[posiciones]}
    )
    return resumen, curva


def cobertura_por_departamento(
    catalogo: pd.DataFrame, entrenamiento: pd.DataFrame, listas: dict[str, tuple[str, pd.DataFrame]], k: int = 10
) -> pd.DataFrame:
    """Productos del catálogo, comprados y recomendados (top K) por departamento."""

    departamento = catalogo.set_index("PRODUCT_ID")["DEPARTMENT"]
    tabla = catalogo.groupby("DEPARTMENT").size().rename("productos_catalogo").to_frame()
    comprados = pd.Series(entrenamiento["PRODUCT_ID"].unique())
    tabla["productos_comprados"] = comprados.map(departamento).value_counts()
    for nombre, (_, lista) in listas.items():
        recomendados = pd.Series(lista.loc[lista["posicion"] <= k, "PRODUCT_ID"].unique())
        tabla[f"recomendados_{nombre}"] = recomendados.map(departamento).value_counts()
    tabla = tabla.fillna(0).astype(int).sort_values("productos_comprados", ascending=False)
    return tabla.reset_index()


def ficha_no_elegibles(
    entrenamiento: pd.DataFrame,
    catalogo: pd.DataFrame,
    reglas: dict,
    lista_sin_reglas: pd.DataFrame,
) -> pd.DataFrame:
    """Peso de combustible y tabaco en las compras y en las listas antes de las reglas."""

    indice = catalogo.set_index("PRODUCT_ID")
    grupos = {
        "Combustible (KIOSK-GAS)": set(indice.index[indice["DEPARTMENT"] == "KIOSK-GAS"]),
        "Tabaco": set(indice.index[indice["COMMODITY_DESC"].isin(reglas.get("categorias_excluidas", {}))]),
    }

    lineas_totales = len(entrenamiento)
    hogares_totales = entrenamiento["household_key"].nunique()
    hogares_con_lista = lista_sin_reglas["household_key"].nunique()

    filas = []
    for nombre, productos in grupos.items():
        compras = entrenamiento.loc[entrenamiento["PRODUCT_ID"].isin(productos)]
        canastas_por_hogar = compras.groupby("household_key")["BASKET_ID"].nunique()
        en_lista = lista_sin_reglas.loc[lista_sin_reglas["PRODUCT_ID"].isin(productos)]
        filas.append(
            {
                "grupo": nombre,
                "productos_en_catalogo": len(productos),
                "porcentaje_de_lineas": round(100 * len(compras) / lineas_totales, 2),
                "porcentaje_de_hogares_compradores": round(100 * len(canastas_por_hogar) / hogares_totales, 1),
                "porcentaje_compradores_que_recompran": round(100 * (canastas_por_hogar >= 2).mean(), 1)
                if len(canastas_por_hogar) else 0.0,
                "canastas_medias_por_comprador": round(float(canastas_por_hogar.mean()), 1) if len(canastas_por_hogar) else 0.0,
                "posiciones_en_listas_sin_reglas": len(en_lista),
                "porcentaje_hogares_con_el_grupo_en_su_lista": round(
                    100 * en_lista["household_key"].nunique() / hogares_con_lista, 1
                ),
            }
        )
    return pd.DataFrame(filas)


def graficar(salida: Path, por_tramo: pd.DataFrame, rendimiento: pd.DataFrame, curva: pd.DataFrame, cobertura: pd.DataFrame) -> None:
    """Genera los cuatro gráficos del análisis."""

    # 1. Repetición según la posición de la canasta.
    datos = por_tramo.loc[por_tramo["tramo"] != "1"]
    figura, eje = plt.subplots(figsize=(10, 6))
    eje.plot(datos["tramo"].astype(str), 100 * datos["proporcion_repetida"], marker="o", label="Mismo código de producto")
    if "proporcion_repetida_variante" in datos:
        eje.plot(datos["tramo"].astype(str), 100 * datos["proporcion_repetida_variante"], marker="o",
                 linestyle="--", label="Misma variante (subcategoría, marca y tamaño)")
    eje.set_ylim(0, 100)
    eje.set_xlabel("Número de canasta dentro del historial del hogar")
    eje.set_ylabel("% de productos de la canasta ya comprados antes")
    eje.set_title("Cuánto de cada canasta es recompra")
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(salida / "defensa_repeticion_por_canasta.png", dpi=150)
    plt.close(figura)

    # 2. Hit Rate@10 por segmento de historial.
    orden = list(dict.fromkeys(rendimiento["segmento"]))
    tabla = rendimiento.pivot(index="segmento", columns="lista", values="hit_rate").reindex(orden)
    figura, eje = plt.subplots(figsize=(10, 6))
    (100 * tabla).plot(kind="bar", ax=eje, rot=0)
    eje.set_xlabel("Longitud del historial en entrenamiento (terciles de hogares)")
    eje.set_ylabel("Hit Rate@10 en validación (%)")
    eje.set_title("Rendimiento según cuánto historial tiene el hogar")
    eje.set_ylim(0, 100 * float(tabla.max().max()) * 1.35)
    eje.legend(title="Lista", loc="upper right", ncol=3)
    eje.grid(axis="y", alpha=0.3)
    figura.tight_layout()
    figura.savefig(salida / "defensa_rendimiento_por_historial.png", dpi=150)
    plt.close(figura)

    # 3. Curva de Lorenz de la popularidad.
    figura, eje = plt.subplots(figsize=(7, 7))
    eje.plot(100 * curva["fraccion_productos"], 100 * curva["fraccion_compras"], label="Productos ordenados por popularidad")
    eje.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Reparto perfectamente parejo")
    eje.set_xlabel("% de productos (de más a menos comprado)")
    eje.set_ylabel("% acumulado de apariciones en canastas")
    eje.set_title("Concentración de la popularidad")
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(salida / "defensa_concentracion_popularidad.png", dpi=150)
    plt.close(figura)

    # 4. Productos distintos recomendados por departamento (top 8 departamentos).
    columnas = [c for c in cobertura.columns if c.startswith("recomendados_explore")]
    datos = cobertura.head(8).set_index("DEPARTMENT")[columnas]
    datos.columns = [c.replace("recomendados_", "") for c in columnas]
    figura, eje = plt.subplots(figsize=(10, 6))
    datos.plot(kind="bar", ax=eje, rot=30)
    eje.set_xlabel("Departamento")
    eje.set_ylabel("Productos distintos recomendados (top 10, validación)")
    eje.set_title("Cobertura del catálogo de las listas explore")
    eje.grid(axis="y", alpha=0.3)
    figura.tight_layout()
    figura.savefig(salida / "defensa_cobertura_por_departamento.png", dpi=150)
    plt.close(figura)


def tabla_markdown(tabla: pd.DataFrame) -> str:
    """Convierte una tabla en Markdown simple."""

    lineas = ["| " + " | ".join(map(str, tabla.columns)) + " |", "| " + " | ".join("---" for _ in tabla.columns) + " |"]
    for fila in tabla.itertuples(index=False):
        lineas.append("| " + " | ".join(str(valor) for valor in fila) + " |")
    return "\n".join(lineas)


def escribir_reporte(ruta: Path, resumen: dict, por_tramo: pd.DataFrame, rendimiento: pd.DataFrame,
                     cobertura: pd.DataFrame, ficha: pd.DataFrame) -> None:
    """Resumen Markdown con las tablas principales y su lectura."""

    repeticion = resumen["repeticion"]
    objetivos = resumen["objetivos_validacion"]
    concentracion = resumen["concentracion"]
    rendimiento_k10 = rendimiento[["lista", "segmento", "hogares_evaluables", "hit_rate", "ndcg"]]

    lineas = [
        "# Análisis exploratorio para la defensa",
        "",
        "Datos: entrenamiento + validación (la prueba no se usa). Generado por `python -m src.analizar_repeticion`.",
        "",
        "## 1. Recompra dentro de cada canasta",
        "",
        f"- Sin contar la primera canasta de cada hogar, en promedio el **{100 * repeticion['media_por_canasta']:.1f} %** "
        f"de los productos de una canasta ya habían sido comprados antes por el hogar "
        f"({100 * repeticion['media_por_linea']:.1f} % si se pondera por línea).",
        f"- A nivel de variante (misma subcategoría, marca y tamaño): **{100 * repeticion['media_por_canasta_variante']:.1f} %**.",
        "",
        tabla_markdown(por_tramo),
        "",
        "## 2. Objetivos de validación por hogar",
        "",
        tabla_markdown(pd.DataFrame([objetivos])),
        "",
        "## 3. Rendimiento según la longitud del historial (K = 10, validación)",
        "",
        tabla_markdown(rendimiento_k10),
        "",
        "## 4. Concentración de la popularidad",
        "",
        f"- El 1 % de los productos más comprados reúne el **{100 * concentracion['participacion_top_1_pct']:.1f} %** "
        f"de las apariciones en canastas; el 10 %, el **{100 * concentracion['participacion_top_10_pct']:.1f} %**.",
        f"- {concentracion['productos_comprados_en_una_sola_canasta']:,} de {concentracion['productos_comprados']:,} "
        f"productos aparecen en una sola canasta. Gini: {concentracion['gini']:.3f}.".replace(",", "."),
        "",
        "### Productos distintos por departamento",
        "",
        tabla_markdown(cobertura.head(12)),
        "",
        "## 5. Combustible y tabaco",
        "",
        tabla_markdown(ficha),
        "",
    ]
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución."""

    analizador = argparse.ArgumentParser(description="EDA de repetición, cold start y concentración para la defensa.")
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)
    analizador.add_argument("--resultados-dir", type=Path, default=RAIZ_DEL_PROYECTO / "outputs")
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)
    analizador.add_argument("--reglas", type=Path, default=RUTA_REGLAS_POR_DEFECTO)
    analizador.add_argument("--configuracion", type=Path, default=RUTA_CONFIGURACION_POR_DEFECTO)
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)
    return analizador.parse_args()


def main() -> None:
    """Calcula todas las tablas y gráficos del análisis."""

    argumentos = interpretar_argumentos()
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")
    catalogo = cargar_productos(argumentos.data_dir)
    with argumentos.reglas.open(encoding="utf-8") as archivo:
        reglas = json.load(archivo)
    with argumentos.configuracion.open(encoding="utf-8") as archivo:
        configuracion = json.load(archivo)
    listas = {
        nombre: (tipo, pd.read_csv(argumentos.resultados_dir / ruta)) for nombre, tipo, ruta in LISTAS_DE_VALIDACION
    }

    # 1. Repetición (entrenamiento + validación: todo lo observado antes de prueba).
    canastas = calcular_repeticion(pd.concat([entrenamiento, validacion], ignore_index=True), catalogo)
    sin_primera = canastas.loc[canastas["orden_de_canasta"] > 1]
    por_tramo = resumir_repeticion_por_tramo(canastas)

    # 2 a 5.
    objetivos = contar_objetivos(entrenamiento, validacion)
    rendimiento = rendimiento_por_historial(entrenamiento, validacion, listas)
    concentracion, curva = calcular_concentracion(entrenamiento)
    cobertura = cobertura_por_departamento(catalogo, entrenamiento, listas)
    lista_sin_reglas = combinar_listas(
        listas["repeat: frecuencia + recencia"][1], listas["explore: ALS"][1],
        configuracion["k"], configuracion["mezcla"]["proporcion_repeat"],
    )
    ficha = ficha_no_elegibles(entrenamiento, catalogo, reglas, lista_sin_reglas)

    resumen = {
        "repeticion": {
            "media_por_canasta": round(float(sin_primera["proporcion_repetida"].mean()), 4),
            "media_por_linea": round(float(1 - sin_primera["nuevos"].sum() / sin_primera["productos"].sum()), 4),
            "media_por_canasta_variante": round(float(sin_primera["proporcion_repetida_variante"].mean()), 4),
            "canastas_analizadas": int(len(sin_primera)),
        },
        "objetivos_validacion": objetivos,
        "concentracion": concentracion,
    }

    salida = argumentos.output_dir
    salida.mkdir(parents=True, exist_ok=True)
    por_tramo.to_csv(salida / "defensa_repeticion_por_tramo.csv", index=False)
    rendimiento.to_csv(salida / "defensa_rendimiento_por_historial.csv", index=False)
    curva.to_csv(salida / "defensa_curva_lorenz.csv", index=False)
    cobertura.to_csv(salida / "defensa_cobertura_por_departamento.csv", index=False)
    ficha.to_csv(salida / "defensa_combustible_y_tabaco.csv", index=False)
    (salida / "defensa_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar(salida, por_tramo, rendimiento, curva, cobertura)
    escribir_reporte(salida / "analisis_defensa.md", resumen, por_tramo, rendimiento, cobertura, ficha)
    print((salida / "analisis_defensa.md").read_text(encoding="utf-8"))


# Ejecutamos solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
