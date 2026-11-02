"""Aplica reglas de negocio configurables a las recomendaciones del modelo.

El modelo responde "qué producto es relevante para este hogar". Las reglas
responden "qué producto se puede ofrecer con cupón". Por eso:

- se aplican después del ranking, sobre listas candidatas más largas que K;
- no modifican datos ni entrenamiento;
- cada producto descartado queda registrado con la regla y el motivo;
- su efecto se mide aparte de la calidad del modelo.

Las reglas se leen de `config/reglas_negocio.json`. Stock, margen y campañas
son puntos de integración: el dataset no trae esos datos, así que vienen vacíos
y solo se activan si el negocio los provee.

Salidas en outputs/reglas/ (medidas sobre validación, sin tocar prueba):
    efecto_reglas_validacion.md
    efecto_reglas_validacion.csv
    filtradas_validacion.csv

Ejemplo de uso:
    python -m src.reglas_negocio
"""

# argparse permite cambiar rutas sin modificar el código.
import argparse

# json lee la configuración de reglas y del modelo final.
import json

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# pandas aplica las reglas de forma vectorizada.
import pandas as pd

# Utilidades ya existentes.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos
from src.evaluar_modelos import (
    DIRECTORIO_DE_DIVISION_POR_DEFECTO,
    RAIZ_DEL_PROYECTO,
    calcular_metricas_de_sistema,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)
from src.seleccionar_modelo import (
    RUTA_CONFIGURACION_POR_DEFECTO,
    combinar_listas,
    ruta_de_recomendaciones,
)


# Rutas por defecto.
RUTA_REGLAS_POR_DEFECTO = RAIZ_DEL_PROYECTO / "config" / "reglas_negocio.json"
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "reglas"


def cargar_reglas(ruta: Path = RUTA_REGLAS_POR_DEFECTO) -> dict:
    """Lee y valida la configuración de reglas."""

    with ruta.open(encoding="utf-8") as archivo:
        reglas = json.load(archivo)

    # Un margen mínimo sin datos de margen no puede aplicarse: preferimos fallar a ignorarlo.
    if reglas.get("margen_minimo") is not None:
        raise NotImplementedError(
            "margen_minimo está configurado pero el dataset no tiene márgenes; "
            "integre una fuente de margen antes de activarlo."
        )
    return reglas


def aplicar_reglas(
    candidatos: pd.DataFrame,
    reglas: dict,
    catalogo: pd.DataFrame,
    soporte_por_producto: dict[int, int] | None = None,
    productos_con_cupon: set[int] | None = None,
    historial: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa candidatos en aceptados y filtrados.

    Parámetros:
        candidatos: recomendaciones con household_key, PRODUCT_ID, posicion y tipo_recomendacion.
        reglas: configuración leída con `cargar_reglas`.
        catalogo: product.csv (PRODUCT_ID, DEPARTMENT, ...).
        soporte_por_producto: canastas de entrenamiento que contienen cada producto.
        productos_con_cupon: productos que alguna vez tuvieron cupón (coupon.csv).
        historial: compras previas (household_key, PRODUCT_ID) para detectar variantes.

    Devuelve (aceptados, filtrados). Los filtrados incluyen `regla` y
    `motivo_filtro`; si un producto incumple varias reglas se registra la primera.
    Los aceptados se renumeran de 1 en adelante, conservando el orden del modelo.
    """

    # Departamento y categoría del catálogo, sin depender de que el candidato ya los traiga.
    catalogo_indexado = catalogo.set_index("PRODUCT_ID")
    datos = candidatos.copy()
    datos["_departamento"] = datos["PRODUCT_ID"].map(catalogo_indexado["DEPARTMENT"])
    datos["_categoria"] = datos["PRODUCT_ID"].map(catalogo_indexado["COMMODITY_DESC"]) if "COMMODITY_DESC" in catalogo_indexado else pd.NA

    # Cada candidato empieza sin motivo de exclusión.
    datos["regla"] = pd.NA
    datos["motivo_filtro"] = pd.NA

    def marcar(mascara: pd.Series, regla: str, motivo) -> None:
        """Registra la regla en los candidatos que la incumplen y aún no fueron filtrados."""

        pendientes = mascara & datos["regla"].isna()
        datos.loc[pendientes, "regla"] = regla
        if isinstance(motivo, pd.Series):
            datos.loc[pendientes, "motivo_filtro"] = motivo[pendientes]
        else:
            datos.loc[pendientes, "motivo_filtro"] = motivo

    # 1. Duplicados: el mismo producto dos veces para el mismo hogar.
    if reglas.get("evitar_duplicados", True):
        marcar(
            datos.duplicated(["household_key", "PRODUCT_ID"], keep="first"),
            "evitar_duplicados",
            "Producto repetido en la lista del hogar.",
        )

    # Clave de variante: misma subcategoría, marca y tamaño = misma oferta para el cliente.
    columnas_variante = ["SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT"]
    tiene_variantes = all(columna in catalogo_indexado.columns for columna in columnas_variante)
    clave = (
        catalogo_indexado[columnas_variante].fillna("").astype(str).agg("|".join, axis=1)
        if tiene_variantes else None
    )

    # 1b. Variantes dentro de la lista: se conserva la de mejor posición.
    if reglas.get("evitar_variantes_en_lista", False) and clave is not None:
        con_clave = datos.assign(_clave=datos["PRODUCT_ID"].map(clave)).sort_values("posicion", kind="stable")
        repetida = con_clave.duplicated(["household_key", "tipo_recomendacion", "_clave"], keep="first")
        repetida &= con_clave["_clave"].notna()
        marcar(
            repetida.reindex(datos.index),
            "evitar_variantes_en_lista",
            "Otra variante del mismo producto ya está más arriba en la lista.",
        )

    # 2. Catálogo: el producto debe existir y tener departamento.
    if reglas.get("exigir_catalogo", True):
        sin_catalogo = datos["_departamento"].isna() | (datos["_departamento"].astype(str).str.strip() == "")
        marcar(sin_catalogo, "exigir_catalogo", "Producto sin catálogo o sin departamento.")

    # 3. Departamentos excluidos, cada uno con su motivo.
    excluidos = reglas.get("departamentos_excluidos", {})
    if excluidos:
        motivos = datos["_departamento"].map(excluidos)
        marcar(motivos.notna(), "departamentos_excluidos", motivos)

    # 3b. Categorías excluidas (p. ej. tabaco), cada una con su motivo.
    categorias = reglas.get("categorias_excluidas", {})
    if categorias:
        motivos = datos["_categoria"].map(categorias)
        marcar(motivos.notna(), "categorias_excluidas", motivos)

    # 4. Stock: integración con inventario (vacía en este dataset).
    sin_stock = set(reglas.get("productos_sin_stock", []))
    if sin_stock:
        marcar(datos["PRODUCT_ID"].isin(sin_stock), "productos_sin_stock", "Producto sin stock.")

    # 5. Soporte mínimo para explore: evitar sugerir productos con muy poca evidencia.
    minimo = reglas.get("soporte_minimo_canastas_explore")
    if minimo and soporte_por_producto is not None:
        es_explore = datos["tipo_recomendacion"].astype(str).str.startswith("explore")
        soporte = datos["PRODUCT_ID"].map(soporte_por_producto).fillna(0)
        marcar(
            es_explore & (soporte < minimo),
            "soporte_minimo_canastas_explore",
            f"Producto nuevo con menos de {minimo} canastas de evidencia.",
        )

    # 5b. Variantes: un explore con la misma subcategoría, marca y tamaño que algo
    # ya comprado no es un producto nuevo para el hogar, aunque cambie el código.
    if reglas.get("evitar_variantes_en_explore", False):
        if historial is None:
            raise ValueError("La regla de variantes requiere el historial de compras.")
        if clave is None:
            raise ValueError("La regla de variantes requiere subcategoría, marca y tamaño en el catálogo.")
        compradas = (
            historial[["household_key", "PRODUCT_ID"]].drop_duplicates()
            .assign(_clave=lambda d: d["PRODUCT_ID"].map(clave))[["household_key", "_clave"]]
            .drop_duplicates()
            .assign(_ya_comprada=True)
        )
        con_clave = datos[["household_key"]].assign(_clave=datos["PRODUCT_ID"].map(clave))
        ya_comprada = con_clave.merge(compradas, on=["household_key", "_clave"], how="left")["_ya_comprada"]
        ya_comprada = ya_comprada.fillna(False).astype(bool).to_numpy()
        es_explore = datos["tipo_recomendacion"].astype(str).str.startswith("explore")
        marcar(
            es_explore & ya_comprada,
            "evitar_variantes_en_explore",
            "Variante de un producto que el hogar ya compra (misma subcategoría, marca y tamaño).",
        )

    # 6. Campañas: limitar a productos que alguna vez tuvieron cupón.
    if reglas.get("solo_productos_con_cupon_historico", False):
        if productos_con_cupon is None:
            raise ValueError("La regla de cupón histórico requiere la lista de productos con cupón.")
        marcar(
            ~datos["PRODUCT_ID"].isin(productos_con_cupon),
            "solo_productos_con_cupon_historico",
            "Producto sin cupones en campañas anteriores.",
        )

    # Separamos y renumeramos posiciones de los aceptados por hogar y tipo.
    datos = datos.drop(columns=["_departamento", "_categoria"])
    filtrados = datos.loc[datos["regla"].notna()].reset_index(drop=True)
    aceptados = datos.loc[datos["regla"].isna()].drop(columns=["regla", "motivo_filtro"])
    aceptados = aceptados.sort_values(["household_key", "tipo_recomendacion", "posicion"])
    aceptados["posicion"] = aceptados.groupby(["household_key", "tipo_recomendacion"]).cumcount() + 1
    return aceptados.reset_index(drop=True), filtrados


def medir_efecto(
    repeat: pd.DataFrame,
    explore: pd.DataFrame,
    reglas: dict,
    catalogo: pd.DataFrame,
    soporte_por_producto: dict[int, int],
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    configuracion: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara la lista final antes y después de las reglas en validación."""

    k = configuracion["k"]
    proporcion = configuracion["mezcla"]["proporcion_repeat"]

    # Aplicamos reglas a cada lista candidata por separado.
    repeat_ok, repeat_fuera = aplicar_reglas(repeat, reglas, catalogo, soporte_por_producto, historial=entrenamiento)
    explore_ok, explore_fuera = aplicar_reglas(explore, reglas, catalogo, soporte_por_producto, historial=entrenamiento)
    filtrados = pd.concat([repeat_fuera, explore_fuera], ignore_index=True)

    relevantes_todos = obtener_productos_relevantes(validacion, entrenamiento, "todos")
    relevantes_explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")
    hogares = set(validacion["household_key"])
    universo = set(entrenamiento["PRODUCT_ID"])
    departamentos = catalogo[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"]]

    # Objetivos elegibles: compras de validación que las reglas permitirían ofrecer.
    validacion_elegible, _ = aplicar_reglas(
        validacion.assign(posicion=1, tipo_recomendacion="objetivo"), reglas, catalogo, historial=entrenamiento
    )
    elegibles_todos = obtener_productos_relevantes(validacion_elegible, entrenamiento, "todos")
    elegibles_explore = obtener_productos_relevantes(validacion_elegible, entrenamiento, "explore")

    filas = []
    for etapa, lista_repeat, lista_explore, objetivo_todos, objetivo_explore in (
        ("sin_reglas", repeat, explore, relevantes_todos, relevantes_explore),
        ("con_reglas", repeat_ok, explore_ok, relevantes_todos, relevantes_explore),
        ("con_reglas_objetivo_elegible", repeat_ok, explore_ok, elegibles_todos, elegibles_explore),
    ):
        lista = combinar_listas(lista_repeat, lista_explore, k, proporcion)
        lista = lista.merge(departamentos, on="PRODUCT_ID", how="left")
        total = calcular_metricas_en_k(lista, objetivo_todos, k)
        descubrimiento = calcular_metricas_en_k(lista.loc[lista["origen"] == "explore"], objetivo_explore, k)
        sistema = calcular_metricas_de_sistema(lista, k, hogares, universo)
        excluidos = set(reglas.get("departamentos_excluidos", {}))
        categorias = set(reglas.get("categorias_excluidas", {}))
        no_elegible = lista["DEPARTMENT"].isin(excluidos) | lista["COMMODITY_DESC"].isin(categorias)
        filas.append(
            {
                "etapa": etapa,
                "hit_rate": total["hit_rate"],
                "ndcg": total["ndcg"],
                "precision": total["precision"],
                "recall": total["recall"],
                "hit_rate_explore": descubrimiento["hit_rate"],
                "cobertura_hogares": sistema["cobertura_hogares"],
                "productos_distintos": sistema["productos_distintos_recomendados"],
                "departamentos_por_lista": sistema["departamentos_distintos_por_lista"],
                "posiciones_no_elegibles": int(no_elegible.sum()),
                "hogares_con_lista_incompleta": int((lista.groupby("household_key").size() < k).sum()),
                "posiciones_explore": int((lista["origen"] == "explore").sum()),
            }
        )

    return pd.DataFrame(filas), filtrados


def escribir_reporte(ruta: Path, efecto: pd.DataFrame, filtrados: pd.DataFrame) -> None:
    """Escribe un resumen Markdown del efecto de las reglas."""

    resumen = (
        filtrados.groupby(["tipo_recomendacion", "regla", "motivo_filtro"], dropna=False)
        .size().reset_index(name="candidatos_filtrados")
        .sort_values("candidatos_filtrados", ascending=False)
    )

    lineas = [
        "# Efecto de las reglas de negocio (validación)",
        "",
        "Las reglas se aplican sobre las listas candidatas (20 repeat y 20 explore por hogar)",
        "antes de armar la lista final de 10. No cambian el modelo; miden el costo de",
        "volver elegibles las recomendaciones. Configuración: `config/reglas_negocio.json`.",
        "",
        "La tercera columna mide contra las compras elegibles: se quitan del objetivo",
        "las compras que las reglas nunca permitirían ofrecer (p. ej. nafta). Separa el",
        "costo de la regla de la calidad del modelo sobre productos ofrecibles.",
        "",
        "| Métrica (K = 10) | Sin reglas | Con reglas | Con reglas, objetivo elegible |",
        "| --- | ---: | ---: | ---: |",
    ]
    for columna in efecto.columns.drop("etapa"):
        valores = " | ".join(str(valor) for valor in efecto[columna])
        lineas.append(f"| {columna} | {valores} |")
    lineas += [
        "",
        "## Candidatos filtrados por regla",
        "",
        "| Tipo | Regla | Motivo | Candidatos |",
        "| --- | --- | --- | ---: |",
    ]
    for fila in resumen.itertuples(index=False):
        lineas.append(f"| {fila.tipo_recomendacion} | {fila.regla} | {fila.motivo_filtro} | {fila.candidatos_filtrados} |")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución."""

    analizador = argparse.ArgumentParser(description="Mide el efecto de las reglas de negocio en validación.")
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)
    analizador.add_argument("--resultados-dir", type=Path, default=RAIZ_DEL_PROYECTO / "outputs")
    analizador.add_argument("--configuracion", type=Path, default=RUTA_CONFIGURACION_POR_DEFECTO)
    analizador.add_argument("--reglas", type=Path, default=RUTA_REGLAS_POR_DEFECTO)
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)
    return analizador.parse_args()


def main() -> None:
    """Aplica las reglas a las listas de validación y reporta su efecto."""

    argumentos = interpretar_argumentos()
    reglas = cargar_reglas(argumentos.reglas)
    with argumentos.configuracion.open(encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    # Listas candidatas de validación de los modelos elegidos.
    repeat = pd.read_csv(ruta_de_recomendaciones(configuracion["repeat"]["modelo"], argumentos.resultados_dir))
    explore = pd.read_csv(ruta_de_recomendaciones(configuracion["explore"]["modelo"], argumentos.resultados_dir))

    # Evidencia por producto: canastas de entrenamiento que lo contienen.
    popularidad = pd.read_csv(argumentos.resultados_dir / "baseline" / "popularidad_entrenamiento.csv")
    soporte = popularidad.set_index("PRODUCT_ID")["canastas_con_producto"].to_dict()

    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")
    catalogo = cargar_productos(argumentos.data_dir)

    efecto, filtrados = medir_efecto(
        repeat, explore, reglas, catalogo, soporte, entrenamiento, validacion, configuracion
    )

    argumentos.output_dir.mkdir(parents=True, exist_ok=True)
    efecto.to_csv(argumentos.output_dir / "efecto_reglas_validacion.csv", index=False)
    filtrados.to_csv(argumentos.output_dir / "filtradas_validacion.csv", index=False)
    escribir_reporte(argumentos.output_dir / "efecto_reglas_validacion.md", efecto, filtrados)
    print((argumentos.output_dir / "efecto_reglas_validacion.md").read_text(encoding="utf-8"))


# Ejecutamos solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
