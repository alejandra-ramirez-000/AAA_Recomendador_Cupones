"""Precalcula todo lo que muestra el prototipo, para que la demo no entrene en vivo.

Simula el uso real: el sistema congelado (D-006) se entrena con **todo** el
historial disponible (entrenamiento + validación + prueba) y recomienda para la
próxima compra de cada hogar. Por eso estas listas no tienen métricas: no existe
una canasta futura contra la cual medirlas. Las métricas que muestra la app
son las de validación y prueba ya reportadas.

Pasos:
1. reentrena repeat (frecuencia + recencia) y explore (ALS congelado);
2. aplica las reglas de negocio (D-008) a 20 + 20 candidatos por hogar;
3. arma la lista final 7 + 3 con una explicación en texto por producto;
4. resume el historial de cada hogar y elige hogares de ejemplo.

Salidas en outputs/demo/.

Ejemplo de uso:
    python -m src.preparar_demo
"""

# argparse permite cambiar rutas sin modificar el código.
import argparse

# json guarda metadatos y hogares de ejemplo.
import json

# datetime registra cuándo se generó la demo.
from datetime import datetime

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# pandas organiza todas las tablas de la demo.
import pandas as pd

# Piezas validadas en etapas anteriores.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos
from src.evaluar_modelos import DIRECTORIO_DE_DIVISION_POR_DEFECTO, RAIZ_DEL_PROYECTO
from src.evaluar_prueba import LARGO_DE_LISTAS_PARCIALES, generar_candidatos
from src.recomendar_baseline import construir_popularidad
from src.reglas_negocio import RUTA_REGLAS_POR_DEFECTO, aplicar_reglas, cargar_reglas
from src.seleccionar_modelo import RUTA_CONFIGURACION_POR_DEFECTO, combinar_listas


# Carpeta que lee la app.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "demo"

# Productos frecuentes que se muestran en el resumen de cada hogar.
CANTIDAD_PRODUCTOS_FRECUENTES = 15


def describir_producto(catalogo: pd.DataFrame) -> pd.Series:
    """Texto corto y legible por producto (el dataset no trae nombres comerciales)."""

    tamano = catalogo["CURR_SIZE_OF_PRODUCT"].fillna("").astype(str).str.strip()
    descripcion = (
        catalogo["SUB_COMMODITY_DESC"].fillna("").str.strip().str.capitalize()
        + " · " + catalogo["BRAND"].fillna("").str.strip()
        + tamano.where(tamano == "", " · " + tamano)
    )
    return pd.Series(descripcion.to_numpy(), index=catalogo["PRODUCT_ID"])


def explicar(fila: pd.Series, descripciones: pd.Series) -> str:
    """Frase que justifica una recomendación según su origen."""

    if fila["origen"] == "repeat":
        frecuencia = int(fila["frecuencia_en_canastas"])
        distancia = int(fila["distancia_desde_ultima_compra"])
        if distancia == 0:
            cuando = "en su última compra"
        elif distancia == 1:
            cuando = "en su penúltima compra"
        else:
            cuando = f"hace {distancia} compras"
        canastas = "1 canasta" if frecuencia == 1 else f"{frecuencia} canastas"
        return f"Lo compró en {canastas}; la última vez {cuando}."
    if pd.isna(fila.get("producto_que_lo_explica")):
        return (
            f"Nuevo para el hogar. Es uno de los productos más comprados: "
            f"aparece en {int(fila['canastas_con_producto'])} canastas."
        )
    similar = descripciones.get(fila["producto_que_lo_explica"], "un producto de su historial")
    if similar == descripciones.get(fila["PRODUCT_ID"]):
        return (
            f"Nuevo para el hogar: es otra variante de «{similar}», que ya compra "
            f"(similitud {fila['similitud_explicativa']:.2f})."
        )
    return (
        f"Nuevo para el hogar. Los hogares que compran «{similar}», como este, "
        f"también lo eligen (similitud {fila['similitud_explicativa']:.2f})."
    )


def generar_listas(
    historial: pd.DataFrame, configuracion: dict, reglas: dict, catalogo: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entrena el sistema congelado, aplica reglas y arma la lista final explicada.

    Devuelve (lista_final, filtrados, respaldo_popular).
    """

    # Repeat y explore congelados, entrenados con todo el historial.
    repeat, explore, _ = generar_candidatos(historial, configuracion, LARGO_DE_LISTAS_PARCIALES)
    popularidad = construir_popularidad(historial)

    # Reglas de negocio con soporte calculado sobre el mismo historial.
    soporte = popularidad.set_index("PRODUCT_ID")["canastas_con_producto"].to_dict()
    repeat_ok, repeat_fuera = aplicar_reglas(repeat, reglas, catalogo, soporte, historial=historial)
    explore_ok, explore_fuera = aplicar_reglas(explore, reglas, catalogo, soporte, historial=historial)
    filtrados = pd.concat([repeat_fuera, explore_fuera], ignore_index=True)

    # Lista final 7 + 3 con sus señales explicativas.
    lista = combinar_listas(repeat_ok, explore_ok, configuracion["k"], configuracion["mezcla"]["proporcion_repeat"])
    lista = lista.merge(
        repeat_ok[["household_key", "PRODUCT_ID", "puntaje_repeat", "frecuencia_en_canastas", "distancia_desde_ultima_compra"]],
        on=["household_key", "PRODUCT_ID"], how="left",
    ).merge(
        explore_ok[["household_key", "PRODUCT_ID", "puntaje_explore", "producto_que_lo_explica",
                    "similitud_explicativa", "canastas_con_producto"]],
        on=["household_key", "PRODUCT_ID"], how="left",
    )
    lista["puntaje"] = lista["puntaje_repeat"].fillna(lista["puntaje_explore"])

    # Respaldo para hogares desconocidos: los más populares que pasan las reglas.
    candidatos_populares = popularidad.head(200).assign(
        household_key=-1, posicion=lambda d: range(1, len(d) + 1), tipo_recomendacion="explore_popularidad"
    )
    respaldo, _ = aplicar_reglas(candidatos_populares, reglas, catalogo, soporte, historial=historial.iloc[0:0])
    # Sin historial conviene variedad: como máximo un producto por subcategoría.
    subcategoria = respaldo["PRODUCT_ID"].map(catalogo.set_index("PRODUCT_ID")["SUB_COMMODITY_DESC"])
    respaldo = respaldo.loc[~subcategoria.duplicated()].reset_index(drop=True)
    respaldo["posicion"] = range(1, len(respaldo) + 1)
    respaldo = respaldo.head(configuracion["k"])[["PRODUCT_ID", "posicion", "canastas_con_producto", "hogares_que_compraron"]]

    return lista, filtrados, respaldo


def resumir_historial(historial: pd.DataFrame, catalogo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resumen por hogar, productos más frecuentes y última canasta."""

    canastas = historial.groupby(["household_key", "BASKET_ID"], as_index=False).agg(
        DAY=("DAY", "first"), orden=("orden_de_canasta", "first"), gasto=("SALES_VALUE", "sum")
    )
    resumen = canastas.groupby("household_key").agg(
        canastas=("BASKET_ID", "nunique"),
        primer_dia=("DAY", "min"),
        ultimo_dia=("DAY", "max"),
        gasto_total=("gasto", "sum"),
        gasto_medio_por_canasta=("gasto", "mean"),
    )
    productos_distintos = historial.groupby("household_key")["PRODUCT_ID"].nunique().rename("productos_distintos")
    con_departamento = historial.merge(catalogo[["PRODUCT_ID", "DEPARTMENT"]], on="PRODUCT_ID", how="left")
    departamento_principal = (
        con_departamento.groupby(["household_key", "DEPARTMENT"]).size().reset_index(name="lineas")
        .sort_values(["household_key", "lineas"], ascending=[True, False])
        .groupby("household_key").head(3)
        .groupby("household_key")["DEPARTMENT"].agg(", ".join).rename("departamentos_principales")
    )
    resumen = resumen.join(productos_distintos).join(departamento_principal).reset_index()
    resumen["gasto_total"] = resumen["gasto_total"].round(2)
    resumen["gasto_medio_por_canasta"] = resumen["gasto_medio_por_canasta"].round(2)

    frecuentes = (
        historial.groupby(["household_key", "PRODUCT_ID"], as_index=False)
        .agg(canastas=("BASKET_ID", "nunique"), ultimo_dia=("DAY", "max"))
        .sort_values(["household_key", "canastas", "ultimo_dia"], ascending=[True, False, False])
        .groupby("household_key").head(CANTIDAD_PRODUCTOS_FRECUENTES)
    )

    ultima = canastas.sort_values("orden").groupby("household_key").tail(1)[["household_key", "BASKET_ID", "DAY"]]
    ultima_canasta = historial.merge(ultima[["BASKET_ID"]], on="BASKET_ID")[
        ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE"]
    ]
    return resumen, frecuentes, ultima_canasta


def elegir_hogares_de_ejemplo(resumen: pd.DataFrame, filtrados: pd.DataFrame, lista: pd.DataFrame) -> list[dict]:
    """Elige hogares deterministas que ilustran casos distintos en la demo."""

    ordenado = resumen.sort_values(["canastas", "household_key"]).reset_index(drop=True)
    largo = ordenado.iloc[-1]
    corto = ordenado.iloc[0]
    tipico = ordenado.iloc[len(ordenado) // 2]

    ejemplos = [
        {"household_key": int(tipico["household_key"]), "caso": f"Historial típico ({int(tipico['canastas'])} canastas)"},
        {"household_key": int(largo["household_key"]), "caso": f"Historial más largo ({int(largo['canastas'])} canastas)"},
        {"household_key": int(corto["household_key"]), "caso": f"Historial más corto ({int(corto['canastas'])} canastas)"},
    ]

    # Los dos casos siguientes solo existen si hay datos que los ilustren.
    excluidos = filtrados.loc[filtrados["regla"] == "departamentos_excluidos"].groupby("household_key").size()
    if not excluidos.empty:
        hogar = int(excluidos.sort_values(ascending=False, kind="stable").index[0])
        ejemplos.append({"household_key": hogar, "caso": "Muchos candidatos filtrados por reglas"})

    explore = lista.loc[lista["origen"] == "explore"].groupby("household_key")["similitud_explicativa"].max().dropna()
    if not explore.empty:
        hogar = int(explore.sort_values(ascending=False, kind="stable").index[0])
        ejemplos.append({"household_key": hogar, "caso": "Sugerencia explore con explicación muy clara"})

    return ejemplos


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución."""

    analizador = argparse.ArgumentParser(description="Precalcula los datos del prototipo.")
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)
    analizador.add_argument("--configuracion", type=Path, default=RUTA_CONFIGURACION_POR_DEFECTO)
    analizador.add_argument("--reglas", type=Path, default=RUTA_REGLAS_POR_DEFECTO)
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)
    analizador.add_argument("--resultados-dir", type=Path, default=RAIZ_DEL_PROYECTO / "outputs")
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)
    return analizador.parse_args()


def main() -> None:
    """Genera todos los archivos que consume app/app.py."""

    argumentos = interpretar_argumentos()
    with argumentos.configuracion.open(encoding="utf-8") as archivo:
        configuracion = json.load(archivo)
    reglas = cargar_reglas(argumentos.reglas)
    catalogo = cargar_productos(argumentos.data_dir)
    descripciones = describir_producto(catalogo)

    # Todo el historial: la demo recomienda para la próxima compra.
    historial = pd.concat(
        [pd.read_csv(argumentos.division_dir / f"interacciones_{nombre}.csv") for nombre in ("entrenamiento", "validacion", "prueba")],
        ignore_index=True,
    )

    lista, filtrados, respaldo = generar_listas(historial, configuracion, reglas, catalogo)
    lista["explicacion"] = lista.apply(explicar, axis=1, descripciones=descripciones)
    resumen, frecuentes, ultima_canasta = resumir_historial(historial, catalogo)

    # Agregamos descripciones y catálogo a todas las tablas de productos.
    columnas_catalogo = ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT"]
    def con_catalogo(tabla: pd.DataFrame) -> pd.DataFrame:
        tabla = tabla.merge(catalogo[columnas_catalogo], on="PRODUCT_ID", how="left")
        tabla["producto"] = tabla["PRODUCT_ID"].map(descripciones)
        return tabla

    lista = con_catalogo(lista)
    filtrados = con_catalogo(filtrados[["household_key", "PRODUCT_ID", "tipo_recomendacion", "posicion", "regla", "motivo_filtro"]])
    respaldo = con_catalogo(respaldo)
    frecuentes = con_catalogo(frecuentes)
    ultima_canasta = con_catalogo(ultima_canasta)

    ejemplos = elegir_hogares_de_ejemplo(resumen, filtrados, lista)

    # Métricas ya reportadas, para la pestaña "Sobre el modelo".
    with (argumentos.resultados_dir / "prueba" / "metricas_prueba.json").open(encoding="utf-8") as archivo:
        metricas_prueba = json.load(archivo)["metricas"]
    metadatos = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "historial": "entrenamiento + validación + prueba (todas las compras disponibles)",
        "configuracion": configuracion,
        "reglas": reglas,
        "metricas_prueba": metricas_prueba,
        "hogares": int(resumen.shape[0]),
    }

    salida = argumentos.output_dir
    salida.mkdir(parents=True, exist_ok=True)
    lista.to_csv(salida / "lista_final.csv", index=False)
    filtrados.to_csv(salida / "filtrados.csv", index=False)
    respaldo.to_csv(salida / "respaldo_popular.csv", index=False)
    resumen.to_csv(salida / "resumen_hogares.csv", index=False)
    frecuentes.to_csv(salida / "productos_frecuentes.csv", index=False)
    ultima_canasta.to_csv(salida / "ultima_canasta.csv", index=False)
    (salida / "hogares_ejemplo.json").write_text(json.dumps(ejemplos, indent=2, ensure_ascii=False), encoding="utf-8")
    (salida / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"Demo generada en {salida}: {lista['household_key'].nunique()} hogares, {len(filtrados)} candidatos filtrados.")
    print("Hogares de ejemplo:")
    for ejemplo in ejemplos:
        print(f"- {ejemplo['household_key']}: {ejemplo['caso']}")


# Ejecutamos solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
