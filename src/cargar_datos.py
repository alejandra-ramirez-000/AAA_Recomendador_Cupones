"""Carga y valida las fuentes principales del recomendador.

Este módulo no modifica los CSV originales. Su objetivo es centralizar las
validaciones iniciales para que los notebooks y modelos posteriores utilicen
las mismas columnas, tipos y mensajes de error.

Ejemplo de uso desde la raíz del proyecto:
    python -m src.cargar_datos --filas-muestra 100000

Para revisar todos los registros de las dos fuentes centrales:
    python -m src.cargar_datos --validar-completo
"""

# argparse permite recibir opciones desde la terminal sin editar el código.
import argparse

# Path permite trabajar con rutas de forma compatible y legible.
from pathlib import Path

# Any describe el contenido heterogéneo del resumen de calidad.
from typing import Any

# pandas se utiliza para leer y analizar las tablas CSV.
import pandas as pd


# Esta ruta apunta a la raíz del proyecto a partir de la ubicación de este archivo.
RAIZ_DEL_PROYECTO = Path(__file__).resolve().parents[1]

# Esta es la ubicación por defecto de los datos originales, que nunca se modifican.
DIRECTORIO_DE_DATOS_POR_DEFECTO = RAIZ_DEL_PROYECTO / "data"

# Estas columnas identifican una compra y son imprescindibles para el recomendador.
COLUMNAS_ESENCIALES_TRANSACCIONES = [
    "household_key",
    "BASKET_ID",
    "DAY",
    "PRODUCT_ID",
    "QUANTITY",
]

# Estas son todas las columnas disponibles en la fuente de transacciones actual.
COLUMNAS_TRANSACCIONES = [
    "household_key",
    "BASKET_ID",
    "DAY",
    "PRODUCT_ID",
    "QUANTITY",
    "SALES_VALUE",
    "STORE_ID",
    "RETAIL_DISC",
    "TRANS_TIME",
    "WEEK_NO",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
]

# Estas columnas describen cada producto y permiten explicar recomendaciones.
COLUMNAS_PRODUCTOS = [
    "PRODUCT_ID",
    "MANUFACTURER",
    "DEPARTMENT",
    "BRAND",
    "COMMODITY_DESC",
    "SUB_COMMODITY_DESC",
    "CURR_SIZE_OF_PRODUCT",
]


def comprobar_archivo(ruta_del_archivo: Path) -> None:
    """Confirma que una fuente esperada existe antes de intentar leerla.

    Args:
        ruta_del_archivo: Ubicación completa del CSV que se desea utilizar.

    Raises:
        FileNotFoundError: Si la ruta no existe o no apunta a un archivo.
    """

    # is_file evita que un directorio con un nombre parecido se acepte por error.
    if not ruta_del_archivo.is_file():
        # El mensaje muestra la ruta exacta que la persona debe revisar.
        raise FileNotFoundError(
            f"No se encontró el archivo esperado: {ruta_del_archivo}"
        )


def validar_columnas_requeridas(
    tabla: pd.DataFrame,
    columnas_requeridas: list[str],
    nombre_de_la_tabla: str,
) -> None:
    """Verifica que una tabla posea todas las columnas necesarias.

    Args:
        tabla: DataFrame que se acaba de cargar desde un CSV.
        columnas_requeridas: Nombres de columna que deben estar presentes.
        nombre_de_la_tabla: Nombre claro de la fuente para el mensaje de error.

    Raises:
        ValueError: Si falta al menos una columna requerida.
    """

    # Convertimos los nombres disponibles a un conjunto para buscar rápidamente.
    columnas_disponibles = set(tabla.columns)

    # Conservamos el orden original de las requeridas para que el mensaje sea claro.
    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in columnas_disponibles
    ]

    # Solo detenemos el proceso si hay información indispensable ausente.
    if columnas_faltantes:
        # Unimos las columnas faltantes para que sea fácil corregir la fuente.
        detalle_faltantes = ", ".join(columnas_faltantes)

        # Informamos qué tabla tiene el problema y qué columnas faltan.
        raise ValueError(
            f"La tabla '{nombre_de_la_tabla}' no contiene: {detalle_faltantes}."
        )


def cargar_transacciones(
    directorio_de_datos: Path = DIRECTORIO_DE_DATOS_POR_DEFECTO,
    filas_a_leer: int | None = None,
) -> pd.DataFrame:
    """Carga las transacciones con los tipos adecuados para el proyecto.

    Args:
        directorio_de_datos: Carpeta que contiene transaction_data.csv.
        filas_a_leer: Límite opcional para trabajar con una muestra reproducible.
            Si es None, se leen todas las transacciones.

    Returns:
        Las transacciones originales en un DataFrame de pandas.
    """

    # Construimos la ruta sin depender de desde qué carpeta se ejecutó Python.
    ruta_transacciones = directorio_de_datos / "transaction_data.csv"

    # Verificamos la existencia antes de que pandas muestre un error menos claro.
    comprobar_archivo(ruta_transacciones)

    # Leemos solo las columnas conocidas para detectar cambios inesperados en el CSV.
    transacciones = pd.read_csv(
        ruta_transacciones,
        usecols=COLUMNAS_TRANSACCIONES,
        nrows=filas_a_leer,
    )

    # Confirmamos que las variables esenciales para el historial estén disponibles.
    validar_columnas_requeridas(
        transacciones,
        COLUMNAS_ESENCIALES_TRANSACCIONES,
        "transaction_data.csv",
    )

    # Devolvemos los datos sin ordenarlos ni filtrarlos para conservar la fuente original.
    return transacciones


def cargar_productos(
    directorio_de_datos: Path = DIRECTORIO_DE_DATOS_POR_DEFECTO,
) -> pd.DataFrame:
    """Carga el catálogo de productos utilizado para describir recomendaciones.

    Args:
        directorio_de_datos: Carpeta que contiene product.csv.

    Returns:
        El catálogo de productos en un DataFrame de pandas.
    """

    # Construimos la ruta de la fuente de productos desde el directorio recibido.
    ruta_productos = directorio_de_datos / "product.csv"

    # Verificamos que la fuente exista antes de solicitarla a pandas.
    comprobar_archivo(ruta_productos)

    # Leemos el catálogo completo porque su tamaño permite cargarlo en memoria.
    productos = pd.read_csv(ruta_productos, usecols=COLUMNAS_PRODUCTOS)

    # Confirmamos que las variables que describen productos estén presentes.
    validar_columnas_requeridas(productos, COLUMNAS_PRODUCTOS, "product.csv")

    # Devolvemos el catálogo sin alterarlo para que otras fases decidan sus filtros.
    return productos


def calcular_resumen_de_calidad(
    transacciones: pd.DataFrame,
    productos: pd.DataFrame,
) -> dict[str, Any]:
    """Calcula controles de calidad iniciales sin eliminar ningún registro.

    Los conteos permiten tomar decisiones documentadas más adelante. Un conteo
    distinto de cero no implica automáticamente que se deba borrar una fila.

    Args:
        transacciones: Compras cargadas desde transaction_data.csv.
        productos: Catálogo cargado desde product.csv.

    Returns:
        Diccionario con tamaños, nulos, duplicados y valores a investigar.
    """

    # Reutilizamos la validación para evitar resultados engañosos en tablas incompletas.
    validar_columnas_requeridas(
        transacciones,
        COLUMNAS_ESENCIALES_TRANSACCIONES,
        "transacciones",
    )

    # Verificamos la clave que permite relacionar compras y catálogo de productos.
    validar_columnas_requeridas(productos, ["PRODUCT_ID"], "productos")

    # Identificamos los productos vendidos que no aparecen en el catálogo disponible.
    productos_sin_catalogo = (
        ~transacciones["PRODUCT_ID"].isin(productos["PRODUCT_ID"])
    ).sum()

    # Reunimos indicadores simples en una estructura fácil de guardar o imprimir.
    resumen = {
        "filas_transacciones": len(transacciones),
        "filas_productos": len(productos),
        "hogares_unicos": transacciones["household_key"].nunique(),
        "canastas_unicas": transacciones["BASKET_ID"].nunique(),
        "productos_comprados_unicos": transacciones["PRODUCT_ID"].nunique(),
        "productos_catalogo_unicos": productos["PRODUCT_ID"].nunique(),
        "nulos_en_transacciones": transacciones.isna().sum().to_dict(),
        "nulos_en_productos": productos.isna().sum().to_dict(),
        "filas_duplicadas_exactas": int(transacciones.duplicated().sum()),
        "productos_sin_catalogo": int(productos_sin_catalogo),
        "cantidades_no_positivas": int((transacciones["QUANTITY"] <= 0).sum()),
        "dias_negativos": int((transacciones["DAY"] < 0).sum()),
    }

    # Entregamos los indicadores sin imprimir para facilitar pruebas automáticas.
    return resumen


def imprimir_resumen_de_calidad(resumen: dict[str, Any]) -> None:
    """Muestra el resumen de calidad en un formato legible en la terminal."""

    # Definimos los indicadores principales en un orden didáctico y estable.
    indicadores_principales = [
        "filas_transacciones",
        "filas_productos",
        "hogares_unicos",
        "canastas_unicas",
        "productos_comprados_unicos",
        "productos_catalogo_unicos",
        "filas_duplicadas_exactas",
        "productos_sin_catalogo",
        "cantidades_no_positivas",
        "dias_negativos",
    ]

    # Mostramos un título que separa este reporte de otros mensajes de ejecución.
    print("\nResumen inicial de calidad de datos")

    # Imprimimos cada indicador principal con una etiqueta fácil de reconocer.
    for nombre_indicador in indicadores_principales:
        print(f"- {nombre_indicador}: {resumen[nombre_indicador]}")

    # Mostramos los nulos por columna para decidir después cómo tratarlos.
    print("- nulos_en_transacciones:", resumen["nulos_en_transacciones"])

    # Mostramos también los nulos de catálogo, relevantes para explicaciones futuras.
    print("- nulos_en_productos:", resumen["nulos_en_productos"])


def interpretar_argumentos() -> argparse.Namespace:
    """Define y lee las opciones disponibles al ejecutar este módulo."""

    # Creamos el analizador que documenta el comando de ejecución.
    analizador = argparse.ArgumentParser(
        description="Carga y valida las fuentes centrales del proyecto AAA."
    )

    # Permitimos elegir otra carpeta para facilitar pruebas con datos pequeños.
    analizador.add_argument(
        "--data-dir",
        type=Path,
        default=DIRECTORIO_DE_DATOS_POR_DEFECTO,
        help="Carpeta que contiene transaction_data.csv y product.csv.",
    )

    # Usamos una muestra por defecto para que la primera ejecución sea ágil.
    analizador.add_argument(
        "--filas-muestra",
        type=int,
        default=100_000,
        help="Cantidad de transacciones a revisar; se ignora con --validar-completo.",
    )

    # Esta opción permite revisar los 2,6 millones de filas cuando sea necesario.
    analizador.add_argument(
        "--validar-completo",
        action="store_true",
        help="Lee todas las transacciones en lugar de una muestra.",
    )

    # Devolvemos las opciones ya validadas por argparse.
    return analizador.parse_args()


def main() -> None:
    """Ejecuta la carga y el reporte inicial desde la terminal."""

    # Leemos las opciones que la persona indicó al invocar el módulo.
    argumentos = interpretar_argumentos()

    # Elegimos None para indicar a pandas que debe leer todas las filas disponibles.
    filas_a_leer = None if argumentos.validar_completo else argumentos.filas_muestra

    # Cargamos transacciones de forma segura, completa o en muestra según la opción.
    transacciones = cargar_transacciones(argumentos.data_dir, filas_a_leer)

    # Cargamos el catálogo que complementa la interpretación de los productos.
    productos = cargar_productos(argumentos.data_dir)

    # Calculamos los controles sin alterar los datos originales.
    resumen = calcular_resumen_de_calidad(transacciones, productos)

    # Presentamos los resultados para registrarlos en la bitácora del proyecto.
    imprimir_resumen_de_calidad(resumen)


# Ejecutamos main solo cuando este archivo se invoca como programa y no al importarlo.
if __name__ == "__main__":
    main()
