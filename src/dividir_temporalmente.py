"""Construye divisiones temporales por hogar sin usar información futura.

Para cada hogar con al menos tres canastas de compras positivas, las canastas
antiguas forman entrenamiento, la penúltima es validación y la última es prueba.
La regla se aplica por hogar para evitar que su fecha de incorporación al dataset
sesgue una división global.

Ejemplo de uso:
    python -m src.dividir_temporalmente
"""

# argparse permite definir carpetas de entrada y salida desde la terminal.
import argparse

# Path facilita construir rutas de manera clara y segura.
from pathlib import Path

# pandas se usa para ordenar canastas, crear particiones y exportar CSV.
import pandas as pd

# Reutilizamos las constantes y el cargador centralizado del proyecto.
from src.cargar_datos import (
    COLUMNAS_ESENCIALES_TRANSACCIONES,
    DIRECTORIO_DE_DATOS_POR_DEFECTO,
    RAIZ_DEL_PROYECTO,
    cargar_transacciones,
    validar_columnas_requeridas,
)


# Esta ruta contiene únicamente resultados derivados que se pueden regenerar.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "division_temporal"


def conservar_interacciones_de_compra(
    transacciones: pd.DataFrame,
) -> pd.DataFrame:
    """Conserva líneas con cantidad positiva para representar compras reales.

    Args:
        transacciones: Líneas originales de transaction_data.csv.

    Returns:
        Copia de las líneas con QUANTITY mayor que cero.
    """

    # Comprobamos que las claves del historial estén disponibles antes de filtrar.
    validar_columnas_requeridas(
        transacciones,
        COLUMNAS_ESENCIALES_TRANSACCIONES,
        "transacciones",
    )

    # Una interacción implícita de compra necesita al menos una unidad adquirida.
    interacciones = transacciones.loc[transacciones["QUANTITY"] > 0].copy()

    # Devolvemos una copia para que la fuente original nunca sea modificada.
    return interacciones


def construir_orden_de_canastas(interacciones: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla de una fila por canasta y su posición por hogar.

    Args:
        interacciones: Líneas con cantidad positiva.

    Returns:
        Canastas ordenadas con una posición temporal por hogar.
    """

    # Agrupamos las líneas de productos para obtener una fila por canasta válida.
    canastas = (
        interacciones.groupby(
            ["household_key", "BASKET_ID"], as_index=False
        )
        .agg(
            # DAY es el momento principal utilizado para ordenar el historial.
            DAY=("DAY", "min"),
            # TRANS_TIME rompe empates dentro de un mismo día cuando está disponible.
            TRANS_TIME=("TRANS_TIME", "min"),
            # WEEK_NO queda disponible para análisis y auditoría posterior.
            WEEK_NO=("WEEK_NO", "min"),
        )
    )

    # Ordenamos de forma determinista por hogar, día, hora y clave de canasta.
    canastas = canastas.sort_values(
        ["household_key", "DAY", "TRANS_TIME", "BASKET_ID"]
    ).reset_index(drop=True)

    # Numeramos las canastas desde uno para que la posición sea fácil de explicar.
    canastas["orden_de_canasta"] = canastas.groupby("household_key").cumcount() + 1

    # Contamos cuántas canastas positivas tiene cada hogar antes de dividirlo.
    canastas["cantidad_de_canastas_del_hogar"] = canastas.groupby(
        "household_key"
    )["BASKET_ID"].transform("size")

    # Devolvemos la secuencia temporal que define entrenamiento, validación y prueba.
    return canastas


def etiquetar_particiones(canastas: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta cada canasta como entrenamiento, validación o prueba.

    Los hogares con menos de tres canastas se excluyen porque no permiten formar
    las tres etapas. Para los demás hogares, solo las dos últimas se reservan.
    """

    # Conservamos hogares que tienen historia suficiente para las tres particiones.
    canastas_incluidas = canastas.loc[
        canastas["cantidad_de_canastas_del_hogar"] >= 3
    ].copy()

    # Inicialmente, las canastas antiguas serán parte del entrenamiento.
    canastas_incluidas["particion"] = "entrenamiento"

    # Identificamos la posición de la última canasta disponible para cada hogar.
    ultima_posicion = canastas_incluidas["cantidad_de_canastas_del_hogar"]

    # Marcamos la última canasta de cada hogar como conjunto de prueba.
    canastas_incluidas.loc[
        canastas_incluidas["orden_de_canasta"] == ultima_posicion,
        "particion",
    ] = "prueba"

    # Marcamos la canasta inmediatamente anterior como conjunto de validación.
    canastas_incluidas.loc[
        canastas_incluidas["orden_de_canasta"] == ultima_posicion - 1,
        "particion",
    ] = "validacion"

    # Devolvemos las canastas incluidas con una etiqueta única por cada una.
    return canastas_incluidas


def crear_division_temporal(
    transacciones: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Asigna cada interacción positiva a una partición temporal por hogar.

    Args:
        transacciones: Líneas originales disponibles para el proyecto.

    Returns:
        Un diccionario con DataFrames de entrenamiento, validación y prueba, y
        la tabla de canastas utilizada para auditar la separación.
    """

    # Excluimos cantidades cero según la decisión D-002 de la bitácora.
    interacciones = conservar_interacciones_de_compra(transacciones)

    # Construimos la secuencia ordenada de compras para cada hogar.
    canastas_ordenadas = construir_orden_de_canastas(interacciones)

    # Etiquetamos las canastas con la partición que les corresponde temporalmente.
    canastas_con_particion = etiquetar_particiones(canastas_ordenadas)

    # Incorporamos la etiqueta de cada canasta a todas sus líneas de productos.
    interacciones_con_particion = interacciones.merge(
        canastas_con_particion[
            ["household_key", "BASKET_ID", "orden_de_canasta", "particion"]
        ],
        on=["household_key", "BASKET_ID"],
        how="inner",
        validate="many_to_one",
    )

    # Elegimos columnas suficientes para modelos y auditoría, evitando datos redundantes.
    columnas_de_salida = [
        "household_key",
        "BASKET_ID",
        "DAY",
        "TRANS_TIME",
        "WEEK_NO",
        "PRODUCT_ID",
        "QUANTITY",
        "SALES_VALUE",
        "RETAIL_DISC",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
        "orden_de_canasta",
    ]

    # Separamos las interacciones según su etiqueta y ordenamos para lectura estable.
    divisiones = {
        nombre_particion: (
            interacciones_con_particion.loc[
                interacciones_con_particion["particion"] == nombre_particion,
                columnas_de_salida,
            ]
            .sort_values(["household_key", "orden_de_canasta", "PRODUCT_ID"])
            .reset_index(drop=True)
        )
        for nombre_particion in ("entrenamiento", "validacion", "prueba")
    }

    # Devolvemos tanto las líneas por partición como la auditoría de canastas.
    return divisiones, canastas_con_particion


def validar_division_temporal(
    divisiones: dict[str, pd.DataFrame], canastas: pd.DataFrame
) -> None:
    """Detiene el proceso si encuentra fuga de información temporal.

    Args:
        divisiones: Interacciones separadas por entrenamiento, validación y prueba.
        canastas: Tabla de canastas con posición y partición asignadas.

    Raises:
        ValueError: Si falta una partición, cambia el orden o hay hogares distintos.
    """

    # Verificamos que las tres particiones requeridas estén presentes en el resultado.
    particiones_requeridas = {"entrenamiento", "validacion", "prueba"}
    if set(divisiones) != particiones_requeridas:
        raise ValueError("La división no contiene las tres particiones requeridas.")

    # Obtenemos hogares presentes en cada partición de interacciones.
    hogares_por_particion = {
        nombre: set(tabla["household_key"].unique())
        for nombre, tabla in divisiones.items()
    }

    # Exigimos que cada hogar incluido contribuya a entrenamiento, validación y prueba.
    if len(set(map(frozenset, hogares_por_particion.values()))) != 1:
        raise ValueError("No todos los hogares aparecen en las tres particiones.")

    # Convertimos etiquetas a números para verificar el orden sin depender de fechas iguales.
    orden_esperado = {"entrenamiento": 1, "validacion": 2, "prueba": 3}

    # Copiamos las columnas necesarias para no alterar la auditoría original.
    auditoria = canastas[["household_key", "orden_de_canasta", "particion"]].copy()

    # Asociamos cada etiqueta con el orden que debe tener en el historial.
    auditoria["orden_de_particion"] = auditoria["particion"].map(orden_esperado)

    # Calculamos el mínimo y máximo orden de partición para cada hogar.
    resumen_por_hogar = auditoria.groupby("household_key", as_index=False).agg(
        minimo=("orden_de_particion", "min"),
        maximo=("orden_de_particion", "max"),
    )

    # Confirmamos que todos los hogares tengan las tres etiquetas esperadas.
    if not ((resumen_por_hogar["minimo"] == 1) & (resumen_por_hogar["maximo"] == 3)).all():
        raise ValueError("Hay hogares sin entrenamiento, validación o prueba.")

    # Ordenamos cada hogar por posición de canasta para revisar cambios de etiqueta.
    auditoria = auditoria.sort_values(["household_key", "orden_de_canasta"])

    # Calculamos el salto entre etiquetas consecutivas de una misma secuencia.
    cambios_por_hogar = auditoria.groupby("household_key")["orden_de_particion"].diff()

    # Un salto negativo indicaría que una canasta futura se usó antes que una pasada.
    if (cambios_por_hogar.dropna() < 0).any():
        raise ValueError("Se detectó una fuga temporal en el orden de las particiones.")


def guardar_division(
    divisiones: dict[str, pd.DataFrame],
    canastas: pd.DataFrame,
    directorio_de_salida: Path,
) -> None:
    """Guarda los CSV derivados que usarán los modelos posteriores."""

    # Creamos la carpeta de resultados si no existe todavía.
    directorio_de_salida.mkdir(parents=True, exist_ok=True)

    # Guardamos cada conjunto con un nombre explícito y sin índice de pandas.
    for nombre_particion, tabla in divisiones.items():
        tabla.to_csv(
            directorio_de_salida / f"interacciones_{nombre_particion}.csv",
            index=False,
            encoding="utf-8",
        )

    # Guardamos la secuencia completa de canastas para auditar la división luego.
    canastas.to_csv(
        directorio_de_salida / "canastas_con_particion.csv",
        index=False,
        encoding="utf-8",
    )

    # Contamos canastas y líneas por partición para dejar un resumen compacto.
    resumen = pd.DataFrame(
        [
            {
                "particion": nombre,
                "hogares": tabla["household_key"].nunique(),
                "canastas": tabla["BASKET_ID"].nunique(),
                "lineas_de_producto": len(tabla),
            }
            for nombre, tabla in divisiones.items()
        ]
    )

    # Exportamos el resumen que se citará en el informe y las métricas posteriores.
    resumen.to_csv(
        directorio_de_salida / "resumen_division.csv",
        index=False,
        encoding="utf-8",
    )


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de terminal de la división temporal."""

    # Creamos el analizador que muestra ayuda para ejecutar el módulo correctamente.
    analizador = argparse.ArgumentParser(
        description="Divide transacciones temporalmente por hogar."
    )

    # Permitimos cambiar la carpeta de fuentes para ejecutar pruebas con muestras.
    analizador.add_argument(
        "--data-dir",
        type=Path,
        default=DIRECTORIO_DE_DATOS_POR_DEFECTO,
        help="Carpeta que contiene transaction_data.csv.",
    )

    # Permitimos cambiar la carpeta de derivados sin tocar el código fuente.
    analizador.add_argument(
        "--output-dir",
        type=Path,
        default=DIRECTORIO_DE_SALIDA_POR_DEFECTO,
        help="Carpeta para guardar las tres particiones temporales.",
    )

    # Devolvemos las opciones interpretadas por argparse.
    return analizador.parse_args()


def main() -> None:
    """Ejecuta la separación temporal completa y muestra un resumen."""

    # Leemos los argumentos de la ejecución actual.
    argumentos = interpretar_argumentos()

    # Cargamos todas las transacciones porque la división debe considerar el historial completo.
    transacciones = cargar_transacciones(argumentos.data_dir)

    # Creamos entrenamiento, validación, prueba y su auditoría temporal.
    divisiones, canastas = crear_division_temporal(transacciones)

    # Verificamos que ningún hogar use una canasta futura para entrenar.
    validar_division_temporal(divisiones, canastas)

    # Guardamos los resultados derivados que utilizarán los próximos modelos.
    guardar_division(divisiones, canastas, argumentos.output_dir)

    # Mostramos tamaños claros para que la persona pueda verificar la ejecución.
    print("\nResumen de la división temporal")
    for nombre, tabla in divisiones.items():
        print(
            f"- {nombre}: {tabla['household_key'].nunique()} hogares, "
            f"{tabla['BASKET_ID'].nunique()} canastas y {len(tabla)} líneas."
        )

    # Indicamos dónde quedaron los CSV derivados para la siguiente fase.
    print(f"\nResultados guardados en: {argumentos.output_dir}")


# Ejecutamos el flujo solo cuando este módulo se invoca directamente.
if __name__ == "__main__":
    main()
