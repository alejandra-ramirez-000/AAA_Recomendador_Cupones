"""Genera el análisis exploratorio reproducible de compras y productos.

El script produce tablas CSV y gráficos PNG dentro de outputs/eda. Todas las
medidas se calculan desde las fuentes originales y no eliminan registros. El
tratamiento final de cantidades iguales a cero se decidirá en una fase posterior.

Ejemplo de uso desde la raíz del proyecto:
    python -m src.analizar_datos
"""

# argparse permite seleccionar carpetas sin cambiar el código fuente.
import argparse

# json permite guardar indicadores generales en un archivo portable y legible.
import json

# Path facilita construir rutas relativas a la raíz del proyecto.
from pathlib import Path

# matplotlib genera gráficos estáticos que pueden incluirse en el informe.
import matplotlib.pyplot as plt

# pandas agrupa, une y exporta las tablas del análisis.
import pandas as pd

# Reutilizamos la carga centralizada para no duplicar validaciones ni rutas.
from src.cargar_datos import (
    DIRECTORIO_DE_DATOS_POR_DEFECTO,
    RAIZ_DEL_PROYECTO,
    cargar_productos,
    cargar_transacciones,
)


# Esta ruta reúne todos los resultados de la primera exploración.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "eda"


def construir_tabla_de_canastas(transacciones: pd.DataFrame) -> pd.DataFrame:
    """Resume cada canasta para estudiar la frecuencia de compra por hogar.

    Args:
        transacciones: Tabla original de líneas de compra.

    Returns:
        Una fila por canasta con hogar, día, semana y cantidad de líneas.
    """

    # Definimos las columnas que deben ser iguales dentro de una misma canasta.
    columnas_de_canasta = ["household_key", "BASKET_ID", "DAY", "WEEK_NO"]

    # Agrupamos las líneas de producto para convertirlas en una unidad de compra.
    canastas = (
        transacciones.groupby(columnas_de_canasta, as_index=False)
        .agg(
            # Contamos cuántas líneas de producto contiene cada canasta.
            lineas_de_producto=("PRODUCT_ID", "size"),
            # Sumamos las unidades para describir el volumen registrado.
            unidades_registradas=("QUANTITY", "sum"),
            # Sumamos el importe para una referencia descriptiva de la canasta.
            valor_de_venta=("SALES_VALUE", "sum"),
        )
    )

    # Devolvemos la tabla resumida para reutilizarla en varios indicadores.
    return canastas


def calcular_tablas_eda(
    transacciones: pd.DataFrame,
    productos: pd.DataFrame,
) -> tuple[dict[str, int | float], dict[str, pd.DataFrame]]:
    """Calcula indicadores y tablas necesarias para el EDA inicial.

    Args:
        transacciones: Todas las líneas históricas de compra disponibles.
        productos: Catálogo descriptivo de productos.

    Returns:
        Una tupla con indicadores generales y tablas para guardar o graficar.
    """

    # Creamos una tabla de una fila por canasta antes de contar compras por hogar.
    canastas = construir_tabla_de_canastas(transacciones)

    # Contamos el número de canastas distintas que tuvo cada hogar en el periodo.
    canastas_por_hogar = (
        canastas.groupby("household_key", as_index=False)
        .agg(cantidad_de_canastas=("BASKET_ID", "nunique"))
        .sort_values("cantidad_de_canastas", ascending=False)
    )

    # Contamos canastas por semana para observar la incorporación temporal de hogares.
    canastas_por_semana = (
        canastas.groupby("WEEK_NO", as_index=False)
        .agg(cantidad_de_canastas=("BASKET_ID", "nunique"))
        .sort_values("WEEK_NO")
    )

    # Unimos el catálogo para interpretar departamentos y categorías de las compras.
    transacciones_con_catalogo = transacciones.merge(
        productos,
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )

    # Conservamos las unidades positivas para describir productos efectivamente adquiridos.
    compras_con_cantidad_positiva = transacciones_con_catalogo[
        transacciones_con_catalogo["QUANTITY"] > 0
    ].copy()

    # Agrupamos las unidades compradas por departamento y las ordenamos de mayor a menor.
    departamentos_mas_comprados = (
        compras_con_cantidad_positiva.groupby("DEPARTMENT", as_index=False)
        .agg(
            unidades_compradas=("QUANTITY", "sum"),
            lineas_de_compra=("PRODUCT_ID", "size"),
        )
        .sort_values("unidades_compradas", ascending=False)
    )

    # Agrupamos las unidades compradas por categoría para conocer mayor detalle.
    categorias_mas_compradas = (
        compras_con_cantidad_positiva.groupby("COMMODITY_DESC", as_index=False)
        .agg(
            unidades_compradas=("QUANTITY", "sum"),
            lineas_de_compra=("PRODUCT_ID", "size"),
        )
        .sort_values("unidades_compradas", ascending=False)
    )

    # Sumamos unidades por producto para obtener los candidatos populares globales.
    productos_mas_comprados = (
        compras_con_cantidad_positiva.groupby("PRODUCT_ID", as_index=False)
        .agg(
            unidades_compradas=("QUANTITY", "sum"),
            lineas_de_compra=("PRODUCT_ID", "size"),
        )
        .merge(
            productos[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC"]],
            on="PRODUCT_ID",
            how="left",
            validate="one_to_one",
        )
        .sort_values("unidades_compradas", ascending=False)
    )

    # Aislamos cantidades cero para investigar su distribución antes de aplicar filtros.
    cantidades_cero_por_semana = (
        transacciones.loc[transacciones["QUANTITY"] == 0]
        .groupby("WEEK_NO", as_index=False)
        .agg(filas_con_cantidad_cero=("QUANTITY", "size"))
        .sort_values("WEEK_NO")
    )

    # Reunimos las cifras que describen la dimensión y la disponibilidad de historiales.
    indicadores = {
        "hogares": int(transacciones["household_key"].nunique()),
        "canastas": int(canastas["BASKET_ID"].nunique()),
        "productos_comprados": int(transacciones["PRODUCT_ID"].nunique()),
        "dias_observados": int(transacciones["DAY"].nunique()),
        "semanas_observadas": int(transacciones["WEEK_NO"].nunique()),
        "hogares_con_al_menos_3_canastas": int(
            (canastas_por_hogar["cantidad_de_canastas"] >= 3).sum()
        ),
        "porcentaje_hogares_con_al_menos_3_canastas": round(
            100
            * (canastas_por_hogar["cantidad_de_canastas"] >= 3).mean(),
            2,
        ),
        "filas_con_cantidad_cero": int((transacciones["QUANTITY"] == 0).sum()),
    }

    # Nombramos cada tabla para poder exportarla en un bucle sencillo y explicable.
    tablas = {
        "canastas_por_hogar": canastas_por_hogar,
        "canastas_por_semana": canastas_por_semana,
        "departamentos_mas_comprados": departamentos_mas_comprados,
        "categorias_mas_compradas": categorias_mas_compradas,
        "productos_mas_comprados": productos_mas_comprados,
        "cantidades_cero_por_semana": cantidades_cero_por_semana,
    }

    # Devolvemos indicadores y tablas para que la exportación quede separada del cálculo.
    return indicadores, tablas


def guardar_tablas(
    indicadores: dict[str, int | float],
    tablas: dict[str, pd.DataFrame],
    directorio_de_salida: Path,
) -> None:
    """Guarda los resultados tabulares del EDA en formatos fáciles de revisar."""

    # Creamos la carpeta y sus padres si todavía no existen.
    directorio_de_salida.mkdir(parents=True, exist_ok=True)

    # Guardamos los indicadores generales con sangría para facilitar su lectura.
    with (directorio_de_salida / "indicadores_generales.json").open(
        "w", encoding="utf-8"
    ) as archivo_de_indicadores:
        json.dump(indicadores, archivo_de_indicadores, indent=2, ensure_ascii=False)

    # Recorremos cada tabla con un nombre que será también su nombre de archivo.
    for nombre_de_tabla, tabla in tablas.items():
        # Exportamos sin índice técnico de pandas para mantener un CSV limpio.
        tabla.to_csv(
            directorio_de_salida / f"{nombre_de_tabla}.csv",
            index=False,
            encoding="utf-8",
        )


def guardar_graficos(
    tablas: dict[str, pd.DataFrame], directorio_de_salida: Path
) -> None:
    """Genera los gráficos principales del EDA con títulos y ejes explícitos."""

    # Seleccionamos la distribución de canastas por hogar para el primer gráfico.
    canastas_por_hogar = tablas["canastas_por_hogar"]

    # Creamos la figura con un tamaño apropiado para el informe y la presentación.
    figura, eje = plt.subplots(figsize=(10, 6))

    # Dibujamos el histograma que muestra la cantidad de canastas por hogar.
    eje.hist(canastas_por_hogar["cantidad_de_canastas"], bins=40, color="#0072B2")

    # Etiquetamos el gráfico para que pueda interpretarse fuera del código.
    eje.set_title("Distribución de canastas por hogar")
    eje.set_xlabel("Cantidad de canastas")
    eje.set_ylabel("Cantidad de hogares")

    # Ajustamos márgenes para evitar que las etiquetas queden cortadas.
    figura.tight_layout()

    # Guardamos el gráfico en alta resolución para incluirlo en documentos.
    figura.savefig(directorio_de_salida / "canastas_por_hogar.png", dpi=150)

    # Cerramos la figura para liberar memoria antes de crear el siguiente gráfico.
    plt.close(figura)

    # Seleccionamos la evolución semanal de canastas para observar la rampa inicial.
    canastas_por_semana = tablas["canastas_por_semana"]

    # Creamos una nueva figura independiente para la serie temporal.
    figura, eje = plt.subplots(figsize=(10, 6))

    # Dibujamos cada semana como una línea para mostrar la evolución temporal.
    eje.plot(
        canastas_por_semana["WEEK_NO"],
        canastas_por_semana["cantidad_de_canastas"],
        color="#009E73",
    )

    # Etiquetamos el gráfico para que sea autoexplicativo en la entrega.
    eje.set_title("Cantidad de canastas registradas por semana")
    eje.set_xlabel("Semana")
    eje.set_ylabel("Cantidad de canastas")

    # Ajustamos el espacio antes de guardar el segundo gráfico.
    figura.tight_layout()

    # Guardamos el gráfico temporal con un nombre claro.
    figura.savefig(directorio_de_salida / "canastas_por_semana.png", dpi=150)

    # Cerramos la segunda figura para liberar sus recursos.
    plt.close(figura)

    # Elegimos los diez departamentos con mayor cantidad de unidades positivas.
    departamentos = tablas["departamentos_mas_comprados"].head(10).sort_values(
        "unidades_compradas"
    )

    # Creamos la figura horizontal para que los nombres largos sean legibles.
    figura, eje = plt.subplots(figsize=(10, 6))

    # Dibujamos barras horizontales para comparar los departamentos principales.
    eje.barh(
        departamentos["DEPARTMENT"],
        departamentos["unidades_compradas"],
        color="#D55E00",
    )

    # Usamos escala logarítmica porque KIOSK-GAS tiene un volumen muy superior.
    eje.set_xscale("log")

    # Indicamos que se consideran unidades positivas y que la escala es logarítmica.
    eje.set_title("Diez departamentos con más unidades positivas (escala log)")
    eje.set_xlabel("Unidades compradas")
    eje.set_ylabel("Departamento")

    # Ajustamos la figura antes de escribirla a disco.
    figura.tight_layout()

    # Guardamos la comparación de departamentos.
    figura.savefig(directorio_de_salida / "departamentos_mas_comprados.png", dpi=150)

    # Cerramos la última figura creada.
    plt.close(figura)


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de terminal de este análisis exploratorio."""

    # Creamos el analizador que documenta el propósito del comando.
    analizador = argparse.ArgumentParser(
        description="Genera las tablas y gráficos iniciales del proyecto AAA."
    )

    # Permitimos una ubicación alternativa de datos para pruebas futuras.
    analizador.add_argument(
        "--data-dir",
        type=Path,
        default=DIRECTORIO_DE_DATOS_POR_DEFECTO,
        help="Carpeta que contiene los CSV originales.",
    )

    # Permitimos cambiar la carpeta de resultados sin tocar el código fuente.
    analizador.add_argument(
        "--output-dir",
        type=Path,
        default=DIRECTORIO_DE_SALIDA_POR_DEFECTO,
        help="Carpeta en la que se guardarán tablas y gráficos.",
    )

    # Devolvemos los argumentos que argparse ya validó según sus tipos definidos.
    return analizador.parse_args()


def main() -> None:
    """Ejecuta el EDA completo y guarda resultados reproducibles."""

    # Leemos las rutas indicadas al ejecutar el módulo.
    argumentos = interpretar_argumentos()

    # Cargamos todas las líneas para que las tablas representen el dataset completo.
    transacciones = cargar_transacciones(argumentos.data_dir)

    # Cargamos el catálogo para incorporar departamentos y categorías al análisis.
    productos = cargar_productos(argumentos.data_dir)

    # Calculamos los indicadores y tablas antes de generar archivos o gráficos.
    indicadores, tablas = calcular_tablas_eda(transacciones, productos)

    # Guardamos los resultados tabulares que respaldan los gráficos del informe.
    guardar_tablas(indicadores, tablas, argumentos.output_dir)

    # Generamos visualizaciones que ayudan a discutir el comportamiento de compra.
    guardar_graficos(tablas, argumentos.output_dir)

    # Mostramos los indicadores principales para registrar la ejecución en terminal.
    print("\nIndicadores generales del EDA")
    for nombre, valor in indicadores.items():
        print(f"- {nombre}: {valor}")

    # Indicamos dónde quedaron los archivos para que el equipo pueda revisarlos.
    print(f"\nResultados guardados en: {argumentos.output_dir}")


# Ejecutamos el análisis solo cuando este archivo se invoca como un programa.
if __name__ == "__main__":
    main()
