"""Pruebas pequeñas para validar las funciones de carga de datos."""

# tempfile crea carpetas temporales que se eliminan al terminar cada prueba.
import tempfile

# Path facilita crear rutas dentro de cada carpeta temporal.
from pathlib import Path

# unittest es la biblioteca estándar para ejecutar estas pruebas sin dependencias extra.
import unittest

# pandas permite construir CSV de ejemplo mínimos y controlados.
import pandas as pd

# Importamos las funciones públicas que se desean verificar.
from src.cargar_datos import (
    calcular_resumen_de_calidad,
    cargar_productos,
    cargar_transacciones,
    validar_columnas_requeridas,
)

# Importamos la división para comprobar que respeta el orden por hogar.
from src.dividir_temporalmente import crear_division_temporal, validar_division_temporal

# Importamos funciones del baseline para validar su comportamiento principal.
from src.evaluar_modelos import calcular_metricas_en_k
from src.recomendar_baseline import (
    construir_perfil_repeat,
    construir_popularidad,
    generar_recomendaciones_explore,
)

# Importamos coocurrencias para comprobar que el soporte mínimo se respeta.
from src.recomendar_coocurrencia import construir_asociaciones


class PruebasDeCargaDeDatos(unittest.TestCase):
    """Comprueba que la carga básica y sus errores sean comprensibles."""

    def crear_fuentes_minimas(self, directorio: Path) -> None:
        """Escribe dos CSV pequeños válidos dentro de una carpeta temporal."""

        # Creamos una transacción con todas las columnas esperadas por el módulo.
        transacciones = pd.DataFrame(
            [{
                "household_key": 1,
                "BASKET_ID": 10,
                "DAY": 1,
                "PRODUCT_ID": 100,
                "QUANTITY": 1,
                "SALES_VALUE": 2.5,
                "STORE_ID": 1,
                "RETAIL_DISC": 0.0,
                "TRANS_TIME": 1200,
                "WEEK_NO": 1,
                "COUPON_DISC": 0.0,
                "COUPON_MATCH_DISC": 0.0,
            }]
        )

        # Creamos un producto con las columnas descriptivas requeridas.
        productos = pd.DataFrame(
            [{
                "PRODUCT_ID": 100,
                "MANUFACTURER": 1,
                "DEPARTMENT": "GROCERY",
                "BRAND": "National",
                "COMMODITY_DESC": "TEST",
                "SUB_COMMODITY_DESC": "TEST ITEM",
                "CURR_SIZE_OF_PRODUCT": "1 EA",
            }]
        )

        # Guardamos la fuente de transacciones con el mismo nombre que usa el proyecto.
        transacciones.to_csv(directorio / "transaction_data.csv", index=False)

        # Guardamos el catálogo mínimo con el nombre esperado por el cargador.
        productos.to_csv(directorio / "product.csv", index=False)

    def test_carga_y_resumen_de_fuentes_validas(self) -> None:
        """La carga debe reconocer una compra, un hogar y un producto válidos."""

        # Creamos una carpeta aislada que se elimina automáticamente al finalizar.
        with tempfile.TemporaryDirectory() as directorio_temporal:
            # Convertimos la ruta temporal en un objeto Path legible.
            ruta_temporal = Path(directorio_temporal)

            # Generamos las fuentes controladas para esta prueba.
            self.crear_fuentes_minimas(ruta_temporal)

            # Cargamos los dos CSV utilizando las funciones que usará el proyecto.
            transacciones = cargar_transacciones(ruta_temporal)
            productos = cargar_productos(ruta_temporal)

            # Calculamos los controles de calidad sobre el conjunto mínimo.
            resumen = calcular_resumen_de_calidad(transacciones, productos)

            # Verificamos que el resumen conserve los conteos esperados.
            self.assertEqual(resumen["filas_transacciones"], 1)
            self.assertEqual(resumen["hogares_unicos"], 1)
            self.assertEqual(resumen["productos_sin_catalogo"], 0)

    def test_columnas_faltantes_generan_un_error_claro(self) -> None:
        """Una fuente incompleta debe indicar qué columna necesaria falta."""

        # Construimos una tabla que no contiene la clave PRODUCT_ID.
        tabla_incompleta = pd.DataFrame({"household_key": [1]})

        # Confirmamos que la validación detenga el flujo con un ValueError.
        with self.assertRaises(ValueError) as contexto_del_error:
            validar_columnas_requeridas(
                tabla_incompleta,
                ["household_key", "PRODUCT_ID"],
                "tabla_de_prueba",
            )

        # Revisamos que el mensaje mencione la columna que requiere corrección.
        self.assertIn("PRODUCT_ID", str(contexto_del_error.exception))

    def test_division_temporal_reserva_las_dos_ultimas_canastas(self) -> None:
        """La división debe excluir ceros y reservar validación y prueba."""

        # Construimos cuatro canastas: tres válidas y una línea de cantidad cero.
        transacciones = pd.DataFrame(
            [
                {"household_key": 1, "BASKET_ID": 10, "DAY": 1, "TRANS_TIME": 900, "WEEK_NO": 1, "PRODUCT_ID": 100, "QUANTITY": 1, "SALES_VALUE": 1.0, "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0},
                {"household_key": 1, "BASKET_ID": 20, "DAY": 2, "TRANS_TIME": 900, "WEEK_NO": 1, "PRODUCT_ID": 200, "QUANTITY": 1, "SALES_VALUE": 1.0, "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0},
                {"household_key": 1, "BASKET_ID": 30, "DAY": 3, "TRANS_TIME": 900, "WEEK_NO": 1, "PRODUCT_ID": 300, "QUANTITY": 1, "SALES_VALUE": 1.0, "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0},
                {"household_key": 1, "BASKET_ID": 40, "DAY": 4, "TRANS_TIME": 900, "WEEK_NO": 1, "PRODUCT_ID": 400, "QUANTITY": 0, "SALES_VALUE": 0.0, "RETAIL_DISC": 0.0, "COUPON_DISC": 0.0, "COUPON_MATCH_DISC": 0.0},
            ]
        )

        # Creamos la partición aplicando la misma función que se usa con el dataset real.
        divisiones, canastas = crear_division_temporal(transacciones)

        # Confirmamos que las validaciones internas no detectan fuga temporal.
        validar_division_temporal(divisiones, canastas)

        # Verificamos que la primera canasta quede en entrenamiento.
        self.assertEqual(divisiones["entrenamiento"]["BASKET_ID"].tolist(), [10])

        # Verificamos que la penúltima canasta quede en validación.
        self.assertEqual(divisiones["validacion"]["BASKET_ID"].tolist(), [20])

        # Verificamos que la última canasta positiva quede en prueba.
        self.assertEqual(divisiones["prueba"]["BASKET_ID"].tolist(), [30])

    def test_explore_no_recomienda_productos_del_historial(self) -> None:
        """Explore debe excluir productos que el hogar ya compró en entrenamiento."""

        # Creamos un entrenamiento pequeño con dos productos populares distintos.
        entrenamiento = pd.DataFrame(
            [
                {"household_key": 1, "BASKET_ID": 1, "PRODUCT_ID": 10, "orden_de_canasta": 1},
                {"household_key": 2, "BASKET_ID": 2, "PRODUCT_ID": 20, "orden_de_canasta": 1},
                {"household_key": 3, "BASKET_ID": 3, "PRODUCT_ID": 20, "orden_de_canasta": 1},
            ]
        )

        # Construimos el ranking global usando exclusivamente esas interacciones.
        popularidad = construir_popularidad(entrenamiento)

        # Pedimos dos recomendaciones nuevas por hogar.
        explore = generar_recomendaciones_explore(entrenamiento, popularidad, 2)

        # Recuperamos las sugerencias del hogar que ya compró el producto 10.
        productos_hogar_uno = set(explore.loc[explore["household_key"] == 1, "PRODUCT_ID"])

        # Confirmamos que el producto histórico no aparezca como una exploración.
        self.assertNotIn(10, productos_hogar_uno)

    def test_metricas_reconocen_un_acierto_en_la_primera_posicion(self) -> None:
        """Las métricas deben otorgar crédito a un producto relevante rankeado primero."""

        # Definimos una lista donde el primer producto recomendado es relevante.
        recomendaciones = pd.DataFrame(
            [
                {"household_key": 1, "PRODUCT_ID": 10, "posicion": 1},
                {"household_key": 1, "PRODUCT_ID": 20, "posicion": 2},
            ]
        )

        # Indicamos que el producto 10 fue la compra real posterior del hogar.
        relevantes = {1: {10}}

        # Evaluamos una lista de dos posiciones para verificar las fórmulas.
        metricas = calcular_metricas_en_k(recomendaciones, relevantes, 2)

        # Precision es un acierto entre dos posiciones recomendadas.
        self.assertEqual(metricas["precision"], 0.5)

        # Recall, Hit Rate y NDCG deben ser perfectos al acertar en primer lugar.
        self.assertEqual(metricas["recall"], 1.0)
        self.assertEqual(metricas["hit_rate"], 1.0)
        self.assertEqual(metricas["ndcg"], 1.0)

    def test_coocurrencia_conserva_pares_con_soporte_suficiente(self) -> None:
        """Las asociaciones deben contar pares que ocurren en varias canastas."""

        # Creamos dos canastas con el mismo par y una tercera con otro producto.
        entrenamiento = pd.DataFrame(
            [
                {"BASKET_ID": 1, "PRODUCT_ID": 10},
                {"BASKET_ID": 1, "PRODUCT_ID": 20},
                {"BASKET_ID": 2, "PRODUCT_ID": 10},
                {"BASKET_ID": 2, "PRODUCT_ID": 20},
                {"BASKET_ID": 3, "PRODUCT_ID": 30},
            ]
        )

        # Pedimos soporte de producto y par de dos canastas para conservar 10-20.
        asociaciones = construir_asociaciones(
            entrenamiento,
            soporte_minimo_producto=2,
            soporte_minimo_par=2,
        )

        # Verificamos que el par repetido se haya contado exactamente dos veces.
        self.assertEqual(len(asociaciones), 1)
        self.assertEqual(asociaciones.iloc[0]["canastas_con_el_par"], 2)


# Permitimos ejecutar el archivo de prueba directamente desde la terminal.
if __name__ == "__main__":
    unittest.main()
