"""Pruebas del análisis exploratorio para la defensa."""

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# pandas construye historiales pequeños.
import pandas as pd

# Funciones del análisis.
from src.analizar_repeticion import (
    calcular_concentracion,
    calcular_repeticion,
    contar_objetivos,
    ficha_no_elegibles,
    segmentar_por_historial,
)


def compras(filas: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    """(hogar, canasta, orden, producto) -> interacciones."""

    return pd.DataFrame(filas, columns=["household_key", "BASKET_ID", "orden_de_canasta", "PRODUCT_ID"])


CATALOGO = pd.DataFrame(
    [
        {"PRODUCT_ID": 1, "DEPARTMENT": "GROCERY", "COMMODITY_DESC": "MILK", "SUB_COMMODITY_DESC": "MILK", "BRAND": "P", "CURR_SIZE_OF_PRODUCT": "1 GA"},
        {"PRODUCT_ID": 2, "DEPARTMENT": "GROCERY", "COMMODITY_DESC": "MILK", "SUB_COMMODITY_DESC": "MILK", "BRAND": "P", "CURR_SIZE_OF_PRODUCT": "1 GA"},
        {"PRODUCT_ID": 3, "DEPARTMENT": "KIOSK-GAS", "COMMODITY_DESC": "GAS", "SUB_COMMODITY_DESC": "GAS", "BRAND": "P", "CURR_SIZE_OF_PRODUCT": ""},
        {"PRODUCT_ID": 4, "DEPARTMENT": "DRUG GM", "COMMODITY_DESC": "CIGARETTES", "SUB_COMMODITY_DESC": "CIG", "BRAND": "N", "CURR_SIZE_OF_PRODUCT": "1 PK"},
    ]
)


class PruebasDeAnalisis(unittest.TestCase):
    """Cada tabla del análisis debe calcularse correctamente en casos pequeños."""

    def test_repeticion_por_codigo_y_por_variante(self) -> None:
        """La canasta 2 repite el producto 1 y compra la variante 2 de la misma leche."""

        canastas = calcular_repeticion(compras([(1, 10, 1, 1), (1, 20, 2, 1), (1, 20, 2, 2), (1, 20, 2, 3)]), CATALOGO)
        segunda = canastas.loc[canastas["orden_de_canasta"] == 2].iloc[0]
        self.assertEqual(segunda["productos"], 3)
        self.assertAlmostEqual(segunda["proporcion_repetida"], 1 / 3)
        self.assertAlmostEqual(segunda["proporcion_repetida_variante"], 2 / 3)
        self.assertEqual(canastas.loc[canastas["orden_de_canasta"] == 1, "proporcion_repetida"].iloc[0], 0)

    def test_objetivos_de_validacion(self) -> None:
        """Hogar 1 recompra y prueba algo nuevo; hogar 2 solo recompra."""

        entrenamiento = compras([(1, 10, 1, 1), (2, 11, 1, 1)])
        validacion = compras([(1, 20, 2, 1), (1, 20, 2, 9), (2, 21, 2, 1)])
        objetivos = contar_objetivos(entrenamiento, validacion)
        self.assertEqual(objetivos["con_ambos"], 1)
        self.assertEqual(objetivos["solo_recompra"], 1)
        self.assertEqual(objetivos["solo_producto_nuevo"], 0)

    def test_segmentos_de_historial(self) -> None:
        """Hogares con 1, 2 y 9 canastas quedan en tres segmentos ordenados."""

        filas = [(1, 100, 1, 1)] + [(2, 200 + i, i, 1) for i in range(1, 3)] + [(3, 300 + i, i, 1) for i in range(1, 10)]
        segmento = segmentar_por_historial(compras(filas))
        self.assertTrue(segmento[1].startswith("corto"))
        self.assertTrue(segmento[3].startswith("largo"))

    def test_concentracion(self) -> None:
        """Un producto en 3 canastas y otro en 1: el más popular reúne 3/4."""

        resumen, curva = calcular_concentracion(compras([(1, 1, 1, 1), (1, 2, 2, 1), (1, 3, 3, 1), (1, 3, 3, 2)]))
        self.assertEqual(resumen["productos_comprados"], 2)
        self.assertEqual(resumen["participacion_top_1_pct"], 0.75)
        self.assertEqual(resumen["productos_comprados_en_una_sola_canasta"], 1)
        self.assertAlmostEqual(curva["fraccion_compras"].iloc[-1], 1.0)

    def test_ficha_de_combustible_y_tabaco(self) -> None:
        """El hogar 1 recompra nafta; el hogar 2 compra tabaco una vez."""

        entrenamiento = compras([(1, 1, 1, 3), (1, 2, 2, 3), (2, 3, 1, 4), (2, 4, 2, 1)])
        lista = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 3, "posicion": 1}, {"household_key": 2, "PRODUCT_ID": 1, "posicion": 1}])
        ficha = ficha_no_elegibles(entrenamiento, CATALOGO, {"categorias_excluidas": {"CIGARETTES": "x"}}, lista).set_index("grupo")
        self.assertEqual(ficha.loc["Combustible (KIOSK-GAS)", "porcentaje_compradores_que_recompran"], 100.0)
        self.assertEqual(ficha.loc["Tabaco", "porcentaje_compradores_que_recompran"], 0.0)
        self.assertEqual(ficha.loc["Combustible (KIOSK-GAS)", "porcentaje_hogares_con_el_grupo_en_su_lista"], 50.0)


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
