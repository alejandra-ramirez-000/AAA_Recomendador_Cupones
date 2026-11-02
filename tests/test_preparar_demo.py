"""Pruebas de las explicaciones en texto que muestra el prototipo."""

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# pandas construye filas y descripciones de ejemplo.
import pandas as pd

# Importamos la función que redacta las explicaciones.
from src.preparar_demo import explicar


DESCRIPCIONES = pd.Series({1: "Leche · Private · 1 GA", 2: "Leche · Private · 1 GA", 3: "Galletas · National"})


class PruebasDeExplicaciones(unittest.TestCase):
    """Cada recomendación debe tener una frase correcta según su origen."""

    def test_repeat_singular_y_ultima_compra(self) -> None:
        fila = pd.Series({"origen": "repeat", "frecuencia_en_canastas": 1, "distancia_desde_ultima_compra": 0})
        self.assertEqual(explicar(fila, DESCRIPCIONES), "Lo compró en 1 canasta; la última vez en su última compra.")

    def test_repeat_plural_y_distancia(self) -> None:
        fila = pd.Series({"origen": "repeat", "frecuencia_en_canastas": 5, "distancia_desde_ultima_compra": 3})
        self.assertEqual(explicar(fila, DESCRIPCIONES), "Lo compró en 5 canastas; la última vez hace 3 compras.")

    def test_explore_cita_el_producto_relacionado(self) -> None:
        fila = pd.Series({"origen": "explore", "PRODUCT_ID": 3, "producto_que_lo_explica": 1, "similitud_explicativa": 0.8})
        texto = explicar(fila, DESCRIPCIONES)
        self.assertIn("«Leche · Private · 1 GA»", texto)
        self.assertIn("0.80", texto)

    def test_explore_por_popularidad(self) -> None:
        fila = pd.Series({"origen": "explore", "PRODUCT_ID": 3, "producto_que_lo_explica": pd.NA,
                          "similitud_explicativa": pd.NA, "canastas_con_producto": 1200})
        self.assertIn("1200 canastas", explicar(fila, DESCRIPCIONES))

    def test_explore_variante_se_explica_como_variante(self) -> None:
        fila = pd.Series({"origen": "explore", "PRODUCT_ID": 2, "producto_que_lo_explica": 1, "similitud_explicativa": 0.99})
        self.assertIn("otra variante", explicar(fila, DESCRIPCIONES))


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
