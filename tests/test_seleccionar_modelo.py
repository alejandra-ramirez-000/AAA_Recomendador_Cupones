"""Pruebas de la combinación de listas repeat/explore y la selección de modelos."""

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# pandas construye listas de recomendación pequeñas.
import pandas as pd

# Importamos las funciones de selección que se quieren verificar.
from src.seleccionar_modelo import combinar_listas, elegir_modelo_por_tipo


def lista(hogar: int, productos: list[int]) -> pd.DataFrame:
    """Crea una lista ordenada de recomendaciones para un hogar."""

    return pd.DataFrame(
        [{"household_key": hogar, "PRODUCT_ID": p, "posicion": i} for i, p in enumerate(productos, start=1)]
    )


class PruebasDeSeleccion(unittest.TestCase):
    """Comprueba cupos, relleno, duplicados y elección por tipo."""

    def test_respeta_cupos_y_orden(self) -> None:
        """Con 60 % repeat y K = 5: 3 repeat primero y luego 2 explore."""

        combinada = combinar_listas(lista(1, [1, 2, 3, 4]), lista(1, [10, 20, 30]), k=5, proporcion_repeat=0.6)
        self.assertEqual(list(combinada["PRODUCT_ID"]), [1, 2, 3, 10, 20])
        self.assertEqual(list(combinada["origen"]), ["repeat"] * 3 + ["explore"] * 2)
        self.assertEqual(list(combinada["posicion"]), [1, 2, 3, 4, 5])

    def test_completa_con_explore_si_falta_historial(self) -> None:
        """Un hogar con un solo producto repeat recibe más explore para llegar a K."""

        combinada = combinar_listas(lista(1, [1]), lista(1, [10, 20, 30, 40]), k=4, proporcion_repeat=0.5)
        self.assertEqual(list(combinada["PRODUCT_ID"]), [1, 10, 20, 30])

    def test_completa_con_repeat_si_falta_explore(self) -> None:
        """Si explore no alcanza su cupo, se completa con más repeat."""

        combinada = combinar_listas(lista(1, [1, 2, 3, 4]), lista(1, [10]), k=4, proporcion_repeat=0.5)
        self.assertEqual(sorted(combinada["PRODUCT_ID"]), [1, 2, 3, 10])

    def test_hogar_sin_historial_recibe_solo_explore(self) -> None:
        """Un hogar ausente de repeat obtiene igualmente su lista explore."""

        combinada = combinar_listas(lista(1, [1, 2]), lista(2, [10, 20]), k=2, proporcion_repeat=0.5)
        self.assertEqual(list(combinada.loc[combinada["household_key"] == 2, "PRODUCT_ID"]), [10, 20])

    def test_elige_baseline_si_ningun_modelo_lo_supera(self) -> None:
        """La elección debe quedarse con el baseline cuando tiene mayor NDCG."""

        comparacion = pd.DataFrame(
            [
                {"modelo": "baseline_x", "tipo": "repeat", "k": 10, "ndcg": 0.3, "hit_rate": 0.6},
                {"modelo": "als_repeat", "tipo": "repeat", "k": 10, "ndcg": 0.1, "hit_rate": 0.4},
            ]
        )
        eleccion = elegir_modelo_por_tipo(comparacion, "repeat")
        self.assertEqual(eleccion["modelo"], "baseline_x")
        self.assertEqual(eleccion["mejora_relativa_ndcg"], 0.0)


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
