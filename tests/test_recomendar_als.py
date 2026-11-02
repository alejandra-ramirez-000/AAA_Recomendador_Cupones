"""Pruebas de la implementación propia de ALS implícito."""

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# numpy y scipy construyen matrices pequeñas de prueba.
import numpy as np
import pandas as pd
from scipy import sparse

# Importamos las piezas de ALS que se quieren verificar.
from src.recomendar_als import (
    actualizar_factores,
    construir_matriz_de_confianza,
    entrenar_als,
    generar_recomendaciones_als,
)


class PruebasDeAls(unittest.TestCase):
    """Comprueba fórmulas, confianza y separación repeat/explore de ALS."""

    def test_gradiente_conjugado_coincide_con_solucion_exacta(self) -> None:
        """Con suficientes pasos, CG debe resolver (Y^T C_u Y + lambda I) x = Y^T C_u p."""

        generador = np.random.default_rng(0)
        matriz = sparse.random(6, 9, density=0.4, format="csr", random_state=1)
        matriz.data = 1.0 + 3.0 * matriz.data
        factores_fijos = generador.normal(size=(9, 4))
        regularizacion = 0.1

        # Muchos pasos de CG en un sistema 4x4 alcanzan la solución exacta.
        aproximada = actualizar_factores(matriz, np.zeros((6, 4)), factores_fijos, regularizacion, pasos=20)

        # Solución exacta fila por fila con la fórmula de Hu, Koren y Volinsky.
        for fila in range(6):
            confianza = matriz[fila].toarray().ravel()
            preferencia = (confianza > 0).astype(float)
            pesos = np.where(confianza > 0, confianza, 1.0)
            a = factores_fijos.T @ (pesos[:, None] * factores_fijos) + regularizacion * np.eye(4)
            b = factores_fijos.T @ (pesos * preferencia)
            np.testing.assert_allclose(aproximada[fila], np.linalg.solve(a, b), atol=1e-8)

    def test_confianza_usa_canastas_y_logaritmo(self) -> None:
        """La confianza debe ser 1 + alpha * log(1 + canastas distintas)."""

        entrenamiento = pd.DataFrame(
            [
                {"household_key": 7, "BASKET_ID": 1, "PRODUCT_ID": 100},
                {"household_key": 7, "BASKET_ID": 2, "PRODUCT_ID": 100},
                {"household_key": 7, "BASKET_ID": 2, "PRODUCT_ID": 200},
            ]
        )
        matriz, hogares, productos, _ = construir_matriz_de_confianza(entrenamiento, alpha=2.0)
        self.assertEqual(list(hogares), [7])
        self.assertEqual(list(productos), [100, 200])
        np.testing.assert_allclose(matriz.toarray()[0], [1 + 2 * np.log(3), 1 + 2 * np.log(2)])

    def test_als_aprende_grupos_y_separa_repeat_explore(self) -> None:
        """Hogares con gustos parecidos deben recibir en explore el producto que les falta."""

        # Grupo A compra 0-3, grupo B compra 4-7. Al hogar 0 le falta el producto 3.
        compras = {0: [0, 1, 2], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3], 3: [4, 5, 6, 7], 4: [4, 5, 6, 7]}
        filas = [{"household_key": h, "BASKET_ID": h * 10 + p, "PRODUCT_ID": p} for h, ps in compras.items() for p in ps]
        entrenamiento = pd.DataFrame(filas)

        matriz, hogares, productos, _ = construir_matriz_de_confianza(entrenamiento, alpha=10.0)
        x, y = entrenar_als(matriz, factores=2, regularizacion=0.01, iteraciones=20, semilla=0)
        repeat, explore = generar_recomendaciones_als(
            matriz, hogares, productos, x, y, cantidad_recomendaciones=2, con_explicacion=True
        )

        # Repeat solo contiene productos ya comprados; explore ninguno.
        for hogar, comprados in compras.items():
            self.assertTrue(set(repeat.loc[repeat["household_key"] == hogar, "PRODUCT_ID"]) <= set(comprados))
            self.assertFalse(set(explore.loc[explore["household_key"] == hogar, "PRODUCT_ID"]) & set(comprados))

        # El primer explore del hogar 0 debe ser el producto 3, explicado por un producto del grupo A.
        primero = explore.loc[(explore["household_key"] == 0) & (explore["posicion"] == 1)].iloc[0]
        self.assertEqual(primero["PRODUCT_ID"], 3)
        self.assertIn(primero["producto_que_lo_explica"], [0, 1, 2])


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
