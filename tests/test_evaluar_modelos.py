"""Pruebas de borde del módulo común de evaluación."""

# Path permite crear carpetas temporales de resultados.
from pathlib import Path

# tempfile crea carpetas temporales que se borran al terminar cada prueba.
import tempfile

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# pandas construye pequeños conjuntos de datos sintéticos.
import pandas as pd

# Importamos las funciones del módulo común que se quieren verificar.
from src.evaluar_modelos import (
    calcular_indice_de_gini,
    calcular_metricas_de_sistema,
    calcular_metricas_en_k,
    construir_tabla_comparativa,
    obtener_productos_relevantes,
)


class PruebasDeEvaluacion(unittest.TestCase):
    """Comprueba casos límite de métricas de ranking y de sistema."""

    def test_hogar_sin_relevantes_no_se_evalua(self) -> None:
        """Un hogar sin compras del tipo analizado no debe afectar los promedios."""

        # El hogar 1 acierta en primera posición; el hogar 2 no tiene relevantes.
        recomendaciones = pd.DataFrame(
            [
                {"household_key": 1, "PRODUCT_ID": 10, "posicion": 1},
                {"household_key": 2, "PRODUCT_ID": 20, "posicion": 1},
            ]
        )
        relevantes = {1: {10}, 2: set()}

        # Evaluamos con K = 1.
        metricas = calcular_metricas_en_k(recomendaciones, relevantes, 1)

        # Solo el hogar 1 cuenta, y su resultado es perfecto.
        self.assertEqual(metricas["hogares_evaluables"], 1)
        self.assertEqual(metricas["hit_rate"], 1.0)

    def test_sin_hogares_evaluables_devuelve_ceros(self) -> None:
        """Si ningún hogar tiene relevantes, las métricas son cero y no hay división por cero."""

        recomendaciones = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 10, "posicion": 1}])
        metricas = calcular_metricas_en_k(recomendaciones, {1: set()}, 5)
        self.assertEqual(metricas["hogares_evaluables"], 0)
        self.assertEqual(metricas["precision"], 0.0)

    def test_lista_menor_que_k_penaliza_precision(self) -> None:
        """Una lista con menos de K productos usa igualmente K como denominador."""

        # Un solo producto recomendado, que además es relevante.
        recomendaciones = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 10, "posicion": 1}])
        metricas = calcular_metricas_en_k(recomendaciones, {1: {10}}, 5)

        # Precision@5 = 1/5; Recall, Hit Rate y NDCG siguen siendo perfectos.
        self.assertEqual(metricas["precision"], 0.2)
        self.assertEqual(metricas["recall"], 1.0)
        self.assertEqual(metricas["ndcg"], 1.0)

    def test_hogar_sin_recomendaciones_cuenta_como_fallo(self) -> None:
        """Un hogar evaluable sin lista recomendada recibe cero, no se omite."""

        recomendaciones = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 10, "posicion": 1}])
        metricas = calcular_metricas_en_k(recomendaciones, {1: {10}, 2: {30}}, 1)
        self.assertEqual(metricas["hogares_evaluables"], 2)
        self.assertEqual(metricas["hit_rate"], 0.5)

    def test_acierto_en_segunda_posicion_reduce_ndcg(self) -> None:
        """NDCG debe valorar menos un acierto en la posición 2 que en la 1."""

        recomendaciones = pd.DataFrame(
            [
                {"household_key": 1, "PRODUCT_ID": 99, "posicion": 1},
                {"household_key": 1, "PRODUCT_ID": 10, "posicion": 2},
            ]
        )
        metricas = calcular_metricas_en_k(recomendaciones, {1: {10}}, 2)

        # 1 / log2(3) ≈ 0,63093.
        self.assertAlmostEqual(metricas["ndcg"], 0.63093, places=5)

    def test_separacion_repeat_explore(self) -> None:
        """Repeat y explore deben dividir las compras de validación según el historial."""

        entrenamiento = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 10}])
        validacion = pd.DataFrame(
            [{"household_key": 1, "PRODUCT_ID": 10}, {"household_key": 1, "PRODUCT_ID": 20}]
        )
        self.assertEqual(obtener_productos_relevantes(validacion, entrenamiento, "repeat"), {1: {10}})
        self.assertEqual(obtener_productos_relevantes(validacion, entrenamiento, "explore"), {1: {20}})
        with self.assertRaises(ValueError):
            obtener_productos_relevantes(validacion, entrenamiento, "otro")

    def test_gini_de_exposicion(self) -> None:
        """Gini vale cero si la exposición es pareja y crece si se concentra."""

        self.assertEqual(calcular_indice_de_gini(pd.Series([3, 3, 3])), 0.0)
        self.assertEqual(calcular_indice_de_gini(pd.Series([0, 0])), 0.0)
        self.assertGreater(calcular_indice_de_gini(pd.Series([0, 0, 9])), 0.6)

    def test_metricas_de_sistema(self) -> None:
        """Cobertura, concentración, diversidad y explicación sobre un caso pequeño."""

        # Dos hogares reciben el mismo producto; un tercer hogar elegible no recibe nada.
        # El producto 30 no tiene catálogo (DEPARTMENT ausente) ni explicación.
        recomendaciones = pd.DataFrame(
            [
                {"household_key": 1, "PRODUCT_ID": 10, "posicion": 1, "DEPARTMENT": "A", "senal": 5},
                {"household_key": 1, "PRODUCT_ID": 30, "posicion": 2, "DEPARTMENT": None, "senal": None},
                {"household_key": 2, "PRODUCT_ID": 10, "posicion": 1, "DEPARTMENT": "A", "senal": 5},
            ]
        )
        metricas = calcular_metricas_de_sistema(
            recomendaciones,
            k=2,
            hogares_elegibles={1, 2, 3},
            productos_de_entrenamiento={10, 20, 30, 40},
            columna_explicacion="senal",
        )

        self.assertAlmostEqual(metricas["cobertura_hogares"], 2 / 3, places=5)
        self.assertEqual(metricas["cobertura_productos"], 0.5)
        self.assertEqual(metricas["productos_distintos_recomendados"], 2)
        self.assertEqual(metricas["participacion_top_10_productos"], 1.0)
        self.assertEqual(metricas["departamentos_distintos_por_lista"], 1.0)
        self.assertAlmostEqual(metricas["porcentaje_con_explicacion"], 2 / 3, places=5)

    def test_tabla_comparativa_omite_modelos_no_ejecutados(self) -> None:
        """La tabla debe incluir modelos con resultados y avisar de los faltantes."""

        entrenamiento = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 10}])
        validacion = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 10}])

        with tempfile.TemporaryDirectory() as carpeta_temporal:
            # Creamos resultados de un único modelo ficticio.
            carpeta_modelo = Path(carpeta_temporal) / "modelo_a"
            carpeta_modelo.mkdir()
            pd.DataFrame(
                [{"household_key": 1, "PRODUCT_ID": 10, "posicion": 1, "senal": 1}]
            ).to_csv(carpeta_modelo / "recs.csv", index=False)

            modelos = [
                {"nombre": "a", "tipo": "repeat", "carpeta": "modelo_a", "archivo": "recs.csv",
                 "columna_explicacion": "senal", "clave_tiempos": "tiempos_segundos"},
                {"nombre": "b", "tipo": "explore", "carpeta": "no_existe", "archivo": "recs.csv",
                 "columna_explicacion": "senal", "clave_tiempos": "tiempos_segundos"},
            ]
            tabla = construir_tabla_comparativa(
                entrenamiento, validacion, Path(carpeta_temporal), modelos, valores_de_k=(1,)
            )

        # Solo aparece el modelo existente, con un acierto perfecto.
        self.assertEqual(list(tabla["modelo"]), ["a"])
        self.assertEqual(tabla.loc[0, "hit_rate"], 1.0)


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
