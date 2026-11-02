"""Pruebas de las reglas de negocio configurables."""

# Path y tempfile permiten probar la lectura de configuración.
import json
import tempfile
from pathlib import Path

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# pandas construye candidatos y catálogo pequeños.
import pandas as pd

# Importamos las funciones de reglas.
from src.reglas_negocio import aplicar_reglas, cargar_reglas


CATALOGO = pd.DataFrame(
    [
        {"PRODUCT_ID": 1, "DEPARTMENT": "GROCERY"},
        {"PRODUCT_ID": 2, "DEPARTMENT": "KIOSK-GAS"},
        {"PRODUCT_ID": 3, "DEPARTMENT": " "},
        {"PRODUCT_ID": 4, "DEPARTMENT": "PRODUCE"},
        {"PRODUCT_ID": 5, "DEPARTMENT": "PRODUCE"},
    ]
)

REGLAS = {
    "exigir_catalogo": True,
    "evitar_duplicados": True,
    "departamentos_excluidos": {"KIOSK-GAS": "Combustible."},
    "soporte_minimo_canastas_explore": 10,
    "productos_sin_stock": [5],
    "margen_minimo": None,
    "solo_productos_con_cupon_historico": False,
}


def candidatos(tipo: str, productos: list[int]) -> pd.DataFrame:
    """Lista de un hogar con el tipo indicado."""

    return pd.DataFrame(
        [{"household_key": 1, "PRODUCT_ID": p, "posicion": i, "tipo_recomendacion": tipo} for i, p in enumerate(productos, 1)]
    )


class PruebasDeReglas(unittest.TestCase):
    """Cada regla debe filtrar, registrar su motivo y respetar el orden del modelo."""

    def test_cada_regla_registra_su_motivo(self) -> None:
        """Gas, sin departamento, inexistente, sin stock y duplicado quedan filtrados con motivo."""

        lista = candidatos("repeat", [2, 1, 3, 99, 5, 1, 4])
        aceptados, filtrados = aplicar_reglas(lista, REGLAS, CATALOGO, {})

        self.assertEqual(list(aceptados["PRODUCT_ID"]), [1, 4])
        self.assertEqual(list(aceptados["posicion"]), [1, 2])
        motivos = dict(zip(filtrados["PRODUCT_ID"], filtrados["regla"]))
        self.assertEqual(motivos[2], "departamentos_excluidos")
        self.assertEqual(motivos[3], "exigir_catalogo")
        self.assertEqual(motivos[99], "exigir_catalogo")
        self.assertEqual(motivos[5], "productos_sin_stock")
        self.assertIn("evitar_duplicados", set(filtrados["regla"]))
        self.assertEqual(filtrados.loc[filtrados["PRODUCT_ID"] == 2, "motivo_filtro"].iloc[0], "Combustible.")

    def test_soporte_minimo_solo_afecta_explore(self) -> None:
        """Un producto con poca evidencia se filtra en explore pero no en repeat."""

        soporte = {1: 3, 4: 50}
        repeat_ok, _ = aplicar_reglas(candidatos("repeat", [1, 4]), REGLAS, CATALOGO, soporte)
        explore_ok, explore_fuera = aplicar_reglas(candidatos("explore_als", [1, 4]), REGLAS, CATALOGO, soporte)

        self.assertEqual(list(repeat_ok["PRODUCT_ID"]), [1, 4])
        self.assertEqual(list(explore_ok["PRODUCT_ID"]), [4])
        self.assertEqual(explore_fuera["regla"].iloc[0], "soporte_minimo_canastas_explore")

    def test_sin_reglas_activas_no_filtra_nada(self) -> None:
        """Con todas las reglas apagadas la lista queda igual."""

        apagadas = {"exigir_catalogo": False, "evitar_duplicados": False, "departamentos_excluidos": {}}
        lista = candidatos("repeat", [2, 3, 99])
        aceptados, filtrados = aplicar_reglas(lista, apagadas, CATALOGO)
        self.assertEqual(list(aceptados["PRODUCT_ID"]), [2, 3, 99])
        self.assertTrue(filtrados.empty)

    def test_cupon_historico_requiere_datos(self) -> None:
        """Activar la regla de cupón sin lista de productos debe fallar explícitamente."""

        reglas = dict(REGLAS, solo_productos_con_cupon_historico=True)
        with self.assertRaises(ValueError):
            aplicar_reglas(candidatos("repeat", [1]), reglas, CATALOGO)
        aceptados, _ = aplicar_reglas(candidatos("repeat", [1, 4]), reglas, CATALOGO, productos_con_cupon={4})
        self.assertEqual(list(aceptados["PRODUCT_ID"]), [4])

    def test_categorias_excluidas_y_variantes(self) -> None:
        """El tabaco se filtra por categoría y una variante ya comprada no cuenta como explore."""

        catalogo = pd.DataFrame(
            [
                {"PRODUCT_ID": 1, "DEPARTMENT": "DRUG GM", "COMMODITY_DESC": "CIGARETTES",
                 "SUB_COMMODITY_DESC": "CIGARETTES", "BRAND": "National", "CURR_SIZE_OF_PRODUCT": "1 PK"},
                {"PRODUCT_ID": 2, "DEPARTMENT": "GROCERY", "COMMODITY_DESC": "MILK",
                 "SUB_COMMODITY_DESC": "WHITE MILK", "BRAND": "Private", "CURR_SIZE_OF_PRODUCT": "1 GA"},
                {"PRODUCT_ID": 3, "DEPARTMENT": "GROCERY", "COMMODITY_DESC": "MILK",
                 "SUB_COMMODITY_DESC": "WHITE MILK", "BRAND": "Private", "CURR_SIZE_OF_PRODUCT": "1 GA"},
                {"PRODUCT_ID": 4, "DEPARTMENT": "GROCERY", "COMMODITY_DESC": "EGGS",
                 "SUB_COMMODITY_DESC": "EGGS LARGE", "BRAND": "Private", "CURR_SIZE_OF_PRODUCT": "1 DZ"},
            ]
        )
        reglas = {"categorias_excluidas": {"CIGARETTES": "Tabaco."}, "evitar_variantes_en_explore": True}
        historial = pd.DataFrame([{"household_key": 1, "PRODUCT_ID": 2}])

        # Repeat: el tabaco se filtra aunque el hogar lo compre.
        repeat_ok, repeat_fuera = aplicar_reglas(candidatos("repeat", [1, 2]), reglas, catalogo, historial=historial)
        self.assertEqual(list(repeat_ok["PRODUCT_ID"]), [2])
        self.assertEqual(repeat_fuera["motivo_filtro"].iloc[0], "Tabaco.")

        # Explore: el producto 3 es la misma leche que el 2 con otro código; el 4 sí es nuevo.
        explore_ok, explore_fuera = aplicar_reglas(candidatos("explore_als", [3, 4]), reglas, catalogo, historial=historial)
        self.assertEqual(list(explore_ok["PRODUCT_ID"]), [4])
        self.assertEqual(explore_fuera["regla"].iloc[0], "evitar_variantes_en_explore")

        # Dentro de una misma lista, dos variantes de la misma leche: queda la mejor posicionada.
        solo_lista = {"evitar_variantes_en_lista": True}
        lista_ok, lista_fuera = aplicar_reglas(candidatos("repeat", [3, 4, 2]), solo_lista, catalogo)
        self.assertEqual(list(lista_ok["PRODUCT_ID"]), [3, 4])
        self.assertEqual(list(lista_fuera["PRODUCT_ID"]), [2])
        self.assertEqual(lista_fuera["regla"].iloc[0], "evitar_variantes_en_lista")

        # Sin historial la regla de variantes no puede aplicarse en silencio.
        with self.assertRaises(ValueError):
            aplicar_reglas(candidatos("explore_als", [3]), reglas, catalogo)

    def test_margen_sin_datos_no_se_ignora_en_silencio(self) -> None:
        """Configurar un margen mínimo sin fuente de márgenes debe dar error."""

        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "reglas.json"
            ruta.write_text(json.dumps(dict(REGLAS, margen_minimo=0.2)), encoding="utf-8")
            with self.assertRaises(NotImplementedError):
                cargar_reglas(ruta)


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
