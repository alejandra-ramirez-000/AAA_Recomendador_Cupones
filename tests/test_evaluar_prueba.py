"""Prueba de punta a punta de la evaluación final con datos sintéticos."""

# unittest ejecuta las pruebas sin dependencias adicionales.
import unittest

# pandas construye un historial y una canasta de prueba pequeños.
import pandas as pd

# Importamos las funciones de la evaluación final.
from src.evaluar_prueba import (
    agregar_explicaciones,
    construir_historial_ampliado,
    evaluar_listas,
    generar_sistema_final,
)
from src.seleccionar_modelo import combinar_listas


def interacciones(compras: dict[int, list[int]], canasta_base: int) -> pd.DataFrame:
    """Crea una canasta por hogar con los productos indicados."""

    return pd.DataFrame(
        [
            {"household_key": h, "BASKET_ID": canasta_base + h, "PRODUCT_ID": p, "orden_de_canasta": canasta_base}
            for h, productos in compras.items()
            for p in productos
        ]
    )


class PruebasDeEvaluacionFinal(unittest.TestCase):
    """Comprueba que el sistema congelado se reentrena y evalúa sin errores."""

    def test_flujo_completo_sintetico(self) -> None:
        """Lista de K posiciones con la mezcla pedida y métricas válidas."""

        entrenamiento = interacciones({h: [1, 2, 3, 4 + h % 2] for h in range(1, 7)}, 1)
        validacion = interacciones({h: [1, 2, 6] for h in range(1, 7)}, 100)
        prueba = interacciones({h: [1, 5, 9] for h in range(1, 7)}, 200)
        configuracion = {
            "k": 4,
            "repeat": {"modelo": "baseline_frecuencia_recencia"},
            "explore": {"modelo": "als_explore"},
            "mezcla": {"proporcion_repeat": 0.75},
            "parametros_als": {"factores": 2, "regularizacion": 0.1, "alpha": 2.0, "iteraciones": 3, "semilla": 0},
        }

        historial = construir_historial_ampliado(entrenamiento, validacion)
        lista, repeat, explore, tiempos = generar_sistema_final(historial, configuracion, largo=5)

        # Cada hogar recibe 4 productos: 3 repeat y 1 explore nuevo para él.
        self.assertTrue((lista.groupby("household_key").size() == 4).all())
        self.assertTrue((lista.groupby("household_key")["origen"].apply(lambda o: (o == "repeat").sum()) == 3).all())
        for hogar, grupo in lista.groupby("household_key"):
            comprados = set(historial.loc[historial["household_key"] == hogar, "PRODUCT_ID"])
            nuevos = set(grupo.loc[grupo["origen"] == "explore", "PRODUCT_ID"])
            self.assertFalse(nuevos & comprados)

        # Cada posición tiene su explicación según el origen.
        explicada = agregar_explicaciones(lista, repeat, explore)
        self.assertTrue(explicada.loc[explicada["origen"] == "repeat", "frecuencia_en_canastas"].notna().all())
        self.assertTrue(explicada.loc[explicada["origen"] == "explore", "producto_que_lo_explica"].notna().all())

        # Las métricas se calculan con la canasta de prueba.
        solo_repeat = combinar_listas(repeat, explore, 4, 1.0)
        metricas = evaluar_listas(explicada, repeat, explore, explore, solo_repeat, historial, prueba, 4)
        self.assertEqual(metricas["sistema_final"]["lista_combinada"]["hogares_evaluables"], 6)
        self.assertGreater(metricas["sistema_final"]["lista_combinada"]["hit_rate"], 0)

    def test_rechaza_configuracion_no_soportada(self) -> None:
        """Una configuración con otros modelos no debe evaluarse en silencio."""

        configuracion = {"repeat": {"modelo": "otro"}, "explore": {"modelo": "als_explore"}}
        with self.assertRaises(NotImplementedError):
            generar_sistema_final(pd.DataFrame(), configuracion)


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
