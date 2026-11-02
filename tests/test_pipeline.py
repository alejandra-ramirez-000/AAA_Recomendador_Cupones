"""Prueba de integración: el pipeline completo sobre un dataset sintético.

Ejecuta todas las etapas (incluidas selección y prueba) en una carpeta
temporal, sin tocar `data/`, `outputs/` ni `config/` del proyecto.
"""

# json, shutil y tempfile preparan y leen la ejecución aislada.
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# pandas lee las salidas para verificarlas.
import pandas as pd

# Pipeline y generador de datos.
from src.pipeline import RAIZ_DEL_PROYECTO, ejecutar, interpretar_argumentos
from tests.datos_sinteticos import generar_datos_sinteticos


class PruebaDeIntegracion(unittest.TestCase):
    """Corre el pipeline una vez y verifica cada salida principal."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.carpeta = Path(tempfile.mkdtemp(prefix="pipeline_"))
        cls.datos = cls.carpeta / "data"
        cls.salida = cls.carpeta / "outputs"
        cls.configuracion = cls.carpeta / "config" / "modelo_final.json"
        generar_datos_sinteticos(cls.datos)

        # Las reglas reales del proyecto, copiadas para no depender de rutas compartidas.
        cls.reglas = cls.carpeta / "config" / "reglas_negocio.json"
        cls.reglas.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(RAIZ_DEL_PROYECTO / "config" / "reglas_negocio.json", cls.reglas)

        cls.opciones_base = [
            "--data-dir", str(cls.datos), "--output-dir", str(cls.salida),
            "--configuracion", str(cls.configuracion), "--reglas", str(cls.reglas),
            "--als-rapido", "--soporte-minimo-coocurrencia", "5",
        ]
        cls.registro = ejecutar(interpretar_argumentos(cls.opciones_base + ["--recongelar", "--incluir-prueba"]))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.carpeta, ignore_errors=True)

    def test_todas_las_etapas_terminan_bien(self) -> None:
        etapas = [etapa["etapa"] for etapa in self.registro["etapas"]]
        self.assertEqual(etapas[0], "calidad")
        self.assertEqual(etapas[-1], "demo")
        self.assertEqual(len(etapas), 12)
        for etapa in self.registro["etapas"]:
            self.assertEqual(etapa["codigo_salida"], 0, etapa["etapa"])

    def test_datos_de_entrada_intactos_y_entorno_registrado(self) -> None:
        self.assertTrue(self.registro["datos_intactos"])
        self.assertIn("pandas", (self.salida / "entorno.txt").read_text(encoding="utf-8"))
        self.assertTrue((self.salida / "registro_pipeline.json").exists())

    def test_division_reserva_dos_canastas_por_hogar(self) -> None:
        resumen = pd.read_csv(self.salida / "division_temporal" / "resumen_division.csv")
        self.assertEqual(len(resumen), 3)
        validacion = pd.read_csv(self.salida / "division_temporal" / "interacciones_validacion.csv")
        self.assertTrue((validacion.groupby("household_key")["BASKET_ID"].nunique() == 1).all())

    def test_comparacion_y_configuracion_congelada(self) -> None:
        comparacion = pd.read_csv(self.salida / "comparacion" / "comparacion_validacion.csv")
        self.assertEqual(comparacion["modelo"].nunique(), 6)
        configuracion = json.loads(self.configuracion.read_text(encoding="utf-8"))
        self.assertEqual(configuracion["k"], 10)
        self.assertEqual(configuracion["mezcla"]["posiciones_explore"], 3)

    def test_prueba_es_unica(self) -> None:
        self.assertTrue((self.salida / "prueba" / "reporte_prueba.md").exists())
        repetir = ejecutar(interpretar_argumentos(self.opciones_base + ["--solo", "prueba", "--incluir-prueba"]))
        self.assertNotEqual(repetir["etapas"][0]["codigo_salida"], 0)

    def test_eda_de_defensa(self) -> None:
        resumen = json.loads((self.salida / "eda" / "defensa_resumen.json").read_text(encoding="utf-8"))
        self.assertGreater(resumen["repeticion"]["media_por_canasta"], 0)
        self.assertEqual(resumen["objetivos_validacion"]["hogares"], 40)
        self.assertTrue((self.salida / "eda" / "defensa_rendimiento_por_historial.png").exists())

    def test_demo_cumple_reglas(self) -> None:
        lista = pd.read_csv(self.salida / "demo" / "lista_final.csv")
        self.assertTrue((lista.groupby("household_key").size() == 10).all())
        self.assertFalse(lista["DEPARTMENT"].eq("KIOSK-GAS").any())
        self.assertFalse(lista["COMMODITY_DESC"].eq("CIGARETTES").any())
        self.assertFalse(lista.duplicated(["household_key", "SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT"]).any())
        self.assertTrue(lista["explicacion"].notna().all())
        filtrados = pd.read_csv(self.salida / "demo" / "filtrados.csv")
        self.assertTrue(filtrados["motivo_filtro"].notna().all())

    def test_etapas_congeladas_no_corren_por_defecto(self) -> None:
        from src.pipeline import elegir_etapas

        elegidas = elegir_etapas(interpretar_argumentos([]))
        self.assertNotIn("seleccion", elegidas)
        self.assertNotIn("prueba", elegidas)


# Permitimos ejecutar este archivo directamente.
if __name__ == "__main__":
    unittest.main()
