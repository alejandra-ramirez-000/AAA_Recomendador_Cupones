"""Ejecuta el proyecto de punta a punta, en orden, con registro reproducible.

Cada etapa se ejecuta como `python -m src.<modulo>`, exactamente igual que los
comandos documentados en el README, y apuntando a las carpetas indicadas. El
pipeline además:

- guarda versiones de Python y librerías, commit de git y fecha en
  `<salida>/entorno.txt`;
- registra cada etapa (comando, duración, resultado) en
  `<salida>/registro_pipeline.json`;
- calcula hashes de los CSV de entrada antes y después, y falla si alguno cambió:
  los datos originales nunca se modifican;
- se detiene en la primera etapa que falle.

Etapas congeladas. Por defecto **no** se ejecutan:
- `seleccion`, que reescribiría `config/modelo_final.json` (D-006). Se activa con
  `--recongelar`.
- `prueba`, la evaluación única sobre prueba (D-007). Se activa con
  `--incluir-prueba`, y además requiere `--forzar-prueba` si ya existe un reporte.

Ejemplos de uso:
    python -m src.pipeline                      # todo menos selección y prueba
    python -m src.pipeline --desde als          # retoma desde una etapa
    python -m src.pipeline --solo reglas demo   # etapas puntuales
    python -m src.pipeline --listar             # muestra las etapas
"""

# argparse define las opciones de la línea de comandos.
import argparse

# hashlib calcula huellas de los archivos de entrada.
import hashlib

# importlib.metadata obtiene versiones instaladas sin importar las librerías.
from importlib import metadata

# json y datetime registran la ejecución.
import json
from datetime import datetime

# os, platform, subprocess y sys ejecutan las etapas con el mismo intérprete.
import os
import platform
import subprocess
import sys

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# perf_counter mide la duración de cada etapa.
from time import perf_counter


# Raíz del repositorio: las etapas se ejecutan desde aquí.
RAIZ_DEL_PROYECTO = Path(__file__).resolve().parents[1]

# Librerías cuyas versiones se registran.
LIBRERIAS_REGISTRADAS = ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "streamlit")

# Etapas en orden de ejecución.
ORDEN_DE_ETAPAS = (
    "calidad", "eda", "division", "baseline", "coocurrencia", "als",
    "comparacion", "seleccion", "eda_defensa", "prueba", "reglas", "demo",
)

# Etapas que no se ejecutan sin una opción explícita.
ETAPAS_CONGELADAS = {"seleccion", "prueba"}


def construir_etapas(argumentos: argparse.Namespace) -> dict[str, list[str]]:
    """Devuelve el comando de cada etapa con las carpetas de esta ejecución."""

    datos = str(argumentos.data_dir)
    salida = argumentos.output_dir
    division = str(salida / "division_temporal")
    configuracion = str(argumentos.configuracion)
    reglas = str(argumentos.reglas)

    etapas = {
        "calidad": ["src.cargar_datos", "--data-dir", datos, "--validar-completo"],
        "eda": ["src.analizar_datos", "--data-dir", datos, "--output-dir", str(salida / "eda")],
        "division": ["src.dividir_temporalmente", "--data-dir", datos, "--output-dir", division],
        "baseline": ["src.recomendar_baseline", "--division-dir", division, "--data-dir", datos,
                     "--output-dir", str(salida / "baseline")],
        "coocurrencia": ["src.recomendar_coocurrencia", "--division-dir", division, "--data-dir", datos,
                         "--output-dir", str(salida / "coocurrencia"),
                         "--soporte-minimo-producto", str(argumentos.soporte_minimo_coocurrencia)],
        "als": ["src.recomendar_als", "--division-dir", division, "--data-dir", datos,
                "--output-dir", str(salida / "als")] + (["--rapido"] if argumentos.als_rapido else []),
        "comparacion": ["src.evaluar_modelos", "--division-dir", division, "--resultados-dir", str(salida),
                        "--output-dir", str(salida / "comparacion")],
        "seleccion": ["src.seleccionar_modelo", "--division-dir", division, "--resultados-dir", str(salida),
                      "--comparacion", str(salida / "comparacion" / "comparacion_validacion.csv"),
                      "--output-dir", str(salida / "seleccion"), "--configuracion", configuracion],
        "eda_defensa": ["src.analizar_repeticion", "--division-dir", division, "--resultados-dir", str(salida),
                        "--data-dir", datos, "--reglas", reglas, "--configuracion", configuracion,
                        "--output-dir", str(salida / "eda")],
        "prueba": ["src.evaluar_prueba", "--division-dir", division, "--resultados-dir", str(salida),
                   "--configuracion", configuracion, "--data-dir", datos, "--output-dir", str(salida / "prueba")]
                  + (["--forzar"] if argumentos.forzar_prueba else []),
        "reglas": ["src.reglas_negocio", "--division-dir", division, "--resultados-dir", str(salida),
                   "--configuracion", configuracion, "--reglas", reglas, "--data-dir", datos,
                   "--output-dir", str(salida / "reglas")],
        "demo": ["src.preparar_demo", "--division-dir", division, "--resultados-dir", str(salida),
                 "--configuracion", configuracion, "--reglas", reglas, "--data-dir", datos,
                 "--output-dir", str(salida / "demo")],
    }
    return {nombre: [sys.executable, "-m", *comando] for nombre, comando in etapas.items()}


def elegir_etapas(argumentos: argparse.Namespace) -> list[str]:
    """Aplica --solo, --desde, --hasta y las protecciones de etapas congeladas."""

    if argumentos.solo:
        elegidas = [etapa for etapa in ORDEN_DE_ETAPAS if etapa in argumentos.solo]
    else:
        inicio = ORDEN_DE_ETAPAS.index(argumentos.desde)
        fin = ORDEN_DE_ETAPAS.index(argumentos.hasta)
        elegidas = list(ORDEN_DE_ETAPAS[inicio:fin + 1])

    habilitadas = {"seleccion": argumentos.recongelar, "prueba": argumentos.incluir_prueba}
    return [etapa for etapa in elegidas if etapa not in ETAPAS_CONGELADAS or habilitadas[etapa]]


def huellas_de_datos(carpeta: Path) -> dict[str, str]:
    """SHA-256 de cada CSV de la carpeta de datos."""

    huellas = {}
    for ruta in sorted(carpeta.glob("*.csv")):
        calculo = hashlib.sha256()
        with ruta.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(1 << 20), b""):
                calculo.update(bloque)
        huellas[ruta.name] = calculo.hexdigest()
    return huellas


def describir_entorno() -> dict[str, str]:
    """Versiones de Python, librerías y commit de git."""

    entorno = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sistema": platform.platform(),
    }
    for libreria in LIBRERIAS_REGISTRADAS:
        try:
            entorno[libreria] = metadata.version(libreria)
        except metadata.PackageNotFoundError:
            entorno[libreria] = "no instalada"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ_DEL_PROYECTO,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        cambios = subprocess.run(
            ["git", "status", "--porcelain"], cwd=RAIZ_DEL_PROYECTO, capture_output=True, text=True, check=True,
        ).stdout.strip()
        entorno["git_commit"] = commit + (" (con cambios sin commitear)" if cambios else "")
    except (OSError, subprocess.CalledProcessError):
        entorno["git_commit"] = "no disponible"
    return entorno


def ejecutar(argumentos: argparse.Namespace) -> dict:
    """Ejecuta las etapas elegidas y devuelve el registro."""

    etapas = elegir_etapas(argumentos)
    comandos = construir_etapas(argumentos)
    salida = argumentos.output_dir
    salida.mkdir(parents=True, exist_ok=True)

    registro = {"entorno": describir_entorno(), "etapas": [], "datos_intactos": None}
    (salida / "entorno.txt").write_text(
        "\n".join(f"{clave}: {valor}" for clave, valor in registro["entorno"].items()) + "\n", encoding="utf-8"
    )

    huellas_antes = huellas_de_datos(argumentos.data_dir)
    print(f"Etapas: {', '.join(etapas)}", flush=True)

    for etapa in etapas:
        print(f"\n=== {etapa} ===", flush=True)
        inicio = perf_counter()
        resultado = subprocess.run(comandos[etapa], cwd=RAIZ_DEL_PROYECTO, env=argumentos.entorno_hijo)
        duracion = round(perf_counter() - inicio, 1)
        registro["etapas"].append(
            {"etapa": etapa, "comando": " ".join(comandos[etapa][1:]), "segundos": duracion,
             "codigo_salida": resultado.returncode}
        )
        print(f"--- {etapa}: {'OK' if resultado.returncode == 0 else 'ERROR'} en {duracion} s", flush=True)
        if resultado.returncode != 0:
            break

    # Los datos de entrada deben quedar idénticos.
    huellas_despues = huellas_de_datos(argumentos.data_dir)
    registro["datos_intactos"] = huellas_antes == huellas_despues
    registro["huellas_datos"] = huellas_despues

    (salida / "registro_pipeline.json").write_text(
        json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return registro


def interpretar_argumentos(lista: list[str] | None = None) -> argparse.Namespace:
    """Define las opciones del pipeline."""

    analizador = argparse.ArgumentParser(description="Ejecuta el proyecto de punta a punta.")
    analizador.add_argument("--data-dir", type=Path, default=RAIZ_DEL_PROYECTO / "data")
    analizador.add_argument("--output-dir", type=Path, default=RAIZ_DEL_PROYECTO / "outputs")
    analizador.add_argument("--configuracion", type=Path, default=RAIZ_DEL_PROYECTO / "config" / "modelo_final.json")
    analizador.add_argument("--reglas", type=Path, default=RAIZ_DEL_PROYECTO / "config" / "reglas_negocio.json")
    analizador.add_argument("--desde", choices=ORDEN_DE_ETAPAS, default=ORDEN_DE_ETAPAS[0])
    analizador.add_argument("--hasta", choices=ORDEN_DE_ETAPAS, default=ORDEN_DE_ETAPAS[-1])
    analizador.add_argument("--solo", nargs="+", choices=ORDEN_DE_ETAPAS)
    analizador.add_argument("--recongelar", action="store_true", help="Ejecuta la selección y reescribe la configuración.")
    analizador.add_argument("--incluir-prueba", action="store_true", help="Ejecuta la evaluación única sobre prueba.")
    analizador.add_argument("--forzar-prueba", action="store_true", help="Permite sobrescribir el reporte de prueba.")
    analizador.add_argument("--als-rapido", action="store_true", help="ALS con una sola configuración pequeña.")
    analizador.add_argument("--soporte-minimo-coocurrencia", type=int, default=500)
    analizador.add_argument("--listar", action="store_true", help="Muestra las etapas y termina.")
    argumentos = analizador.parse_args(lista)
    # Las etapas imprimen tildes y símbolos: forzamos UTF-8 para la consola de Windows.
    argumentos.entorno_hijo = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return argumentos


def main() -> None:
    """Punto de entrada desde la terminal."""

    argumentos = interpretar_argumentos()
    if argumentos.listar:
        for etapa in ORDEN_DE_ETAPAS:
            nota = " (congelada: requiere opción explícita)" if etapa in ETAPAS_CONGELADAS else ""
            print(f"- {etapa}{nota}")
        return

    registro = ejecutar(argumentos)
    print("\nResumen:")
    for etapa in registro["etapas"]:
        print(f"- {etapa['etapa']}: código {etapa['codigo_salida']}, {etapa['segundos']} s")
    print(f"- datos de entrada intactos: {registro['datos_intactos']}")

    fallo = any(etapa["codigo_salida"] != 0 for etapa in registro["etapas"])
    if fallo or not registro["datos_intactos"]:
        sys.exit(1)


# Ejecutamos solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
