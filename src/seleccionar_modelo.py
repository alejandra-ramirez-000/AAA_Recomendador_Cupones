"""Selecciona los modelos finales y la mezcla repeat/explore usando validación.

Pasos:
1. Lee la tabla comparativa (`outputs/comparacion/comparacion_validacion.csv`)
   y elige, para cada tipo, el modelo con mayor NDCG@10. Un modelo más complejo
   solo reemplaza al baseline si lo supera.
2. Combina las listas repeat y explore elegidas en una única lista de K
   productos, para varias proporciones repeat/explore, y mide la curva de
   compromiso en validación.
3. Aplica la regla de decisión de la mezcla y congela la configuración final en
   `config/modelo_final.json`. A partir de ahí no se ajusta nada más antes de
   evaluar sobre prueba.

El conjunto de prueba no se abre en ningún paso.

Ejemplo de uso:
    python -m src.seleccionar_modelo
"""

# argparse permite cambiar rutas y la regla de decisión sin modificar el código.
import argparse

# json guarda la configuración congelada.
import json

# date registra cuándo se congeló la decisión.
from datetime import date

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# pandas organiza listas, curvas y tablas.
import pandas as pd

# Evaluación común y rutas por defecto.
from src.evaluar_modelos import (
    DIRECTORIO_DE_DIVISION_POR_DEFECTO,
    MODELOS_REGISTRADOS,
    RAIZ_DEL_PROYECTO,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)


# Carpetas de entrada y salida.
RUTA_COMPARACION_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "comparacion" / "comparacion_validacion.csv"
DIRECTORIO_DE_SALIDA_POR_DEFECTO = RAIZ_DEL_PROYECTO / "outputs" / "seleccion"
RUTA_CONFIGURACION_POR_DEFECTO = RAIZ_DEL_PROYECTO / "config" / "modelo_final.json"

# Proporciones repeat evaluadas en la curva (0 = solo explore, 1 = solo repeat).
PROPORCIONES_REPEAT = tuple(round(valor / 10, 1) for valor in range(11))

# Largo de la lista mostrada al área comercial en la demo.
K_PRINCIPAL = 10

# Regla de mezcla: el mínimo de posiciones explore que el negocio exige en la lista.
# Es un requisito del producto (que la lista no sea solo lo ya conocido), no un
# resultado del modelo: con métricas de aciertos la mezcla óptima sería casi todo repeat.
MINIMO_POSICIONES_EXPLORE = 3


def combinar_listas(
    recomendaciones_repeat: pd.DataFrame,
    recomendaciones_explore: pd.DataFrame,
    k: int,
    proporcion_repeat: float,
) -> pd.DataFrame:
    """Combina por hogar una lista repeat y una explore en K posiciones.

    Se toman las primeras round(K * proporción) posiciones repeat y el resto
    explore. Las repeat van primero porque su probabilidad de acierto es mucho
    mayor. Si una lista no alcanza su cupo (por ejemplo, un hogar con poco
    historial), las posiciones libres se completan con la otra. Se descartan
    productos duplicados.
    """

    # Cupos de cada tipo dentro de la lista.
    cupo_repeat = round(k * proporcion_repeat)

    # Listas ordenadas de productos por hogar.
    repeat_por_hogar = (
        recomendaciones_repeat.sort_values(["household_key", "posicion"])
        .groupby("household_key")["PRODUCT_ID"].agg(list).to_dict()
    )
    explore_por_hogar = (
        recomendaciones_explore.sort_values(["household_key", "posicion"])
        .groupby("household_key")["PRODUCT_ID"].agg(list).to_dict()
    )

    filas: list[dict] = []
    for household_key in sorted(set(repeat_por_hogar) | set(explore_por_hogar)):
        lista_repeat = repeat_por_hogar.get(household_key, [])
        lista_explore = explore_por_hogar.get(household_key, [])

        # Primero cada tipo llena su cupo; luego el sobrante completa con el otro.
        elegidos_repeat = lista_repeat[:cupo_repeat]
        elegidos_explore = lista_explore[: k - len(elegidos_repeat)]
        faltan = k - len(elegidos_repeat) - len(elegidos_explore)
        if faltan > 0:
            elegidos_repeat += lista_repeat[len(elegidos_repeat): len(elegidos_repeat) + faltan]

        # Armamos la lista final sin duplicados, repeat primero.
        vistos: set[int] = set()
        posicion = 1
        for origen, productos in (("repeat", elegidos_repeat), ("explore", elegidos_explore)):
            for producto_id in productos:
                if producto_id in vistos:
                    continue
                vistos.add(producto_id)
                filas.append(
                    {
                        "household_key": household_key,
                        "PRODUCT_ID": producto_id,
                        "posicion": posicion,
                        "origen": origen,
                    }
                )
                posicion += 1

    # Devolvemos la lista combinada con el origen de cada posición.
    return pd.DataFrame(filas, columns=["household_key", "PRODUCT_ID", "posicion", "origen"])


def calcular_curva_de_mezcla(
    recomendaciones_repeat: pd.DataFrame,
    recomendaciones_explore: pd.DataFrame,
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    valores_de_k: tuple[int, ...] = (K_PRINCIPAL, 20),
    proporciones: tuple[float, ...] = PROPORCIONES_REPEAT,
) -> pd.DataFrame:
    """Mide la lista combinada para cada proporción repeat y cada K."""

    # Objetivos: cualquier compra, solo recompras y solo productos nuevos.
    relevantes_todos = obtener_productos_relevantes(validacion, entrenamiento, "todos")
    relevantes_explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")

    filas: list[dict] = []
    for k in valores_de_k:
        for proporcion in proporciones:
            combinada = combinar_listas(recomendaciones_repeat, recomendaciones_explore, k, proporcion)
            total = calcular_metricas_en_k(combinada, relevantes_todos, k)

            # Aciertos explore dentro de la lista combinada: ¿se descubre algo nuevo?
            solo_explore = combinada.loc[combinada["origen"] == "explore"]
            descubrimiento = calcular_metricas_en_k(solo_explore, relevantes_explore, k)

            filas.append(
                {
                    "k": k,
                    "proporcion_repeat": proporcion,
                    "posiciones_repeat": round(k * proporcion),
                    "posiciones_explore": k - round(k * proporcion),
                    "precision": total["precision"],
                    "recall": total["recall"],
                    "hit_rate": total["hit_rate"],
                    "ndcg": total["ndcg"],
                    "hit_rate_explore": descubrimiento["hit_rate"],
                    "hogares_evaluables": total["hogares_evaluables"],
                }
            )

    return pd.DataFrame(filas)


def elegir_modelo_por_tipo(comparacion: pd.DataFrame, tipo: str, k: int = K_PRINCIPAL) -> dict:
    """Elige el modelo con mayor NDCG@k del tipo pedido y lo compara con el baseline."""

    candidatos = comparacion.loc[(comparacion["tipo"] == tipo) & (comparacion["k"] == k)]
    candidatos = candidatos.sort_values(["ndcg", "hit_rate"], ascending=False)
    mejor = candidatos.iloc[0]
    baseline = candidatos.loc[candidatos["modelo"].str.startswith("baseline")].iloc[0]
    return {
        "modelo": mejor["modelo"],
        f"ndcg@{k}": float(mejor["ndcg"]),
        f"hit_rate@{k}": float(mejor["hit_rate"]),
        "baseline": baseline["modelo"],
        f"ndcg@{k}_baseline": float(baseline["ndcg"]),
        "mejora_relativa_ndcg": round(float(mejor["ndcg"] / baseline["ndcg"] - 1), 4),
    }


def ruta_de_recomendaciones(nombre_modelo: str, directorio_resultados: Path) -> Path:
    """Devuelve el CSV de recomendaciones de un modelo registrado."""

    for modelo in MODELOS_REGISTRADOS:
        if modelo["nombre"] == nombre_modelo:
            return directorio_resultados / modelo["carpeta"] / modelo["archivo"]
    raise KeyError(f"Modelo no registrado: {nombre_modelo}")


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución de la selección."""

    analizador = argparse.ArgumentParser(description="Selecciona modelos y mezcla repeat/explore en validación.")
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)
    analizador.add_argument("--comparacion", type=Path, default=RUTA_COMPARACION_POR_DEFECTO)
    analizador.add_argument("--resultados-dir", type=Path, default=RAIZ_DEL_PROYECTO / "outputs")
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)
    analizador.add_argument("--configuracion", type=Path, default=RUTA_CONFIGURACION_POR_DEFECTO)
    analizador.add_argument("--minimo-explore", type=int, default=MINIMO_POSICIONES_EXPLORE)
    return analizador.parse_args()


def main() -> None:
    """Elige modelos, calcula la curva de mezcla y congela la configuración."""

    argumentos = interpretar_argumentos()

    # 1. Mejor modelo por tipo según la tabla comparativa.
    comparacion = pd.read_csv(argumentos.comparacion)
    eleccion_repeat = elegir_modelo_por_tipo(comparacion, "repeat")
    eleccion_explore = elegir_modelo_por_tipo(comparacion, "explore")
    print(f"Repeat elegido: {eleccion_repeat}")
    print(f"Explore elegido: {eleccion_explore}")

    # 2. Curva de mezcla con los modelos elegidos.
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")
    recomendaciones_repeat = pd.read_csv(ruta_de_recomendaciones(eleccion_repeat["modelo"], argumentos.resultados_dir))
    recomendaciones_explore = pd.read_csv(ruta_de_recomendaciones(eleccion_explore["modelo"], argumentos.resultados_dir))
    curva = calcular_curva_de_mezcla(recomendaciones_repeat, recomendaciones_explore, entrenamiento, validacion)

    argumentos.output_dir.mkdir(parents=True, exist_ok=True)
    curva.to_csv(argumentos.output_dir / "curva_mezcla_validacion.csv", index=False)

    # 3. Regla: con K principal, respetar el mínimo explore y maximizar NDCG total.
    curva_k = curva.loc[curva["k"] == K_PRINCIPAL]
    admisibles = curva_k.loc[curva_k["posiciones_explore"] >= argumentos.minimo_explore]
    elegida = admisibles.sort_values(["ndcg", "hit_rate"], ascending=False).iloc[0]
    sin_restriccion = curva_k.sort_values(["ndcg", "hit_rate"], ascending=False).iloc[0]

    print("\nCurva de mezcla (K = 10)")
    print(curva_k.to_string(index=False))
    print(f"\nMezcla elegida: {int(elegida['posiciones_repeat'])} repeat + {int(elegida['posiciones_explore'])} explore")

    # 4. Congelamos la configuración final.
    configuracion = {
        "fecha_de_congelamiento": date.today().isoformat(),
        "k": K_PRINCIPAL,
        "repeat": eleccion_repeat,
        "explore": eleccion_explore,
        "mezcla": {
            "proporcion_repeat": float(elegida["proporcion_repeat"]),
            "posiciones_repeat": int(elegida["posiciones_repeat"]),
            "posiciones_explore": int(elegida["posiciones_explore"]),
            "orden": "repeat primero, luego explore; si un tipo no alcanza su cupo se completa con el otro",
            "regla": f"máximo NDCG@{K_PRINCIPAL} total con al menos {argumentos.minimo_explore} posiciones explore",
            "metricas_validacion": {
                "ndcg": float(elegida["ndcg"]),
                "hit_rate": float(elegida["hit_rate"]),
                "hit_rate_explore": float(elegida["hit_rate_explore"]),
            },
            "referencia_sin_minimo_explore": {
                "posiciones_repeat": int(sin_restriccion["posiciones_repeat"]),
                "ndcg": float(sin_restriccion["ndcg"]),
                "hit_rate": float(sin_restriccion["hit_rate"]),
            },
        },
    }

    # Parámetros de ALS, si fue elegido, para poder reentrenarlo sin ambigüedad.
    ruta_als = argumentos.resultados_dir / "als" / "metricas_validacion.json"
    if "als" in (eleccion_repeat["modelo"] + eleccion_explore["modelo"]) and ruta_als.exists():
        with ruta_als.open(encoding="utf-8") as archivo:
            configuracion["parametros_als"] = json.load(archivo)["parametros"]

    argumentos.configuracion.parent.mkdir(parents=True, exist_ok=True)
    with argumentos.configuracion.open("w", encoding="utf-8") as archivo:
        json.dump(configuracion, archivo, indent=2, ensure_ascii=False)
    print(f"\nConfiguración congelada en {argumentos.configuracion}")


# Ejecutamos la selección solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
