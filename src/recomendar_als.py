"""Genera y evalúa recomendaciones con ALS implícito (filtrado colaborativo).

ALS (Alternating Least Squares) para retroalimentación implícita aprende un
vector latente por hogar (X) y otro por producto (Y) tal que X_u · Y_i se acerque
a 1 cuando el hogar compró el producto y a 0 cuando no lo hizo. Cada compra
observada recibe una confianza mayor cuanto más veces se repitió:

    preferencia p_ui = 1 si el hogar compró el producto, 0 si no
    confianza  c_ui = 1 + alpha * log(1 + canastas_con_el_producto)

y se minimiza (Hu, Koren y Volinsky, 2008):

    sum_ui c_ui * (p_ui - X_u · Y_i)^2 + lambda * (||X||^2 + ||Y||^2)

Se alterna: con Y fijo, cada X_u es la solución de un sistema lineal; luego con
X fijo se resuelve Y. Para no invertir una matriz por hogar y por producto se
usan pocos pasos de gradiente conjugado, como hace la librería `implicit`
(Takács, Pilászy y Tikk, 2011).

Implementación propia: la librería `implicit` no puede usarse en el equipo
porque el Control de aplicaciones de Windows bloquea sus DLL compiladas (ver
D-005 en la bitácora). Esta versión usa solo numpy y scipy.sparse.

Solo se usa entrenamiento para aprender; validación sirve para elegir la
configuración. El conjunto de prueba no se abre.

Salidas en outputs/als/:
    grilla_validacion.csv                     métricas de todas las configuraciones
    recomendaciones_repeat_validacion.csv     ranking repeat de la mejor configuración
    recomendaciones_explore_validacion.csv    ranking explore de la mejor configuración
    metricas_validacion.json                  métricas, parámetros y tiempos

Ejemplos de uso:
    python -m src.recomendar_als                    # grilla completa
    python -m src.recomendar_als --rapido           # una sola configuración pequeña
"""

# argparse permite elegir carpetas y parámetros sin modificar el código.
import argparse

# itertools.product recorre todas las combinaciones de la grilla.
from itertools import product

# json guarda métricas y parámetros en un formato sencillo de consultar.
import json

# Path permite construir rutas legibles dentro del proyecto.
from pathlib import Path

# perf_counter mide tiempos de entrenamiento y recomendación.
from time import perf_counter

# numpy realiza el álgebra lineal de los factores latentes.
import numpy as np

# pandas organiza interacciones y resultados.
import pandas as pd

# scipy.sparse guarda la matriz hogar x producto sin ocupar memoria por los ceros.
from scipy import sparse

# Reutilizamos catálogo, evaluación común y utilidades del baseline.
from src.cargar_datos import DIRECTORIO_DE_DATOS_POR_DEFECTO, cargar_productos
from src.evaluar_modelos import (
    VALORES_DE_K,
    calcular_metricas_en_k,
    obtener_productos_relevantes,
)
from src.recomendar_baseline import (
    DIRECTORIO_DE_DIVISION_POR_DEFECTO,
    enriquecer_con_catalogo,
)


# Los derivados de ALS se guardan separados del resto de modelos.
DIRECTORIO_DE_SALIDA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "outputs" / "als"

# Grilla pequeña y fija, decidida antes de mirar resultados (ver D-005).
# Las iteraciones se evalúan dentro de una misma corrida: se mide en 15 y se continúa hasta 30.
GRILLA_POR_DEFECTO = {
    "factores": (32, 64),
    "regularizacion": (0.05, 0.10),
    "alpha": (2.0, 10.0),
    "iteraciones": (15, 30),
}

# Configuración pequeña para pruebas rápidas.
GRILLA_RAPIDA = {
    "factores": (32,),
    "regularizacion": (0.10,),
    "alpha": (10.0,),
    "iteraciones": (10,),
}

# Semilla fija para que los factores iniciales, y por lo tanto los resultados, sean reproducibles.
SEMILLA_POR_DEFECTO = 42

# Pasos de gradiente conjugado por actualización; 3 es el valor por defecto de `implicit`.
PASOS_GRADIENTE_CONJUGADO = 3

# Tamaño de bloque para cálculos fila a fila, para acotar la memoria usada.
TAMANO_DE_BLOQUE = 200_000


def construir_matriz_de_confianza(
    entrenamiento: pd.DataFrame, alpha: float
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, pd.DataFrame]:
    """Construye la matriz dispersa hogar x producto con confianzas.

    Devuelve:
        matriz: CSR de forma (hogares, productos) con c_ui en las compras observadas;
        hogares: household_key de cada fila;
        productos: PRODUCT_ID de cada columna;
        frecuencias: tabla hogar-producto con la cantidad de canastas.
    """

    # La señal es en cuántas canastas aparece el producto, no la suma de unidades
    # (las unidades no son comparables entre departamentos, p. ej. KIOSK-GAS).
    frecuencias = (
        entrenamiento.groupby(["household_key", "PRODUCT_ID"], as_index=False)
        .agg(frecuencia_en_canastas=("BASKET_ID", "nunique"))
    )

    # Asignamos un índice entero consecutivo a cada hogar y a cada producto.
    hogares = np.sort(frecuencias["household_key"].unique())
    productos = np.sort(frecuencias["PRODUCT_ID"].unique())
    filas = np.searchsorted(hogares, frecuencias["household_key"].to_numpy())
    columnas = np.searchsorted(productos, frecuencias["PRODUCT_ID"].to_numpy())

    # Transformamos la frecuencia en confianza con un logaritmo para atenuar productos muy repetidos.
    confianza = 1.0 + alpha * np.log1p(frecuencias["frecuencia_en_canastas"].to_numpy(dtype=float))

    # Creamos la matriz dispersa; los ceros (no compras) no se almacenan.
    matriz = sparse.csr_matrix(
        (confianza, (filas, columnas)), shape=(len(hogares), len(productos))
    )

    # Ordenamos índices para que las operaciones dispersas sean deterministas.
    matriz.sort_indices()

    # Devolvemos la matriz y los mapeos necesarios para traducir índices a IDs.
    return matriz, hogares, productos, frecuencias


def producto_punto_por_fila(
    matriz: sparse.csr_matrix, factores_filas: np.ndarray, factores_columnas: np.ndarray
) -> np.ndarray:
    """Calcula X_u · Y_i para cada celda no nula (u, i) de la matriz, por bloques."""

    # Recuperamos el número de fila de cada celda no nula a partir de indptr.
    filas = np.repeat(np.arange(matriz.shape[0]), np.diff(matriz.indptr))
    columnas = matriz.indices

    # Reservamos el resultado: un producto punto por celda no nula.
    resultado = np.empty(matriz.nnz)

    # Procesamos por bloques para no crear de una vez arreglos de (nnz x factores).
    for inicio in range(0, matriz.nnz, TAMANO_DE_BLOQUE):
        fin = min(inicio + TAMANO_DE_BLOQUE, matriz.nnz)
        resultado[inicio:fin] = np.einsum(
            "ij,ij->i",
            factores_filas[filas[inicio:fin]],
            factores_columnas[columnas[inicio:fin]],
        )

    # Devolvemos los productos punto alineados con matriz.data.
    return resultado


def actualizar_factores(
    matriz: sparse.csr_matrix,
    factores_a_actualizar: np.ndarray,
    factores_fijos: np.ndarray,
    regularizacion: float,
    pasos: int = PASOS_GRADIENTE_CONJUGADO,
) -> np.ndarray:
    """Actualiza los factores de las filas de `matriz` con los de las columnas fijos.

    Para cada fila u resuelve aproximadamente
        (Y^T C_u Y + lambda I) x_u = Y^T C_u p_u
    con gradiente conjugado, partiendo de la solución anterior. Como
    Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y, el primer término se calcula una sola vez
    y el segundo solo involucra las compras observadas de u.
    """

    # Y^T Y + lambda I es común a todas las filas.
    cantidad_factores = factores_fijos.shape[1]
    base = factores_fijos.T @ factores_fijos + regularizacion * np.eye(cantidad_factores)

    # Copiamos los pesos (c_ui - 1) de las compras observadas, alineados con la matriz.
    pesos_extra = matriz.data - 1.0

    def multiplicar_por_a(vectores: np.ndarray) -> np.ndarray:
        """Calcula A_u v_u para todas las filas a la vez."""

        # Término común: v_u (Y^T Y + lambda I).
        resultado = vectores @ base

        # Término específico: Y^T (C_u - I) Y v_u, usando solo las celdas no nulas.
        valores = pesos_extra * producto_punto_por_fila(matriz, vectores, factores_fijos)
        matriz_ponderada = sparse.csr_matrix(
            (valores, matriz.indices, matriz.indptr), shape=matriz.shape
        )
        return resultado + matriz_ponderada @ factores_fijos

    # Lado derecho: Y^T C_u p_u = suma de c_ui * y_i sobre las compras observadas (p_ui = 1).
    lado_derecho = matriz @ factores_fijos

    # Gradiente conjugado vectorizado: cada fila es un sistema independiente.
    solucion = factores_a_actualizar.copy()
    residuo = lado_derecho - multiplicar_por_a(solucion)
    direccion = residuo.copy()
    norma_residuo = np.einsum("ij,ij->i", residuo, residuo)

    for _ in range(pasos):
        # Si todos los residuos son ~0, la solución ya es exacta.
        if np.all(norma_residuo < 1e-20):
            break

        # Paso óptimo a lo largo de la dirección actual.
        a_por_direccion = multiplicar_por_a(direccion)
        denominador = np.einsum("ij,ij->i", direccion, a_por_direccion)
        paso = np.divide(norma_residuo, denominador, out=np.zeros_like(norma_residuo), where=denominador > 0)

        # Actualizamos solución y residuo.
        solucion += paso[:, None] * direccion
        residuo -= paso[:, None] * a_por_direccion

        # Nueva dirección conjugada.
        nueva_norma = np.einsum("ij,ij->i", residuo, residuo)
        beta = np.divide(nueva_norma, norma_residuo, out=np.zeros_like(nueva_norma), where=norma_residuo > 0)
        direccion = residuo + beta[:, None] * direccion
        norma_residuo = nueva_norma

    # Devolvemos los factores actualizados.
    return solucion


def entrenar_als(
    matriz: sparse.csr_matrix,
    factores: int,
    regularizacion: float,
    iteraciones: int,
    semilla: int = SEMILLA_POR_DEFECTO,
    al_terminar_iteracion=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Entrena ALS implícito y devuelve (factores_hogar, factores_producto).

    `al_terminar_iteracion(numero, X, Y)` se invoca tras cada iteración; permite
    evaluar varias cantidades de iteraciones en una sola corrida.
    """

    # Inicializamos factores pequeños y aleatorios con semilla fija.
    generador = np.random.default_rng(semilla)
    factores_hogar = generador.normal(0, 0.01, size=(matriz.shape[0], factores))
    factores_producto = generador.normal(0, 0.01, size=(matriz.shape[1], factores))

    # La transpuesta permite resolver productos con la misma función que hogares.
    matriz_transpuesta = matriz.T.tocsr()
    matriz_transpuesta.sort_indices()

    # Alternamos: hogares con productos fijos, luego productos con hogares fijos.
    for numero in range(1, iteraciones + 1):
        factores_hogar = actualizar_factores(matriz, factores_hogar, factores_producto, regularizacion)
        factores_producto = actualizar_factores(
            matriz_transpuesta, factores_producto, factores_hogar, regularizacion
        )
        if al_terminar_iteracion is not None:
            al_terminar_iteracion(numero, factores_hogar, factores_producto)

    # Devolvemos ambos conjuntos de factores latentes.
    return factores_hogar, factores_producto


def generar_recomendaciones_als(
    matriz: sparse.csr_matrix,
    hogares: np.ndarray,
    productos: np.ndarray,
    factores_hogar: np.ndarray,
    factores_producto: np.ndarray,
    cantidad_recomendaciones: int,
    con_explicacion: bool = False,
    tamano_lote: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera rankings repeat y explore a partir del puntaje X_u · Y_i.

    Repeat ordena solo productos del historial del hogar; explore solo productos
    nuevos. Con `con_explicacion=True`, cada sugerencia explore indica el
    producto del historial más parecido en el espacio latente.
    """

    # Normalizamos factores de producto para medir similitud coseno en las explicaciones.
    normas = np.linalg.norm(factores_producto, axis=1, keepdims=True)
    factores_normalizados = factores_producto / np.where(normas > 0, normas, 1.0)

    # Acumulamos filas de ambos rankings.
    filas_repeat: list[dict] = []
    filas_explore: list[dict] = []
    k = cantidad_recomendaciones

    # Procesamos hogares por lotes para no crear la matriz completa hogares x productos.
    for inicio in range(0, len(hogares), tamano_lote):
        fin = min(inicio + tamano_lote, len(hogares))
        puntajes = factores_hogar[inicio:fin] @ factores_producto.T

        for desplazamiento in range(fin - inicio):
            indice_hogar = inicio + desplazamiento
            fila = puntajes[desplazamiento]
            historial = matriz.indices[matriz.indptr[indice_hogar]:matriz.indptr[indice_hogar + 1]]

            # Repeat: mejores productos dentro del historial.
            puntajes_historial = fila[historial]
            orden_repeat = np.argsort(-puntajes_historial, kind="stable")[:k]
            for posicion, indice_local in enumerate(orden_repeat, start=1):
                indice_producto = historial[indice_local]
                filas_repeat.append(
                    {
                        "household_key": hogares[indice_hogar],
                        "PRODUCT_ID": productos[indice_producto],
                        "posicion": posicion,
                        "puntaje_als": float(puntajes_historial[indice_local]),
                    }
                )

            # Explore: anulamos el historial y tomamos los mejores productos restantes.
            fila_explore = fila.copy()
            fila_explore[historial] = -np.inf
            candidatos = np.argpartition(-fila_explore, k)[:k]
            candidatos = candidatos[np.argsort(-fila_explore[candidatos], kind="stable")]
            for posicion, indice_producto in enumerate(candidatos, start=1):
                registro = {
                    "household_key": hogares[indice_hogar],
                    "PRODUCT_ID": productos[indice_producto],
                    "posicion": posicion,
                    "puntaje_als": float(fila_explore[indice_producto]),
                }

                # Explicación: producto comprado más similar al recomendado.
                if con_explicacion and len(historial):
                    similitudes = factores_normalizados[historial] @ factores_normalizados[indice_producto]
                    mejor = int(np.argmax(similitudes))
                    registro["producto_que_lo_explica"] = productos[historial[mejor]]
                    registro["similitud_explicativa"] = float(similitudes[mejor])

                filas_explore.append(registro)

    # Convertimos y etiquetamos ambos rankings.
    recomendaciones_repeat = pd.DataFrame(filas_repeat)
    recomendaciones_repeat["tipo_recomendacion"] = "repeat_als"
    recomendaciones_explore = pd.DataFrame(filas_explore)
    recomendaciones_explore["tipo_recomendacion"] = "explore_als"
    return recomendaciones_repeat, recomendaciones_explore


def evaluar_en_k10(
    recomendaciones_repeat: pd.DataFrame,
    recomendaciones_explore: pd.DataFrame,
    relevantes_repeat: dict[int, set[int]],
    relevantes_explore: dict[int, set[int]],
) -> dict[str, float]:
    """Resume las métricas en K = 10 de ambos tipos para la tabla de la grilla."""

    # Calculamos métricas comunes para repeat y explore.
    repeat = calcular_metricas_en_k(recomendaciones_repeat, relevantes_repeat, 10)
    explore = calcular_metricas_en_k(recomendaciones_explore, relevantes_explore, 10)

    # Aplanamos con prefijos para que cada configuración sea una fila.
    resumen = {f"repeat_{clave}@10": valor for clave, valor in repeat.items() if clave not in ("k", "hogares_evaluables")}
    resumen.update({f"explore_{clave}@10": valor for clave, valor in explore.items() if clave not in ("k", "hogares_evaluables")})
    return resumen


def ejecutar_grilla(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    grilla: dict[str, tuple],
    cantidad_recomendaciones: int,
    semilla: int,
) -> pd.DataFrame:
    """Entrena cada configuración y la evalúa en validación (K = 10)."""

    # Los objetivos reales no dependen de la configuración: se calculan una vez.
    relevantes_repeat = obtener_productos_relevantes(validacion, entrenamiento, "repeat")
    relevantes_explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")

    # Las iteraciones a evaluar se toman dentro de una misma corrida.
    iteraciones_a_evaluar = set(grilla["iteraciones"])
    iteraciones_maximas = max(iteraciones_a_evaluar)

    filas: list[dict] = []
    for alpha in grilla["alpha"]:
        # La matriz depende solo de alpha.
        matriz, hogares, productos, _ = construir_matriz_de_confianza(entrenamiento, alpha)

        for factores, regularizacion in product(grilla["factores"], grilla["regularizacion"]):
            inicio = perf_counter()
            tiempo_evaluacion = 0.0

            def evaluar(numero: int, x: np.ndarray, y: np.ndarray) -> None:
                nonlocal tiempo_evaluacion
                if numero not in iteraciones_a_evaluar:
                    return
                inicio_eval = perf_counter()
                repeat, explore = generar_recomendaciones_als(
                    matriz, hogares, productos, x, y, cantidad_recomendaciones
                )
                tiempo_evaluacion += perf_counter() - inicio_eval
                fila = {
                    "factores": factores,
                    "regularizacion": regularizacion,
                    "alpha": alpha,
                    "iteraciones": numero,
                    "segundos_entrenamiento": round(perf_counter() - inicio - tiempo_evaluacion, 2),
                    **evaluar_en_k10(repeat, explore, relevantes_repeat, relevantes_explore),
                }
                filas.append(fila)
                print(
                    f"factores={factores} reg={regularizacion} alpha={alpha} it={numero} "
                    f"-> repeat NDCG@10={fila['repeat_ndcg@10']:.4f} "
                    f"explore NDCG@10={fila['explore_ndcg@10']:.4f} "
                    f"explore HR@10={fila['explore_hit_rate@10']:.4f} "
                    f"({fila['segundos_entrenamiento']} s)",
                    flush=True,
                )

            entrenar_als(matriz, factores, regularizacion, iteraciones_maximas, semilla, evaluar)

    # Devolvemos la grilla ordenada por el criterio de selección.
    return pd.DataFrame(filas).sort_values(
        ["explore_ndcg@10", "repeat_ndcg@10"], ascending=False
    ).reset_index(drop=True)


def interpretar_argumentos() -> argparse.Namespace:
    """Define las opciones de ejecución de ALS."""

    analizador = argparse.ArgumentParser(description="Entrena y evalúa ALS implícito en validación.")
    analizador.add_argument("--division-dir", type=Path, default=DIRECTORIO_DE_DIVISION_POR_DEFECTO)
    analizador.add_argument("--output-dir", type=Path, default=DIRECTORIO_DE_SALIDA_POR_DEFECTO)
    analizador.add_argument("--data-dir", type=Path, default=DIRECTORIO_DE_DATOS_POR_DEFECTO)
    analizador.add_argument("--cantidad-recomendaciones", type=int, default=20)
    analizador.add_argument("--semilla", type=int, default=SEMILLA_POR_DEFECTO)
    analizador.add_argument("--rapido", action="store_true", help="Usa una sola configuración pequeña.")
    return analizador.parse_args()


def main() -> None:
    """Ejecuta la grilla, reentrena la mejor configuración y guarda sus resultados."""

    argumentos = interpretar_argumentos()
    if argumentos.cantidad_recomendaciones <= 0:
        raise ValueError("La cantidad de recomendaciones debe ser mayor que cero.")

    # Cargamos únicamente entrenamiento y validación; prueba queda reservada.
    entrenamiento = pd.read_csv(argumentos.division_dir / "interacciones_entrenamiento.csv")
    validacion = pd.read_csv(argumentos.division_dir / "interacciones_validacion.csv")

    # 1. Grilla: cada configuración se evalúa solo en validación.
    grilla = GRILLA_RAPIDA if argumentos.rapido else GRILLA_POR_DEFECTO
    resultados_grilla = ejecutar_grilla(
        entrenamiento, validacion, grilla, argumentos.cantidad_recomendaciones, argumentos.semilla
    )
    argumentos.output_dir.mkdir(parents=True, exist_ok=True)
    resultados_grilla.to_csv(argumentos.output_dir / "grilla_validacion.csv", index=False)

    # 2. Elegimos la configuración con mejor NDCG@10 explore (desempate: repeat).
    mejor = resultados_grilla.iloc[0]
    parametros = {
        "factores": int(mejor["factores"]),
        "regularizacion": float(mejor["regularizacion"]),
        "alpha": float(mejor["alpha"]),
        "iteraciones": int(mejor["iteraciones"]),
        "semilla": argumentos.semilla,
        "pasos_gradiente_conjugado": PASOS_GRADIENTE_CONJUGADO,
        "criterio_de_seleccion": "max explore NDCG@10 en validación",
    }
    print(f"\nMejor configuración: {parametros}")

    # 3. Reentrenamos la mejor configuración midiendo tiempos.
    inicio_entrenamiento = perf_counter()
    matriz, hogares, productos, frecuencias = construir_matriz_de_confianza(entrenamiento, parametros["alpha"])
    factores_hogar, factores_producto = entrenar_als(
        matriz, parametros["factores"], parametros["regularizacion"], parametros["iteraciones"], parametros["semilla"]
    )
    segundos_entrenamiento = perf_counter() - inicio_entrenamiento

    inicio_recomendacion = perf_counter()
    recomendaciones_repeat, recomendaciones_explore = generar_recomendaciones_als(
        matriz, hogares, productos, factores_hogar, factores_producto,
        argumentos.cantidad_recomendaciones, con_explicacion=True,
    )
    segundos_recomendacion = perf_counter() - inicio_recomendacion

    # Repeat se explica con la frecuencia histórica del producto en el hogar.
    recomendaciones_repeat = recomendaciones_repeat.merge(
        frecuencias, on=["household_key", "PRODUCT_ID"], how="left", validate="one_to_one"
    )

    # 4. Métricas completas en K = 5, 10 y 20 con el módulo común.
    relevantes_repeat = obtener_productos_relevantes(validacion, entrenamiento, "repeat")
    relevantes_explore = obtener_productos_relevantes(validacion, entrenamiento, "explore")
    metricas = {
        "repeat": [calcular_metricas_en_k(recomendaciones_repeat, relevantes_repeat, k) for k in VALORES_DE_K],
        "explore": [calcular_metricas_en_k(recomendaciones_explore, relevantes_explore, k) for k in VALORES_DE_K],
        "parametros": parametros,
        "configuraciones_evaluadas": len(resultados_grilla),
        "tiempos_segundos": {
            "entrenamiento": round(segundos_entrenamiento, 3),
            "recomendacion_total": round(segundos_recomendacion, 3),
            "ms_recomendacion_por_hogar": round(1000 * segundos_recomendacion / len(hogares), 3),
        },
    }

    # 5. Agregamos catálogo para interpretar y guardamos todo.
    productos_catalogo = cargar_productos(argumentos.data_dir)
    recomendaciones_repeat = enriquecer_con_catalogo(recomendaciones_repeat, productos_catalogo)
    recomendaciones_explore = enriquecer_con_catalogo(recomendaciones_explore, productos_catalogo)
    recomendaciones_repeat.to_csv(argumentos.output_dir / "recomendaciones_repeat_validacion.csv", index=False)
    recomendaciones_explore.to_csv(argumentos.output_dir / "recomendaciones_explore_validacion.csv", index=False)
    with (argumentos.output_dir / "metricas_validacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, indent=2, ensure_ascii=False)

    print("\nMétricas de ALS en validación")
    for tipo in ("repeat", "explore"):
        for resultado in metricas[tipo]:
            print(f"- {tipo}@{resultado['k']}: {resultado}")
    print(f"- tiempos: {metricas['tiempos_segundos']}")


# Ejecutamos el flujo solo cuando el módulo se invoca directamente.
if __name__ == "__main__":
    main()
