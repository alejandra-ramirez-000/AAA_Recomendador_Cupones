"""Prototipo de cara al cliente: simula ser socio de un supermercado y recibir
cupones personalizados que puede reclamar, descartar o canjear en caja.

Dos formas de probar el sistema:

    - Cliente real: cualquier household_key de la base (dunnhumby). Lee los
      archivos precalculados por `python -m src.preparar_demo`.
    - Cliente simulado: se crea desde la app, se le arma un historial de
      compras a mano (o con un botón de demo rápida) y sus cupones se generan
      EN VIVO con el mismo motor de reglas de negocio del proyecto. Repeat usa
      la misma fórmula que `src/recomendar_baseline.py` (frecuencia + recencia);
      explore usa el modelo de coocurrencia ya evaluado (`src/recomendar_coocurrencia.py`)
      porque ALS no puede anotar un hogar que no existía al entrenar sin
      reentrenar el modelo completo.

También incluye una "Vista analista" (oculta por defecto) con el detalle
técnico — puntajes del modelo y candidatos descartados por las reglas de
negocio — para el equipo comercial y la defensa del proyecto.

Ejecutar desde la raíz del proyecto:
    streamlit run app/app.py
"""

# csv y datetime registran el feedback simulado (reclamos, descartes, canjes).
import base64
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# pandas y streamlit construyen la interfaz.
import pandas as pd
import streamlit as st

# Se agrega la raíz del proyecto al path para reutilizar el motor de reglas y
# la fórmula de repeat del sistema real (en vez de reimplementarlas aparte).
_RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(_RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROYECTO))

from src.reglas_negocio import aplicar_reglas, cargar_reglas  # noqa: E402
from src.recomendar_baseline import construir_perfil_repeat  # noqa: E402


# Carpeta con los archivos precalculados de la demo y carpeta con los CSV crudos.
DIRECTORIO_DEMO = _RAIZ_PROYECTO / "outputs" / "demo"
DIRECTORIO_DATOS = _RAIZ_PROYECTO / "data"

# Archivo donde se acumula el feedback simulado.
RUTA_FEEDBACK = DIRECTORIO_DEMO / "feedback_simulado.csv"

# Precio de referencia por producto (`python -m src.calcular_precios_catalogo`).
RUTA_PRECIOS = DIRECTORIO_DEMO / "precios_estimados.csv"

# Asociaciones por coocurrencia y popularidad de entrenamiento: se reutilizan
# para generar explore y aplicar las reglas de negocio a clientes simulados.
RUTA_ASOCIACIONES = _RAIZ_PROYECTO / "outputs" / "coocurrencia" / "asociaciones_principales.csv"
RUTA_POPULARIDAD = _RAIZ_PROYECTO / "outputs" / "baseline" / "popularidad_entrenamiento.csv"

# Columnas visibles en las tablas técnicas (vista analista).
COLUMNAS_LISTA = {
    "posicion": "#",
    "producto": "Producto",
    "DEPARTMENT": "Departamento",
    "puntaje": "Puntaje",
    "explicacion": "Por qué se recomienda",
}

# Encabezados de cada sección de cupones según el origen de la recomendación.
TITULOS_ORIGEN = {
    "repeat": (":material/autorenew: Porque siempre los comprás", "repeat"),
    "explore": (":material/auto_awesome: Para que descubras algo nuevo", "explore"),
}

# Un ícono Material por departamento, para que el catálogo se vea menos a tabla y más a góndola.
# (nombre tal como lo espera Streamlit en ":material/<nombre>:" o la fuente Material Symbols)
ICONOS_DEPARTAMENTO = {
    "GROCERY": "shopping_cart", "PRODUCE": "nutrition", "DRUG GM": "medication",
    "MEAT": "kebab_dining", "MEAT-PCKGD": "kebab_dining",
    "DELI": "lunch_dining", "DELI/SNACK BAR": "lunch_dining", "NUTRITION": "medication",
    "PASTRY": "bakery_dining", "SEAFOOD": "set_meal", "SEAFOOD-PCKGD": "set_meal",
    "FLORAL": "local_florist", "RESTAURANT": "restaurant", "SALAD BAR": "eco",
    "KIOSK-GAS": "local_gas_station", "PHOTO": "photo_camera", "SPIRITS": "liquor",
    "GRO BAKERY": "bakery_dining", "TOYS": "toys", "AUTOMOTIVE": "directions_car",
    "HOUSEWARES": "chair", "CHARITABLE CONT": "volunteer_activism", "VIDEO RENTAL": "movie",
    "POSTAL CENTER": "local_post_office", "CHEF SHOPPE": "soup_kitchen",
    "COUP/STR & MFG": "confirmation_number", "PROD-WHS SALES": "nutrition",
}
ICONO_DEPARTAMENTO_POR_DEFECTO = "sell"

# Categorías con más productos: las que se muestran como pastillas de filtro rápido.
CATEGORIAS_DESTACADAS = ["GROCERY", "DRUG GM", "PRODUCE", "MEAT", "DELI", "PASTRY"]

# Fotos de alimentos por departamento (app/assets/alimentos/<nombre>.jpg). Los
# departamentos sin foto conservan el ícono vectorial (ICONOS_DEPARTAMENTO).
DIRECTORIO_ASSETS = Path(__file__).parent / "assets" / "alimentos"
IMAGENES_DEPARTAMENTO = {
    "GROCERY": "grocery", "PROD-WHS SALES": "grocery",
    "PRODUCE": "produce",
    "MEAT": "meat", "MEAT-PCKGD": "meat",
    "DELI": "deli", "DELI/SNACK BAR": "deli",
    "PASTRY": "pastry", "GRO BAKERY": "pastry",
    "SEAFOOD": "seafood", "SEAFOOD-PCKGD": "seafood",
    "SPIRITS": "bebidas",
    "DRUG GM": "farmacia", "NUTRITION": "farmacia",
}


def icono_md(nombre: str) -> str:
    """Shortcode de ícono Material para texto y widgets nativos de Streamlit
    (botones, tabs, markdown, captions, alertas): se renderiza como un ícono
    vectorial real, no como emoji."""

    return f":material/{nombre}:"


def icono_html(nombre: str, tamano: int = 22) -> str:
    """El mismo ícono Material, como HTML, para usar dentro de bloques con
    unsafe_allow_html=True (el shortcode ':material/...:' no se interpreta ahí)."""

    return f'<span class="material-symbols-outlined" style="font-size:{tamano}px;vertical-align:middle;">{nombre}</span>'


def _nombre_icono_departamento(departamento: str) -> str:
    return ICONOS_DEPARTAMENTO.get(str(departamento).strip().upper(), ICONO_DEPARTAMENTO_POR_DEFECTO)


def icono_departamento_md(departamento: str) -> str:
    return icono_md(_nombre_icono_departamento(departamento))


def icono_departamento_html(departamento: str, tamano: int = 22) -> str:
    return icono_html(_nombre_icono_departamento(departamento), tamano)


@st.cache_data
def _imagen_base64(nombre: str) -> str | None:
    ruta = DIRECTORIO_ASSETS / f"{nombre}.jpg"
    if not ruta.exists():
        return None
    return base64.b64encode(ruta.read_bytes()).decode("ascii")


def imagen_departamento_html(departamento: str) -> str:
    """Foto de la categoría si existe; si no, el ícono vectorial sobre un
    fondo degradado. Pensada para llenar el banner superior de una tarjeta
    (clase .coupon-banner, que le da el tamaño y el overflow oculto)."""

    nombre = IMAGENES_DEPARTAMENTO.get(str(departamento).strip().upper())
    b64 = _imagen_base64(nombre) if nombre else None
    if not b64:
        return f'<div class="coupon-banner-fallback">{icono_departamento_html(departamento, 40)}</div>'
    return f'<img src="data:image/jpeg;base64,{b64}" alt="{nombre}">'


def imagen_hero_html() -> str | None:
    b64 = _imagen_base64("hero_alimentos")
    if not b64:
        return None
    return f'<img class="hero-img" src="data:image/jpeg;base64,{b64}" alt="Alimentos frescos">'


def icono_caso(caso: str) -> str:
    """Ícono (shortcode Material) según la descripción del hogar de ejemplo (login)."""

    caso = caso.lower()
    if "más largo" in caso:
        return icono_md("workspace_premium")
    if "más corto" in caso:
        return icono_md("fiber_new")
    if "filtrados" in caso:
        return icono_md("block")
    if "explicación" in caso:
        return icono_md("auto_awesome")
    return icono_md("badge")


st.set_page_config(page_title="SuperCoop — Cupones para vos", page_icon="🛒", layout="wide")


def aplicar_estilos() -> None:
    """Sistema visual de la app: tipografía, paleta, navbar y tarjetas."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,500,0,0&display=block');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Íconos Material dentro de HTML propio (navbar, hero, tarjetas con unsafe_allow_html). */
        .material-symbols-outlined {
            font-family: 'Material Symbols Outlined';
            font-weight: normal; font-style: normal;
            line-height: 1; letter-spacing: normal; text-transform: none;
            display: inline-block; white-space: nowrap; word-wrap: normal; direction: ltr;
            -webkit-font-smoothing: antialiased;
        }
        h1, h2, h3, h4, h5, .hero h1, button[kind] p, button[kind] div {
            font-family: 'Poppins', sans-serif !important;
        }

        :root {
            --color-primario: #4A1D96;
            --color-primario-oscuro: #371569;
            --color-secundario: #7ED957;
            --color-secundario-oscuro: #5FBF3A;
            --color-fondo-seccion: #EDE7F6;
            --color-fondo-general: #F8F9FB;
            --color-texto-principal: #2D1B4E;
            --color-texto-secundario: #6B7280;
            --radio: 16px;
            --sombra: 0 2px 14px rgba(45, 27, 78, 0.08);
        }

        .stApp { background: var(--color-fondo-general); }

        /* --- Navbar --- */
        .navbar {
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 12px;
            padding: 14px 26px; margin: -1rem -1rem 1.6rem -1rem;
            background: white; border: 1px solid #ECEAF5;
            border-radius: 0 0 20px 20px;
            box-shadow: var(--sombra);
        }
        .navbar-brand { display: flex; align-items: center; gap: 12px; }
        .navbar-logo {
            width: 42px; height: 42px; border-radius: 12px; font-size: 1.3rem;
            display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, var(--color-primario), var(--color-secundario));
        }
        .navbar-brand-text { font-size: 1.35rem; font-weight: 800; color: var(--color-primario); letter-spacing: -0.02em; }
        .navbar-brand-text span { color: var(--color-secundario-oscuro); }
        .navbar-right { display: flex; gap: 10px; flex-wrap: wrap; }
        .chip-cliente {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--color-fondo-seccion); color: var(--color-primario); font-weight: 700;
            padding: 8px 16px; border-radius: 999px; font-size: 0.88rem;
        }
        .chip-carrito {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--color-secundario); color: white; font-weight: 700;
            padding: 8px 16px; border-radius: 999px; font-size: 0.88rem;
        }

        /* --- Hero (login) --- */
        .hero {
            background: linear-gradient(120deg, var(--color-primario) 0%, #6A3BC9 55%, var(--color-secundario-oscuro) 150%);
            border-radius: 24px; padding: 46px 44px; margin-bottom: 26px; position: relative; overflow: hidden;
        }
        .hero-badge { position: absolute; right: 36px; top: 50%; transform: translateY(-50%); font-size: 4.2rem; opacity: 0.92; }
        .hero-img {
            position: absolute; right: 40px; top: 50%; transform: translateY(-50%);
            width: 220px; height: 220px; border-radius: 20px; object-fit: cover;
            box-shadow: 0 16px 34px rgba(0,0,0,0.28); border: 4px solid rgba(255,255,255,0.3);
        }
        @media (max-width: 900px) { .hero-img { display: none; } }
        .hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0 0 10px 0; color: white; max-width: 70%; }
        .hero h1 span { color: var(--color-secundario); }
        .hero p { font-size: 1.05rem; color: rgba(255,255,255,0.88); max-width: 480px; margin: 0; }

        /* --- Franja de confianza --- */
        .trust-row { display: flex; gap: 30px; justify-content: center; flex-wrap: wrap; padding: 6px 0 14px 0; }
        .trust-item { display: flex; align-items: center; gap: 10px; color: var(--color-texto-principal); font-weight: 600; font-size: 0.92rem; }
        .trust-icon { font-size: 1.5rem; }

        /* --- Tarjetas de cupón: banner de foto arriba, con el precio/descuento superpuesto --- */
        .coupon-banner {
            position: relative; width: 100%; height: 120px; border-radius: 14px;
            overflow: hidden; margin-bottom: 12px;
        }
        .coupon-banner img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .coupon-banner-fallback {
            width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, var(--color-fondo-seccion), #E3DAF5);
        }
        .coupon-discount-badge {
            position: absolute; top: 10px; right: 10px;
            background: var(--color-primario); color: white; font-weight: 800; font-size: 1.05rem;
            padding: 6px 14px; border-radius: 999px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .coupon-discount-badge.bienvenida { background: var(--color-secundario-oscuro); }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--radio) !important;
            box-shadow: var(--sombra);
            border-color: #ECEAF5 !important;
        }

        /* --- Botones: primario sólido (verde), secundario tipo pastilla (violeta) --- */
        button[kind] {
            border-radius: 999px !important;
            font-weight: 600 !important;
            transition: transform 0.05s ease, box-shadow 0.15s ease;
        }
        button[kind]:hover { transform: translateY(-1px); }
        button[kind="secondary"] {
            background: white !important;
            border: 1.5px solid var(--color-primario) !important;
            color: var(--color-primario) !important;
        }
        button[kind="secondary"]:hover { background: var(--color-fondo-seccion) !important; }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--color-texto-secundario); }
        .stTabs [aria-selected="true"] { color: var(--color-primario) !important; }

        /* --- Alertas: se reúsan mucho para feedback, van con la paleta de marca --- */
        [data-testid="stAlertContainer"] { border-radius: 14px !important; border: none !important; }
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: var(--color-fondo-seccion) !important; }
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) * { color: var(--color-primario) !important; }
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background: #E8F8E3 !important; }
        [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) * { color: var(--color-secundario-oscuro) !important; }

        /* --- Métricas (st.metric): número en Poppins y color de marca --- */
        [data-testid="stMetricValue"] { font-family: 'Poppins', sans-serif; color: var(--color-primario); }
        [data-testid="stMetricLabel"] { color: var(--color-texto-secundario); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def cargar_demo() -> dict:
    """Carga una sola vez todas las tablas precalculadas."""

    if not (DIRECTORIO_DEMO / "lista_final.csv").exists():
        return {}
    tablas = {
        nombre: pd.read_csv(DIRECTORIO_DEMO / f"{nombre}.csv")
        for nombre in ("lista_final", "filtrados", "respaldo_popular", "resumen_hogares", "productos_frecuentes", "ultima_canasta")
    }
    tablas["ejemplos"] = json.loads((DIRECTORIO_DEMO / "hogares_ejemplo.json").read_text(encoding="utf-8"))
    tablas["metadatos"] = json.loads((DIRECTORIO_DEMO / "metadatos.json").read_text(encoding="utf-8"))
    return tablas


@st.cache_data
def cargar_precios() -> pd.Series:
    """Precio de referencia por producto (`python -m src.calcular_precios_catalogo`)."""

    if not RUTA_PRECIOS.exists():
        return pd.Series(dtype=float)
    precios = pd.read_csv(RUTA_PRECIOS)
    return precios.set_index("PRODUCT_ID")["precio_referencia"]


@st.cache_data
def cargar_catalogo() -> pd.DataFrame:
    """Catálogo general: todos los productos con precio de referencia conocido."""

    if not RUTA_PRECIOS.exists():
        return pd.DataFrame()
    precios = pd.read_csv(RUTA_PRECIOS)
    productos = pd.read_csv(
        DIRECTORIO_DATOS / "product.csv",
        usecols=["PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT"],
    )
    catalogo = productos.merge(precios, on="PRODUCT_ID", how="inner")
    catalogo = catalogo.loc[catalogo["DEPARTMENT"].notna() & (catalogo["DEPARTMENT"].str.strip() != "")]
    tamano = catalogo["CURR_SIZE_OF_PRODUCT"].fillna("").astype(str).str.strip()
    catalogo["producto"] = (
        catalogo["SUB_COMMODITY_DESC"].fillna("").str.strip().str.capitalize()
        + " · " + catalogo["BRAND"].fillna("").str.strip()
        + tamano.where(tamano == "", " · " + tamano)
    )
    return catalogo


@st.cache_data
def catalogo_deduplicado(pendientes_ids: frozenset) -> pd.DataFrame:
    """Un producto por descripción (varios PRODUCT_ID distintos comparten
    descripción cuando no tienen tamaño cargado). Si alguno de ese grupo
    tiene un cupón pendiente para el hogar, se usa ese como representante.
    """

    catalogo = cargar_catalogo()
    if catalogo.empty:
        return catalogo
    catalogo = catalogo.copy()
    catalogo["_con_cupon"] = ~catalogo["PRODUCT_ID"].isin(pendientes_ids)
    catalogo = catalogo.sort_values(["producto", "DEPARTMENT", "_con_cupon", "PRODUCT_ID"])
    return (
        catalogo.drop_duplicates(subset=["producto", "DEPARTMENT"], keep="first")
        .drop(columns="_con_cupon")
        .reset_index(drop=True)
    )


@st.cache_data
def cargar_asociaciones() -> pd.DataFrame:
    """Asociaciones por coocurrencia (lift) entre productos, ya evaluadas."""

    if not RUTA_ASOCIACIONES.exists():
        return pd.DataFrame()
    return pd.read_csv(RUTA_ASOCIACIONES)


@st.cache_data
def cargar_popularidad() -> pd.DataFrame:
    """Popularidad de entrenamiento por producto (respaldo y soporte mínimo)."""

    if not RUTA_POPULARIDAD.exists():
        return pd.DataFrame()
    return pd.read_csv(RUTA_POPULARIDAD)


@st.cache_data
def soporte_por_producto_dict() -> dict:
    return cargar_popularidad().set_index("PRODUCT_ID")["canastas_con_producto"].to_dict()


@st.cache_data
def opciones_clientes_reales(resumen: pd.DataFrame) -> dict:
    """Texto de cada cliente real para el buscador del login -> household_key."""

    ordenado = resumen.sort_values("canastas", ascending=False)
    return {
        f"N.º {int(fila.household_key)} — {int(fila.canastas)} canastas · ${fila.gasto_total:,.0f}": int(fila.household_key)
        for fila in ordenado.itertuples()
    }


def registrar_feedback(hogar: int, producto_id: int, origen: str, valoracion: str) -> None:
    """Agrega una fila al registro local de feedback simulado."""

    nuevo = not RUTA_FEEDBACK.exists()
    with RUTA_FEEDBACK.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if nuevo:
            escritor.writerow(["fecha", "household_key", "PRODUCT_ID", "origen", "valoracion"])
        escritor.writerow([datetime.now().isoformat(timespec="seconds"), hogar, producto_id, origen, valoracion])


def voz_cliente(explicacion: str) -> str:
    """Reescribe la explicación técnica (en tercera persona) en segunda persona."""

    reemplazos = [
        ("Lo compró en", "Lo compraste en"),
        ("su última compra", "tu última compra"),
        ("su penúltima compra", "tu penúltima compra"),
        ("Nuevo para el hogar.", "¡Nuevo para vos!"),
        ("Nuevo para el hogar:", "Nuevo para vos:"),
        ("que ya compra ", "que ya comprás "),
        ("como este,", "como vos,"),
    ]
    for antes, despues in reemplazos:
        explicacion = explicacion.replace(antes, despues)
    return explicacion


def calcular_descuento(puntaje: float, tipo: str) -> int:
    """Descuento simulado para la demo (no hay datos de margen por producto).

    Repeat (fidelización) se mueve entre 10-20%; explore (incentivo a probar
    algo nuevo) entre 15-30%. Escala con el puntaje del modelo.
    """

    if tipo == "bienvenida":
        return 10
    puntaje = 0.0 if pd.isna(puntaje) else max(0.0, min(1.0, float(puntaje)))
    base, rango = (10, 10) if tipo == "repeat" else (15, 15)
    return round(base + puntaje * rango)


def precio_de(producto_id: int) -> float | None:
    precios = cargar_precios()
    if producto_id not in precios.index:
        return None
    return float(precios.loc[producto_id])


def billetera_de(hogar: int) -> dict:
    return st.session_state.setdefault("billetera", {}).setdefault(hogar, {})


def descartes_de(hogar: int) -> set:
    return st.session_state.setdefault("descartes", {}).setdefault(hogar, set())


def carrito_de(hogar: int) -> dict:
    return st.session_state.setdefault("carrito", {}).setdefault(hogar, {})


# --- Clientes simulados: se crean desde la app, con su propio historial ----

def clientes_simulados() -> dict:
    return st.session_state.setdefault("clientes_simulados", {})


def crear_cliente_simulado(nombre: str) -> int:
    siguiente = st.session_state.get("siguiente_id_simulado", 900_001)
    st.session_state["siguiente_id_simulado"] = siguiente + 1
    clientes_simulados()[siguiente] = {
        "nombre": nombre or f"Cliente simulado {siguiente - 900_000}",
        "canastas": [],
        "canasta_en_construccion": {},
        "lista_final": pd.DataFrame(),
        "filtrados": pd.DataFrame(),
        "generado_para_n_canastas": -1,
    }
    return siguiente


def historial_df_de(hogar: int) -> pd.DataFrame:
    """Historial de un cliente simulado en el mismo formato que usa el pipeline real."""

    canastas = clientes_simulados()[hogar]["canastas"]
    filas = [
        {"household_key": hogar, "BASKET_ID": hogar * 10_000 + indice, "orden_de_canasta": indice, "PRODUCT_ID": producto_id}
        for indice, canasta in enumerate(canastas)
        for producto_id in canasta
    ]
    return pd.DataFrame(filas, columns=["household_key", "BASKET_ID", "orden_de_canasta", "PRODUCT_ID"])


def registrar_canasta(hogar: int) -> None:
    """Pasa la canasta en construcción al historial (una compra más, ya pasada)."""

    datos = clientes_simulados()[hogar]
    if not datos["canasta_en_construccion"]:
        return
    datos["canastas"].append(list(datos["canasta_en_construccion"].keys()))
    datos["canasta_en_construccion"] = {}


def generar_historial_automatico(hogar: int) -> None:
    """Historial de ejemplo instantáneo, para hacer una demo sin cargar nada a mano."""

    popularidad = cargar_popularidad()
    pool = popularidad.sort_values("canastas_con_producto", ascending=False).head(200)["PRODUCT_ID"].tolist()
    rng = random.Random(hogar)
    favoritos = rng.sample(pool[:80], k=6)
    datos = clientes_simulados()[hogar]
    for _ in range(6):
        canasta = set(rng.sample(favoritos, k=rng.randint(3, 5)))
        canasta |= set(rng.sample(pool, k=rng.randint(1, 3)))
        datos["canastas"].append(sorted(canasta))


def tipo_cliente(demo: dict, hogar: int) -> str:
    """'real' (base de datos), 'simulado' (creado en la app) o 'desconocido'."""

    if hogar in clientes_simulados():
        return "simulado"
    if hogar in set(demo["resumen_hogares"]["household_key"]):
        return "real"
    return "desconocido"


def generar_recomendaciones_simuladas(demo: dict, hogar: int) -> None:
    """Calcula la lista final para un cliente simulado, con su propio historial.

    Repeat: misma fórmula que `src.recomendar_baseline.construir_perfil_repeat`
    (frecuencia + recencia). Explore: coocurrencia (lift) sobre los productos
    del historial, con respaldo en popularidad si faltan candidatos. Reglas de
    negocio: `src.reglas_negocio.aplicar_reglas`, igual que en el sistema real.
    """

    datos = clientes_simulados()[hogar]
    columnas_lista = ["household_key", "PRODUCT_ID", "posicion", "origen", "puntaje", "explicacion", "DEPARTMENT", "producto"]
    historial = historial_df_de(hogar)
    if historial.empty:
        datos["lista_final"] = pd.DataFrame(columns=columnas_lista)
        datos["filtrados"] = pd.DataFrame()
        datos["generado_para_n_canastas"] = 0
        return

    catalogo = cargar_catalogo()
    catalogo_indexado = catalogo.set_index("PRODUCT_ID")
    propios = set(historial["PRODUCT_ID"])

    # --- repeat ---
    def explicar_repeat(fila) -> str:
        frecuencia = int(fila["frecuencia_en_canastas"])
        distancia = int(fila["distancia_desde_ultima_compra"])
        if distancia == 0:
            cuando = "en su última compra"
        elif distancia == 1:
            cuando = "en su penúltima compra"
        else:
            cuando = f"hace {distancia} compras"
        canastas = "1 canasta" if frecuencia == 1 else f"{frecuencia} canastas"
        return f"Lo compró en {canastas}; la última vez {cuando}."

    perfil = construir_perfil_repeat(historial)
    repeat = perfil.rename(columns={"puntaje_repeat": "puntaje"}).sort_values("puntaje", ascending=False).reset_index(drop=True)
    repeat["explicacion"] = repeat.apply(explicar_repeat, axis=1)
    repeat["posicion"] = range(1, len(repeat) + 1)
    repeat["tipo_recomendacion"] = "repeat"
    repeat = repeat[["household_key", "PRODUCT_ID", "posicion", "tipo_recomendacion", "puntaje", "explicacion"]]

    # --- explore: coocurrencia sobre los productos propios ---
    asociaciones = cargar_asociaciones()
    candidatos = asociaciones.loc[
        asociaciones["PRODUCT_ID_ORIGEN"].isin(propios) & ~asociaciones["PRODUCT_ID_DESTINO"].isin(propios)
    ].copy()
    if not candidatos.empty:
        candidatos = candidatos.sort_values("puntaje_asociacion", ascending=False).drop_duplicates("PRODUCT_ID_DESTINO", keep="first")
        maximo = candidatos["puntaje_asociacion"].max() or 1.0
        candidatos["puntaje"] = candidatos["puntaje_asociacion"] / maximo
        candidatos["PRODUCT_ID"] = candidatos["PRODUCT_ID_DESTINO"]
        candidatos["explicacion"] = candidatos.apply(
            lambda f: (
                f"Nuevo para el hogar. Los hogares que compran «"
                f"{catalogo_indexado['producto'].get(f['PRODUCT_ID_ORIGEN'], 'un producto de su historial')}», "
                f"como este, también lo eligen (asociación {f['puntaje']:.2f})."
            ),
            axis=1,
        )
        candidatos = candidatos[["PRODUCT_ID", "puntaje", "explicacion"]]

    pool_necesario = 20
    faltan = pool_necesario - len(candidatos)
    if faltan > 0:
        popularidad = cargar_popularidad()
        vistos = propios | (set(candidatos["PRODUCT_ID"]) if not candidatos.empty else set())
        extra = popularidad.loc[~popularidad["PRODUCT_ID"].isin(vistos)].sort_values(
            "puntaje_popularidad", ascending=False
        ).head(faltan).copy()
        extra["puntaje"] = extra["puntaje_popularidad"] * 0.3
        extra["explicacion"] = extra["canastas_con_producto"].apply(
            lambda n: f"Nuevo para el hogar. Es uno de los productos más comprados: aparece en {int(n)} canastas."
        )
        candidatos = pd.concat([candidatos, extra[["PRODUCT_ID", "puntaje", "explicacion"]]], ignore_index=True)

    explore = candidatos.sort_values("puntaje", ascending=False).head(pool_necesario).reset_index(drop=True)
    explore["household_key"] = hogar
    explore["posicion"] = range(1, len(explore) + 1)
    explore["tipo_recomendacion"] = "explore"
    explore = explore[["household_key", "PRODUCT_ID", "posicion", "tipo_recomendacion", "puntaje", "explicacion"]]

    candidatos_totales = pd.concat([repeat, explore], ignore_index=True)

    # --- reglas de negocio reales del proyecto ---
    reglas = cargar_reglas()
    aceptados, filtrados = aplicar_reglas(
        candidatos_totales, reglas, catalogo, soporte_por_producto=soporte_por_producto_dict(), historial=historial,
    )

    # --- mezcla final: misma proporción repeat/explore que el modelo congelado ---
    mezcla = demo["metadatos"]["configuracion"]["mezcla"]
    cupo_repeat = mezcla["posiciones_repeat"]
    k = mezcla["posiciones_repeat"] + mezcla["posiciones_explore"]
    rep_acept = aceptados.loc[aceptados["tipo_recomendacion"] == "repeat"].sort_values("posicion")
    exp_acept = aceptados.loc[aceptados["tipo_recomendacion"] == "explore"].sort_values("posicion")
    elegidos_rep = rep_acept.head(cupo_repeat)
    elegidos_exp = exp_acept.head(k - len(elegidos_rep))
    faltan_final = k - len(elegidos_rep) - len(elegidos_exp)
    if faltan_final > 0:
        elegidos_rep = pd.concat([elegidos_rep, rep_acept.iloc[len(elegidos_rep): len(elegidos_rep) + faltan_final]])
    final = pd.concat([elegidos_rep, elegidos_exp], ignore_index=True)
    final["origen"] = final["tipo_recomendacion"]
    final["posicion"] = range(1, len(final) + 1)
    final["DEPARTMENT"] = final["PRODUCT_ID"].map(catalogo_indexado["DEPARTMENT"])
    final["producto"] = final["PRODUCT_ID"].map(catalogo_indexado["producto"])
    final = final[columnas_lista]

    datos["lista_final"] = final
    datos["filtrados"] = filtrados
    datos["generado_para_n_canastas"] = len(datos["canastas"])


def asegurar_recomendaciones_simuladas(demo: dict, hogar: int) -> None:
    datos = clientes_simulados()[hogar]
    if datos.get("generado_para_n_canastas") != len(datos["canastas"]):
        generar_recomendaciones_simuladas(demo, hogar)


def lista_final_de(demo: dict, hogar: int) -> pd.DataFrame:
    """Lista final del hogar, sea un cliente real o uno simulado."""

    if hogar in clientes_simulados():
        asegurar_recomendaciones_simuladas(demo, hogar)
        return clientes_simulados()[hogar]["lista_final"]
    return demo["lista_final"].loc[demo["lista_final"]["household_key"] == hogar]


def pendientes(lista: pd.DataFrame, hogar: int) -> pd.DataFrame:
    """Candidatos que el cliente todavía no reclamó ni descartó."""

    vistos = set(billetera_de(hogar)) | descartes_de(hogar)
    return lista.loc[~lista["PRODUCT_ID"].isin(vistos)]


def ids_pendientes_de(demo: dict, hogar: int) -> frozenset:
    """IDs con cupón pendiente para este hogar (sin reclamar ni descartar)."""

    lista = lista_final_de(demo, hogar)
    vistos = set(billetera_de(hogar)) | descartes_de(hogar)
    return frozenset(pid for pid in lista["PRODUCT_ID"] if pid not in vistos)


def buscar_cupon_pendiente(demo: dict, hogar: int, producto_id: int) -> dict | None:
    """Si el producto está entre los recomendados y no fue reclamado/descartado, su cupón."""

    if producto_id in billetera_de(hogar) or producto_id in descartes_de(hogar):
        return None
    lista = lista_final_de(demo, hogar)
    coincidencia = lista.loc[lista["PRODUCT_ID"] == producto_id]
    if coincidencia.empty:
        return None
    fila = coincidencia.iloc[0]
    return {"descuento": calcular_descuento(fila["puntaje"], fila["origen"]), "origen": fila["origen"]}


def tarjeta_cupon(fila, tipo: str, hogar: int) -> None:
    """Una tarjeta de cupón con descuento, explicación y botones de acción."""

    producto_id = int(fila.PRODUCT_ID)
    descuento = calcular_descuento(getattr(fila, "puntaje", None), tipo)
    precio = precio_de(producto_id)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="coupon-banner">
                {imagen_departamento_html(fila.DEPARTMENT)}
                <div class="coupon-discount-badge">{descuento}% OFF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{fila.producto}**")
        st.caption(str(fila.DEPARTMENT).title())
        if precio is not None:
            st.caption(f"Ahorrás \\${precio * descuento / 100:,.2f} (precio de referencia \\${precio:,.2f})")
        st.write(voz_cliente(fila.explicacion))
        c1, c2 = st.columns(2)
        if c1.button(f"{icono_md('confirmation_number')} Aplicar cupón", key=f"reclamar-{hogar}-{producto_id}", width="stretch", type="primary"):
            billetera_de(hogar)[producto_id] = {
                "producto": fila.producto, "departamento": fila.DEPARTMENT, "precio_referencia": precio,
                "descuento": descuento, "origen": tipo, "estado": "reclamado",
            }
            registrar_feedback(hogar, producto_id, tipo, "util")
            st.rerun()
        if c2.button("No, gracias", key=f"descartar-{hogar}-{producto_id}", width="stretch"):
            descartes_de(hogar).add(producto_id)
            registrar_feedback(hogar, producto_id, tipo, "no_util")
            st.rerun()


def tarjeta_bienvenida(fila, hogar: int) -> None:
    """Tarjeta de respaldo (sin historial): productos populares elegibles."""

    producto_id = int(fila.PRODUCT_ID)
    precio = precio_de(producto_id)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="coupon-banner">
                <div class="coupon-banner-fallback">{icono_html('celebration', 40)}</div>
                <div class="coupon-discount-badge bienvenida">10% OFF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{fila.producto}**")
        st.caption(str(fila.DEPARTMENT).title())
        if precio is not None:
            st.caption(f"Ahorrás \\${precio * 0.10:,.2f} (precio de referencia \\${precio:,.2f})")
        st.write(f"Es uno de los productos favoritos de nuestros clientes: lo eligieron {int(fila.hogares_que_compraron):,} hogares.".replace(",", "."))
        c1, c2 = st.columns(2)
        if c1.button(f"{icono_md('confirmation_number')} Aplicar cupón", key=f"reclamar-bv-{hogar}-{producto_id}", width="stretch", type="primary"):
            billetera_de(hogar)[producto_id] = {
                "producto": fila.producto, "departamento": fila.DEPARTMENT, "precio_referencia": precio,
                "descuento": 10, "origen": "bienvenida", "estado": "reclamado",
            }
            registrar_feedback(hogar, producto_id, "bienvenida", "util")
            st.rerun()
        if c2.button("No, gracias", key=f"descartar-bv-{hogar}-{producto_id}", width="stretch"):
            descartes_de(hogar).add(producto_id)
            registrar_feedback(hogar, producto_id, "bienvenida", "no_util")
            st.rerun()


def mostrar_grilla(filas: pd.DataFrame, tipo: str, hogar: int) -> None:
    columnas = st.columns(3)
    for i, fila in enumerate(filas.itertuples()):
        with columnas[i % 3]:
            if tipo == "bienvenida":
                tarjeta_bienvenida(fila, hogar)
            else:
                tarjeta_cupon(fila, tipo, hogar)


def tarjeta_catalogo(demo: dict, fila, hogar: int, tipo: str) -> None:
    """Una tarjeta de producto del catálogo general, con botón para agregar al carrito."""

    producto_id = int(fila.PRODUCT_ID)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="coupon-banner">
                {imagen_departamento_html(fila.DEPARTMENT)}
                <div class="coupon-discount-badge">${fila.precio_referencia:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{fila.producto}**")
        st.caption(str(fila.DEPARTMENT).title())
        sugerencia = buscar_cupon_pendiente(demo, hogar, producto_id) if tipo in ("real", "simulado") else None
        if sugerencia:
            st.success(f"{icono_md('confirmation_number')} Tenés un cupón de {sugerencia['descuento']}% para este producto")
        if st.button(f"{icono_md('add_shopping_cart')} Agregar al carrito", key=f"agregar-{hogar}-{producto_id}", width="stretch"):
            carrito = carrito_de(hogar)
            if producto_id in carrito:
                carrito[producto_id]["cantidad"] += 1
            else:
                carrito[producto_id] = {
                    "producto": fila.producto, "DEPARTMENT": fila.DEPARTMENT,
                    "precio_referencia": float(fila.precio_referencia), "cantidad": 1,
                }
            st.toast(f"Agregado al carrito: {fila.producto}")
            st.rerun()


def seccion_catalogo(demo: dict, hogar: int, tipo: str) -> None:
    catalogo = catalogo_deduplicado(ids_pendientes_de(demo, hogar))
    if catalogo.empty:
        st.info(
            "No hay catálogo de precios disponible todavía. Ejecutá "
            "`python -m src.calcular_precios_catalogo` y recargá la página."
        )
        return
    st.caption(
        "Explorá productos y agregalos a tu carrito. Si alguno tiene un cupón disponible para vos, "
        "te lo mostramos acá mismo."
    )
    busqueda = st.text_input(
        "Buscar producto", placeholder="Buscar producto, ej. leche, pan, café…", label_visibility="collapsed"
    )

    disponibles = set(catalogo["DEPARTMENT"].unique())
    categorias_pill = [("Todos", icono_md("apps"))] + [
        (depto, icono_departamento_md(depto)) for depto in CATEGORIAS_DESTACADAS if depto in disponibles
    ]
    depto_activo = st.session_state.get("catalogo_depto", "Todos")
    columnas_pill = st.columns(len(categorias_pill))
    for columna, (depto, icono) in zip(columnas_pill, categorias_pill):
        etiqueta = "Todos" if depto == "Todos" else depto.title()
        if columna.button(
            f"{icono} {etiqueta}", key=f"pill-depto-{depto}", width="stretch",
            type="primary" if depto_activo == depto else "secondary",
        ):
            st.session_state["catalogo_depto"] = depto
            depto_activo = depto
            st.rerun()

    filtrado = catalogo
    if depto_activo != "Todos":
        filtrado = filtrado.loc[filtrado["DEPARTMENT"] == depto_activo]
    if busqueda:
        filtrado = filtrado.loc[filtrado["producto"].str.contains(busqueda, case=False, na=False)]
    total = len(filtrado)
    if total == 0:
        st.warning("No encontramos productos con ese filtro.")
        return
    filtrado = filtrado.sort_values("producto").head(60)
    st.caption(f"Mostrando {len(filtrado)} de {total:,} resultados.".replace(",", "."))

    columnas = st.columns(3)
    for i, fila in enumerate(filtrado.itertuples()):
        with columnas[i % 3]:
            tarjeta_catalogo(demo, fila, hogar, tipo)


def seccion_carrito(demo: dict, hogar: int) -> None:
    mensaje = st.session_state.pop("mensaje_compra", None)
    if mensaje:
        st.success(mensaje)

    carrito = carrito_de(hogar)
    if not carrito:
        st.info(f"Tu carrito está vacío. Mirá la pestaña **{icono_md('storefront')} Catálogo** para agregar productos.")
        return

    billetera = billetera_de(hogar)
    subtotal = 0.0
    descuento_total = 0.0
    for producto_id, item in list(carrito.items()):
        with st.container(border=True):
            cols = st.columns([4, 1, 2, 1, 1])
            cols[0].markdown(f"{icono_departamento_md(item['DEPARTMENT'])} **{item['producto']}**")
            cols[0].caption(str(item["DEPARTMENT"]).title())
            nueva_cantidad = cols[1].number_input(
                "Cantidad", min_value=0, value=int(item["cantidad"]), step=1,
                key=f"cantidad-{hogar}-{producto_id}", label_visibility="collapsed",
            )
            if nueva_cantidad == 0:
                del carrito[producto_id]
                st.rerun()
            item["cantidad"] = int(nueva_cantidad)
            importe = item["precio_referencia"] * item["cantidad"]

            cupon = billetera.get(producto_id)
            sugerencia = None
            if cupon:
                descuento_linea = importe * cupon["descuento"] / 100
                cols[2].markdown(f"~~\\${importe:,.2f}~~ **\\${importe - descuento_linea:,.2f}**")
                cols[2].caption(f"{icono_md('sell')} Cupón -{cupon['descuento']}% aplicado")
                subtotal += importe
                descuento_total += descuento_linea
            else:
                cols[2].markdown(f"**${importe:,.2f}**")
                subtotal += importe
                sugerencia = buscar_cupon_pendiente(demo, hogar, producto_id)
                if sugerencia and cols[3].button(
                    icono_md("confirmation_number"), key=f"reclamar-carrito-{hogar}-{producto_id}",
                    help=f"Reclamar cupón de {sugerencia['descuento']}% para este producto",
                ):
                    billetera[producto_id] = {
                        "producto": item["producto"], "departamento": item["DEPARTMENT"],
                        "precio_referencia": item["precio_referencia"],
                        "descuento": sugerencia["descuento"], "origen": sugerencia["origen"], "estado": "reclamado",
                    }
                    registrar_feedback(hogar, producto_id, sugerencia["origen"], "util")
                    st.rerun()
            if cols[4].button(icono_md("delete"), key=f"quitar-carrito-{hogar}-{producto_id}", help="Quitar del carrito"):
                del carrito[producto_id]
                st.rerun()

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Subtotal", f"${subtotal:,.2f}")
    r2.metric("Descuento por cupones", f"-${descuento_total:,.2f}")
    r3.metric("Total a pagar", f"${subtotal - descuento_total:,.2f}")
    if st.button(f"{icono_md('check_circle')} Finalizar compra", type="primary", width="stretch"):
        for producto_id in list(carrito):
            cupon = billetera.get(producto_id)
            if cupon:
                cupon["estado"] = "canjeado"
                registrar_feedback(hogar, producto_id, cupon["origen"], "canjeado")
        carrito.clear()
        st.session_state["mensaje_compra"] = (
            f"¡Compra realizada! Pagaste \\${subtotal - descuento_total:,.2f} y ahorraste "
            f"\\${descuento_total:,.2f} con tus cupones."
        )
        st.balloons()
        st.rerun()


def pantalla_login(demo: dict) -> None:
    """Pantalla inicial: elegir un cliente real de la base o crear uno simulado."""

    imagen_hero = imagen_hero_html()
    insignia_hero = imagen_hero if imagen_hero else f'<div class="hero-badge">{icono_html("shopping_cart", 64)}</div>'
    st.markdown(
        f"""
        <div class="hero">
            {insignia_hero}
            <h1>Ahorrá más <span>en tus compras</span></h1>
            <p>SuperCoop arma, para cada cliente, los cupones que más le sirven a partir de su propio historial de compras — en tiempo real.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="trust-row">
            <div class="trust-item"><span class="trust-icon">{icono_html('target')}</span> Cupones a medida según tu historial real</div>
            <div class="trust-item"><span class="trust-icon">{icono_html('balance')}</span> Reglas de negocio reales, sin tabaco ni combustible</div>
            <div class="trust-item"><span class="trust-icon">{icono_html('science')}</span> Probá el sistema con un cliente inventado, en vivo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    pestana_real, pestana_simulado = st.tabs([
        f"{icono_md('person')} Cliente real (base de datos)", f"{icono_md('science')} Cliente simulado",
    ])

    with pestana_real:
        st.caption("Buscá cualquier cliente real de la base por número de socio, o probá un caso de ejemplo.")
        opciones = opciones_clientes_reales(demo["resumen_hogares"])
        elegido = st.selectbox(
            "Buscar cliente por número de socio", options=list(opciones), index=None,
            placeholder="Escribí un número de socio para buscar…", label_visibility="collapsed",
        )
        if st.button("Ingresar", key="ingresar-real", type="primary", width="stretch", disabled=elegido is None):
            st.session_state.cliente = opciones[elegido]
            st.rerun()
        st.markdown("**Demos rápidas:**")
        columnas = st.columns(3)
        for i, ejemplo in enumerate(demo["ejemplos"]):
            with columnas[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{icono_caso(ejemplo['caso'])} N.º {ejemplo['household_key']}**")
                    st.caption(ejemplo["caso"])
                    if st.button("Ingresar", key=f"login-{ejemplo['household_key']}", width="stretch"):
                        st.session_state.cliente = ejemplo["household_key"]
                        st.rerun()

    with pestana_simulado:
        st.caption(
            "Creá un cliente desde cero: armás su historial de compras vos mismo y la app genera "
            "sus cupones en vivo, con el mismo motor de reglas de negocio que el sistema real."
        )
        simulados = clientes_simulados()
        if simulados:
            st.markdown("**Clientes simulados ya creados:**")
            columnas = st.columns(3)
            for i, (id_simulado, info) in enumerate(simulados.items()):
                with columnas[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{icono_md('science')} {info['nombre']}**")
                        st.caption(f"{len(info['canastas'])} canastas registradas")
                        if st.button("Ingresar", key=f"login-sim-{id_simulado}", width="stretch"):
                            st.session_state.cliente = id_simulado
                            st.rerun()
            st.divider()
        nombre_nuevo = st.text_input("Nombre para el nuevo cliente simulado", placeholder="Ej. Cliente de prueba 1")
        if st.button(f"{icono_md('person_add')} Crear cliente simulado", type="primary", width="stretch"):
            nuevo = crear_cliente_simulado(nombre_nuevo.strip())
            st.session_state.cliente = nuevo
            st.rerun()


def seccion_cupones_recomendados(demo: dict, hogar: int) -> None:
    lista = lista_final_de(demo, hogar)
    if lista.empty:
        st.info(f"Todavía no armaste tu historial de compras. Mirá la pestaña **{icono_md('science')} Historial simulado** para empezar.")
        return
    lista = lista.sort_values("posicion")
    hubo_contenido = False
    for origen in ("repeat", "explore"):
        candidatos = pendientes(lista.loc[lista["origen"] == origen], hogar)
        if candidatos.empty:
            continue
        hubo_contenido = True
        titulo, tipo = TITULOS_ORIGEN[origen]
        st.markdown(f"### {titulo}")
        mostrar_grilla(candidatos, tipo, hogar)
    if not hubo_contenido:
        st.success(f"{icono_md('celebration')} Ya revisaste todos tus cupones de hoy. Mirá la pestaña **{icono_md('account_balance_wallet')} Mis cupones** para canjearlos.")


def seccion_mis_cupones(hogar: int) -> None:
    billetera = billetera_de(hogar)
    if not billetera:
        st.info(f"Todavía no reclamaste ningún cupón. Mirá la pestaña **{icono_md('confirmation_number')} Cupones para vos**.")
        return
    st.caption(
        "Mostralos en caja para aplicar el descuento. Simulación: en producción esto se "
        "integraría con el sistema de cajas del supermercado."
    )
    for producto_id, info in list(billetera.items()):
        with st.container(border=True):
            cols = st.columns([4, 2, 1])
            clase = "coupon-discount-badge bienvenida" if info["origen"] == "bienvenida" else "coupon-discount-badge"
            cols[0].markdown(
                f"""
                <div class="coupon-banner">
                    {imagen_departamento_html(info['departamento'])}
                    <div class="{clase}">{info['descuento']}% OFF</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols[0].markdown(f"**{info['producto']}**")
            cols[0].caption(str(info["departamento"]).title())
            precio = info.get("precio_referencia")
            if precio is not None:
                cols[0].caption(f"Ahorrás \\${precio * info['descuento'] / 100:,.2f} (precio de referencia \\${precio:,.2f})")
            if info["estado"] == "reclamado":
                if cols[1].button(f"{icono_md('check_circle')} Canjear en caja", key=f"canjear-{hogar}-{producto_id}", width="stretch"):
                    info["estado"] = "canjeado"
                    registrar_feedback(hogar, producto_id, info["origen"], "canjeado")
                    st.rerun()
            else:
                cols[1].success(f"Canjeado {icono_md('check_circle')}")
            if cols[2].button(icono_md("delete"), key=f"quitar-{hogar}-{producto_id}", help="Quitar de la billetera"):
                del billetera[producto_id]
                st.rerun()


def seccion_mis_compras(demo: dict, hogar: int) -> None:
    st.markdown("#### Lo que más comprás")
    frecuentes = demo["productos_frecuentes"].query("household_key == @hogar")
    st.dataframe(
        frecuentes[["producto", "DEPARTMENT", "canastas", "ultimo_dia"]].rename(
            columns={"producto": "Producto", "DEPARTMENT": "Departamento", "canastas": "Veces compradas", "ultimo_dia": "Último día"}
        ),
        hide_index=True, width="stretch",
    )
    ultima = demo["ultima_canasta"].query("household_key == @hogar")
    st.markdown(f"#### Tu última compra (día {int(ultima['DAY'].iloc[0])})")
    st.dataframe(
        ultima[["producto", "DEPARTMENT", "QUANTITY", "SALES_VALUE"]].rename(
            columns={"producto": "Producto", "DEPARTMENT": "Departamento", "QUANTITY": "Cantidad", "SALES_VALUE": "Importe"}
        ),
        hide_index=True, width="stretch",
    )


def seccion_historial_simulado(demo: dict, hogar: int) -> None:
    """Armar (y seguir ampliando) el historial de un cliente simulado."""

    datos = clientes_simulados()[hogar]
    st.caption(
        "Armá un historial de compras para este cliente y generá sus cupones en vivo: repeat usa la "
        "misma fórmula del sistema real (frecuencia + recencia); explore usa el modelo de coocurrencia "
        "ya evaluado en el proyecto (ALS no puede anotar un cliente que no existía al entrenar)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Canastas registradas", len(datos["canastas"]))
    c2.metric("Productos distintos", len({pid for canasta in datos["canastas"] for pid in canasta}))

    if st.button(f"{icono_md('casino')} Generar historial de ejemplo (demo rápida)", width="stretch"):
        generar_historial_automatico(hogar)
        st.rerun()

    st.divider()
    st.markdown(f"#### {icono_md('add_circle')} Agregar una canasta (compra pasada)")
    busqueda = st.text_input(
        "Buscar producto para agregar", placeholder="Buscar producto, ej. leche, pan, café…",
        key=f"buscar-canasta-{hogar}",
    )
    if busqueda:
        catalogo = catalogo_deduplicado(frozenset())
        resultados = catalogo.loc[catalogo["producto"].str.contains(busqueda, case=False, na=False)].head(8)
        if resultados.empty:
            st.caption("No encontramos productos con ese filtro.")
        for fila in resultados.itertuples():
            cols = st.columns([5, 1])
            cols[0].write(f"{icono_departamento_md(fila.DEPARTMENT)} {fila.producto}")
            if cols[1].button(icono_md("add"), key=f"agregar-canasta-{hogar}-{fila.PRODUCT_ID}"):
                datos["canasta_en_construccion"][int(fila.PRODUCT_ID)] = {
                    "producto": fila.producto, "DEPARTMENT": fila.DEPARTMENT,
                }
                st.rerun()

    if datos["canasta_en_construccion"]:
        st.markdown(f"**{icono_md('shopping_basket')} Canasta en construcción:**")
        for producto_id, info in list(datos["canasta_en_construccion"].items()):
            cols = st.columns([5, 1])
            cols[0].write(info["producto"])
            if cols[1].button(icono_md("delete"), key=f"quitar-canasta-{hogar}-{producto_id}"):
                del datos["canasta_en_construccion"][producto_id]
                st.rerun()
        if st.button(f"{icono_md('check_circle')} Registrar esta canasta como compra pasada", type="primary", width="stretch"):
            registrar_canasta(hogar)
            st.rerun()

    if datos["canastas"]:
        st.divider()
        st.markdown(f"#### {icono_md('receipt_long')} Historial registrado")
        nombres = cargar_catalogo().set_index("PRODUCT_ID")["producto"]
        filas = [
            {"Canasta": indice + 1, "Productos": ", ".join(nombres.get(pid, f"#{pid}") for pid in canasta)}
            for indice, canasta in enumerate(datos["canastas"])
        ]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


def pagina_cliente_desconocido(demo: dict, hogar: int) -> None:
    """Respaldo para un número de socio sin historial: productos populares."""

    st.warning(
        f"No encontramos historial de compras para el socio N.º {hogar}. Puede ser tu primera "
        "compra. Mientras tanto, te dejamos una selección de nuestros productos más elegidos."
    )
    respaldo = pendientes(demo["respaldo_popular"], hogar)
    if respaldo.empty:
        st.success(f"{icono_md('celebration')} Ya revisaste todos los cupones de bienvenida disponibles.")
        return
    mostrar_grilla(respaldo, "bienvenida", hogar)


def tabla_filtrados(filtrados: pd.DataFrame, catalogo_indexado: pd.DataFrame) -> None:
    """Tabla de candidatos descartados por reglas (reutilizable para real y simulado)."""

    if filtrados is None or filtrados.empty:
        st.success("Ningún candidato de este hogar fue descartado por las reglas.")
        return
    datos = filtrados.copy()
    if "producto" not in datos.columns:
        datos["producto"] = datos["PRODUCT_ID"].map(catalogo_indexado["producto"])
    if "DEPARTMENT" not in datos.columns:
        datos["DEPARTMENT"] = datos["PRODUCT_ID"].map(catalogo_indexado["DEPARTMENT"])
    st.info(f"{len(datos)} candidatos descartados antes de armar la lista. El modelo los consideraba relevantes, pero no son elegibles para cupón.")
    tabla = datos[["tipo_recomendacion", "posicion", "producto", "DEPARTMENT", "motivo_filtro"]].rename(
        columns={"tipo_recomendacion": "Lista", "posicion": "Posición original", "producto": "Producto",
                 "DEPARTMENT": "Departamento", "motivo_filtro": "Motivo"}
    )
    tabla["Lista"] = tabla["Lista"].map({"repeat": "Repeat", "explore_als": "Explore", "explore": "Explore"}).fillna(tabla["Lista"])
    tabla["Posición original"] = tabla["Posición original"].astype(str)
    st.table(tabla, hide_index=True)


def pagina_analista(demo: dict, hogar: int, tipo: str) -> None:
    """Panel técnico: puntajes del modelo y candidatos descartados por reglas."""

    st.caption(
        "Panel para el equipo comercial: puntajes del modelo detrás de cada cupón y "
        "candidatos que las reglas de negocio descartaron antes de llegar al cliente."
    )
    catalogo_indexado = cargar_catalogo().set_index("PRODUCT_ID")

    if tipo == "real":
        resumen = demo["resumen_hogares"].set_index("household_key").loc[hogar]
        filtrados = demo["filtrados"].query("household_key == @hogar")
        metricas = st.columns(5)
        metricas[0].metric("Canastas", int(resumen["canastas"]))
        metricas[1].metric("Productos distintos", int(resumen["productos_distintos"]))
        metricas[2].metric("Gasto total", f"${resumen['gasto_total']:,.2f}")
        metricas[3].metric("Gasto medio por canasta", f"${resumen['gasto_medio_por_canasta']:,.2f}")
        metricas[4].metric("Días con compras", f"{int(resumen['primer_dia'])}–{int(resumen['ultimo_dia'])}")
        st.write(f"**Departamentos principales:** {resumen['departamentos_principales']}")
    else:
        historial = historial_df_de(hogar)
        filtrados = clientes_simulados()[hogar].get("filtrados", pd.DataFrame())
        precios = cargar_precios()
        metricas = st.columns(3)
        metricas[0].metric("Canastas", historial["BASKET_ID"].nunique() if not historial.empty else 0)
        metricas[1].metric("Productos distintos", historial["PRODUCT_ID"].nunique() if not historial.empty else 0)
        gasto = sum(precios.get(pid, 0.0) or 0.0 for pid in historial["PRODUCT_ID"]) if not historial.empty else 0.0
        metricas[2].metric("Gasto estimado", f"${gasto:,.2f}")
        st.caption("Cliente simulado: estas métricas vienen del historial armado en la app, no del dataset.")

    lista = lista_final_de(demo, hogar)
    if lista.empty:
        st.info("Todavía no hay recomendaciones para mostrar.")
        return
    lista = lista.sort_values("posicion")
    repeat = lista.loc[lista["origen"] == "repeat"]
    explore = lista.loc[lista["origen"] == "explore"]
    st.markdown(f"#### Repeat — productos que ya compra ({len(repeat)})")
    st.caption("Puntaje = 0,7 × frecuencia normalizada + 0,3 × recencia.")
    tabla_de_productos(repeat)
    st.markdown(f"#### Explore — productos nuevos para el hogar ({len(explore)})")
    if tipo == "real":
        st.caption("Puntaje ALS: afinidad estimada a partir de hogares con compras parecidas.")
    else:
        st.caption("Puntaje de coocurrencia: asociación (lift) con los productos del historial, normalizada 0-1.")
    tabla_de_productos(explore)

    st.markdown(f"#### {icono_md('block')} Filtrados por reglas")
    tabla_filtrados(filtrados, catalogo_indexado)


def tabla_de_productos(datos: pd.DataFrame) -> None:
    """Muestra una tabla de recomendaciones con nombres de columna legibles (vista analista)."""

    tabla = datos[list(COLUMNAS_LISTA)].rename(columns=COLUMNAS_LISTA)
    tabla["Puntaje"] = tabla["Puntaje"].map("{:.3f}".format)
    tabla["#"] = tabla["#"].astype(str)
    st.table(tabla, hide_index=True)


def pagina_modelo(demo: dict) -> None:
    """Resume cómo funciona el sistema y sus resultados medidos."""

    meta = demo["metadatos"]
    configuracion = meta["configuracion"]
    prueba = meta["metricas_prueba"]
    st.subheader("Cómo funciona")
    st.markdown(
        f"""
1. **Datos:** compras anonimizadas de {format(meta['hogares'], ',').replace(',', '.')} hogares (dunnhumby — The Complete Journey).
2. **Repeat:** frecuencia + recencia de los productos que el hogar ya compra.
3. **Explore:** ALS implícito (filtrado colaborativo) con {configuracion['parametros_als']['factores']} factores para clientes reales;
   coocurrencia (lift) para clientes simulados, porque ALS no puede anotar un hogar que no existía al entrenar.
4. **Reglas de negocio:** se excluyen productos no elegibles para cupón (por ejemplo, combustible) y se registra el motivo.
5. **Lista final:** {configuracion['mezcla']['posiciones_repeat']} repeat + {configuracion['mezcla']['posiciones_explore']} explore = {configuracion['k']} candidatos.

Las listas de esta demo se entrenaron con todo el historial disponible y recomiendan para la próxima compra.
Los descuentos (% OFF) son **simulados**: el proyecto no cuenta con datos de margen por producto.
"""
    )
    st.subheader("Resultados medidos (evaluación única sobre prueba, sin reglas)")
    final = prueba["sistema_final"]["lista_combinada"]
    repeat = prueba["componentes"]["repeat_frecuencia_recencia"][1]
    explore = prueba["componentes"]["explore_als"][1]
    popularidad = prueba["referencias_no_seleccionadas"]["explore_popularidad"][1]
    columnas = st.columns(4)
    columnas[0].metric("Hit Rate@10 lista final", f"{final['hit_rate']:.1%}")
    columnas[1].metric("Hit Rate@10 repeat", f"{repeat['hit_rate']:.1%}")
    columnas[2].metric("Hit Rate@10 explore (ALS)", f"{explore['hit_rate']:.1%}",
                       delta=f"{100 * (explore['hit_rate'] - popularidad['hit_rate']):+.1f} pp vs. popularidad")
    columnas[3].metric("Hogares evaluados", f"{final['hogares_evaluables']:,}".replace(",", "."))
    st.caption(
        "Hit Rate@10: proporción de hogares que compraron al menos uno de los productos sugeridos en su "
        "siguiente canasta. Mide coincidencia con compras futuras, no el efecto causal de un cupón; "
        "eso requeriría una prueba A/B con clientes reales."
    )


def main() -> None:
    """Punto de entrada de la app."""

    aplicar_estilos()
    demo = cargar_demo()
    if not demo:
        st.error("Faltan los datos de la demo. Ejecutar primero: `python -m src.preparar_demo`.")
        st.stop()

    if "cliente" not in st.session_state:
        st.session_state.cliente = None

    if st.session_state.cliente is None:
        pantalla_login(demo)
        st.stop()

    hogar = st.session_state.cliente
    tipo = tipo_cliente(demo, hogar)
    carrito = carrito_de(hogar)
    etiqueta = clientes_simulados()[hogar]["nombre"] if tipo == "simulado" else f"N.º {hogar}"

    with st.sidebar:
        st.markdown(f"##### {icono_md('badge')} Tarjeta de socio")
        st.markdown(f"### {etiqueta}")
        if tipo == "simulado":
            st.caption("Cliente simulado (demo en vivo)")
        if st.button(f"{icono_md('sync_alt')} Cambiar de cliente", width="stretch"):
            st.session_state.cliente = None
            st.rerun()
        st.divider()
        vista_analista = st.toggle(
            f"{icono_md('tune')} Vista analista", value=False,
            help="Panel técnico: puntajes del modelo y reglas de negocio aplicadas.",
        )
        if carrito:
            st.divider()
            st.caption(f"{icono_md('shopping_basket')} {sum(item['cantidad'] for item in carrito.values())} producto(s) en el carrito")

    unidades_carrito = sum(item["cantidad"] for item in carrito.values())
    st.markdown(
        f"""
        <div class="navbar">
            <div class="navbar-brand">
                <div class="navbar-logo">{icono_html('shopping_cart', 22)}</div>
                <div class="navbar-brand-text">Super<span>Coop</span></div>
            </div>
            <div class="navbar-right">
                <span class="chip-cliente">{icono_html('badge', 16)} {etiqueta}</span>
                <span class="chip-carrito">{icono_html('shopping_basket', 16)} {unidades_carrito}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if tipo == "desconocido":
        pagina_cliente_desconocido(demo, hogar)
        if vista_analista:
            st.divider()
            st.info(f"{icono_md('tune')} Vista analista: este socio no tiene historial, no hay datos técnicos para mostrar.")
        return

    nombre_tab_historial = f"{icono_md('science')} Historial simulado" if tipo == "simulado" else f"{icono_md('receipt_long')} Mis compras"
    nombres_tabs = [
        f"{icono_md('confirmation_number')} Cupones para vos", f"{icono_md('storefront')} Catálogo",
        f"{icono_md('shopping_basket')} Mi carrito", f"{icono_md('account_balance_wallet')} Mis cupones",
        nombre_tab_historial,
    ]
    if vista_analista:
        nombres_tabs.append(f"{icono_md('tune')} Vista analista")
    nombres_tabs.append(f"{icono_md('info')} Cómo funciona")
    tabs = st.tabs(nombres_tabs)

    with tabs[0]:
        seccion_cupones_recomendados(demo, hogar)
    with tabs[1]:
        seccion_catalogo(demo, hogar, tipo)
    with tabs[2]:
        seccion_carrito(demo, hogar)
    with tabs[3]:
        seccion_mis_cupones(hogar)
    with tabs[4]:
        if tipo == "simulado":
            seccion_historial_simulado(demo, hogar)
        else:
            seccion_mis_compras(demo, hogar)
    siguiente = 5
    if vista_analista:
        with tabs[siguiente]:
            pagina_analista(demo, hogar, tipo)
        siguiente += 1
    with tabs[siguiente]:
        pagina_modelo(demo)


main()
