"""
EDA inicial de dunnhumby - The Complete Journey.

Corré desde la raíz del proyecto:
    python src/01_eda.py

Genera:
    reports/eda_summary.md          -> métricas clave en texto
    reports/fig_basket_size.png     -> distribución de tamaño de canasta
    reports/fig_baskets_per_hh.png  -> distribución de nº de canastas por cliente
    reports/fig_txns_per_week.png   -> evolución temporal de transacciones
"""
import matplotlib
matplotlib.use("Agg")  # no requiere display, solo guardamos figuras a disco
import matplotlib.pyplot as plt
import pandas as pd

from data_loading import load_products, load_transactions
from paths import REPORTS_DIR


def main() -> None:
    txns = load_transactions()
    products = load_products()

    lines: list[str] = ["# EDA inicial — dunnhumby The Complete Journey\n"]

    # --- 1. Tamaño general del dataset -----------------------------------
    n_rows = len(txns)
    n_households = txns["household_key"].nunique()
    n_baskets = txns["basket_id"].nunique()
    n_products_txn = txns["product_id"].nunique()
    n_products_catalog = products["product_id"].nunique()
    n_stores = txns["store_id"].nunique()

    lines.append("## 1. Tamaño general\n")
    lines.append(f"- Líneas de transacción (filas): **{n_rows:,}**")
    lines.append(f"- Hogares (household_key) distintos: **{n_households:,}**")
    lines.append(f"- Canastas (basket_id) distintas: **{n_baskets:,}**")
    lines.append(f"- Productos distintos comprados: **{n_products_txn:,}** (catálogo completo: {n_products_catalog:,})")
    lines.append(f"- Tiendas distintas: **{n_stores:,}**")
    lines.append(f"- Promedio de canastas por hogar: **{n_baskets / n_households:.2f}**")
    lines.append(f"- Promedio de líneas por canasta: **{n_rows / n_baskets:.2f}**\n")
    print("\n".join(lines[-8:]))

    # Sanity check: cada basket_id debería pertenecer a un único household_key
    baskets_per_hh_check = txns.groupby("basket_id")["household_key"].nunique()
    n_baskets_multi_hh = (baskets_per_hh_check > 1).sum()
    lines.append(
        f"- Chequeo de integridad: canastas con más de 1 household_key asociado: "
        f"**{n_baskets_multi_hh}** (debería ser 0)\n"
    )
    if n_baskets_multi_hh:
        print(f"[warn] {n_baskets_multi_hh} basket_id están asociados a más de un household_key — revisar.")

    # --- 2. Distribución temporal ------------------------------------------
    day_min, day_max = txns["day"].min(), txns["day"].max()
    week_min, week_max = txns["week_no"].min(), txns["week_no"].max()
    lines.append("## 2. Cobertura temporal\n")
    lines.append(f"- Rango de `day`: {day_min} a {day_max} ({day_max - day_min} días ≈ {(day_max - day_min) / 7:.0f} semanas)")
    lines.append(f"- Rango de `week_no`: {week_min} a {week_max}\n")

    txns_per_week = txns.groupby("week_no")["basket_id"].nunique()
    fig, ax = plt.subplots(figsize=(9, 4))
    txns_per_week.plot(ax=ax)
    ax.set_xlabel("week_no")
    ax.set_ylabel("nº de canastas distintas")
    ax.set_title("Canastas por semana a lo largo del dataset")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig_txns_per_week.png", dpi=120)
    plt.close(fig)

    # --- 3. Tamaño de canasta (nº de productos distintos por basket_id) ---
    basket_size = txns.groupby("basket_id")["product_id"].nunique()
    lines.append("## 3. Tamaño de canasta (productos distintos por ticket)\n")
    lines.append("```")
    lines.append(str(basket_size.describe()))
    lines.append("```\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    basket_size.clip(upper=basket_size.quantile(0.99)).hist(bins=50, ax=ax)
    ax.set_xlabel("productos distintos por canasta (recortado al p99)")
    ax.set_ylabel("frecuencia")
    ax.set_title("Distribución del tamaño de canasta")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig_basket_size.png", dpi=120)
    plt.close(fig)

    # --- 4. Nº de canastas por hogar (clave para el split leave-last-out) -
    baskets_per_hh = txns.groupby("household_key")["basket_id"].nunique()
    lines.append("## 4. Canastas por hogar\n")
    lines.append("```")
    lines.append(str(baskets_per_hh.describe()))
    lines.append("```\n")

    n_hh_min3 = (baskets_per_hh >= 3).sum()
    n_hh_lt3 = (baskets_per_hh < 3).sum()
    lines.append(
        f"- Hogares con **≥ 3 canastas** (usables para split train/val/test leave-last-out): "
        f"**{n_hh_min3:,}** ({n_hh_min3 / n_households:.1%})"
    )
    lines.append(
        f"- Hogares con **< 3 canastas** (se excluyen del split): "
        f"**{n_hh_lt3:,}** ({n_hh_lt3 / n_households:.1%})\n"
    )
    print(f"\nHogares con >= 3 canastas (usables para el split): {n_hh_min3:,} / {n_households:,}")

    fig, ax = plt.subplots(figsize=(7, 4))
    baskets_per_hh.clip(upper=baskets_per_hh.quantile(0.99)).hist(bins=50, ax=ax)
    ax.set_xlabel("nº de canastas por hogar (recortado al p99)")
    ax.set_ylabel("frecuencia")
    ax.set_title("Distribución de canastas por hogar")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig_baskets_per_hh.png", dpi=120)
    plt.close(fig)

    # --- 5. Popularidad de productos (para entender la cola larga) --------
    product_popularity = txns.groupby("product_id")["basket_id"].nunique().sort_values(ascending=False)
    top10_share = product_popularity.head(10).sum() / n_baskets
    lines.append("## 5. Concentración de popularidad\n")
    lines.append(
        f"- Los 10 productos más comprados aparecen en el **{top10_share:.1%}** de todas las canastas "
        f"(referencia de cuán larga es la cola de productos poco frecuentes)."
    )
    lines.append(
        f"- Productos comprados en una sola canasta en todo el dataset: "
        f"**{(product_popularity == 1).sum():,}** de {len(product_popularity):,}\n"
    )

    out_path = REPORTS_DIR / "eda_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen guardado en {out_path}")
    print(f"Figuras guardadas en {REPORTS_DIR}")


if __name__ == "__main__":
    main()
