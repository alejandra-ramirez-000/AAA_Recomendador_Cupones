"""
Grid search de w_lift (peso del componente de co-ocurrencia en el híbrido Explore)
evaluado SOLO en 'val' — nunca en test, para no elegir el peso mirando el set de
evaluación final.

Reutiliza el mismo cómputo caro (matriz de lift, W @ lift) para todos los pesos del
grid — solo la combinación final cambia por punto del grid, así que corre rápido.

Corré desde la raíz del proyecto (después de 02_build_baskets_and_split.py y
03_build_targets.py):
    python src/11_tune_hybrid_weight.py

Genera:
    reports/hybrid_weight_tuning.md
"""
import pandas as pd

from basket_utils import exploded_long
from cooccurrence_scoring import prepare_blend_components, rank_from_blend_components
from evaluation import evaluate
from paths import DATA_PROCESSED_DIR, REPORTS_DIR

WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K_FOR_SELECTION = 20  # métrica usada para elegir el mejor peso


def main() -> None:
    baskets = pd.read_parquet(DATA_PROCESSED_DIR / "baskets.parquet")
    baskets = baskets[baskets["split"] != "excluded"]
    long = exploded_long(baskets)
    history_long = long[long["split"] == "train"]  # historial permitido para predecir val

    targets = pd.read_parquet(DATA_PROCESSED_DIR / "targets.parquet")

    print("Precalculando componentes de co-ocurrencia y popularidad (una sola vez)...")
    components = prepare_blend_components(history_long)

    rows = []
    for w_lift in WEIGHT_GRID:
        w_pop = 1 - w_lift
        predictions = rank_from_blend_components(components, w_lift=w_lift, w_pop=w_pop)
        predictions["eval_split"] = "val"
        predictions["segment"] = "explore"

        result = evaluate(predictions, targets, "val", "explore", ks=(5, 10, K_FOR_SELECTION))
        result.insert(0, "w_lift", w_lift)
        rows.append(result)
        row20 = result[result["k"] == K_FOR_SELECTION].iloc[0]
        print(
            f"w_lift={w_lift:.1f} (w_pop={w_pop:.1f}) -> "
            f"recall@{K_FOR_SELECTION}={row20['recall']:.4f}  "
            f"hit_rate@{K_FOR_SELECTION}={row20['hit_rate']:.4f}  "
            f"ndcg@{K_FOR_SELECTION}={row20['ndcg']:.4f}"
        )

    full = pd.concat(rows, ignore_index=True)
    best_row = full[full["k"] == K_FOR_SELECTION].sort_values("ndcg", ascending=False).iloc[0]
    print(
        f"\nMejor w_lift por NDCG@{K_FOR_SELECTION} en val: {best_row['w_lift']:.1f} "
        f"(ndcg={best_row['ndcg']:.4f}, recall={best_row['recall']:.4f}, hit_rate={best_row['hit_rate']:.4f})"
    )

    lines = [
        "# Grid search de w_lift (Explore híbrido) — evaluado en val\n",
        f"Mejor peso por NDCG@{K_FOR_SELECTION}: **w_lift={best_row['w_lift']:.1f}** "
        f"(w_pop={1 - best_row['w_lift']:.1f})\n",
        full.to_markdown(index=False, floatfmt=".4f"),
    ]
    out_path = REPORTS_DIR / "hybrid_weight_tuning.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
