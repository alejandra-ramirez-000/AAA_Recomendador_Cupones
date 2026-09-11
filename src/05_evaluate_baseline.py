"""
Evalúa el Modelo 1 (baseline) con Precision@K, Recall@K, Hit Rate@K, NDCG@K
para K=5,10,20, por separado para Repeat y Explore, en val y test.

Corré desde la raíz del proyecto (después de 03_build_targets.py y 04_baseline_model.py):
    python src/05_evaluate_baseline.py

Genera:
    reports/baseline_metrics.md
"""
import pandas as pd

from evaluation import evaluate
from paths import DATA_PROCESSED_DIR, REPORTS_DIR

KS = (5, 10, 20)


def main() -> None:
    predictions = pd.read_parquet(DATA_PROCESSED_DIR / "baseline_predictions.parquet")
    targets = pd.read_parquet(DATA_PROCESSED_DIR / "targets.parquet")

    lines = ["# Modelo 1 — Baseline (frecuencia+recencia / popularidad global)\n"]
    lines.append(
        "Métricas calculadas por separado para Repeat y Explore. `n_households` indica "
        "cuántos hogares tenían al menos un producto verdadero de ese segmento en su "
        "canasta objetivo (los que no, quedan fuera del promedio).\n"
    )

    all_results = []
    for eval_split in ("val", "test"):
        for segment in ("repeat", "explore"):
            result = evaluate(predictions, targets, eval_split, segment, ks=KS)
            result.insert(0, "segment", segment)
            result.insert(0, "eval_split", eval_split)
            all_results.append(result)

            lines.append(f"## {eval_split} — {segment}\n")
            lines.append(result.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")

            print(f"\n[{eval_split} | {segment}]")
            print(result.to_string(index=False))

    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(DATA_PROCESSED_DIR / "baseline_metrics.csv", index=False)

    out_path = REPORTS_DIR / "baseline_metrics.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {out_path}")
    print(f"Guardado: {DATA_PROCESSED_DIR / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
