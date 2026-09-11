"""
Evalúa el Modelo 2b (co-ocurrencia híbrida) con las mismas métricas y K que los anteriores.

Corré desde la raíz del proyecto (después de 09_cooccurrence_hybrid_model.py):
    python src/10_evaluate_cooccurrence_hybrid.py

Genera:
    reports/cooccurrence_hybrid_metrics.md
    data/processed/cooccurrence_hybrid_metrics.csv
"""
import pandas as pd

from evaluation import evaluate
from paths import DATA_PROCESSED_DIR, REPORTS_DIR

KS = (5, 10, 20)


def main() -> None:
    predictions = pd.read_parquet(DATA_PROCESSED_DIR / "cooccurrence_hybrid_predictions.parquet")
    targets = pd.read_parquet(DATA_PROCESSED_DIR / "targets.parquet")

    lines = ["# Modelo 2b — Co-ocurrencia híbrida (piso de popularidad + lift) para Explore\n"]

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
    full.to_csv(DATA_PROCESSED_DIR / "cooccurrence_hybrid_metrics.csv", index=False)

    out_path = REPORTS_DIR / "cooccurrence_hybrid_metrics.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
