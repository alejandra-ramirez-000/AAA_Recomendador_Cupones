"""
Evalúa el Modelo 2 (co-ocurrencia) con las mismas métricas y K que el baseline.

Corré desde la raíz del proyecto (después de 03_build_targets.py y 06_cooccurrence_model.py):
    python src/07_evaluate_cooccurrence.py

Genera:
    reports/cooccurrence_metrics.md
    data/processed/cooccurrence_metrics.csv
"""
import pandas as pd

from evaluation import evaluate
from paths import DATA_PROCESSED_DIR, REPORTS_DIR

KS = (5, 10, 20)


def main() -> None:
    predictions = pd.read_parquet(DATA_PROCESSED_DIR / "cooccurrence_predictions.parquet")
    targets = pd.read_parquet(DATA_PROCESSED_DIR / "targets.parquet")

    lines = ["# Modelo 2 — Co-ocurrencia (lift) para Explore\n"]
    lines.append(
        "Repeat sin cambios respecto al Modelo 1 (frecuencia+recencia). Explore ahora usa "
        "lift entre productos en vez de popularidad global. `n_households` = hogares con "
        "al menos un producto verdadero de ese segmento en su canasta objetivo.\n"
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
    full.to_csv(DATA_PROCESSED_DIR / "cooccurrence_metrics.csv", index=False)

    out_path = REPORTS_DIR / "cooccurrence_metrics.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {out_path}")
    print(f"Guardado: {DATA_PROCESSED_DIR / 'cooccurrence_metrics.csv'}")


if __name__ == "__main__":
    main()
