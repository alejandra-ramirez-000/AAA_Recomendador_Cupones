"""
Compara las métricas de todos los modelos evaluados hasta el momento
(lee los *_metrics.csv que va dejando cada 0N_evaluate_*.py).

Corré desde la raíz del proyecto (después de correr los evaluate_* de cada modelo):
    python src/08_compare_models.py

Genera:
    reports/model_comparison.md
"""
import pandas as pd

from paths import DATA_PROCESSED_DIR, REPORTS_DIR

MODEL_FILES = {
    "1_baseline": "baseline_metrics.csv",
    "2_cooccurrence_lift": "cooccurrence_metrics.csv",
    "2b_cooccurrence_hybrid": "cooccurrence_hybrid_metrics.csv",
    "3_als": "als_metrics.csv",
}


def main() -> None:
    frames = []
    for model_name, filename in MODEL_FILES.items():
        path = DATA_PROCESSED_DIR / filename
        if not path.exists():
            print(f"(salteado: {filename} no existe todavía)")
            continue
        df = pd.read_csv(path)
        df.insert(0, "model", model_name)
        frames.append(df)

    if not frames:
        print("No hay métricas para comparar todavía. Corré los evaluate_*.py primero.")
        return

    comparison = pd.concat(frames, ignore_index=True)

    lines = ["# Comparación de modelos\n"]
    for eval_split in ("val", "test"):
        for segment in ("repeat", "explore"):
            subset = comparison[
                (comparison["eval_split"] == eval_split) & (comparison["segment"] == segment)
            ].drop(columns=["eval_split", "segment"])
            if subset.empty:
                continue
            lines.append(f"## {eval_split} — {segment}\n")
            lines.append(subset.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
            print(f"\n[{eval_split} | {segment}]")
            print(subset.to_string(index=False))

    out_path = REPORTS_DIR / "model_comparison.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
