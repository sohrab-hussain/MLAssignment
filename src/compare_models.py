"""Collect the saved metrics of all five models into one comparison table.

Run from the project root after training every model:
    python -m src.compare_models

Prints a Markdown table (paste it into README.md) and writes
model/comparison.csv.
"""
import json

import pandas as pd

from src.common.config import MODEL_DIR, MODEL_NAMES, model_dir


def main() -> None:
    rows = []
    for key in MODEL_NAMES:
        path = model_dir(key) / "metrics.json"
        if not path.exists():
            print(f"Missing {path} - run: python -m src.models.{key}.train")
            continue
        summary = json.loads(path.read_text())
        rows.append({"ML Model Name": summary["model"], **summary["metrics"]})

    table = pd.DataFrame(rows)
    table.to_csv(MODEL_DIR / "comparison.csv", index=False)
    print(table.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
