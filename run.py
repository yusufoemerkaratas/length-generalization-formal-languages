"""Train and evaluate every model and seed of one experiment.

Usage: python run.py --config configs/dyck1.yaml

Writes to the config's output_dir:
  metrics.csv      accuracy per model, seed, test bin and example kind
  runs.csv         training summary per model and seed
  config_used.yaml the config plus software versions
"""

import argparse
import csv
import platform
from pathlib import Path

import torch
import yaml

from flg.data import make_datasets
from flg.languages import LANGUAGES
from flg.models import build_model
from flg.train import accuracy, train_model


def majority_accuracy(train, examples):
    """Accuracy of always predicting the most common label in the training set
    (label 1 on a tie)."""
    ones = sum(label for _, label, _ in train)
    guess = 1 if ones >= len(train) - ones else 0
    result = {}
    for key in ["all"] + sorted({kind for _, _, kind in examples}):
        subset = [label for _, label, kind in examples if key in ("all", kind)]
        result[key] = sum(label == guess for label in subset) / len(subset)
    return result


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    lang = LANGUAGES[cfg["language"]]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version(), "torch": str(torch.__version__)}
    (out / "config_used.yaml").write_text(yaml.safe_dump({**cfg, "versions": versions}))

    metric_rows, run_rows = [], []
    for seed in cfg["seeds"]:
        data = make_datasets(lang, cfg["data"], seed)
        for bin_name, examples in data["test"].items():
            for kind, acc in majority_accuracy(data["train"], examples).items():
                metric_rows.append({"model": "majority", "seed": seed, "bin": bin_name,
                                    "kind": kind, "accuracy": round(acc, 4)})

        for model_name, model_cfg in cfg["models"].items():
            torch.manual_seed(seed)
            model = build_model(model_cfg)
            info = train_model(model, data, lang.alphabet, cfg["training"], seed)
            run_rows.append({"model": model_name, "seed": seed, **info})
            for bin_name, examples in data["test"].items():
                for kind, acc in accuracy(model, examples, lang.alphabet).items():
                    metric_rows.append({"model": model_name, "seed": seed, "bin": bin_name,
                                        "kind": kind, "accuracy": round(acc, 4)})
            print(f"{cfg['language']} seed={seed} {model_name}: {info}", flush=True)
            # Rewrite after every run so finished runs are saved if the job stops.
            write_csv(out / "metrics.csv", metric_rows)
            write_csv(out / "runs.csv", run_rows)


if __name__ == "__main__":
    main()
