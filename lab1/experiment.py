"""Запуск серии и сохранение исходных результатов без готового оптимизатора."""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from ga import genetic_algorithm, random_search

ROOT = Path(__file__).resolve().parent


def load_config(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if settings["variant"] != 9 or settings["dimension"] != 8 or settings["bounds"] != [-10, 10]:
        raise ValueError("Эта программа реализует вариант 9, d = 8, границы [-10, 10]")
    if not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Нужен непустой список различных seed")
    return settings


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, help="Один запуск каждой конфигурации и случайного поиска")
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.seed is not None:
        settings["seeds"] = [args.seed]
    args.output.mkdir(parents=True, exist_ok=True)

    runs, history = [], []
    methods = [(item["name"], item["sigma"]) for item in settings["configurations"]]
    methods.append(("random_search", None))
    for name, sigma in methods:
        for seed in settings["seeds"]:
            start = perf_counter()
            result = (random_search(settings, seed) if sigma is None
                      else genetic_algorithm(settings, sigma, seed))
            seconds = perf_counter() - start
            row = dict(method=name, seed=seed, best_f=result.value,
                       evaluations=result.evaluations, seconds=seconds)
            row.update({f"x{i + 1}": value for i, value in enumerate(result.x)})
            runs.append(row)
            for generation, evaluations, value in result.history:
                history.append(dict(method=name, seed=seed, generation=generation,
                                    evaluations=evaluations, best_f=value))
        print(f"{name}: {len(settings['seeds'])} запусков выполнено", flush=True)

    summary = []
    for name, _ in methods:
        selected = [row for row in runs if row["method"] == name]
        values = np.array([row["best_f"] for row in selected])
        summary.append(dict(method=name, runs=len(values), best=values.min(),
                            mean=values.mean(), median=np.median(values),
                            std=values.std(ddof=1) if len(values) > 1 else 0.0,
                            worst=values.max(), below_0_01=int(np.sum(values < 0.01)),
                            mean_seconds=np.mean([row["seconds"] for row in selected])))
    write_csv(args.output / "runs.csv", runs, list(runs[0]))
    write_csv(args.output / "history.csv", history, list(history[0]))
    write_csv(args.output / "summary.csv", summary, list(summary[0]))
    (args.output / "config_used.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       platform=platform.platform(), processor=platform.machine())
    (args.output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in summary:
        print(f"{row['method']}: mean={row['mean']:.6g}, median={row['median']:.6g}, "
              f"best={row['best']:.6g}, worst={row['worst']:.6g}")


if __name__ == "__main__":
    main()
