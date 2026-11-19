"""Графики строятся из сохранённых CSV, повторно запускать поиск не нужно."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
METHODS = ["ga_sigma_0.05", "ga_sigma_0.20", "random_search"]
LABELS = ["ГА, σ = 0.05", "ГА, σ = 0.20", "Случайный поиск"]
COLORS = ["#2563a6", "#d17922", "#666666"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def trajectories(rows, method):
    selected = [row for row in rows if row["method"] == method]
    seeds = sorted({int(row["seed"]) for row in selected})
    curves = []
    for seed in seeds:
        curve = sorted((row for row in selected if int(row["seed"]) == seed),
                       key=lambda row: int(row["generation"]))
        curves.append([float(row["best_f"]) for row in curve])
    generations = np.array([int(row["generation"]) for row in curve])
    evaluations = np.array([int(row["evaluations"]) for row in curve])
    return generations, evaluations, np.array(curves)


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def flowchart(path):
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set(xlim=(0, 10), ylim=(0, 12))
    ax.axis("off")
    steps = [
        (10.8, "Задать параметры и seed"),
        (9.5, "100 случайных особей в [−10, 10]⁸\nОценить функцию; сохранить лучшее значение"),
        (8.2, "Проверить остаток бюджета\nОсталось B − evaluations вычислений"),
        (6.9, "Сохранить 2 элитные особи\nВыбрать родителей турнирами по 3"),
        (5.6, "BLX-α-кроссовер (p = 0.9, α = 0.5)\nГауссовская мутация (p = 1/8 на координату)"),
        (4.3, "Ограничить координаты интервалом [−10, 10]\nОценить только новых потомков"),
        (3.0, "Собрать новое поколение\nОбновить счётчик и историю лучшего f"),
    ]
    for y, text in steps:
        ax.add_patch(FancyBboxPatch((1.5, y - 0.42), 6.8, 0.84,
                                   boxstyle="round,pad=0.08", facecolor="white",
                                   edgecolor="#333333", linewidth=1.2))
        ax.text(4.9, y, text, ha="center", va="center", fontsize=10)
    for (y1, _), (y2, _) in zip(steps, steps[1:]):
        ax.add_patch(FancyArrowPatch((4.9, y1 - 0.52), (4.9, y2 + 0.52),
                                    arrowstyle="->", mutation_scale=14))
    ax.text(5.05, 7.56, "Есть остаток", fontsize=9)
    ax.plot([8.4, 9.2, 9.2, 8.5], [3, 3, 8.2, 8.2], color="#333333")
    ax.add_patch(FancyArrowPatch((8.5, 8.2), (8.38, 8.2), arrowstyle="->", mutation_scale=14))
    ax.add_patch(FancyArrowPatch((1.4, 8.2), (0.6, 8.2), arrowstyle="-", mutation_scale=14))
    ax.plot([0.6, 0.6, 4.9], [8.2, 1.9, 1.9], color="#333333")
    ax.add_patch(FancyArrowPatch((4.9, 1.9), (4.9, 1.6), arrowstyle="->", mutation_scale=14))
    ax.text(1.0, 8.4, "Ноль", fontsize=9, ha="center")
    ax.add_patch(FancyBboxPatch((2.0, 0.6), 5.8, 0.9, boxstyle="round,pad=0.08",
                               facecolor="white", edgecolor="#333333"))
    ax.text(4.9, 1.05, "Сохранить лучшее решение и историю", ha="center", va="center", fontsize=10)
    ax.set_title("Схема генетического алгоритма", pad=10)
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.results / "history.csv")
    runs = read_csv(args.results / "runs.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
    for index, ax in enumerate(axes):
        generations, _, values = trajectories(rows, METHODS[index])
        ax.fill_between(generations, values.min(axis=0), values.max(axis=0),
                        color=COLORS[index], alpha=0.18, label="Минимум–максимум")
        ax.plot(generations, values.mean(axis=0), color=COLORS[index], label="Среднее")
        ax.set(title=LABELS[index], xlabel="Поколение", yscale="log")
        ax.legend()
    axes[0].set_ylabel("Лучшее найденное f (логарифмическая шкала)")
    fig.suptitle(f"Сходимость ГА по {len(values)} независимым запускам")
    save_figure(fig, args.output / "convergence.png")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for method, label, color in zip(METHODS, LABELS, COLORS):
        _, evaluations, values = trajectories(rows, method)
        ax.plot(evaluations, values.mean(axis=0), label=label, color=color)
    ax.set(xlabel="Число вычислений функции", ylabel="Среднее лучшее найденное f",
           yscale="log", title=f"Сравнение при одинаковом бюджете: {evaluations[-1]:,} оценок")
    ax.legend()
    save_figure(fig, args.output / "comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    values = [[float(row["best_f"]) for row in runs if row["method"] == name]
              for name in METHODS]
    boxes = ax.boxplot(values, tick_labels=LABELS, showfliers=False, patch_artist=True)
    for box, color in zip(boxes["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.25)
    for index, (points, color) in enumerate(zip(values, COLORS), start=1):
        ax.scatter(index + np.linspace(-0.12, 0.12, len(points)), points,
                   color=color, s=20, alpha=0.75, zorder=3)
    ax.set(yscale="log", ylabel="Итоговое лучшее f", title=f"Результаты всех {len(runs)} запусков")
    save_figure(fig, args.output / "distribution.png")
    flowchart(args.output / "flowchart.png")
    print(f"Графики сохранены: {args.output}")


if __name__ == "__main__":
    main()
