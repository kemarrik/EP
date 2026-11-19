"""Графики, схема алгоритма и расписание по сохранённой серии."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problem import ROOT

METHODS = ["penalty_reassign", "penalty_swap", "repair_reassign", "repair_swap", "random_feasible"]
LABELS = ["Штраф, переназначение", "Штраф, обмен", "Ремонт, переназначение", "Ремонт, обмен", "Случайный допустимый"]
COLORS = ["#2166ac", "#67a9cf", "#b35806", "#e6ab55", "#555555"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def curves(rows, method, column="best_f"):
    selected = [r for r in rows if r["method"] == method]
    seeds = sorted({int(r["seed"]) for r in selected})
    result = []
    for seed in seeds:
        curve = sorted((r for r in selected if int(r["seed"]) == seed), key=lambda r: int(r["generation"]))
        result.append([float(r[column]) for r in curve])
    return np.array([int(r["generation"]) for r in curve]), np.array([int(r["evaluations"]) for r in curve]), np.array(result)


def save(fig, output, name):
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.results / "history.csv")
    metadata = json.loads((args.results / "metadata.json").read_text())
    data = json.loads((args.results / "data_used.json").read_text())
    optimum = metadata["optimum"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    for ax, operator, title in zip(axes, ("reassign", "swap"), ("Мутация переназначения", "Мутация обмена")):
        for index, method in enumerate(METHODS[:4]):
            if not method.endswith(operator):
                continue
            generation, _, values = curves(rows, method)
            ax.fill_between(generation, values.min(axis=0), values.max(axis=0), color=COLORS[index], alpha=0.16)
            ax.plot(generation, values.mean(axis=0), label=LABELS[index], color=COLORS[index])
        ax.axhline(optimum, color="black", linestyle="--", linewidth=1, label="Точный минимум")
        ax.set(title=title, xlabel="Поколение")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Лучшее допустимое значение цели")
    fig.suptitle("Среднее и диапазон минимум–максимум по серии")
    save(fig, args.output, "convergence.png")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for method, label, color in zip(METHODS, LABELS, COLORS):
        _, evaluations, values = curves(rows, method)
        ax.plot(evaluations, values.mean(axis=0), label=label, color=color)
    ax.axhline(optimum, color="black", linestyle="--", label="Точный минимум")
    ax.set(xlabel="Число оценённых решений", ylabel="Среднее лучшее допустимое значение", title="Сравнение при одинаковом бюджете")
    ax.legend(fontsize=9)
    save(fig, args.output, "comparison.png")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for method, label, color in zip(METHODS[:4], LABELS, COLORS):
        generations, _, values = curves(rows, method, "feasible_fraction")
        ax.plot(generations, values.mean(axis=0), label=label, color=color,
                linestyle="--" if method.endswith("swap") else "-")
    ax.set(ylim=(0, 1.04), xlabel="Поколение", ylabel="Доля допустимых особей в популяции", title="Работа с ограничениями")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, args.output, "feasibility.png")

    fig, ax = plt.subplots(figsize=(12, 5.3), layout="constrained")
    matrix = np.full((len(data["rooms"]), len(data["slots"])), -1, dtype=int)
    for i, room in enumerate(metadata["best_assignment"]):
        matrix[room, data["events"][i]["slot"]] = i
    ax.grid(False)
    for room, spec in enumerate(data["rooms"]):
        for slot in range(len(data["slots"])):
            i = matrix[room, slot]
            blocked = slot in spec["unavailable_slots"]
            color = "#cccccc" if blocked else "#e0edf4" if i >= 0 else "white"
            ax.add_patch(plt.Rectangle((slot - .5, room - .5), 1, 1, facecolor=color, edgecolor="#999999"))
            if i >= 0:
                event = data["events"][i]
                text = f"{event['id']} · {event['students']} чел."
                if event["fixed_room"] is not None:
                    text += "\nзакреплено"
            else:
                text = "Закрыта" if blocked else "Свободна"
            ax.text(slot, room, text, ha="center", va="center", fontsize=9)
    ax.set(xlim=(-.5, 5.5), ylim=(5.5, -.5), xticks=range(6), yticks=range(6),
           xticklabels=data["slots"], yticklabels=[f"{r['id']} ({r['capacity']} мест)" for r in data["rooms"]],
           title=f"Лучшее расписание: стоимость {optimum if min(float(r['best_f']) for r in rows) == optimum else min(float(r['best_f']) for r in rows):g}")
    save(fig, args.output, "schedule.png")

    fig, ax = plt.subplots(figsize=(10, 8.4))
    ax.axis("off")
    steps = ["Параметры и seed; допустимая начальная популяция",
             "Оценить мягкую стоимость и жёсткие нарушения",
             "Сохранить элиту; выбрать родителей турниром",
             "Равномерный кроссовер вектора назначений",
             "Мутация: переназначение или обмен в одном слоте",
             "Ремонт: восстановить паросочетание по слотам\nШтраф: оставить потомков и учесть M · H",
             "Оценить потомков; обновить популяцию и историю",
             "Если бюджет остался — повторить с отбора\nИначе сохранить лучшее допустимое расписание"]
    for i, text in enumerate(steps):
        y = .94 - i * .125
        ax.text(.5, y, text, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.65", facecolor="white", edgecolor="#333333"), fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(.5, y - .09), xytext=(.5, y - .035), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
    ax.set_title("Схема ГА для назначения аудиторий", pad=20)
    save(fig, args.output, "flowchart.png")
    print("Сохранены 5 рисунков")


if __name__ == "__main__":
    main()
