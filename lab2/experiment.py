"""Полная серия 2 × 2 и случайный допустимый поиск."""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from ga import genetic_algorithm, random_feasible
from problem import ROOT, Problem


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data.json")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    problem = Problem.load(args.data)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        settings["seeds"] = [args.seed]
    if not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("Нужны различные seed")
    args.output.mkdir(parents=True, exist_ok=True)
    runs, history, solutions = [], [], {}
    methods = [(a, m) for a in ("penalty", "repair") for m in ("reassign", "swap")]
    methods.append(("random", "feasible"))
    for approach, operator in methods:
        method = f"{approach}_{operator}"
        for seed in settings["seeds"]:
            start = perf_counter()
            result = (random_feasible(problem, settings, seed) if approach == "random" else
                      genetic_algorithm(problem, settings, approach, operator, seed))
            seconds = perf_counter() - start
            runs.append(dict(method=method, seed=seed, best_f=result.value,
                             violations=sum(problem.violations(result.x).values()),
                             evaluations=result.evaluations, repair_calls=result.repair_calls,
                             seconds=seconds, assignment=json.dumps(result.x.tolist())))
            solutions[method, seed] = result.x
            for generation, evaluations, value, feasible in result.history:
                history.append(dict(method=method, seed=seed, generation=generation,
                                    evaluations=evaluations, best_f=value, feasible_fraction=feasible))
        print(f"{method}: {len(settings['seeds'])} запусков", flush=True)

    # Точный метод применяется после серий и не передаёт свой ответ ГА или baseline.
    exact_x, optimum = problem.exact_solution()
    summary = []
    for approach, operator in methods:
        name = f"{approach}_{operator}"
        selected = [row for row in runs if row["method"] == name]
        values = np.array([row["best_f"] for row in selected])
        summary.append(dict(method=name, runs=len(values), best=int(values.min()),
                            mean=float(values.mean()), median=float(np.median(values)),
                            std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                            worst=int(values.max()), exact_hits=int(np.sum(values == optimum)),
                            mean_seconds=float(np.mean([row["seconds"] for row in selected]))))
    best = min(runs, key=lambda row: row["best_f"])
    schedule = []
    for i, room_index in enumerate(solutions[best["method"], best["seed"]]):
        event = problem.events[i]
        room = problem.rooms[room_index]
        schedule.append(dict(event=event["id"], time=problem.data["slots"][event["slot"]],
                             students=event["students"], room=room["id"], capacity=room["capacity"],
                             empty_seats=room["capacity"]-event["students"],
                             preference_cost=event["preference_cost"][room_index],
                             cost=int(problem.cost[i, room_index]), fixed=event["fixed_room"] is not None))
    write_csv(args.output / "runs.csv", runs)
    write_csv(args.output / "history.csv", history)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "best_schedule.csv", schedule)
    examples = []
    for seed in (100, 101):
        rng = np.random.default_rng(seed)
        x = problem.repair(rng.integers(problem.m, size=problem.n), rng)
        examples.append(dict(name=f"valid_{seed}", assignment=x.tolist(), violations=problem.violations(x)))
    x = np.array(examples[0]["assignment"])
    x[1] = x[0]
    examples.append(dict(name="same_room_same_time", assignment=x.tolist(), violations=problem.violations(x)))
    x = np.array(examples[0]["assignment"])
    for i, event in enumerate(problem.events):
        used = {int(x[j]) for j, other in enumerate(problem.events) if other["slot"] == event["slot"]}
        small_free = [j for j, room in enumerate(problem.rooms)
                      if j not in used and room["capacity"] < event["students"]
                      and event["slot"] not in room["unavailable_slots"]]
        if small_free and event["fixed_room"] is None:
            x[i] = small_free[0]
            break
    else:
        raise ValueError("Не удалось построить пример отдельного нарушения вместимости")
    examples.append(dict(name="insufficient_capacity", assignment=x.tolist(), violations=problem.violations(x)))
    metadata = dict(optimum=optimum, exact_assignment=exact_x.tolist(), best_method=best["method"],
                    best_seed=best["seed"], best_assignment=solutions[best["method"], best["seed"]].tolist(),
                    penalty=problem.penalty, python=platform.python_version(), numpy=np.__version__,
                    platform=platform.platform())
    for filename, data in (("config_used.json", settings), ("data_used.json", problem.data),
                           ("metadata.json", metadata), ("examples.json", examples)):
        (args.output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Точный минимум:", optimum)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
