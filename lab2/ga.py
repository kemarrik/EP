"""ГА: штраф/ремонт и переназначение/обмен в одном временном слоте."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    x: np.ndarray
    value: int
    evaluations: int
    repair_calls: int
    history: list


def mutate(x, problem, operator, rng):
    x = x.copy()
    if operator == "reassign":
        for i in np.flatnonzero(rng.random(problem.n) < 1 / problem.n):
            choices = [room for room in problem.allowed[i] if room != x[i]]
            if choices:
                x[i] = rng.choice(choices)
    elif operator == "swap":
        if rng.random() < 0.5:
            indices = problem.indices[int(rng.integers(len(problem.indices)))]
            a, b = rng.choice(indices, size=2, replace=False)
            x[a], x[b] = x[b], x[a]
    else:
        raise ValueError("Неизвестная мутация")
    return x


def genetic_algorithm(problem, settings, approach, operator, seed):
    size, elite, budget = settings["population_size"], settings["elite_count"], settings["budget"]
    if not 30 <= size <= budget or not 1 <= elite < size:
        raise ValueError("Нужны 30 <= population_size <= budget и 1 <= elite_count < population_size")
    if approach not in ("penalty", "repair"):
        raise ValueError("Неизвестный способ обработки ограничений")
    rng = np.random.default_rng(seed)
    population = np.array([problem.repair(rng.integers(problem.m, size=problem.n), rng)
                           for _ in range(size)])
    repairs = size
    values, hard = problem.evaluate(population)
    scores = values + problem.penalty * hard
    evaluations, generation = size, 0
    history = [(0, evaluations, int(values[hard == 0].min()), float(np.mean(hard == 0)))]
    while evaluations < budget:
        count = min(size - elite, budget - evaluations)
        candidates = rng.integers(size, size=(2 * count, settings["tournament_size"]))
        parents = population[candidates[np.arange(2 * count), np.argmin(scores[candidates], axis=1)]]
        children = []
        for i in range(count):
            child = parents[2 * i].copy()
            if rng.random() < settings["crossover_probability"]:
                mask = rng.random(problem.n) < 0.5
                child[mask] = parents[2 * i + 1, mask]
            child = mutate(child, problem, operator, rng)
            if approach == "repair":
                child = problem.repair(child, rng)
                repairs += 1
            children.append(child)
        children = np.array(children)
        new_values, new_hard = problem.evaluate(children)
        keep = np.argsort(scores, kind="stable")[:size - count]
        population = np.concatenate((population[keep], children))
        values = np.concatenate((values[keep], new_values))
        hard = np.concatenate((hard[keep], new_hard))
        scores = values + problem.penalty * hard
        evaluations += count
        generation += 1
        history.append((generation, evaluations, int(values[hard == 0].min()), float(np.mean(hard == 0))))
    best = np.argmin(scores)
    assert hard[best] == 0
    return Result(population[best].copy(), int(values[best]), evaluations, repairs, history)


def random_feasible(problem, settings, seed):
    rng = np.random.default_rng(seed)
    best_x, best = None, float("inf")
    history = []
    evaluations = 0
    while evaluations < settings["budget"]:
        batch = settings["population_size"] if evaluations == 0 else settings["population_size"] - settings["elite_count"]
        count = min(batch, settings["budget"] - evaluations)
        points = np.array([problem.repair(rng.integers(problem.m, size=problem.n), rng)
                           for _ in range(count)])
        values, hard = problem.evaluate(points)
        assert np.all(hard == 0)
        index = np.argmin(values)
        if values[index] < best:
            best_x, best = points[index].copy(), int(values[index])
        evaluations += count
        history.append((len(history), evaluations, best, 1.0))
    return Result(best_x, best, evaluations, evaluations, history)
