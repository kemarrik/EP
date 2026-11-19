"""Генетический алгоритм с вещественными особями и случайный поиск."""

from dataclasses import dataclass

import numpy as np

from problem import DIMENSION, LOWER, UPPER, objective


@dataclass
class Result:
    x: np.ndarray
    value: float
    evaluations: int
    # Строка истории: поколение (для случайного поиска — шаг), оценки, лучший f.
    history: list


def tournament(population, values, count, size, rng):
    candidates = rng.integers(0, len(population), size=(count, size))
    winners = candidates[np.arange(count), np.argmin(values[candidates], axis=1)]
    return population[winners].copy()


def blx_crossover(first, second, alpha, probability, rng):
    """Для каждой координаты выбираем точку из расширенного интервала родителей."""
    distance = np.abs(first - second)
    left = np.minimum(first, second) - alpha * distance
    right = np.maximum(first, second) + alpha * distance
    mixed = rng.uniform(left, right)
    use_crossover = rng.random((len(first), 1)) < probability
    return np.where(use_crossover, mixed, first)


def mutate(children, probability, sigma, rng):
    mask = rng.random(children.shape) < probability
    return children + mask * rng.normal(0, sigma, children.shape)


def genetic_algorithm(settings, sigma, seed):
    population_size = settings["population_size"]
    budget = settings["budget"]
    elite_count = settings["elite_count"]
    if not (30 <= population_size <= budget):
        raise ValueError("Нужно 30 <= population_size <= budget")
    if not (1 <= elite_count < population_size):
        raise ValueError("elite_count должен быть от 1 до population_size - 1")
    if settings["tournament_size"] < 1 or sigma <= 0 or settings["blx_alpha"] < 0:
        raise ValueError("Некорректные параметры отбора, мутации или BLX")
    for key in ("mutation_probability", "crossover_probability"):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f"{key} должен лежать в [0, 1]")

    rng = np.random.default_rng(seed)
    population = rng.uniform(LOWER, UPPER, (population_size, DIMENSION))
    values = objective(population)
    evaluations = population_size
    generation = 0
    history = [(generation, evaluations, float(values.min()))]

    while evaluations < budget:
        # Элиту не оцениваем повторно. Последнее поколение может быть неполным.
        count = min(population_size - elite_count, budget - evaluations)
        first = tournament(population, values, count, settings["tournament_size"], rng)
        second = tournament(population, values, count, settings["tournament_size"], rng)
        children = blx_crossover(
            first, second, settings["blx_alpha"], settings["crossover_probability"], rng
        )
        children = mutate(children, settings["mutation_probability"], sigma, rng)
        children = np.clip(children, LOWER, UPPER)
        child_values = objective(children)

        # В последнем неполном поколении оставляем больше старых особей.
        keep = np.argsort(values)[:population_size - count]
        population = np.concatenate((population[keep], children))
        values = np.concatenate((values[keep], child_values))
        evaluations += count
        generation += 1
        history.append((generation, evaluations, float(values.min())))

    best = np.argmin(values)
    return Result(population[best].copy(), float(values[best]), evaluations, history)


def random_search(settings, seed):
    """Независимые равномерные точки; сохраняется лучшая из всех проверенных."""
    rng = np.random.default_rng(seed)
    budget = settings["budget"]
    population_size = settings["population_size"]
    elite_count = settings["elite_count"]
    best_x, best_value = None, float("inf")
    evaluations, step = 0, 0
    history = []
    while evaluations < budget:
        # Те же контрольные точки бюджета, что у ГА, без повторного отбора точек.
        batch = population_size if evaluations == 0 else population_size - elite_count
        count = min(batch, budget - evaluations)
        points = rng.uniform(LOWER, UPPER, (count, DIMENSION))
        values = objective(points)
        index = np.argmin(values)
        if values[index] < best_value:
            best_x, best_value = points[index].copy(), float(values[index])
        evaluations += count
        history.append((step, evaluations, best_value))
        step += 1
    return Result(best_x, best_value, evaluations, history)
