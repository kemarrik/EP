"""Вариант 9: функция Диксона — Прайса."""

import numpy as np

DIMENSION = 8
LOWER = -10.0
UPPER = 10.0


def objective(x):
    """Принимает один вектор или матрицу, в которой особи записаны по строкам."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != DIMENSION:
        raise ValueError(f"В особи должно быть {DIMENSION} координат")
    weights = np.arange(2, DIMENSION + 1)
    return (x[..., 0] - 1) ** 2 + np.sum(
        weights * (2 * x[..., 1:] ** 2 - x[..., :-1]) ** 2, axis=-1
    )


def known_minimum():
    """Положительный глобальный минимум, только для проверки и анализа."""
    return 2.0 ** (-1 + 2.0 ** (1 - np.arange(1, DIMENSION + 1)))
