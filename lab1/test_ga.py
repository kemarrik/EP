"""Проверки формулы, границ, бюджета и воспроизводимости."""

import unittest
from unittest.mock import patch

import numpy as np

from experiment import ROOT, load_config
from ga import blx_crossover, genetic_algorithm, random_search
from problem import DIMENSION, LOWER, UPPER, known_minimum, objective


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_config(ROOT / "config.json")
        self.settings.update(population_size=30, budget=347)

    def test_objective_known_values(self):
        self.assertEqual(objective(np.zeros(DIMENSION)), 1)
        self.assertEqual(objective(np.ones(DIMENSION)), 35)
        np.testing.assert_array_equal(objective(np.array([np.zeros(8), np.ones(8)])), [1, 35])

    def test_global_minimum(self):
        point = known_minimum()
        self.assertLess(objective(point), 1e-25)
        point[-1] *= -1
        self.assertLess(objective(point), 1e-25)

    def test_blx_interval_and_disabled_crossover(self):
        first, second = np.zeros((100, 8)), np.ones((100, 8))
        children = blx_crossover(first, second, 0.5, 1.0, np.random.default_rng(42))
        self.assertTrue(np.all((children >= -0.5) & (children <= 1.5)))
        children = blx_crossover(first, second, 0.5, 0.0, np.random.default_rng(42))
        np.testing.assert_array_equal(children, first)

    def test_actual_budget_and_all_evaluated_points(self):
        for function in (genetic_algorithm, random_search):
            for budget in (30, 31, 347):
                self.settings["budget"] = budget
                evaluated = []

                def counted(points):
                    evaluated.append(points.copy())
                    return objective(points)

                with patch("ga.objective", side_effect=counted):
                    result = (function(self.settings, 0.2, 4) if function is genetic_algorithm
                              else function(self.settings, 4))
                points = np.concatenate(evaluated)
                self.assertEqual(len(points), budget)
                self.assertEqual(result.evaluations, budget)
                self.assertTrue(np.all((points >= LOWER) & (points <= UPPER)))
                self.assertEqual(result.value, objective(result.x))
                self.assertTrue(np.all(np.diff([row[2] for row in result.history]) <= 0))

    def test_reproducibility_and_new_seed(self):
        for function in (genetic_algorithm, random_search):
            def run(seed):
                return (function(self.settings, 0.05, seed) if function is genetic_algorithm
                        else function(self.settings, seed))
            first, repeat, other = run(7), run(7), run(8)
            np.testing.assert_array_equal(first.x, repeat.x)
            self.assertEqual(first.history, repeat.history)
            self.assertFalse(np.array_equal(first.x, other.x))

    def test_invalid_settings(self):
        self.settings["elite_count"] = self.settings["population_size"]
        with self.assertRaises(ValueError):
            genetic_algorithm(self.settings, 0.05, 0)


if __name__ == "__main__":
    unittest.main()
