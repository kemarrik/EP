"""Проверки ограничений, восстановления, эталона и бюджета."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

from ga import genetic_algorithm, mutate, random_feasible
from problem import Problem


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem.load()
        self.settings = dict(population_size=30, elite_count=2, budget=103,
                             tournament_size=3, crossover_probability=0.9)

    def test_repair_and_independent_constraint_check(self):
        p = self.problem
        for seed in range(30):
            rng = np.random.default_rng(seed)
            x = p.repair(rng.integers(p.m, size=p.n), rng)
            self.assertEqual(sum(p.violations(x).values()), 0)
            soft, hard = p.evaluate(x[None, :])
            self.assertEqual(hard[0], 0)
            expected = 0
            for i, event in enumerate(p.events):
                room = p.rooms[x[i]]
                self.assertGreaterEqual(room["capacity"], event["students"])
                self.assertNotIn(event["slot"], room["unavailable_slots"])
                if event["fixed_room"] is not None:
                    self.assertEqual(x[i], event["fixed_room"])
                expected += room["capacity"] - event["students"] + event["preference_cost"][x[i]]
            for indices in p.indices:
                self.assertEqual(len(set(x[indices])), len(indices))
            self.assertEqual(soft[0], expected)
            np.testing.assert_array_equal(p.repair(x, rng), x)

    def test_each_constraint(self):
        p = self.problem
        rng = np.random.default_rng(1)
        x = p.repair(rng.integers(p.m, size=p.n), rng)
        cases = []
        bad = x.copy()
        bad[1] = bad[0]
        cases.append((bad, "room_conflicts"))
        bad = x.copy()
        bad[np.argmax(p.size)] = np.argmin(p.capacity)
        cases.append((bad, "capacity"))
        room = next(i for i, r in enumerate(p.rooms) if r["unavailable_slots"])
        slot = p.rooms[room]["unavailable_slots"][0]
        index = next(i for i, e in enumerate(p.events) if e["slot"] == slot)
        bad = x.copy()
        bad[index] = room
        cases.append((bad, "unavailable"))
        index = next(i for i, e in enumerate(p.events) if e["fixed_room"] is not None)
        bad = x.copy()
        bad[index] = (bad[index] + 1) % p.m
        cases.append((bad, "fixed_room"))
        for bad, kind in cases:
            violations = p.violations(bad)
            self.assertGreater(violations[kind], 0)
            self.assertEqual(p.evaluate(bad[None, :])[1][0], sum(violations.values()))

    def test_exact_solution_against_enumeration(self):
        p = self.problem
        answer, optimum = p.exact_solution()
        brute_total = 0
        for indices in p.indices:
            best = float("inf")
            for rooms in itertools.permutations(range(p.m), len(indices)):
                if all(room in p.allowed[i] for i, room in zip(indices, rooms)):
                    best = min(best, sum(p.cost[i, room] for i, room in zip(indices, rooms)))
            brute_total += best
        self.assertEqual(optimum, brute_total)
        self.assertEqual(p.evaluate(answer[None, :])[0][0], optimum)
        self.assertEqual(sum(p.violations(answer).values()), 0)

    def test_budget_reproducibility_and_elitism(self):
        p = self.problem
        for approach in ("penalty", "repair"):
            for operator in ("reassign", "swap"):
                for budget in (30, 31, 103):
                    self.settings["budget"] = budget
                    original = p.evaluate
                    counts = []

                    def counted(points):
                        counts.append(len(points))
                        return original(points)

                    with patch.object(p, "evaluate", side_effect=counted):
                        a = genetic_algorithm(p, self.settings, approach, operator, 5)
                    b = genetic_algorithm(p, self.settings, approach, operator, 5)
                    self.assertEqual(sum(counts), budget)
                    self.assertEqual(a.evaluations, budget)
                    np.testing.assert_array_equal(a.x, b.x)
                    self.assertEqual(a.history, b.history)
                    self.assertTrue(np.all(np.diff([row[2] for row in a.history]) <= 0))
                    self.assertEqual(sum(p.violations(a.x).values()), 0)
                    if approach == "repair":
                        self.assertTrue(all(row[3] == 1 for row in a.history))

    def test_random_budget_and_score_dominance(self):
        p = self.problem
        original = p.evaluate
        counts = []

        def counted(points):
            counts.append(len(points))
            return original(points)

        with patch.object(p, "evaluate", side_effect=counted):
            result = random_feasible(p, self.settings, 1)
        self.assertEqual(sum(counts), self.settings["budget"])
        self.assertEqual(sum(p.violations(result.x).values()), 0)
        self.assertGreater(p.penalty, p.cost.max(axis=1).sum())

    def test_infeasible_data(self):
        data = dict(rooms=[dict(id="A1", capacity=100, unavailable_slots=[])],
                    events=[dict(id=str(i), slot=0, students=10, fixed_room=None,
                                 preference_cost=[0]) for i in range(2)])
        p = Problem(data)
        with self.assertRaises(ValueError):
            p.repair([0, 0], np.random.default_rng(1))
        with self.assertRaises(ValueError):
            p.exact_solution()

    def test_mutation_indices(self):
        p = self.problem
        rng = np.random.default_rng(42)
        x = p.repair(rng.integers(p.m, size=p.n), rng)
        for operator in ("reassign", "swap"):
            for _ in range(20):
                mutated = mutate(x, p, operator, rng)
                self.assertTrue(np.all((0 <= mutated) & (mutated < p.m)))
                if operator == "swap":
                    for indices in p.indices:
                        np.testing.assert_array_equal(np.sort(mutated[indices]), np.sort(x[indices]))


if __name__ == "__main__":
    unittest.main()
