"""Назначение аудиторий при фиксированном времени занятий."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


class Problem:
    def __init__(self, data):
        self.data = data
        self.rooms = data["rooms"]
        self.events = data["events"]
        self.n = len(self.events)
        self.m = len(self.rooms)
        self.slots = sorted({e["slot"] for e in self.events})
        self.indices = [np.array([i for i, e in enumerate(self.events) if e["slot"] == s])
                        for s in self.slots]
        self.capacity = np.array([r["capacity"] for r in self.rooms])
        self.size = np.array([e["students"] for e in self.events])
        self.allowed = []
        self.cost = np.zeros((self.n, self.m), dtype=int)
        for i, event in enumerate(self.events):
            choices = []
            for j, room in enumerate(self.rooms):
                self.cost[i, j] = max(room["capacity"] - event["students"], 0) + event["preference_cost"][j]
                if (room["capacity"] >= event["students"]
                        and event["slot"] not in room["unavailable_slots"]
                        and (event["fixed_room"] is None or event["fixed_room"] == j)):
                    choices.append(j)
            if not choices:
                raise ValueError(f"Для {event['id']} нет подходящей аудитории")
            self.allowed.append(choices)
        # Любое допустимое решение имеет стоимость меньше штрафа за одно нарушение.
        self.penalty = int(self.cost.max(axis=1).sum()) + 1

    @classmethod
    def load(cls, path=ROOT / "data.json"):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def evaluate(self, population):
        population = np.asarray(population)
        if population.ndim != 2 or population.shape[1] != self.n:
            raise ValueError("Неверная длина вектора назначений")
        if not np.issubdtype(population.dtype, np.integer):
            raise ValueError("Номера аудиторий должны быть целыми")
        if np.any(population < 0) or np.any(population >= self.m):
            raise ValueError("Номер аудитории вне диапазона")
        soft = self.cost[np.arange(self.n), population].sum(axis=1)
        hard = np.zeros(len(population), dtype=int)
        for i, event in enumerate(self.events):
            assigned = population[:, i]
            hard += self.capacity[assigned] < event["students"]
            hard += np.array([event["slot"] in self.rooms[r]["unavailable_slots"]
                              for r in assigned])
            if event["fixed_room"] is not None:
                hard += assigned != event["fixed_room"]
        for indices in self.indices:
            for pos, i in enumerate(indices):
                hard += (population[:, i, None] == population[:, indices[pos + 1:]]).sum(axis=1)
        return soft, hard

    def violations(self, x):
        x = np.asarray(x, dtype=int)
        result = {"capacity": 0, "unavailable": 0, "fixed_room": 0, "room_conflicts": 0}
        for i, event in enumerate(self.events):
            room = self.rooms[x[i]]
            result["capacity"] += int(room["capacity"] < event["students"])
            result["unavailable"] += int(event["slot"] in room["unavailable_slots"])
            result["fixed_room"] += int(event["fixed_room"] is not None and x[i] != event["fixed_room"])
        for indices in self.indices:
            for pos, i in enumerate(indices):
                result["room_conflicts"] += sum(int(x[i] == x[j]) for j in indices[pos + 1:])
        return result

    def repair(self, x, rng):
        """Паросочетание отдельно для каждого слота; без оптимизации мягкой цели."""
        x = np.array(x, dtype=int, copy=True)
        for indices in self.indices:
            order = sorted(indices, key=lambda i: len(self.allowed[i]))
            choices = {}
            for i in order:
                candidates = list(rng.permutation(self.allowed[i]))
                if x[i] in candidates:
                    candidates.remove(x[i])
                    candidates.insert(0, int(x[i]))
                choices[i] = candidates
            owner = {}

            def place(i, visited):
                for room in choices[i]:
                    if room in visited:
                        continue
                    visited.add(room)
                    if room not in owner or place(owner[room], visited):
                        owner[room] = i
                        return True
                return False

            for i in order:
                if not place(i, set()):
                    raise ValueError("Для временного слота нет допустимого назначения")
            for room, i in owner.items():
                x[i] = room
        return x

    def exact_solution(self):
        """Отдельный эталон: динамика по маске занятых комнат, не часть ГА."""
        x = np.empty(self.n, dtype=int)
        total = 0
        for indices in self.indices:
            states = {0: (0, [])}
            for i in indices:
                following = {}
                for mask, (value, rooms) in states.items():
                    for room in self.allowed[i]:
                        if mask & (1 << room):
                            continue
                        new_mask = mask | (1 << room)
                        candidate = value + int(self.cost[i, room])
                        if new_mask not in following or candidate < following[new_mask][0]:
                            following[new_mask] = (candidate, rooms + [room])
                states = following
            if not states:
                raise ValueError("Нет допустимого решения")
            value, rooms = min(states.values(), key=lambda pair: pair[0])
            x[indices] = rooms
            total += value
        return x, total
