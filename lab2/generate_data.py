"""Воспроизводимый синтетический экземпляр варианта 9."""

import json

import numpy as np

from problem import ROOT, Problem


def main():
    rng = np.random.default_rng(926)
    rooms = [dict(id=f"A{j + 1}", capacity=c, unavailable_slots=[])
             for j, c in enumerate([25, 35, 45, 60, 80, 100])]
    # В первых трёх слотах одна аудитория закрыта.
    for slot in range(3):
        rooms[int(rng.integers(6))]["unavailable_slots"].append(slot)
    events = []
    witness = []
    for slot in range(6):
        available = [j for j, room in enumerate(rooms) if slot not in room["unavailable_slots"]]
        selected = rng.choice(available, size=5, replace=False)
        for pos, room in enumerate(selected):
            i = len(events)
            # Размер группы помещается в исходную аудиторию; сам ответ в данные не записывается.
            students = int(rng.integers(max(15, rooms[room]["capacity"] - 25), rooms[room]["capacity"] + 1))
            events.append(dict(id=f"Z{i + 1:02d}", slot=slot, students=students,
                               fixed_room=int(room) if slot in (0, 2, 4) and pos == 0 else None,
                               preference_cost=rng.integers(0, 41, size=6).tolist()))
            witness.append(int(room))
    data = dict(seed=926, description="30 занятий, 6 фиксированных временных слотов, 6 аудиторий",
                slots=[f"День {s // 3 + 1}, пара {s % 3 + 1}" for s in range(6)],
                rooms=rooms, events=events)
    problem = Problem(data)
    assert sum(problem.violations(witness).values()) == 0
    (ROOT / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("data.json: 30 занятий, 6 аудиторий, 60 пар конфликтующих по времени занятий")


if __name__ == "__main__":
    main()
