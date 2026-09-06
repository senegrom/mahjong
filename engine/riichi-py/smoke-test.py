"""Exercise the actual Python extension without training/NumPy dependencies."""
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("riichi_py", root / "target/debug/libriichi_py.so")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arena = module.Arena(games=2, seed=81)
assert arena.games == 2
assert len(arena.observations()) == 2 * module.PLANES * module.POSITIONS * 4
assert len(arena.legal_mask()) == 2 * module.ACTIONS
assert len(arena.seats()) == 2
assert not arena.all_finished()
snapshot = arena.observations()
saved_hex = snapshot.hex()
for _ in range(50):
    mask = arena.legal_mask()
    actions = [next(i for i, legal in enumerate(mask[g * module.ACTIONS:(g + 1) * module.ACTIONS]) if legal) for g in range(2)]
    arena.step(actions)
assert len(arena.observations()) == len(snapshot)
assert snapshot.hex() == saved_hex
print("Python binding import, ABI, byte buffers and 100 game decisions passed")
