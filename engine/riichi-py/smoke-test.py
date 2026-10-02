"""Exercise the actual Python extension without training/NumPy dependencies."""
import importlib.util
import sys
from importlib.machinery import ExtensionFileLoader
from pathlib import Path

root = Path(__file__).resolve().parents[2]
# What `cargo build -p riichi-py` names the library on each platform. Only
# the Linux name ends in a suffix Python recognises as an extension, so the
# loader is named rather than guessed from the suffix.
built = {"win32": "riichi_py.dll", "darwin": "libriichi_py.dylib"}.get(sys.platform, "libriichi_py.so")
library = root / "target" / "debug" / built
if not library.is_file():
    sys.exit(f"{library} is missing: run `cargo build --locked -p riichi-py` first")
spec = importlib.util.spec_from_file_location(
    "riichi_py", library, loader=ExtensionFileLoader("riichi_py", str(library)))
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
