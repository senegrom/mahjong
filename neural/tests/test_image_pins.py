"""The Modal training image (`neural.modal_app.image`) is built with the
tools CI tests: the Rust toolchain every workflow installs, PyTorch at the
release CI's python-engines action installs, numpy, maturin and PyTorch's
own dependencies at the pins of that action's constraints.txt, which is
copied into the image first, and both engines from their lockfiles. Modal
keeps every layer it has built, so an image left to take whatever is newest
kept the versions of its first build, and any rebuild took releases that no
test had run (review finding infra-9). The image is read through a stand-in
for the Modal client that keeps every step it is built with, in order."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github" / "actions" / "python-engines"
PINS = ACTION / "constraints.txt"


def canonical(name: str) -> str:
    """A distribution's name as package indexes compare it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path: Path) -> dict[str, str]:
    """A pin file's `name==version` lines, by canonical name."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            name, _, version = text.partition("==")
            found[canonical(name.strip())] = version.strip()
    return found


class Image:
    """Stands in for `modal.Image`, keeping each step it is built with as
    (method, arguments, keywords)."""

    def __init__(self, steps: list):
        self.steps = steps

    def __getattr__(self, method: str):
        def step(*arguments, **keywords):
            self.steps.append((method, arguments, keywords))
            return self
        return step


class App:
    def __init__(self, *arguments, **keywords): pass
    def function(self, **keywords): return lambda f: f
    def local_entrypoint(self, **keywords): return lambda f: f


def image_steps() -> list[tuple[str, tuple, dict]]:
    """Every step `neural.modal_app` builds its image with, in order."""
    steps: list = []

    def debian_slim(**keywords):
        steps.append(("debian_slim", (), keywords))
        return Image(steps)

    modal = SimpleNamespace(
        Image=SimpleNamespace(debian_slim=debian_slim), App=App,
        Volume=SimpleNamespace(from_name=lambda *arguments, **keywords: SimpleNamespace()))
    spec = importlib.util.spec_from_file_location("image_pins_modal_app",
                                                  ROOT / "neural" / "modal_app.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"modal": modal}):
        spec.loader.exec_module(module)
    return steps


def shell_commands(steps) -> list[str]:
    """The commands of the image's `run_commands` steps."""
    return [command for method, arguments, _ in steps if method == "run_commands"
            for command in arguments]


def action_torch() -> str:
    """The PyTorch release CI's python-engines action installs."""
    return re.search(r"torch==(\S+)", (ACTION / "action.yml").read_text(encoding="utf-8")).group(1)


class ImagePinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steps = image_steps()

    def test_rust_is_the_toolchain_ci_builds_the_engines_with(self):
        """rustup installed `stable`, whichever release that was on the day
        the layer was built. The workflows that build the Python engines
        with the python-engines action install one toolchain in every job,
        and the image must install the same."""
        toolchains = set()
        for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            text = workflow.read_text(encoding="utf-8")
            if "./.github/actions/python-engines" in text:
                toolchains |= {found.strip("'\"") for found in
                               re.findall(r"^\s*toolchain:\s*(\S+)", text, re.MULTILINE)}
        self.assertEqual(len(toolchains), 1, f"the workflows install {sorted(toolchains)}")
        installs = [re.search(r"--default-toolchain\s+(\S+)", command)
                    for command in shell_commands(self.steps) if "rustup" in command]
        self.assertEqual(len(installs), 1, "the image installs Rust once, with rustup")
        self.assertIsNotNone(installs[0], "rustup must be told which toolchain")
        self.assertEqual(installs[0].group(1), toolchains.pop())

    def test_python_packages_are_installed_at_the_pins_ci_tests(self):
        """numpy and maturin were open ranges and PyTorch's dependencies
        whatever was newest. constraints.txt, copied in before the install,
        now holds every one of them to CI's release, and PyTorch is the
        release CI installs."""
        methods = [method for method, _, _ in self.steps]
        self.assertEqual(methods.count("pip_install"), 1)
        install = methods.index("pip_install")
        packages, keywords = self.steps[install][1], self.steps[install][2]
        self.assertIn("-c /src/constraints.txt", keywords.get("extra_options", ""))
        copies = [index for index, (method, arguments, keywords) in enumerate(self.steps)
                  if method == "add_local_file" and Path(arguments[0]).resolve() == PINS
                  and arguments[1] == "/src/constraints.txt" and keywords.get("copy") is True]
        self.assertTrue(copies and copies[0] < install,
                        "constraints.txt must be built into the image before the install")
        torch = [package for package in packages
                 if canonical(re.split(r"[\s=<>!~;\[]", package, maxsplit=1)[0]) == "torch"]
        self.assertEqual(torch, [f"torch=={action_torch()}"])
        pinned = pins(PINS)
        for package in packages:
            if package in torch:
                continue
            self.assertRegex(package, r"^[A-Za-z0-9._-]+$",
                             "a version for the image belongs in constraints.txt")
            self.assertIn(canonical(package), pinned)

    def test_engines_build_from_their_lockfiles(self):
        """maturin built without --locked, so a lockfile that no longer
        matched its Cargo.toml was re-resolved to the newest crates instead
        of failing the build, as it fails CI's."""
        builds = [command for command in shell_commands(self.steps) if "maturin build" in command]
        self.assertEqual(len(builds), 2, "riichi-py and libriichi")
        for build in builds:
            self.assertIn("--locked", build.split())

    def test_pytorchs_dependencies_are_pinned_where_the_image_reads_them(self):
        """CI installs PyTorch without its dependencies and the image with
        constraints.txt, so each dependency PyTorch leaves open must be
        pinned there, and not only among the export tools, which the image
        never reads. Those PyTorch pins exactly itself, its CUDA libraries,
        are fixed by its release. Read from the PyTorch installed here,
        which in CI is the release the action pins."""
        try:
            installed = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("PyTorch is not installed")
        if installed.split("+", 1)[0] != action_torch():
            self.skipTest(f"PyTorch {installed} is installed, not the pinned {action_torch()}")
        pinned = pins(PINS)
        for requirement in importlib.metadata.requires("torch") or []:
            specifier = requirement.split(";", 1)[0]
            if "extra ==" in requirement or "==" in specifier:
                continue
            name = re.match(r"[A-Za-z0-9._-]+", specifier).group(0)
            self.assertIn(canonical(name), pinned,
                          f"PyTorch needs {requirement!r}, which constraints.txt does not pin")


if __name__ == "__main__":
    unittest.main()
