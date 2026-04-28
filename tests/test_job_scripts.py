from __future__ import annotations

import importlib
import inspect
import shlex
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest
from typer.models import OptionInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = REPO_ROOT / "jobs"


def _logical_shell_lines(path: Path) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        lines.append((current + line).strip())
        current = ""
    if current:
        lines.append(current.strip())
    return lines


def _script_invocations(path: Path) -> Iterable[tuple[str, list[str]]]:
    for line in _logical_shell_lines(path):
        if "scripts/" not in line:
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as exc:
            raise AssertionError(f"{path}: invalid shell quoting in line: {line}") from exc

        for i, token in enumerate(tokens):
            if token.startswith("scripts/") and token.endswith(".py"):
                yield token.replace("\\", "/"), tokens[i + 1 :]


def _split_param_decl(decl: str) -> list[str]:
    return [part for part in decl.split("/") if part.startswith("-")]


def _is_bool_annotation(annotation: Any) -> bool:
    if annotation is bool:
        return True
    return bool(get_origin(annotation) is None and annotation == "bool")


def _enum_type(annotation: Any) -> type[Enum] | None:
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return annotation
    for arg in get_args(annotation):
        if inspect.isclass(arg) and issubclass(arg, Enum):
            return arg
    return None


def _cli_options(script: str) -> dict[str, tuple[str, Any, bool]]:
    module_name = script.removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        return {}

    options: dict[str, tuple[str, Any, bool]] = {}
    type_hints = get_type_hints(main)
    for name, param in inspect.signature(main).parameters.items():
        default = param.default
        if not isinstance(default, OptionInfo):
            continue
        annotation = type_hints.get(name, param.annotation)
        requires_value = not _is_bool_annotation(annotation)
        for decl in default.param_decls:
            for flag in _split_param_decl(decl):
                options[flag] = (name, annotation, requires_value)
    return options


def _job_scripts() -> list[Path]:
    return sorted(path for path in JOBS_DIR.glob("*.sh") if path.name != "_env.sh")


@pytest.mark.parametrize("job_path", _job_scripts(), ids=lambda p: p.name)
def test_job_python_invocations_use_valid_cli_options(job_path: Path) -> None:
    invocations = list(_script_invocations(job_path))
    if not invocations:
        pytest.skip(f"{job_path.name} uses an external CLI entrypoint")

    for script, args in invocations:
        options = _cli_options(script)
        if not options:
            continue

        i = 0
        while i < len(args):
            token = args[i]
            if token == "--":
                break
            if not token.startswith("-"):
                i += 1
                continue

            flag, sep, inline_value = token.partition("=")
            assert flag in options, f"{job_path.name}: {script} uses unknown option {flag!r}"

            _, annotation, requires_value = options[flag]
            value: str | None = inline_value if sep else None
            if requires_value and value is None:
                assert i + 1 < len(args), f"{job_path.name}: {script} option {flag!r} is missing a value"
                value = args[i + 1]
                assert not value.startswith("--"), f"{job_path.name}: {script} option {flag!r} is missing a value"
                i += 1

            enum_cls = _enum_type(annotation)
            if enum_cls is not None and value is not None and not value.startswith("$"):
                choices = {str(member.value) for member in enum_cls}
                assert value in choices, (
                    f"{job_path.name}: {script} option {flag!r} has invalid value {value!r}; "
                    f"expected one of {sorted(choices)}"
                )

            i += 1


def test_all_train_srpo_update_methods_are_explicitly_supported() -> None:
    options = _cli_options("scripts/train_srpo.py")
    _, annotation, _ = options["--update-method"]
    enum_cls = _enum_type(annotation)
    assert enum_cls is not None

    seen: set[str] = set()
    for job_path in _job_scripts():
        for script, args in _script_invocations(job_path):
            if script != "scripts/train_srpo.py":
                continue
            for i, token in enumerate(args):
                if token == "--update-method" and i + 1 < len(args):
                    seen.add(args[i + 1])
                elif token.startswith("--update-method="):
                    seen.add(token.split("=", 1)[1])

    supported = {str(member.value) for member in enum_cls}
    assert seen <= supported
    assert "success_bc" in supported
