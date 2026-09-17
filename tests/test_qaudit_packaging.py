"""Package metadata, dependency bounds, and installable source-layout contracts.

These tests inspect packaging declarations against imports and files in the
repository. They check package scope, supported interpreters, version metadata,
and the pytest import path.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

try:  # tomllib is 3.11+; requires-python floor is 3.10, so fall back.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 without tomli
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PKG_DIR = REPO_ROOT / "src" / "qaudit"

# Declared third-party dependencies are checked against imports by
# test_declared_deps_match_measured_imports.
MEASURED_THIRD_PARTY = {"numpy", "pandas", "scipy"}

pytestmark = pytest.mark.skipif(
    tomllib is None, reason="no TOML parser on this interpreter (py3.10 sans tomli)"
)


def _load_pyproject() -> dict:
    assert PYPROJECT.is_file(), f"root pyproject.toml missing at {PYPROJECT}"
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _req_name(req: str) -> str:
    m = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", req.strip())
    assert m, f"unparseable requirement: {req!r}"
    return m.group(0).lower()


def _version_tuple(v: str) -> tuple[int, ...]:
    """Numeric prefix of a version string, e.g. '1.26.4' -> (1, 26, 4)."""
    parts: list[int] = []
    for piece in v.split("."):
        m = re.match(r"^\d+", piece)
        if not m:
            break
        parts.append(int(m.group(0)))
    assert parts, f"no numeric version prefix in {v!r}"
    return tuple(parts)


def _third_party_imports(pkg_dir: Path) -> set[str]:
    """Top-level absolute imports across the package, minus stdlib and self."""
    found: set[str] = set()
    for py in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)  # py3.10+
    return {n for n in found if n not in stdlib and n != "qaudit"}


# ---------------------------------------------------------------- metadata


def test_distribution_name_and_package_scope() -> None:
    # Only qaudit packages belong to this distribution.
    full = _load_pyproject()
    proj = full["project"]
    # The short ``qaudit`` distribution name belongs to an unrelated PyPI
    # project.  Keep the public import package while publishing under an
    # unambiguous distribution name.
    assert proj["name"] == "qaudit-backtest"
    include = full["tool"]["setuptools"]["packages"]["find"].get("include", [])
    assert include and all(pat.startswith("qaudit") for pat in include), (
        f"packages.find must be pinned to qaudit*, got {include!r}"
    )


def test_declared_deps_match_measured_imports() -> None:
    # An undeclared import breaks a fresh install; a declared dependency that
    # is never imported is dead weight. Both directions must agree.
    proj = _load_pyproject()["project"]
    declared = {_req_name(r) for r in proj["dependencies"]}
    measured = _third_party_imports(PKG_DIR)
    assert measured == MEASURED_THIRD_PARTY, (
        f"qaudit's third-party imports changed: {sorted(measured)} vs measured "
        f"{sorted(MEASURED_THIRD_PARTY)}; update pyproject dependencies AND this test"
    )
    assert declared == measured, (
        f"pyproject dependencies {sorted(declared)} != imports actually used "
        f"{sorted(measured)} - bounds must be measured, not guessed"
    )


def test_dependency_bounds_are_fail_closed_and_satisfiable() -> None:
    # The installed stack must satisfy declared bounds, and the CI matrix must cover
    # supported dependency majors.
    import numpy
    import pandas
    import scipy

    installed = {
        "numpy": _version_tuple(numpy.__version__),
        "pandas": _version_tuple(pandas.__version__),
        "scipy": _version_tuple(scipy.__version__),
    }
    proj = _load_pyproject()["project"]
    seen: set[str] = set()
    for req in proj["dependencies"]:
        name = _req_name(req)
        seen.add(name)
        specs = re.findall(r"(>=|<=|<|>|==)\s*([0-9][0-9.]*)", req)
        assert specs, f"{req!r} has no version bounds - deps must be bounded"
        ops = {op for op, _ in specs}
        assert ">=" in ops, f"{req!r} missing a floor"
        assert "<" in ops or "<=" in ops or "==" in ops, f"{req!r} missing a cap (fail-closed)"
        for op, ver in specs:
            v, have = _version_tuple(ver), installed[name]
            ok = {
                ">=": have >= v,
                ">": have > v,
                "<=": have <= v,
                "<": have < v,
                "==": have == v,
            }[op]
            assert ok, (
                f"installed {name} {installed[name]} violates declared "
                f"bound {op}{ver} - environment and metadata disagree"
            )
    assert seen == MEASURED_THIRD_PARTY
    caps = {_req_name(r): re.search(r"<\s*([0-9]+)", r).group(1)  # type: ignore[union-attr]
            for r in proj["dependencies"]}
    assert caps == {"numpy": "3", "pandas": "4", "scipy": "2"}, (
        f"dependency caps moved ({caps!r}): admit a new major only after a "
        f"green full-suite run on it, add it to the CI matrix, and update "
        f"this regression"
    )


def test_requires_python_floor() -> None:
    # The floor is measured from syntax. There is no upper cap: 3.13 needs
    # NumPy 2 (numpy 1.26 ships no cp313 wheels), the dependency bounds admit
    # it, and the CI matrix tests 3.13 as a blocking job.
    proj = _load_pyproject()["project"]
    assert proj["requires-python"] == ">=3.10"
    assert sys.version_info >= (3, 10), "suite running below the declared floor"
    classifiers = proj.get("classifiers", [])
    for minor in (10, 11, 12, 13):
        assert f"Programming Language :: Python :: 3.{minor}" in classifiers
    assert "Typing :: Typed" in classifiers, "py.typed ships, advertise it"


def test_ci_matrix_covers_every_supported_interpreter_and_major() -> None:
    # Every classifier-advertised Python runs as a blocking job, and both
    # NumPy majors are exercised: the ``minimum`` job pins numpy 1.24 and the
    # ``numpy1`` job installs the last 1.x, while the ``latest`` jobs resolve
    # NumPy 2 and pandas 3. No ``continue-on-error`` canary may hide a red
    # interpreter behind a green badge.
    ci = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text()
    for minor in (10, 11, 12, 13):
        assert f'"3.{minor}"' in ci, f"CI matrix does not test Python 3.{minor}"
    assert "continue-on-error" not in ci, "no non-blocking canary jobs"
    assert "--ignore-requires-python" not in ci, "no requires-python bypass"
    assert "numpy==1.24.0" in ci and "pandas==2.0.0" in ci, "minimum-pins job missing"
    assert "'numpy<2'" in ci, "NumPy 1.x job missing"
    assert "python -m pytest -W error -q" in ci


def test_version_single_source() -> None:
    # dynamic version must point at qaudit.__version__ (single source of truth).
    data = _load_pyproject()
    assert data["project"]["dynamic"] == ["version"]
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "qaudit.__version__"
    import qaudit

    assert re.fullmatch(r"\d+(\.\d+)+([abc]|rc)?\d*", qaudit.__version__), (
        f"qaudit.__version__ = {qaudit.__version__!r} is not PEP 440-ish"
    )


def test_packages_pinned_no_autofind_leak() -> None:
    # Automatic package discovery would ship any stray src/ package in the
    # qaudit distribution.
    find = _load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert find["include"] == ["qaudit*"], (
        f"packages.find include widened to {find['include']}"
    )


def test_license_declared_and_shipped() -> None:
    # PEP 639 license expression plus the LICENSE file in every distribution.
    proj = _load_pyproject()["project"]
    assert proj["license"] == "MIT"
    assert proj["license-files"] == ["LICENSE"]
    assert not any(c.startswith("License ::") for c in proj.get("classifiers", [])), (
        "PEP 639: the license expression replaces the deprecated classifier"
    )
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Permission is hereby granted" in text


def test_py_typed_marker_ships() -> None:
    # PEP 561: downstream mypy must see the inline annotations.
    assert (PKG_DIR / "py.typed").is_file()
    data = _load_pyproject()
    assert data["tool"]["setuptools"]["package-data"]["qaudit"] == ["py.typed"]


def test_warnings_are_errors_in_pytest_config() -> None:
    # A plain local ``pytest`` must be as strict as the CI command line.
    ini = _load_pyproject()["tool"]["pytest"]["ini_options"]
    assert ini["filterwarnings"] == ["error"]


# ------------------------------------------------------ source-layout imports


def test_pytest_imports_the_source_tree() -> None:
    ini = _load_pyproject()["tool"]["pytest"]["ini_options"]
    assert ini["pythonpath"] == ["src"], "tests must import qaudit from src/ without an install"
    assert ini["testpaths"] == ["tests"], "bare `pytest` must collect tests/ only"


def test_qaudit_importable_and_self_contained() -> None:
    # qaudit must import only itself, the stdlib, and its declared
    # dependencies - an import of any other src/ module would make the
    # installed dist (which ships only qaudit*) broken at runtime.
    declared = {_req_name(r) for r in _load_pyproject()["project"]["dependencies"]}
    stray = _third_party_imports(PKG_DIR) - declared
    assert not stray, f"qaudit imports undeclared modules {sorted(stray)}"
    import qaudit
    from qaudit import AuditConfig, BacktestArtifacts, audit  # noqa: F401

    assert callable(audit) and qaudit.__version__
