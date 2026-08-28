from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_pure_override_helper():
    """Load only the pure helper without importing the CLI-style evaluator module."""
    script_path = Path(__file__).parents[1] / "scripts" / "tracking" / "evaluate_frame0_macro.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_apply_concrete_newton_overrides"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {}
    exec(compile(isolated_module, str(script_path), "exec"), namespace)  # noqa: S102
    return namespace["_apply_concrete_newton_overrides"]


def test_diagnostic_overrides_modify_selected_concrete_newton_cfg() -> None:
    apply_overrides = _load_pure_override_helper()
    physics_cfg = SimpleNamespace(
        num_substeps=1,
        default_shape_cfg=SimpleNamespace(margin=0.0),
    )

    apply_overrides(physics_cfg, num_substeps=4, contact_margin=0.025)

    assert physics_cfg.num_substeps == 4
    assert physics_cfg.default_shape_cfg.margin == pytest.approx(0.025)


@pytest.mark.parametrize(
    ("kwargs", "missing_attribute"),
    [
        ({"num_substeps": 2, "contact_margin": None}, "num_substeps"),
        ({"num_substeps": None, "contact_margin": 0.01}, "default_shape_cfg.margin"),
    ],
)
def test_diagnostic_overrides_reject_non_newton_cfg(kwargs, missing_attribute: str) -> None:
    apply_overrides = _load_pure_override_helper()

    with pytest.raises(ValueError, match=rf"physics=newton_mjwarp.*{missing_attribute}"):
        apply_overrides(SimpleNamespace(), **kwargs)
