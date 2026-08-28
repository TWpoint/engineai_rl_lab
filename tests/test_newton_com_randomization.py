from __future__ import annotations

from collections.abc import Callable

import newton
import numpy as np
import pytest
import warp as wp
from engineai_rl_lab.tasks.tracking.mdp.events import _require_safe_newton_com_randomization
from newton.solvers import SolverMuJoCo

_COM_OFFSETS = np.asarray([0.2, -0.2, 0.05, 0.0], dtype=np.float32)


def _versions(**overrides: str) -> Callable[[str], str]:
    package_versions = {"newton": "1.5.0", "mujoco-warp": "3.11.0", **overrides}
    return package_versions.__getitem__


def _pendulum_template(com_x: float) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    body = builder.add_link(
        mass=1.0,
        com=wp.vec3(com_x, 0.0, 0.0),
        inertia=wp.diag(wp.vec3(0.05, 0.05, 0.05)),
    )
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
    builder.add_articulation([joint])
    return builder


def _runtime_com_model() -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    builder.replicate(_pendulum_template(0.0), len(_COM_OFFSETS))
    return builder.finalize(device="cpu")


def _precompiled_com_model() -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    for world_id, com_x in enumerate(_COM_OFFSETS):
        builder.add_world(
            _pendulum_template(float(com_x)),
            xform=wp.transform((2.0 * world_id, 0.0, 0.0), wp.quat_identity()),
        )
    return builder.finalize(device="cpu")


def _initialize_solver(model: newton.Model) -> tuple[newton.State, newton.State, SolverMuJoCo]:
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver = SolverMuJoCo(model, iterations=10, ls_iterations=5, disable_contacts=True)
    return state_in, state_out, solver


def _step_precompiled_model(model: newton.Model) -> tuple[newton.State, SolverMuJoCo]:
    state_in, state_out, solver = _initialize_solver(model)
    solver.step(state_in, state_out, model.control(), None, 0.002)
    return state_out, solver


def _update_com_and_step(model: newton.Model) -> tuple[newton.State, SolverMuJoCo]:
    state_in, state_out, solver = _initialize_solver(model)
    body_com = model.body_com.numpy().copy()
    body_com[:, 0] = _COM_OFFSETS
    model.body_com.assign(body_com)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    solver.step(state_in, state_out, model.control(), None, 0.002)
    return state_out, solver


def test_com_version_guard_bypasses_non_newton_backends() -> None:
    def fail_if_called(_package_name: str) -> str:
        raise AssertionError("PhysX must not query Newton package versions")

    _require_safe_newton_com_randomization("PhysXManager", fail_if_called)


def test_com_version_guard_accepts_supported_stack() -> None:
    _require_safe_newton_com_randomization("NewtonManager", _versions())


def test_com_version_guard_accepts_installed_training_stack() -> None:
    _require_safe_newton_com_randomization("NewtonManager")


def test_com_version_guard_rejects_unvalidated_prerelease() -> None:
    with pytest.raises(RuntimeError, match="Cannot validate newton version"):
        _require_safe_newton_com_randomization("NewtonManager", _versions(newton="1.5.0rc1"))


@pytest.mark.parametrize(
    ("overrides", "expected_package"),
    [
        ({"newton": "1.4.9"}, "newton"),
        ({"mujoco-warp": "3.10.9"}, "mujoco-warp"),
    ],
)
def test_com_version_guard_rejects_unsafe_stack(overrides: dict[str, str], expected_package: str) -> None:
    with pytest.raises(RuntimeError, match=rf"{expected_package}>="):
        _require_safe_newton_com_randomization("NewtonManager", _versions(**overrides))


def test_runtime_com_update_matches_precompiled_dynamics_on_cpu() -> None:
    reference_state, reference_solver = _step_precompiled_model(_precompiled_com_model())
    runtime_state, runtime_solver = _update_com_and_step(_runtime_com_model())

    reference_mass_matrix = reference_solver.mjw_data.M.numpy().copy()
    runtime_mass_matrix = runtime_solver.mjw_data.M.numpy().copy()
    reference_joint_velocity = reference_state.joint_qd.numpy().copy()
    runtime_joint_velocity = runtime_state.joint_qd.numpy().copy()

    assert np.isfinite(runtime_mass_matrix).all()
    assert np.isfinite(runtime_joint_velocity).all()
    np.testing.assert_allclose(runtime_mass_matrix, reference_mass_matrix, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(runtime_joint_velocity, reference_joint_velocity, rtol=2.0e-5, atol=2.0e-6)
