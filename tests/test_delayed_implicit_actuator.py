from __future__ import annotations

import torch
from engineai_rl_lab.tasks.tracking.robots.actuator import (
    DelayedImplicitActuator,
    DelayedImplicitActuatorCfg,
)

from isaaclab.utils.types import ArticulationActions


def test_reset_pose_seed_precedes_first_policy_target_at_configured_lag() -> None:
    cfg = DelayedImplicitActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=1.0,
        damping=0.1,
        joint_effort_limit=10.0,
        min_delay=1,
        max_delay=3,
    )
    actuator = DelayedImplicitActuator(
        cfg,
        joint_names=["joint"],
        joint_ids=slice(None),
        num_envs=3,
        device="cpu",
        stiffness=1.0,
        damping=0.1,
        joint_effort_limit=10.0,
    )
    actuator.reset(None)
    for buffer in (
        actuator.positions_delay_buffer,
        actuator.velocities_delay_buffer,
        actuator.efforts_delay_buffer,
    ):
        buffer.set_time_lag(torch.tensor([1, 2, 3], dtype=torch.int))

    reset_pose = torch.tensor([[10.0], [20.0], [30.0]])
    policy_target = torch.tensor([[100.0], [200.0], [300.0]])
    actuator.seed_position_history(reset_pose, None)

    outputs = []
    for _ in range(4):
        result = actuator.compute(
            ArticulationActions(
                joint_positions=policy_target.clone(),
                joint_velocities=torch.zeros_like(policy_target),
                joint_efforts=torch.zeros_like(policy_target),
            ),
            joint_pos=reset_pose,
            joint_vel=torch.zeros_like(reset_pose),
        )
        outputs.append(result.joint_positions.clone())

    expected = (
        reset_pose,
        torch.tensor([[100.0], [20.0], [30.0]]),
        torch.tensor([[100.0], [200.0], [30.0]]),
        policy_target,
    )
    for output, expected_output in zip(outputs, expected, strict=True):
        torch.testing.assert_close(output, expected_output)
