from __future__ import annotations

import re
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.envs.mdp.events import randomize_rigid_body_com as _IsaacLabRandomizeRigidBodyCom
from isaaclab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_SAFE_NEWTON_COM_VERSIONS = {
    "newton": (1, 5, 0),
    "mujoco-warp": (3, 11, 0),
}


def _release_tuple(package_name: str, package_version: str) -> tuple[int, int, int]:
    # Accept stable PEP 440 releases (and harmless local build metadata) while
    # conservatively rejecting prerelease/dev builds at the minimum boundary.
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?(?:\+[A-Za-z0-9.-]+)?\s*", package_version)
    if match is None:
        raise RuntimeError(f"Cannot validate {package_name} version {package_version!r} for Newton CoM randomization.")
    return tuple(int(value or 0) for value in match.groups())


def _require_safe_newton_com_randomization(
    physics_manager_name: str,
    version_resolver: Callable[[str], str] = version,
) -> None:
    """Reject Newton stacks that predate safe per-world runtime CoM updates."""
    if "newton" not in physics_manager_name.lower():
        return

    for package_name, minimum_version in _SAFE_NEWTON_COM_VERSIONS.items():
        try:
            installed_version = version_resolver(package_name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Newton CoM randomization requires {package_name}>={'.'.join(map(str, minimum_version))}, "
                f"but {package_name} is not installed."
            ) from exc
        if _release_tuple(package_name, installed_version) < minimum_version:
            raise RuntimeError(
                f"Newton CoM randomization requires {package_name}>={'.'.join(map(str, minimum_version))}; "
                f"found {installed_version}. Upgrade the Newton stack before training."
            )


class randomize_rigid_body_com(_IsaacLabRandomizeRigidBodyCom):
    """Use Isaac Lab CoM randomization only with a Newton stack that safely rebuilds dynamics."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        _require_safe_newton_com_randomization(env.sim.physics_manager.__name__)
        super().__init__(cfg, env)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = asset.data.default_joint_pos.torch[0].clone()

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.torch.clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        asset_env_ids = env_ids[:, None] if joint_ids != slice(None) else env_ids
        asset.data.default_joint_pos.torch[asset_env_ids, joint_ids] = pos

        # Action offsets are action-local, so asset-global joint IDs cannot be
        # used directly when the policy controls only a joint subset.
        action = env.action_manager.get_term("joint_pos")
        if isinstance(action._offset, torch.Tensor):
            if joint_ids == slice(None):
                selected_joint_names = list(asset.joint_names)
            else:
                selected_joint_names = [asset.joint_names[index] for index in asset_cfg.joint_ids]
            action_name_to_id = {name: index for index, name in enumerate(action._joint_names)}
            selected_columns = [column for column, name in enumerate(selected_joint_names) if name in action_name_to_id]
            action_joint_ids = [action_name_to_id[selected_joint_names[column]] for column in selected_columns]
            if action_joint_ids:
                action_env_ids = env_ids[:, None]
                action._offset[action_env_ids, torch.tensor(action_joint_ids, device=asset.device)] = pos[
                    :, selected_columns
                ]
