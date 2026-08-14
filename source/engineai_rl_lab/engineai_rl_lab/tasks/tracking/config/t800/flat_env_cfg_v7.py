from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v6 import (
    T800FlatObservationsCfgV6,
    T800FlatWoStateEstimationEnvCfgV6,
)


DEFAULT_FRAME_OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


@configclass
class T800FlatObservationsCfgV7(T800FlatObservationsCfgV6):
    """V6 commands grouped into one temporal trajectory per body entity."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pose_b = ObsTerm(
            func=mdp.motion_body_pose_b_window_by_entity,
            params={"command_name": "motion", "frame_offsets": DEFAULT_FRAME_OFFSETS},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV7(T800FlatWoStateEstimationEnvCfgV6):
    """T800 V7 environment with entity-major temporal pose commands."""

    observations: T800FlatObservationsCfgV7 = T800FlatObservationsCfgV7()

    def set_command_offsets(self, frame_offsets: list[int] | tuple[int, ...]) -> None:
        """Select command frames by offsets relative to the current motion frame."""
        if not frame_offsets:
            raise ValueError("frame_offsets must contain at least one frame")
        if any(not isinstance(offset, int) for offset in frame_offsets):
            raise TypeError("frame_offsets must contain integers only")
        self.observations.command.link_pose_b.params["frame_offsets"] = list(frame_offsets)
