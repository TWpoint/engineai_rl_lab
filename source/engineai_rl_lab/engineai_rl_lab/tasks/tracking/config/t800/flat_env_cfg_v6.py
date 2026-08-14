from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v5 import (
    T800FlatObservationsCfgV5,
    T800FlatWoStateEstimationEnvCfgV5,
)


@configclass
class T800FlatObservationsCfgV6(T800FlatObservationsCfgV5):
    """V5 observations with a configurable temporal reference-command window."""

    @configclass
    class CommandCfg(ObsGroup):
        link_pos_b = ObsTerm(
            func=mdp.motion_body_pos_b_window,
            params={"command_name": "motion", "frame_offsets": [-4, -3, -2, -1, 0, 1, 2, 3, 4]},
        )
        link_ori_b = ObsTerm(
            func=mdp.motion_body_ori_b_window,
            params={"command_name": "motion", "frame_offsets": [-4, -3, -2, -1, 0, 1, 2, 3, 4]},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV6(T800FlatWoStateEstimationEnvCfgV5):
    """T800 V6 environment with past/current/future key-body pose commands."""

    observations: T800FlatObservationsCfgV6 = T800FlatObservationsCfgV6()

    def set_command_offsets(self, frame_offsets: list[int] | tuple[int, ...]) -> None:
        """Select command frames by offsets relative to the current motion frame."""
        if not frame_offsets:
            raise ValueError("frame_offsets must contain at least one frame")
        if any(not isinstance(offset, int) for offset in frame_offsets):
            raise TypeError("frame_offsets must contain integers only")
        for term in (self.observations.command.link_pos_b, self.observations.command.link_ori_b):
            term.params["frame_offsets"] = list(frame_offsets)
