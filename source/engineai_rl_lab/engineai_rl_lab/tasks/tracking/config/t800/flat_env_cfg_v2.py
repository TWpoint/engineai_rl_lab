from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg import T800FlatWoStateEstimationEnvCfg
from engineai_rl_lab.tasks.tracking.tracking_env_cfg import ObservationsCfg


@configclass
class T800FlatObservationsCfgV2(ObservationsCfg):
    """Observations with separate proprioception history and unstacked commands."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Command-related terms are exposed through CommandCfg instead.
        command = None
        motion_anchor_pos_b = None
        motion_anchor_ori_b = None

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    command: CommandCfg = CommandCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV2(T800FlatWoStateEstimationEnvCfg):
    """T800 environment with temporal proprioception and unstacked commands."""

    observations: T800FlatObservationsCfgV2 = T800FlatObservationsCfgV2()
