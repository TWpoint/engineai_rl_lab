from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg import T800FlatWoStateEstimationEnvCfg
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v2 import T800FlatObservationsCfgV2


@configclass
class T800FlatObservationsCfgV3(T800FlatObservationsCfgV2):
    """V2 observations with a shorter history and an unstacked last action."""

    @configclass
    class PolicyCfg(T800FlatObservationsCfgV2.PolicyCfg):
        # The last action is exposed through LastActionCfg so it is not stacked.
        actions = None

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5

    @configclass
    class LastActionCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 4
            self.flatten_history_dim = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    last_action: LastActionCfg = LastActionCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV3(T800FlatWoStateEstimationEnvCfg):
    """T800 V3 environment with five observations and four previous actions."""

    observations: T800FlatObservationsCfgV3 = T800FlatObservationsCfgV3()
