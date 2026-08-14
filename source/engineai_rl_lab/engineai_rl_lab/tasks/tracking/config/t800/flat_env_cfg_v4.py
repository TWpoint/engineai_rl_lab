from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v3 import (
    T800FlatObservationsCfgV3,
    T800FlatWoStateEstimationEnvCfgV3,
)


@configclass
class T800FlatObservationsCfgV4(T800FlatObservationsCfgV3):
    """V3 observations with last actions in a separate four-frame group."""

    @configclass
    class PolicyCfg(T800FlatObservationsCfgV3.PolicyCfg):
        actions = None

    @configclass
    class ActionCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 4
            self.flatten_history_dim = False

    policy: PolicyCfg = PolicyCfg()
    action: ActionCfg = ActionCfg()


@configclass
class T800FlatWoStateEstimationEnvCfgV4(T800FlatWoStateEstimationEnvCfgV3):
    """T800 V3 environment with a separate four-frame action history."""

    observations: T800FlatObservationsCfgV4 = T800FlatObservationsCfgV4()
