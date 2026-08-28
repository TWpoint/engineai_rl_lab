from isaaclab.utils.configclass import configclass

from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v25 import (
    T800_V25_FRAME_OFFSETS,
    T800FlatWoStateEstimationEnvCfgV25Scale,
)

T800_V26_FRAME_OFFSETS = T800_V25_FRAME_OFFSETS.copy()


@configclass
class T800FlatWoStateEstimationEnvCfgV26Scale(T800FlatWoStateEstimationEnvCfgV25Scale):
    """V25 task unchanged; V26 differs only in adaptive rollout sampling."""
