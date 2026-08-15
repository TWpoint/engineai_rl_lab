from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg_v7_n3_scale import T800FlatV7N3ScalePPORunnerCfg


@configclass
class T800FlatV8ScalePPORunnerCfg(T800FlatV7N3ScalePPORunnerCfg):
    """V8 Scale/LaFAN runner using the V7-N3 agent architecture."""

    run_name = "v8_scale_lafan"
