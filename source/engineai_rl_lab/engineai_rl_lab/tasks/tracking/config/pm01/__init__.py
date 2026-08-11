import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-PM01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PM01FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-PM01-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PM01FlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-PM01-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:PM01FlatLowFreqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PM01FlatLowFreqPPORunnerCfg",
    },
)
