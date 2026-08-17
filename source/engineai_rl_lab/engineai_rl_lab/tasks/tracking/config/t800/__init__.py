import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-T800-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T800FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T800FlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v1:T800FlatWoStateEstimationEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v1:T800FlatV1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v1-n1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v1:T800FlatWoStateEstimationEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v1:T800FlatV1N1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v2:T800FlatWoStateEstimationEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v2:T800FlatV2PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v3:T800FlatWoStateEstimationEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v3:T800FlatV3PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v3-n1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v3:T800FlatWoStateEstimationEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v3:T800FlatV3N1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v4:T800FlatWoStateEstimationEnvCfgV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v4:T800FlatV4PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v5:T800FlatWoStateEstimationEnvCfgV5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v5:T800FlatV5PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v6:T800FlatWoStateEstimationEnvCfgV6",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v6:T800FlatV6PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v7:T800FlatWoStateEstimationEnvCfgV7",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v7:T800FlatV7PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v7-n1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v7:T800FlatWoStateEstimationEnvCfgV7",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v7:T800FlatV7N1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v7-n2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v7:T800FlatWoStateEstimationEnvCfgV7",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v7_n2:T800FlatV7N2PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v7-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v7_scale:T800FlatWoStateEstimationEnvCfgV7Scale",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v7_scale:T800FlatV7ScalePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v7-n2-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v7_scale:T800FlatWoStateEstimationEnvCfgV7Scale",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg_v7_n2_scale:T800FlatV7N2ScalePPORunnerCfg"),
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v7-n3-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v7_scale:T800FlatWoStateEstimationEnvCfgV7Scale",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg_v7_n3_scale:T800FlatV7N3ScalePPORunnerCfg"),
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v8:T800FlatWoStateEstimationEnvCfgV8",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v8:T800FlatV8PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v8-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v8:T800FlatWoStateEstimationEnvCfgV8Scale",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v8_scale:T800FlatV8ScalePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v8-n1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v8:T800FlatWoStateEstimationEnvCfgV8",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v8_n1:T800FlatV8N1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v8-n1-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v8:T800FlatWoStateEstimationEnvCfgV8Scale",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg_v8_n1:T800FlatV8N1ScalePPORunnerCfg"),
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v8-n2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v8:T800FlatWoStateEstimationEnvCfgV8",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v8_n2:T800FlatV8N2PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v8-n2-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v8:T800FlatWoStateEstimationEnvCfgV8Scale",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg_v8_n2:T800FlatV8N2ScalePPORunnerCfg"),
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v9:T800FlatWoStateEstimationEnvCfgV9",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v9:T800FlatV9PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v9-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v9:T800FlatWoStateEstimationEnvCfgV9Scale",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v9:T800FlatV9ScalePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v9-n1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v9_n1:T800FlatWoStateEstimationEnvCfgV9N1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v9_n1:T800FlatV9N1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v10:T800FlatWoStateEstimationEnvCfgV10",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v10:T800FlatV10PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v11:T800FlatWoStateEstimationEnvCfgV11",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_v11:T800FlatV11PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v11-n1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v11_n1:T800FlatWoStateEstimationEnvCfgV11N1",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg_v11_n1:T800FlatV11N1PPORunnerCfg"),
    },
)

gym.register(
    id="Tracking-Flat-T800-Wo-State-Estimation-v11-n1-scale-lafan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (f"{__name__}.flat_env_cfg_v11_n1:T800FlatWoStateEstimationEnvCfgV11N1ScaleLafan"),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg_v11_n1:T800FlatV11N1ScaleLafanPPORunnerCfg"),
    },
)

gym.register(
    id="Tracking-Flat-T800-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T800FlatLowFreqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T800FlatLowFreqPPORunnerCfg",
    },
)
