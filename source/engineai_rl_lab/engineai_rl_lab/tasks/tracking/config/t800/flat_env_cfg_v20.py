from copy import deepcopy

from isaaclab.utils.configclass import configclass

import engineai_rl_lab.tasks.tracking.mdp as mdp
from engineai_rl_lab.tasks.tracking.config.t800.flat_env_cfg_v19 import (
    T800FlatWoStateEstimationEnvCfgV19Scale,
)
from engineai_rl_lab.tasks.tracking.mdp.command_v1 import (
    AdaptiveSamplerV1Cfg,
    MotionCommandV1Cfg,
)


@configclass
class T800FlatWoStateEstimationEnvCfgV20Scale(T800FlatWoStateEstimationEnvCfgV19Scale):
    """V19 task settings with scaled actor velocities, target-only commands, and the V1 sampler."""

    def __post_init__(self):
        super().__post_init__()
        # Match the ScaleBFM input scale so raw joint velocities do not
        # dominate the actor's direct linear proprioception projection.
        self.observations.proprioception.joint_vel.scale = 0.05
        # The actor receives only the reference trajectory. The privileged
        # critic observation retains its current-frame target error.
        self.observations.command.link_pose_b.func = mdp.motion_body_pose_b_window_by_entity_xz
        previous = self.commands.motion
        self.commands.motion = MotionCommandV1Cfg(
            resampling_time_range=previous.resampling_time_range,
            debug_vis=previous.debug_vis,
            cmd_kind=previous.cmd_kind,
            element_names=deepcopy(previous.element_names),
            asset_name=previous.asset_name,
            motion_file=previous.motion_file,
            motion_data_device=previous.motion_data_device,
            max_num_load_motions=previous.max_num_load_motions,
            working_set_replacement=previous.working_set_replacement,
            motion_load_workers=previous.motion_load_workers,
            motion_chunk_frames=previous.motion_chunk_frames,
            anchor_body_name=previous.anchor_body_name,
            body_names=deepcopy(previous.body_names),
            pose_range=deepcopy(previous.pose_range),
            velocity_range=deepcopy(previous.velocity_range),
            joint_position_range=previous.joint_position_range,
            resample_at_motion_end=previous.resample_at_motion_end,
            start_at_motion_beginning=previous.start_at_motion_beginning,
            playback_start_frame=previous.playback_start_frame,
            adaptive_sampling=AdaptiveSamplerV1Cfg(
                bin_size=50,
                equal_motion_weighting=False,
                coverage_fraction=0.20,
                pre_failure_window=0,
                tracking_error_scale=0.30,
                fast_half_life=32.0,
                slow_half_life=64.0,
                uncertainty_exposure=32.0,
                learnability_full_scale=0.05,
                max_learnable_fraction=0.75,
                probability_cap_ratio=200.0,
            ),
            anchor_visualizer_cfg=deepcopy(previous.anchor_visualizer_cfg),
            body_visualizer_cfg=deepcopy(previous.body_visualizer_cfg),
        )
