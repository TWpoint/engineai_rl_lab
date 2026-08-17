import os
import time
from itertools import count

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from engineai_rl_lab.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
    get_actor_obs_normalizer,
)


def _get_logger_type(runner: OnPolicyRunner) -> str | None:
    if hasattr(runner, "logger_type"):
        return runner.logger_type
    logger = getattr(runner, "logger", None)
    return getattr(logger, "logger_type", None)


def _get_policy(runner: OnPolicyRunner):
    if hasattr(runner.alg, "get_policy"):
        return runner.alg.get_policy()
    return runner.alg.policy


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if _get_logger_type(self) in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(
                _get_policy(self),
                normalizer=get_actor_obs_normalizer(self),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def _motion_command(self):
        command_manager = getattr(self.env.unwrapped, "command_manager", None)
        if command_manager is None:
            return None
        try:
            return command_manager.get_term("motion")
        except (KeyError, ValueError):
            return None

    def _sync_adaptive_sampling(self, iteration: int) -> None:
        command = self._motion_command()
        if command is None or not hasattr(command, "sync_and_compute_adaptive_sampling"):
            return
        frequency = int(self.cfg.get("sync_adaptive_sampling_all_gpus_freq", 200))
        sync_across_ranks = frequency > 0 and (iteration + 1) % frequency == 0
        command.sync_and_compute_adaptive_sampling(sync_across_ranks=sync_across_ranks)

    def _resample_motion_working_set(self, iteration: int, obs):
        frequency = int(self.cfg.get("motion_resample_frequency", 250))
        if frequency <= 0 or (iteration + 1) % frequency != 0:
            return obs
        command = self._motion_command()
        if command is None or not hasattr(command, "resample_motion_working_set"):
            return obs
        if command.resample_motion_working_set():
            obs, _ = self.env.reset()
            return obs.to(self.device)
        return obs

    def learn(self, num_learning_iterations: int | None, init_at_random_ep_len: bool = False) -> None:
        """RSL-RL loop with SONIC's per-iteration AS update and working-set callback."""

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations if num_learning_iterations is not None else None
        iterations = range(start_it, total_it) if total_it is not None else count(start_it)
        for it in iterations:
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            self._sync_adaptive_sampling(it)
            obs = self._resample_motion_working_set(it, obs)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        command = self._motion_command()
        checkpoint_infos = {
            "runner_infos": infos,
            "motion_adaptive_sampling": (
                command.get_adaptive_sampling_state()
                if command is not None and hasattr(command, "get_adaptive_sampling_state")
                else None
            ),
        }
        super().save(path, checkpoint_infos)
        if _get_logger_type(self) in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                _get_policy(self),
                normalizer=get_actor_obs_normalizer(self),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        checkpoint_infos = super().load(path, load_cfg, strict, map_location)
        if not isinstance(checkpoint_infos, dict) or "motion_adaptive_sampling" not in checkpoint_infos:
            return checkpoint_infos
        command = self._motion_command()
        adaptive_state = checkpoint_infos.get("motion_adaptive_sampling")
        if command is not None and adaptive_state is not None and command.load_adaptive_sampling_state(adaptive_state):
            if command.resample_motion_working_set():
                self.env.reset()
        return checkpoint_infos.get("runner_infos")
