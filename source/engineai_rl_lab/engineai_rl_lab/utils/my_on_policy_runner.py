import os
import re
import time
from dataclasses import dataclass
from itertools import count
from typing import Any

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


def invalid_state_diagnostic_metrics(statistics: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert ``[invalid events, terminations, transitions]`` into stable rollout metrics."""
    invalid_events, terminations, transitions = statistics.unbind()
    return {
        "Diagnostics/invalid_robot_state_events": invalid_events,
        "Diagnostics/invalid_robot_state_events_per_million_transitions": (
            invalid_events * 1.0e6 / transitions.clamp_min(1.0)
        ),
        "Diagnostics/invalid_robot_state_fraction_of_terminations": (invalid_events / terminations.clamp_min(1.0)),
    }


def accumulate_invalid_state_statistics(
    statistics: torch.Tensor,
    termination_manager,
    dones: torch.Tensor,
) -> None:
    """Accumulate the current-step invalid-state mask and rollout denominators."""
    try:
        invalid_states = termination_manager.get_term("invalid_robot_state")
    except (AttributeError, KeyError, ValueError):
        invalid_states = torch.zeros_like(dones, dtype=torch.bool)
    statistics[0] += invalid_states.sum().to(statistics.device)
    statistics[1] += dones.bool().sum().to(statistics.device)
    statistics[2] += dones.numel()


@dataclass
class RolloutExtrema:
    """Device-resident rollout extrema and non-finite flags.

    Row zero stores finite absolute maxima and row one stores whether a
    non-finite value was observed. The final two columns are the raw policy
    action and the processed action respectively.
    """

    reward_term_names: tuple[str, ...]
    statistics: torch.Tensor


def _as_torch_tensor(value: Any) -> torch.Tensor | None:
    """Return a tensor view for Torch-like action buffers without moving it to CPU."""
    if isinstance(value, torch.Tensor):
        return value
    if value is None:
        return None
    try:
        torch_value = getattr(value, "torch", None)
        if callable(torch_value):
            torch_value = torch_value()
        if isinstance(torch_value, torch.Tensor):
            return torch_value
        if hasattr(value, "__dlpack__"):
            return torch.from_dlpack(value)
    except (AttributeError, RuntimeError, TypeError):
        return None
    return None


def resolve_reward_term_names(reward_manager: Any) -> tuple[str, ...]:
    """Resolve a stable name for every column in ``RewardManager._step_reward``."""
    names = tuple(str(name) for name in getattr(reward_manager, "_term_names", ()))
    step_reward = _as_torch_tensor(getattr(reward_manager, "_step_reward", None))
    if step_reward is None or step_reward.ndim == 0:
        return names
    num_terms = step_reward.shape[-1]
    return tuple(names[index] if index < len(names) else f"term_{index}" for index in range(num_terms))


def create_rollout_extrema(reward_term_names: tuple[str, ...], device: torch.device | str) -> RolloutExtrema:
    """Create an empty rollout-extrema accumulator on ``device``."""
    statistics = torch.zeros((2, len(reward_term_names) + 2), dtype=torch.float64, device=device)
    return RolloutExtrema(reward_term_names=reward_term_names, statistics=statistics)


def processed_action_tensors(action_manager: Any) -> tuple[torch.Tensor, ...]:
    """Read processed actions across current and legacy ActionManager APIs."""
    if action_manager is None:
        return ()

    try:
        direct_actions = getattr(action_manager, "processed_actions", None)
        if callable(direct_actions):
            direct_actions = direct_actions()
    except (AttributeError, KeyError, NotImplementedError, RuntimeError, TypeError):
        direct_actions = None
    direct_tensor = _as_torch_tensor(direct_actions)
    if direct_tensor is not None:
        return (direct_tensor,)

    try:
        term_names = getattr(action_manager, "active_terms", ())
        if callable(term_names):
            term_names = term_names()
        terms = [action_manager.get_term(name) for name in term_names]
    except (AttributeError, KeyError, NotImplementedError, RuntimeError, TypeError):
        try:
            terms = list(action_manager._terms.values())
        except (AttributeError, RuntimeError, TypeError):
            return ()

    tensors = []
    for term in terms:
        try:
            tensor = _as_torch_tensor(getattr(term, "processed_actions", None))
        except (AttributeError, NotImplementedError, RuntimeError, TypeError):
            tensor = None
        if tensor is not None:
            tensors.append(tensor)
    return tuple(tensors)


def _accumulate_scalar_extrema(statistics: torch.Tensor, column: int, values: torch.Tensor | None) -> None:
    if values is None or values.numel() == 0:
        return
    values = values.to(device=statistics.device)
    finite = torch.isfinite(values)
    finite_max = torch.where(finite, values.abs(), torch.zeros_like(values)).amax().to(statistics.dtype)
    statistics[0, column].copy_(torch.maximum(statistics[0, column], finite_max))
    nonfinite = (~finite).any().to(statistics.dtype)
    statistics[1, column].copy_(torch.maximum(statistics[1, column], nonfinite))


def accumulate_rollout_extrema(
    extrema: RolloutExtrema,
    reward_rates: torch.Tensor | None,
    policy_actions: torch.Tensor | None,
    processed_actions: tuple[torch.Tensor, ...] = (),
) -> None:
    """Accumulate finite maxima and non-finite flags without a host synchronization."""
    num_reward_terms = len(extrema.reward_term_names)
    reward_rates = _as_torch_tensor(reward_rates)
    if reward_rates is not None and reward_rates.ndim > 0 and reward_rates.numel() > 0:
        num_columns = min(num_reward_terms, reward_rates.shape[-1])
        if num_columns > 0:
            flat_rates = reward_rates[..., :num_columns].reshape(-1, num_columns).to(extrema.statistics.device)
            finite = torch.isfinite(flat_rates)
            finite_max = torch.where(finite, flat_rates.abs(), torch.zeros_like(flat_rates)).amax(dim=0)
            finite_max = finite_max.to(extrema.statistics.dtype)
            current_max = extrema.statistics[0, :num_columns]
            current_max.copy_(torch.maximum(current_max, finite_max))
            nonfinite = (~finite).any(dim=0).to(extrema.statistics.dtype)
            current_nonfinite = extrema.statistics[1, :num_columns]
            current_nonfinite.copy_(torch.maximum(current_nonfinite, nonfinite))

    _accumulate_scalar_extrema(extrema.statistics, num_reward_terms, _as_torch_tensor(policy_actions))
    for processed_action in processed_actions:
        _accumulate_scalar_extrema(extrema.statistics, num_reward_terms + 1, _as_torch_tensor(processed_action))


def _safe_metric_components(names: tuple[str, ...]) -> tuple[str, ...]:
    """Make readable, unique metric path components from arbitrary term names."""
    components = []
    used = set()
    for index, name in enumerate(names):
        component = re.sub(r"[^\w.-]+", "_", name).strip("_.") or f"term_{index}"
        if component in used:
            component = f"{component}_{index}"
        used.add(component)
        components.append(component)
    return tuple(components)


def rollout_extrema_metrics(extrema: RolloutExtrema) -> dict[str, torch.Tensor]:
    """Convert a reduced accumulator into logger-ready diagnostic metrics."""
    metrics = {}
    for index, term_name in enumerate(_safe_metric_components(extrema.reward_term_names)):
        metrics[f"Diagnostics/reward_rate_max_abs/{term_name}"] = extrema.statistics[0, index]
        metrics[f"Diagnostics/reward_rate_nonfinite_detected/{term_name}"] = extrema.statistics[1, index]

    action_offset = len(extrema.reward_term_names)
    metrics["Diagnostics/policy_action_max_abs"] = extrema.statistics[0, action_offset]
    metrics["Diagnostics/policy_action_nonfinite_detected"] = extrema.statistics[1, action_offset]
    metrics["Diagnostics/processed_action_max_abs"] = extrema.statistics[0, action_offset + 1]
    metrics["Diagnostics/processed_action_nonfinite_detected"] = extrema.statistics[1, action_offset + 1]
    return metrics


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

    def _sync_adaptive_sampling(self, iteration: int, force_sync: bool = False) -> None:
        command = self._motion_command()
        if command is None or not hasattr(command, "sync_and_compute_adaptive_sampling"):
            return
        frequency = int(self.cfg.get("sync_adaptive_sampling_all_gpus_freq", 200))
        save_interval = int(self.cfg.get("save_interval", 0))
        periodic_sync = frequency > 0 and (iteration + 1) % frequency == 0
        save_boundary_sync = save_interval > 0 and iteration % save_interval == 0
        sync_across_ranks = periodic_sync or save_boundary_sync or force_sync
        command.sync_and_compute_adaptive_sampling(sync_across_ranks=sync_across_ranks)

    def _resample_motion_working_set(self, iteration: int, obs):
        frequency = int(self.cfg.get("motion_resample_frequency", 250))
        if frequency <= 0 or (iteration + 1) % frequency != 0:
            return obs
        if self.cfg.get("stagger_motion_working_set_refresh", False):
            local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
            local_rank = int(getattr(self, "gpu_local_rank", os.getenv("LOCAL_RANK", "0")))
            refresh_event = (iteration + 1) // frequency - 1
            if local_rank != refresh_event % local_world_size:
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
            invalid_state_statistics = torch.zeros(3, dtype=torch.float64, device=self.device)
            reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
            action_manager = getattr(self.env.unwrapped, "action_manager", None)
            rollout_extrema = create_rollout_extrema(resolve_reward_term_names(reward_manager), self.device)
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    accumulate_rollout_extrema(
                        rollout_extrema,
                        _as_torch_tensor(getattr(reward_manager, "_step_reward", None)),
                        actions,
                        processed_action_tensors(action_manager),
                    )
                    accumulate_invalid_state_statistics(
                        invalid_state_statistics,
                        getattr(self.env.unwrapped, "termination_manager", None),
                        dones,
                    )
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
                if self.is_distributed:
                    torch.distributed.all_reduce(invalid_state_statistics, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(rollout_extrema.statistics, op=torch.distributed.ReduceOp.MAX)
                diagnostic_metrics = invalid_state_diagnostic_metrics(invalid_state_statistics)
                diagnostic_metrics.update(rollout_extrema_metrics(rollout_extrema))
                self.logger.ep_extras.append(diagnostic_metrics)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            is_final_iteration = total_it is not None and it == total_it - 1
            self._sync_adaptive_sampling(it, force_sync=is_final_iteration)
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
            command.resample_motion_working_set()
            self.env.reset()
        return checkpoint_infos.get("runner_infos")
