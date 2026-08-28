from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import socket
import stat
import tempfile
import time
import uuid
import zipfile
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import translate as translate_glob
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import yaml

_MOTION_CATALOG_SCHEMA_VERSION = 1
_MOTION_CATALOG_LOCK_POLL_SECONDS = 0.1
_MOTION_CATALOG_LOCK_WAIT_SECONDS = 30.0 * 60.0
_MOTION_CATALOG_LOCK_STALE_SECONDS = 6.0 * 60.0 * 60.0

MOTION_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
MOTION_FIELD_GROUPS = {
    "pose": ("body_pos_w", "body_quat_w"),
    "state": ("joint_pos", "joint_vel", "body_lin_vel_w", "body_ang_vel_w"),
}


def _resolve_motion_files_and_dependencies(
    manifest: str | os.PathLike[str],
) -> tuple[list[str], dict[str, str]]:
    """Resolve a motion source and fingerprint each YAML document that was parsed."""

    root = Path(manifest).expanduser().resolve()
    if root.suffix.lower() not in {".yaml", ".yml"}:
        if not root.is_file():
            raise FileNotFoundError(f"Motion file does not exist: {root}")
        return [str(root)], {}

    visiting: set[Path] = set()
    dependency_digests: dict[str, str] = {}

    def _walk(path: Path) -> tuple[list[Path], list[str]]:
        path = path.resolve()
        if path in visiting:
            raise ValueError(f"Recursive motion manifest include detected at {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Motion manifest does not exist: {path}")
        visiting.add(path)
        raw_document = path.read_bytes()
        dependency_digests[str(path)] = hashlib.sha256(raw_document).hexdigest()
        safe_loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        document = yaml.load(raw_document.decode("utf-8"), Loader=safe_loader) or {}
        if not isinstance(document, dict):
            raise ValueError(f"Motion manifest {path} must contain a mapping")

        motions: list[Path] = []
        exclusions: list[str] = []
        manifest_keys = {"files", "exclude_files"}
        direct_mapping = not manifest_keys.intersection(document)
        if direct_mapping:
            if not all(isinstance(name, str) and isinstance(value, str) for name, value in document.items()):
                raise ValueError(f"Motion mapping {path} must contain string motion-name to file-path entries")
            file_entries = list(document.values())
            exclude_entries: list[str] = []
        else:
            unknown_keys = set(document) - manifest_keys
            if unknown_keys:
                raise ValueError(f"Unknown keys in motion manifest {path}: {sorted(unknown_keys)}")
            file_entries = document.get("files", [])
            exclude_entries = document.get("exclude_files", [])
            if not isinstance(file_entries, list) or not isinstance(exclude_entries, list):
                raise ValueError(f"files and exclude_files in {path} must be lists")

        for entry in file_entries:
            candidate = Path(os.path.abspath(path.parent / os.fspath(entry)))
            if direct_mapping and not candidate.exists() and not Path(entry).is_absolute():
                cwd_candidate = Path(os.path.abspath(os.fspath(entry)))
                if cwd_candidate.exists():
                    candidate = cwd_candidate
            if candidate.suffix.lower() in {".yaml", ".yml"}:
                child_motions, child_exclusions = _walk(candidate)
                motions.extend(child_motions)
                exclusions.extend(child_exclusions)
            else:
                motions.append(candidate)
        for entry in exclude_entries:
            candidate = Path(os.path.abspath(path.parent / os.fspath(entry)))
            if candidate.suffix.lower() in {".yaml", ".yml"}:
                _, child_exclusions = _walk(candidate)
                exclusions.extend(child_exclusions)
            else:
                exclusions.append(candidate.as_posix())
        visiting.remove(path)
        return motions, exclusions

    motions, exclusions = _walk(root)
    exact_exclusions = {pattern for pattern in exclusions if not any(char in pattern for char in "*?[")}
    glob_exclusions = [pattern for pattern in exclusions if pattern not in exact_exclusions]
    glob_exclusion_regex = (
        re.compile("|".join(f"(?:{translate_glob(pattern)})" for pattern in glob_exclusions))
        if glob_exclusions
        else None
    )
    unique: list[str] = []
    seen: set[Path] = set()
    for motion in motions:
        motion_path = motion.as_posix()
        if (
            motion in seen
            or motion_path in exact_exclusions
            or (glob_exclusion_regex is not None and glob_exclusion_regex.match(motion_path) is not None)
        ):
            continue
        if motion.suffix.lower() != ".npz":
            raise ValueError(f"Unsupported motion file in {root}: {motion}")
        seen.add(motion)
        unique.append(str(motion))
    if not unique:
        raise ValueError(f"Motion manifest {root} did not resolve to any .npz files")
    return unique, dependency_digests


def resolve_motion_files(manifest: str | os.PathLike[str]) -> list[str]:
    """Resolve one NPZ or a recursive ``files``/``exclude_files`` YAML manifest."""

    motion_files, _ = _resolve_motion_files_and_dependencies(manifest)
    return motion_files


def shard_motion_files(motion_files: Sequence[str], *, world_size: int, rank: int) -> tuple[list[str], list[int]]:
    """Return the deterministic global-rank slice of an ordered motion list.

    Small datasets are replicated when they contain fewer motions than ranks so
    no worker receives an empty collection.
    """

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed motion rank/world size: {rank}/{world_size}")
    if not motion_files:
        raise ValueError("Motion file list must not be empty")
    if world_size == 1 or len(motion_files) < world_size:
        global_ids = list(range(len(motion_files)))
    else:
        global_ids = list(range(rank, len(motion_files), world_size))
    return [motion_files[index] for index in global_ids], global_ids


def read_motion_length(path: str | os.PathLike[str]) -> int:
    """Read a motion frame count from its NPZ header without decoding its arrays."""

    with zipfile.ZipFile(path) as archive:
        try:
            stream = archive.open("joint_pos.npy")
        except KeyError as exc:
            raise ValueError(f"Motion file {os.fspath(path)!r} is missing field 'joint_pos'") from exc
        with stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Motion file {os.fspath(path)!r} uses unsupported NPY header version {version}")
    if len(shape) != 2 or shape[0] < 2:
        raise ValueError(f"Motion file {os.fspath(path)!r} has invalid joint_pos shape {shape}")
    return int(shape[0])


def read_motion_lengths(motion_files: Sequence[str], *, max_workers: int = 4) -> list[int]:
    """Read the complete manifest's frame counts in deterministic order."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_workers == 1:
        return [read_motion_length(path) for path in motion_files]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="motion-header") as executor:
        return list(executor.map(read_motion_length, motion_files))


def _catalog_digest(motion_files: Sequence[str], motion_lengths: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for path, length in zip(motion_files, motion_lengths, strict=True):
        encoded_path = os.fsencode(path)
        digest.update(len(encoded_path).to_bytes(8, byteorder="little"))
        digest.update(encoded_path)
        digest.update(int(length).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _direct_motion_signature(path: Path) -> dict[str, str | int]:
    file_stat = path.stat()
    return {
        "path": str(path),
        "size": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
    }


def _catalog_payload(
    manifest: Path,
    motion_files: list[str],
    motion_lengths: list[int],
    dependency_digests: dict[str, str],
) -> dict[str, object]:
    is_yaml = manifest.suffix.lower() in {".yaml", ".yml"}
    return {
        "schema_version": _MOTION_CATALOG_SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_dependencies": [{"path": path, "sha256": digest} for path, digest in dependency_digests.items()],
        "direct_motion": None if is_yaml else _direct_motion_signature(manifest),
        "motion_files": motion_files,
        "motion_lengths": motion_lengths,
        "catalog_sha256": _catalog_digest(motion_files, motion_lengths),
    }


def _valid_manifest_dependencies(payload: dict[str, object], manifest: Path) -> bool:
    dependencies = payload.get("manifest_dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return False
    dependency_paths: set[str] = set()
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            return False
        path = dependency.get("path")
        expected_digest = dependency.get("sha256")
        if (
            not isinstance(path, str)
            or path in dependency_paths
            or not Path(path).is_absolute()
            or Path(path).suffix.lower() not in {".yaml", ".yml"}
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            return False
        dependency_paths.add(path)
        try:
            actual_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return False
        if actual_digest != expected_digest:
            return False
    return str(manifest) in dependency_paths


def _valid_direct_motion(payload: dict[str, object], manifest: Path) -> bool:
    direct_motion = payload.get("direct_motion")
    if not isinstance(direct_motion, dict) or direct_motion.get("path") != str(manifest):
        return False
    expected_size = direct_motion.get("size")
    expected_mtime_ns = direct_motion.get("mtime_ns")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or not isinstance(expected_mtime_ns, int)
        or isinstance(expected_mtime_ns, bool)
    ):
        return False
    try:
        actual = manifest.stat()
    except OSError:
        return False
    return actual.st_size == expected_size and actual.st_mtime_ns == expected_mtime_ns


def _validated_catalog_payload(
    payload: object,
    manifest: Path,
) -> tuple[list[str], list[int]] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != _MOTION_CATALOG_SCHEMA_VERSION:
        return None
    if payload.get("manifest") != str(manifest):
        return None

    motion_files = payload.get("motion_files")
    motion_lengths = payload.get("motion_lengths")
    if (
        not isinstance(motion_files, list)
        or not motion_files
        or not isinstance(motion_lengths, list)
        or len(motion_files) != len(motion_lengths)
    ):
        return None
    if any(
        not isinstance(path, str) or not Path(path).is_absolute() or Path(path).suffix.lower() != ".npz"
        for path in motion_files
    ):
        return None
    if len(set(motion_files)) != len(motion_files):
        return None
    if any(not isinstance(length, int) or isinstance(length, bool) or length < 2 for length in motion_lengths):
        return None
    expected_digest = payload.get("catalog_sha256")
    if not isinstance(expected_digest, str) or _catalog_digest(motion_files, motion_lengths) != expected_digest:
        return None

    if manifest.suffix.lower() in {".yaml", ".yml"}:
        if payload.get("direct_motion") is not None or not _valid_manifest_dependencies(payload, manifest):
            return None
    elif payload.get("manifest_dependencies") != [] or not _valid_direct_motion(payload, manifest):
        return None
    return motion_files, motion_lengths


def _load_motion_catalog_cache(cache_path: Path, manifest: Path) -> tuple[list[str], list[int]] | None:
    try:
        with cache_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _validated_catalog_payload(payload, manifest)


def _cache_file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        file_stat = path.stat()
    except OSError:
        return None
    return (file_stat.st_dev, file_stat.st_ino, file_stat.st_size, file_stat.st_mtime_ns)


def _try_create_catalog_lock(lock_path: Path) -> tuple[str | None, bool]:
    token = uuid.uuid4().hex
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock_path, flags, 0o644)
    except FileExistsError:
        return None, True
    except OSError:
        return None, False
    try:
        owner = {
            "token": token,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_ns": time.time_ns(),
        }
        with os.fdopen(descriptor, mode="w", encoding="utf-8") as stream:
            json.dump(owner, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        with contextlib.suppress(OSError):
            lock_path.unlink()
        raise
    return token, True


def _catalog_lock_is_stale(lock_path: Path) -> bool:
    try:
        lock_stat = lock_path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(lock_stat.st_mode):
        return False
    if time.time() - lock_stat.st_mtime > _MOTION_CATALOG_LOCK_STALE_SECONDS:
        return True
    try:
        with lock_path.open(encoding="utf-8") as stream:
            owner = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(owner, dict) or owner.get("hostname") != socket.gethostname():
        return False
    pid = owner.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, PermissionError):
        return False
    return False


def _remove_stale_catalog_lock(lock_path: Path) -> None:
    try:
        observed = lock_path.stat()
    except OSError:
        return
    if not _catalog_lock_is_stale(lock_path):
        return
    try:
        current = lock_path.stat()
    except OSError:
        return
    if (current.st_dev, current.st_ino, current.st_mtime_ns) != (
        observed.st_dev,
        observed.st_ino,
        observed.st_mtime_ns,
    ):
        return
    with contextlib.suppress(OSError):
        lock_path.unlink()


def _release_catalog_lock(lock_path: Path, token: str) -> None:
    try:
        with lock_path.open(encoding="utf-8") as stream:
            owner = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(owner, dict) or owner.get("token") != token:
        return
    with contextlib.suppress(OSError):
        lock_path.unlink()


def _atomic_write_catalog(cache_path: Path, payload: dict[str, object]) -> bool:
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, mode="w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, cache_path)
        temporary_path = None
        try:
            directory_descriptor = os.open(cache_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            pass
        else:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink()


def _build_motion_catalog(
    manifest: Path,
    *,
    max_workers: int,
) -> tuple[list[str], list[int], dict[str, object]]:
    motion_files, dependency_digests = _resolve_motion_files_and_dependencies(manifest)
    motion_lengths = read_motion_lengths(motion_files, max_workers=max_workers)
    payload = _catalog_payload(manifest, motion_files, motion_lengths, dependency_digests)
    return motion_files, motion_lengths, payload


def resolve_motion_catalog(
    manifest: str | os.PathLike[str],
    *,
    cache_path: str | os.PathLike[str] | None = None,
    max_workers: int = 4,
) -> tuple[list[str], list[int]]:
    """Resolve ordered motion paths and frame lengths, optionally through a safe JSON cache.

    A cache hit re-hashes only the recursive YAML documents, not every referenced
    NPZ. YAML motion corpora are therefore treated as immutable/versioned: change
    a manifest (or delete the sidecar) when replacing NPZ data in place. The
    subsequent ``MotionCollection`` decode still checks the length of every NPZ
    it actually loads against this catalog.

    The cache is opt-in. Writers coordinate with an ``O_EXCL`` lock file, which
    works on the shared NFS deployment where advisory locks are disabled, and
    publish with an atomic replace. A corrupt, stale, or unwritable cache falls
    back to rebuilding the catalog without changing motion-loading semantics.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    resolved_manifest = Path(manifest).expanduser().resolve()
    if cache_path is None:
        motion_files, _ = _resolve_motion_files_and_dependencies(resolved_manifest)
        return motion_files, read_motion_lengths(motion_files, max_workers=max_workers)

    resolved_cache_path = Path(os.path.abspath(os.path.expanduser(os.fspath(cache_path))))
    if resolved_cache_path == resolved_manifest:
        raise ValueError("cache_path must not overwrite the motion manifest")
    cached = _load_motion_catalog_cache(resolved_cache_path, resolved_manifest)
    if cached is not None:
        return cached

    try:
        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        motion_files, _ = _resolve_motion_files_and_dependencies(resolved_manifest)
        return motion_files, read_motion_lengths(motion_files, max_workers=max_workers)

    lock_path = resolved_cache_path.with_name(f"{resolved_cache_path.name}.lock")
    initial_cache_identity = _cache_file_identity(resolved_cache_path)
    deadline = time.monotonic() + _MOTION_CATALOG_LOCK_WAIT_SECONDS
    token: str | None = None
    while token is None:
        token, lock_available = _try_create_catalog_lock(lock_path)
        if token is not None:
            break
        if not lock_available:
            motion_files, _ = _resolve_motion_files_and_dependencies(resolved_manifest)
            return motion_files, read_motion_lengths(motion_files, max_workers=max_workers)
        if _catalog_lock_is_stale(lock_path):
            _remove_stale_catalog_lock(lock_path)
            continue
        current_cache_identity = _cache_file_identity(resolved_cache_path)
        if current_cache_identity != initial_cache_identity:
            cached = _load_motion_catalog_cache(resolved_cache_path, resolved_manifest)
            if cached is not None:
                return cached
            initial_cache_identity = current_cache_identity
        if time.monotonic() >= deadline:
            motion_files, _ = _resolve_motion_files_and_dependencies(resolved_manifest)
            return motion_files, read_motion_lengths(motion_files, max_workers=max_workers)
        time.sleep(_MOTION_CATALOG_LOCK_POLL_SECONDS)

    try:
        cached = _load_motion_catalog_cache(resolved_cache_path, resolved_manifest)
        if cached is not None:
            return cached
        motion_files, motion_lengths, payload = _build_motion_catalog(
            resolved_manifest,
            max_workers=max_workers,
        )
        # Do not publish a catalog from a YAML snapshot that changed while the
        # (potentially long) NPZ header scan was in progress.
        if resolved_manifest.suffix.lower() not in {".yaml", ".yml"} or _valid_manifest_dependencies(
            payload, resolved_manifest
        ):
            _atomic_write_catalog(resolved_cache_path, payload)
        return motion_files, motion_lengths
    finally:
        _release_catalog_lock(lock_path, token)


@dataclass(frozen=True)
class MotionShardSelection:
    files: tuple[str, ...]
    global_ids: tuple[int, ...]
    global_num_motions: int
    rank: int
    world_size: int
    is_distributed_shard: bool


def select_motion_shard(
    motion_files: Sequence[str],
    *,
    shard_by_rank: bool = True,
    world_size: int | None = None,
    rank: int | None = None,
) -> MotionShardSelection:
    """Resolve rank metadata before torch.distributed is initialized."""

    if (world_size is None) != (rank is None):
        raise ValueError("world_size and rank must be provided together")
    if world_size is None:
        env_world_size = os.environ.get("WORLD_SIZE")
        env_rank = os.environ.get("RANK")
        if (env_world_size is None) != (env_rank is None):
            raise ValueError("WORLD_SIZE and RANK must be set together")
        world_size = int(env_world_size) if env_world_size is not None else 1
        rank = int(env_rank) if env_rank is not None else 0
    else:
        world_size = int(world_size)
        rank = int(rank)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed motion rank/world size: {rank}/{world_size}")
    if shard_by_rank and world_size > 1 and len(motion_files) >= world_size:
        files, global_ids = shard_motion_files(motion_files, world_size=world_size, rank=rank)
        is_distributed_shard = True
    else:
        files = list(motion_files)
        global_ids = list(range(len(motion_files)))
        is_distributed_shard = False
    if not files:
        raise RuntimeError(f"Motion sharding assigned no files to rank {rank}/{world_size}")
    return MotionShardSelection(
        files=tuple(files),
        global_ids=tuple(global_ids),
        global_num_motions=len(motion_files),
        rank=rank,
        world_size=world_size,
        is_distributed_shard=is_distributed_shard,
    )


class MotionLoader:
    """Load and canonicalize one original motion NPZ."""

    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int] | torch.Tensor,
        device: str | torch.device = "cpu",
        *,
        joint_names: Sequence[str] | None = None,
        body_names: Sequence[str] | None = None,
    ):
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Invalid motion file path: {motion_file}")
        with np.load(motion_file, allow_pickle=False) as data:
            required = {
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"Motion file {motion_file!r} is missing fields: {missing}")
            self.fps = float(np.asarray(data["fps"]).item())
            if not math.isfinite(self.fps) or self.fps <= 0.0:
                raise ValueError(f"Motion file {motion_file!r} has invalid fps={self.fps}")
            file_joint_names = data["joint_names"].astype(str).tolist() if "joint_names" in data.files else None
            file_body_names = data["body_names"].astype(str).tolist() if "body_names" in data.files else None

            joint_pos = np.asarray(data["joint_pos"])
            joint_vel = np.asarray(data["joint_vel"])
            body_pos_w = np.asarray(data["body_pos_w"])
            body_quat_w = np.asarray(data["body_quat_w"])
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"])
            quaternion_order = "wxyz"
            if "quaternion_order" in data.files:
                quaternion_order_value = np.asarray(data["quaternion_order"]).item()
                if isinstance(quaternion_order_value, bytes):
                    quaternion_order_value = quaternion_order_value.decode("ascii")
                quaternion_order = str(quaternion_order_value).lower()

        if joint_pos.ndim != 2 or joint_pos.shape != joint_vel.shape:
            raise ValueError(
                f"Motion file {motion_file!r} has invalid joint shapes {joint_pos.shape} and {joint_vel.shape}"
            )
        if joint_pos.shape[0] < 2:
            raise ValueError(f"Motion clips must contain at least two frames: {motion_file}")
        body_shape = body_pos_w.shape[:2]
        if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
            raise ValueError(f"Motion file {motion_file!r} has invalid body_pos_w shape {body_pos_w.shape}")
        expected_shapes = (
            (body_quat_w, 4, "body_quat_w"),
            (body_lin_vel_w, 3, "body_lin_vel_w"),
            (body_ang_vel_w, 3, "body_ang_vel_w"),
        )
        for values, width, field in expected_shapes:
            if values.ndim != 3 or values.shape[:2] != body_shape or values.shape[-1] != width:
                raise ValueError(f"Motion file {motion_file!r} has invalid {field} shape {values.shape}")
        if body_shape[0] != joint_pos.shape[0]:
            raise ValueError(f"Motion file {motion_file!r} has inconsistent joint/body frame counts")

        requested_joint_names = list(joint_names) if joint_names is not None else None
        if file_joint_names is not None and len(file_joint_names) != joint_pos.shape[1]:
            raise ValueError(
                f"Motion file {motion_file!r} has {len(file_joint_names)} joint names but {joint_pos.shape[1]} joints"
            )
        if requested_joint_names is not None and file_joint_names is not None:
            joint_indices = self._resolve_name_indexes(
                file_joint_names, requested_joint_names, "joint", require_complete=True
            )
            joint_pos = joint_pos[:, joint_indices]
            joint_vel = joint_vel[:, joint_indices]
        elif requested_joint_names is not None and joint_pos.shape[1] != len(requested_joint_names):
            raise ValueError(
                f"Motion file {motion_file!r} has {joint_pos.shape[1]} joints but the robot has "
                f"{len(requested_joint_names)}; joint_names metadata is required"
            )
        self.joint_names = requested_joint_names if requested_joint_names is not None else file_joint_names

        if file_body_names is not None and len(file_body_names) != body_shape[1]:
            raise ValueError(
                f"Motion file {motion_file!r} has {len(file_body_names)} body names but {body_shape[1]} bodies"
            )
        if body_names is not None and file_body_names is not None:
            selected_body_indices = self._resolve_name_indexes(
                file_body_names, list(body_names), "body", require_complete=False
            )
        else:
            selected_body_indices = torch.as_tensor(body_indexes, dtype=torch.long).cpu().tolist()
            if selected_body_indices and max(selected_body_indices) >= body_shape[1]:
                raise ValueError(
                    f"Motion file {motion_file!r} has {body_shape[1]} bodies, but requested index "
                    f"{max(selected_body_indices)}"
                )
        body_pos_w = body_pos_w[:, selected_body_indices]
        body_quat_w = body_quat_w[:, selected_body_indices]
        body_lin_vel_w = body_lin_vel_w[:, selected_body_indices]
        body_ang_vel_w = body_ang_vel_w[:, selected_body_indices]
        if quaternion_order == "wxyz":
            body_quat_w = np.roll(body_quat_w, shift=-1, axis=-1)
        elif quaternion_order != "xyzw":
            raise ValueError(
                f"Unsupported quaternion_order {quaternion_order!r} in {motion_file!r}; expected wxyz or xyzw"
            )

        target_device = torch.device(device)
        arrays = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
        }
        for field, values in arrays.items():
            contiguous = np.ascontiguousarray(values, dtype=np.float32)
            if not np.all(np.isfinite(contiguous)):
                raise ValueError(f"Motion file {motion_file!r} contains non-finite values in {field}")
            setattr(self, field, torch.from_numpy(contiguous).to(target_device))
        self.time_step_total = int(joint_pos.shape[0])

    @staticmethod
    def _resolve_name_indexes(
        file_names: Sequence[str], requested_names: Sequence[str], kind: str, *, require_complete: bool
    ) -> list[int]:
        if len(file_names) != len(set(file_names)):
            raise ValueError(f"Motion file contains duplicate {kind} names: {file_names}")
        index_by_name = {name: index for index, name in enumerate(file_names)}
        requested_name_set = set(requested_names)
        missing = [name for name in requested_names if name not in index_by_name]
        extra = [name for name in file_names if name not in requested_name_set] if require_complete else []
        if missing or extra:
            raise ValueError(f"Motion {kind} names do not match the robot. Missing={missing}, extra={extra}")
        return [index_by_name[name] for name in requested_names]


@dataclass
class _PendingMotion:
    path: str
    global_id: int
    loader: MotionLoader


class MotionCollection:
    """Rank-sharded, chunked in-memory view of an unchanged YAML/NPZ corpus."""

    _FIELDS = MOTION_FIELDS
    fields = MOTION_FIELDS

    def __init__(
        self,
        motion_files: Sequence[str],
        body_indexes: Sequence[int] | torch.Tensor,
        device: str | torch.device,
        storage_device: str | torch.device,
        joint_names: Sequence[str],
        body_names: Sequence[str],
        *,
        world_size: int | None = None,
        rank: int | None = None,
        shard_by_rank: bool = True,
        selected_global_ids: Sequence[int] | None = None,
        global_time_totals: Sequence[int] | torch.Tensor | None = None,
        max_chunk_frames: int = 262_144,
        max_workers: int | None = None,
    ):
        if max_chunk_frames <= 0:
            raise ValueError("max_chunk_frames must be positive")
        manifest_digest = hashlib.sha256()
        for path in motion_files:
            encoded_path = os.fsencode(os.path.abspath(path))
            manifest_digest.update(len(encoded_path).to_bytes(8, byteorder="little"))
            manifest_digest.update(encoded_path)
        self.manifest_fingerprint = manifest_digest.hexdigest()
        self.manifest_fingerprint_words = tuple(
            int.from_bytes(manifest_digest.digest()[offset : offset + 8], byteorder="little", signed=True)
            for offset in range(0, manifest_digest.digest_size, 8)
        )
        if selected_global_ids is None:
            selection = select_motion_shard(
                motion_files,
                shard_by_rank=shard_by_rank,
                world_size=world_size,
                rank=rank,
            )
        else:
            if (world_size is None) != (rank is None):
                raise ValueError("world_size and rank must be provided together")
            if world_size is None:
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                rank = int(os.environ.get("RANK", "0"))
            assert rank is not None
            selected_global_ids = [int(global_id) for global_id in selected_global_ids]
            if not selected_global_ids:
                raise ValueError("selected_global_ids must not be empty")
            invalid_ids = [global_id for global_id in selected_global_ids if not 0 <= global_id < len(motion_files)]
            if invalid_ids:
                raise ValueError(f"selected_global_ids contains invalid indices: {invalid_ids[:8]}")
            selection = MotionShardSelection(
                files=tuple(motion_files[global_id] for global_id in selected_global_ids),
                global_ids=tuple(selected_global_ids),
                global_num_motions=len(motion_files),
                rank=rank,
                world_size=world_size,
                is_distributed_shard=(
                    len(selected_global_ids) != len(motion_files)
                    or len(set(selected_global_ids)) != len(selected_global_ids)
                ),
            )
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device)
        self.global_num_motions = selection.global_num_motions
        self.rank = selection.rank
        self.world_size = selection.world_size
        self.is_distributed_shard = selection.is_distributed_shard
        self.max_chunk_frames = int(max_chunk_frames)
        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() or 1))
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        # CUDA allocations and copies must stay on the constructing thread.
        self.max_workers = 1 if self.storage_device.type != "cpu" else int(max_workers)
        if global_time_totals is None:
            resolved_global_time_totals = read_motion_lengths(motion_files, max_workers=self.max_workers)
        elif isinstance(global_time_totals, torch.Tensor):
            resolved_global_time_totals = [int(value) for value in global_time_totals.tolist()]
        else:
            resolved_global_time_totals = [int(value) for value in global_time_totals]
        if len(resolved_global_time_totals) != len(motion_files):
            raise ValueError(
                "global_time_totals must contain one length per manifest motion: "
                f"{len(resolved_global_time_totals)} != {len(motion_files)}"
            )
        self.global_time_totals = torch.tensor(resolved_global_time_totals, dtype=torch.long, device=self.device)

        self.names: list[str] = []
        global_ids: list[int] = []
        time_totals: list[int] = []
        motion_chunk_ids: list[int] = []
        motion_chunk_offsets: list[int] = []
        self._chunks: list[dict[str, torch.Tensor]] = []
        self.field_shapes: dict[str, tuple[int, ...]] = {}
        self._field_groups: dict[str, str] = {}
        self._field_slices: dict[str, slice] = {}
        self._group_widths: dict[str, int] = {}
        self.fps: float | None = None
        self.resident_bytes = 0

        entries = list(zip(selection.files, selection.global_ids, strict=True))
        planned_lengths = [resolved_global_time_totals[global_id] for global_id in selection.global_ids]
        planned_chunk_frames = self._plan_chunk_frames(planned_lengths)
        loader_iterator = iter(
            self._iter_loaders(
                entries,
                body_indexes=body_indexes,
                joint_names=joint_names,
                body_names=body_names,
            )
        )
        first_item = next(loader_iterator)
        self._validate_loader(first_item.path, first_item.loader)
        chunk_id = 0
        chunk = self._allocate_chunk(planned_chunk_frames[chunk_id])
        chunk_frame_offset = 0

        for item_index, item in enumerate(chain((first_item,), loader_iterator)):
            length = item.loader.time_step_total
            expected_length = planned_lengths[item_index]
            if length != expected_length:
                raise RuntimeError(
                    f"Motion length changed while loading {item.path!r}: header={expected_length}, decoded={length}"
                )
            if chunk_frame_offset == planned_chunk_frames[chunk_id]:
                self._chunks.append(chunk)
                chunk_id += 1
                chunk = self._allocate_chunk(planned_chunk_frames[chunk_id])
                chunk_frame_offset = 0
            if chunk_frame_offset + length > planned_chunk_frames[chunk_id]:
                raise RuntimeError("Internal motion chunk planning mismatch")
            self._append_motion(
                item,
                chunk=chunk,
                chunk_id=chunk_id,
                chunk_frame_offset=chunk_frame_offset,
                global_ids=global_ids,
                time_totals=time_totals,
                motion_chunk_ids=motion_chunk_ids,
                motion_chunk_offsets=motion_chunk_offsets,
            )
            chunk_frame_offset += length
        if chunk_frame_offset != planned_chunk_frames[chunk_id]:
            raise RuntimeError("Final motion chunk was not filled to its planned size")
        self._chunks.append(chunk)
        if not self._chunks:
            raise RuntimeError("Motion collection did not load any chunks")

        self.global_ids = torch.tensor(global_ids, dtype=torch.long, device=self.device)
        self.time_totals = torch.tensor(time_totals, dtype=torch.long, device=self.device)
        self.time_offsets = torch.zeros(len(time_totals), dtype=torch.long, device=self.device)
        if len(time_totals) > 1:
            self.time_offsets[1:] = torch.cumsum(self.time_totals[:-1], dim=0)
        self.motion_chunk_ids = torch.tensor(motion_chunk_ids, dtype=torch.long, device=self.device)
        self.motion_chunk_offsets = torch.tensor(motion_chunk_offsets, dtype=torch.long, device=self.device)
        self.time_step_total = int(sum(time_totals))
        self.rank_num_files = len(self.names)
        self.rank_num_frames = self.time_step_total
        self.num_chunks = len(self._chunks)
        assert self.fps is not None

    def _iter_loaders(
        self,
        entries: Sequence[tuple[str, int]],
        *,
        body_indexes: Sequence[int] | torch.Tensor,
        joint_names: Sequence[str],
        body_names: Sequence[str],
    ) -> Iterator[_PendingMotion]:
        def _load(path: str) -> MotionLoader:
            return MotionLoader(
                path,
                body_indexes,
                device=self.storage_device,
                joint_names=joint_names,
                body_names=body_names,
            )

        if self.max_workers == 1:
            for path, global_id in entries:
                yield _PendingMotion(path, global_id, _load(path))
            return

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="motion-npz") as executor:
            iterator = iter(entries)
            inflight: deque[tuple[str, int, Future[MotionLoader]]] = deque()
            for _ in range(min(2 * self.max_workers, len(entries))):
                path, global_id = next(iterator)
                inflight.append((path, global_id, executor.submit(_load, path)))
            while inflight:
                path, global_id, future = inflight.popleft()
                yield _PendingMotion(path, global_id, future.result())
                try:
                    next_path, next_global_id = next(iterator)
                except StopIteration:
                    continue
                inflight.append((next_path, next_global_id, executor.submit(_load, next_path)))

    def _plan_chunk_frames(self, motion_lengths: Sequence[int]) -> list[int]:
        chunk_frames: list[int] = []
        current_frames = 0
        for length in motion_lengths:
            if length > self.max_chunk_frames:
                raise ValueError(
                    f"A motion contains {length} frames, exceeding motion_chunk_frames="
                    f"{self.max_chunk_frames}; raise the configured chunk limit"
                )
            if current_frames and current_frames + length > self.max_chunk_frames:
                chunk_frames.append(current_frames)
                current_frames = 0
            current_frames += length
            if current_frames >= self.max_chunk_frames:
                chunk_frames.append(current_frames)
                current_frames = 0
        if current_frames:
            chunk_frames.append(current_frames)
        return chunk_frames

    def _initialize_layout(self, loader: MotionLoader) -> None:
        self.field_shapes = {field: tuple(getattr(loader, field).shape[1:]) for field in self._FIELDS}
        for group, fields in MOTION_FIELD_GROUPS.items():
            cursor = 0
            for field in fields:
                width = math.prod(self.field_shapes[field])
                self._field_groups[field] = group
                self._field_slices[field] = slice(cursor, cursor + width)
                cursor += width
            self._group_widths[group] = cursor

    def _validate_loader(self, path: str, loader: MotionLoader) -> None:
        if self.fps is None:
            self.fps = loader.fps
        elif not math.isclose(loader.fps, self.fps):
            raise ValueError(f"All motions must have the same fps; {path} has {loader.fps}, expected {self.fps}")
        if not self.field_shapes:
            self._initialize_layout(loader)
        for field in self._FIELDS:
            shape = tuple(getattr(loader, field).shape[1:])
            if shape != self.field_shapes[field]:
                raise ValueError(
                    f"All motions must share {field} trailing shape; {path} has {shape}, "
                    f"expected {self.field_shapes[field]}"
                )

    def _allocate_chunk(self, total_frames: int) -> dict[str, torch.Tensor]:
        chunk = {
            group: torch.empty(
                (total_frames, self._group_widths[group]),
                dtype=torch.float32,
                device=self.storage_device,
            )
            for group in MOTION_FIELD_GROUPS
        }
        self.resident_bytes += sum(values.numel() * values.element_size() for values in chunk.values())
        return chunk

    def _append_motion(
        self,
        item: _PendingMotion,
        *,
        chunk: dict[str, torch.Tensor],
        chunk_id: int,
        chunk_frame_offset: int,
        global_ids: list[int],
        time_totals: list[int],
        motion_chunk_ids: list[int],
        motion_chunk_offsets: list[int],
    ) -> None:
        self._validate_loader(item.path, item.loader)
        loader = item.loader
        length = loader.time_step_total
        for group, fields in MOTION_FIELD_GROUPS.items():
            target = chunk[group][chunk_frame_offset : chunk_frame_offset + length]
            for field in fields:
                values = getattr(loader, field).reshape(length, -1)
                target[:, self._field_slices[field]].copy_(values)
        self.names.append(item.path)
        global_ids.append(item.global_id)
        time_totals.append(length)
        motion_chunk_ids.append(chunk_id)
        motion_chunk_offsets.append(chunk_frame_offset)

    @property
    def num_motions(self) -> int:
        return len(self.names)

    def lengths(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.time_totals[motion_ids]

    def sample_many(
        self,
        fields: Sequence[str],
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fields = tuple(fields)
        unknown = set(fields) - set(self._FIELDS)
        if unknown:
            raise KeyError(f"Unknown motion fields: {sorted(unknown)}")
        if motion_ids.shape != time_steps.shape:
            raise ValueError(
                f"motion_ids and time_steps must have the same shape, got {motion_ids.shape} and {time_steps.shape}"
            )
        original_shape = tuple(motion_ids.shape)
        flat_motion_ids = motion_ids.reshape(-1).to(self.device, dtype=torch.long)
        flat_time_steps = time_steps.reshape(-1).to(self.device, dtype=torch.long)
        lengths = self.time_totals[flat_motion_ids]
        clamped = torch.minimum(torch.clamp_min(flat_time_steps, 0), lengths - 1)
        if self.num_chunks == 1:
            chunk_ids = None
            chunk_rows = self.motion_chunk_offsets[flat_motion_ids] + clamped
            unique_chunks: list[int] = [0]
        else:
            chunk_ids = self.motion_chunk_ids[flat_motion_ids]
            chunk_rows = self.motion_chunk_offsets[flat_motion_ids] + clamped
            unique_chunks = torch.unique(chunk_ids).tolist()

        requested_groups = tuple(dict.fromkeys(self._field_groups[field] for field in fields))
        group_values: dict[str, torch.Tensor] = {}
        for group in requested_groups:
            width = self._group_widths[group]
            if not flat_motion_ids.numel():
                group_values[group] = torch.empty((0, width), dtype=torch.float32, device=self.device)
                continue
            if self.num_chunks == 1:
                source_rows = chunk_rows.to(self.storage_device)
                group_values[group] = (
                    self._chunks[0][group].index_select(0, source_rows).to(self.device, non_blocking=True)
                )
                continue
            output = torch.empty((len(flat_motion_ids), width), dtype=torch.float32, device=self.device)
            assert chunk_ids is not None
            for chunk_id_value in unique_chunks:
                chunk_id = int(chunk_id_value)
                positions = torch.where(chunk_ids == chunk_id)[0]
                source_rows = chunk_rows[positions].to(self.storage_device)
                values = self._chunks[chunk_id][group].index_select(0, source_rows).to(self.device, non_blocking=True)
                output.index_copy_(0, positions, values)
            group_values[group] = output

        return {
            field: group_values[self._field_groups[field]][:, self._field_slices[field]].reshape(
                *original_shape, *self.field_shapes[field]
            )
            for field in fields
        }

    def sample(self, field: str, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        return self.sample_many((field,), motion_ids, time_steps)[field]
