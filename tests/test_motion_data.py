from __future__ import annotations

import importlib.util
import json
import multiprocessing
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[1]
    / "source"
    / "engineai_rl_lab"
    / "engineai_rl_lab"
    / "tasks"
    / "tracking"
    / "mdp"
    / "motion_data.py"
)
SPEC = importlib.util.spec_from_file_location("engineai_motion_data_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
motion_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_data
SPEC.loader.exec_module(motion_data)

MotionCollection = motion_data.MotionCollection
MotionLoader = motion_data.MotionLoader
resolve_motion_catalog = motion_data.resolve_motion_catalog
resolve_motion_files = motion_data.resolve_motion_files
select_motion_shard = motion_data.select_motion_shard
shard_motion_files = motion_data.shard_motion_files


JOINT_FILE_ORDER = ("j1", "j0")
JOINT_OUTPUT_ORDER = ("j0", "j1")
BODY_FILE_ORDER = ("b1", "unused", "b0")
BODY_OUTPUT_ORDER = ("b0", "b1")


def _write_motion(path: Path, frames: int, offset: float, *, quaternion_order: str = "wxyz") -> None:
    frame = np.arange(frames, dtype=np.float32)[:, None]
    joint_pos = np.concatenate((frame + offset + 10.0, frame + offset + 20.0), axis=1)
    joint_vel = joint_pos + 100.0
    body_base = np.arange(frames * 3 * 3, dtype=np.float32).reshape(frames, 3, 3) + offset
    body_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    if quaternion_order == "xyzw":
        body_quat = np.roll(body_quat, shift=-1, axis=-1)
    np.savez(
        path,
        fps=np.asarray(50.0, dtype=np.float32),
        joint_names=np.asarray(JOINT_FILE_ORDER),
        body_names=np.asarray(BODY_FILE_ORDER),
        quaternion_order=np.asarray(quaternion_order),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_base,
        body_quat_w=body_quat,
        body_lin_vel_w=body_base + 1000.0,
        body_ang_vel_w=body_base + 2000.0,
    )


def _resolve_catalog_process(
    manifest: str,
    cache_path: str,
    start_event,
    results,
) -> None:
    start_event.wait()
    try:
        results.put(resolve_motion_catalog(manifest, cache_path=cache_path, max_workers=1))
    except BaseException as error:  # pragma: no cover - asserted through the child exit/result.
        results.put((type(error).__name__, str(error)))
        raise


def test_recursive_yaml_resolver_preserves_order_deduplicates_and_excludes(tmp_path: Path) -> None:
    motions = tmp_path / "motions"
    motions.mkdir()
    for name in ("a.npz", "b.npz", "c.npz"):
        _write_motion(motions / name, 2, 0.0)
    child = tmp_path / "child.yaml"
    child.write_text(
        "files:\n  - motions/a.npz\n  - motions/b.npz\nexclude_files:\n  - motions/b.npz\n",
        encoding="utf-8",
    )
    root = tmp_path / "root.yaml"
    root.write_text(
        "files:\n  - child.yaml\n  - motions/a.npz\n  - motions/c.npz\n",
        encoding="utf-8",
    )

    assert resolve_motion_files(root) == [str(motions / "a.npz"), str(motions / "c.npz")]


def test_missing_nested_exclusion_manifest_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root.yaml"
    root.write_text(
        "files:\n  - motion.npz\nexclude_files:\n  - missing-exclusions.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing-exclusions.yaml"):
        resolve_motion_files(root)


def test_yaml_resolver_applies_glob_exclusions(tmp_path: Path) -> None:
    motions = tmp_path / "motions"
    motions.mkdir()
    for name in ("keep.npz", "skip-one.npz", "skip-two.npz"):
        _write_motion(motions / name, 2, 0.0)
    root = tmp_path / "root.yaml"
    root.write_text(
        "files:\n  - motions/keep.npz\n  - motions/skip-one.npz\n  - motions/skip-two.npz\n"
        "exclude_files:\n  - motions/skip-*.npz\n",
        encoding="utf-8",
    )

    assert resolve_motion_files(root) == [str(motions / "keep.npz")]


def test_motion_catalog_cache_hit_skips_manifest_resolution_and_npz_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motions = tmp_path / "motions"
    motions.mkdir()
    first = motions / "a.npz"
    second = motions / "b.npz"
    _write_motion(first, 2, 0.0)
    _write_motion(second, 3, 1.0)
    child = tmp_path / "child.yaml"
    child.write_text("files:\n  - motions/a.npz\n  - motions/b.npz\n", encoding="utf-8")
    root = tmp_path / "root.yaml"
    root.write_text("files:\n  - child.yaml\n", encoding="utf-8")
    cache = tmp_path / "cache" / "catalog.json"

    expected = ([str(first), str(second)], [2, 3])
    assert resolve_motion_catalog(root, cache_path=cache, max_workers=2) == expected
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["manifest"] == str(root)
    assert {item["path"] for item in payload["manifest_dependencies"]} == {str(root), str(child)}
    assert "motion_stats" not in payload

    monkeypatch.setattr(
        motion_data,
        "_resolve_motion_files_and_dependencies",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache hit resolved YAML")),
    )
    monkeypatch.setattr(
        motion_data,
        "read_motion_lengths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache hit read NPZ headers")),
    )
    assert resolve_motion_catalog(root, cache_path=cache, max_workers=2) == expected


def test_motion_catalog_cache_rebuilds_after_nested_manifest_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _write_motion(first, 2, 0.0)
    _write_motion(second, 4, 1.0)
    child = tmp_path / "child.yaml"
    child.write_text("files:\n  - a.npz\n", encoding="utf-8")
    root = tmp_path / "root.yaml"
    root.write_text("files:\n  - child.yaml\n", encoding="utf-8")
    cache = tmp_path / "catalog.json"

    calls = 0
    original_read_lengths = motion_data.read_motion_lengths

    def counted_read_lengths(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_read_lengths(*args, **kwargs)

    monkeypatch.setattr(motion_data, "read_motion_lengths", counted_read_lengths)
    assert resolve_motion_catalog(root, cache_path=cache) == ([str(first)], [2])
    assert calls == 1

    child.write_text("files:\n  - b.npz\n", encoding="utf-8")
    assert resolve_motion_catalog(root, cache_path=cache) == ([str(second)], [4])
    assert calls == 2


def test_motion_catalog_cache_recovers_from_corrupt_json(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    _write_motion(motion, 3, 0.0)
    manifest = tmp_path / "motions.yaml"
    manifest.write_text("files:\n  - motion.npz\n", encoding="utf-8")
    cache = tmp_path / "catalog.json"
    cache.write_text("{not valid json", encoding="utf-8")

    assert resolve_motion_catalog(manifest, cache_path=cache) == ([str(motion)], [3])
    assert json.loads(cache.read_text(encoding="utf-8"))["motion_lengths"] == [3]


def test_motion_catalog_direct_npz_cache_invalidates_on_file_change(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    cache = tmp_path / "catalog.json"
    _write_motion(motion, 2, 0.0)
    assert resolve_motion_catalog(motion, cache_path=cache) == ([str(motion)], [2])

    _write_motion(motion, 5, 0.0)
    assert resolve_motion_catalog(motion, cache_path=cache) == ([str(motion)], [5])


def test_motion_catalog_reclaims_dead_same_host_lock(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    _write_motion(motion, 2, 0.0)
    manifest = tmp_path / "motions.yaml"
    manifest.write_text("files:\n  - motion.npz\n", encoding="utf-8")
    cache = tmp_path / "catalog.json"
    lock = tmp_path / "catalog.json.lock"
    lock.write_text(
        json.dumps(
            {
                "token": "abandoned",
                "hostname": socket.gethostname(),
                "pid": 2**31 - 1,
                "created_ns": 0,
            }
        ),
        encoding="utf-8",
    )

    assert resolve_motion_catalog(manifest, cache_path=cache) == ([str(motion)], [2])
    assert not lock.exists()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_motion_catalog_concurrent_processes_build_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motions = []
    for index, frames in enumerate((2, 3, 4)):
        path = tmp_path / f"motion-{index}.npz"
        _write_motion(path, frames, float(index))
        motions.append(path)
    manifest = tmp_path / "motions.yaml"
    manifest.write_text(
        "files:\n" + "".join(f"  - {path.name}\n" for path in motions),
        encoding="utf-8",
    )
    cache = tmp_path / "catalog.json"
    build_log = tmp_path / "catalog-builds.txt"
    original_read_lengths = motion_data.read_motion_lengths

    def logged_read_lengths(*args, **kwargs):
        with build_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{multiprocessing.current_process().pid}\n")
        return original_read_lengths(*args, **kwargs)

    monkeypatch.setattr(motion_data, "read_motion_lengths", logged_read_lengths)
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_resolve_catalog_process,
            args=(str(manifest), str(cache), start_event, results),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    expected = ([str(path) for path in motions], [2, 3, 4])
    assert [results.get(timeout=2) for _ in processes] == [expected] * len(processes)
    assert len(build_log.read_text(encoding="utf-8").splitlines()) == 1
    assert not cache.with_name(f"{cache.name}.lock").exists()


def test_rank_shards_are_disjoint_complete_and_use_global_rank() -> None:
    files = [f"motion-{index}" for index in range(10)]
    shards = [shard_motion_files(files, world_size=4, rank=rank) for rank in range(4)]
    ids = [set(global_ids) for _, global_ids in shards]
    assert set.union(*ids) == set(range(10))
    assert all(ids[left].isdisjoint(ids[right]) for left in range(4) for right in range(left + 1, 4))
    assert max(map(len, ids)) - min(map(len, ids)) <= 1
    assert shards[2][1] == [2, 6]


def test_more_ranks_than_motions_replicates_small_dataset() -> None:
    files = ["a", "b"]
    selection = select_motion_shard(files, world_size=4, rank=3)
    assert selection.files == tuple(files)
    assert selection.global_ids == (0, 1)
    assert not selection.is_distributed_shard


@pytest.mark.parametrize(("world_size", "rank"), [(0, 0), (2, -1), (2, 2)])
def test_invalid_rank_metadata_is_rejected(world_size: int, rank: int) -> None:
    with pytest.raises(ValueError):
        select_motion_shard(["a"], world_size=world_size, rank=rank)


def test_rank_metadata_must_be_paired(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="provided together"):
        select_motion_shard(["a"], world_size=2)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(ValueError, match="set together"):
        select_motion_shard(["a"])


def test_motion_loader_reorders_names_selects_bodies_and_converts_quaternion(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path, 3, 7.0)
    loader = MotionLoader(
        str(path),
        body_indexes=(0, 1),
        joint_names=JOINT_OUTPUT_ORDER,
        body_names=BODY_OUTPUT_ORDER,
    )

    assert loader.joint_pos.shape == (3, 2)
    torch.testing.assert_close(loader.joint_pos[0], torch.tensor([27.0, 17.0]))
    assert loader.body_pos_w.shape == (3, 2, 3)
    # b0 is file body index 2, b1 is index 0.
    torch.testing.assert_close(loader.body_pos_w[0, :, 0], torch.tensor([13.0, 7.0]))
    torch.testing.assert_close(loader.body_quat_w[..., 3], torch.ones(3, 2))
    torch.testing.assert_close(loader.body_quat_w[..., :3], torch.zeros(3, 2, 3))


def test_motion_loader_rejects_nonfinite_data(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_motion(path, 3, 0.0)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["joint_vel"] = arrays["joint_vel"].copy()
    arrays["joint_vel"][1, 0] = np.nan
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="non-finite values in joint_vel"):
        MotionLoader(
            str(path),
            body_indexes=(0, 1),
            joint_names=JOINT_OUTPUT_ORDER,
            body_names=BODY_OUTPUT_ORDER,
        )


def test_chunked_collection_sample_many_matches_individual_loaders(tmp_path: Path) -> None:
    paths: list[str] = []
    lengths = (2, 3, 4, 5)
    for index, length in enumerate(lengths):
        path = tmp_path / f"motion-{index}.npz"
        _write_motion(path, length, 100.0 * index)
        paths.append(str(path))

    collection = MotionCollection(
        paths,
        body_indexes=(0, 1),
        device="cpu",
        storage_device="cpu",
        joint_names=JOINT_OUTPUT_ORDER,
        body_names=BODY_OUTPUT_ORDER,
        world_size=1,
        rank=0,
        max_chunk_frames=5,
        max_workers=2,
    )
    assert collection.num_chunks == 3
    assert collection.num_motions == 4
    assert collection.global_ids.tolist() == [0, 1, 2, 3]
    assert len(collection.manifest_fingerprint) == 64
    assert len(collection.manifest_fingerprint_words) == 4
    assert collection.rank_num_frames == sum(lengths)
    assert collection.resident_bytes == sum(lengths) * 30 * 4

    motion_ids = torch.tensor([0, 1, 2, 3])
    time_steps = torch.tensor([-3, 2, 999, 1])
    samples = collection.sample_many(collection._FIELDS, motion_ids, time_steps)
    expected_times = (0, 2, 3, 1)
    for field in collection._FIELDS:
        expected = []
        for path, time_step in zip(paths, expected_times, strict=True):
            loader = MotionLoader(
                path,
                body_indexes=(0, 1),
                joint_names=JOINT_OUTPUT_ORDER,
                body_names=BODY_OUTPUT_ORDER,
            )
            expected.append(getattr(loader, field)[time_step])
        torch.testing.assert_close(samples[field], torch.stack(expected))


def test_single_chunk_preserves_each_motion_offset(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"motion-{index}.npz"
        _write_motion(path, 3, 100.0 * index)
        paths.append(str(path))
    collection = MotionCollection(
        paths,
        body_indexes=(0, 1),
        device="cpu",
        storage_device="cpu",
        joint_names=JOINT_OUTPUT_ORDER,
        body_names=BODY_OUTPUT_ORDER,
        max_chunk_frames=9,
    )
    assert collection.num_chunks == 1

    sampled = collection.sample("joint_pos", torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long))

    torch.testing.assert_close(sampled[:, 0], torch.tensor([20.0, 120.0, 220.0]))


def test_sonic_working_set_preserves_replacement_sampled_duplicates(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"motion-{index}.npz"
        _write_motion(path, 3, 100.0 * index)
        paths.append(str(path))

    collection = MotionCollection(
        paths,
        body_indexes=(0, 1),
        device="cpu",
        storage_device="cpu",
        joint_names=JOINT_OUTPUT_ORDER,
        body_names=BODY_OUTPUT_ORDER,
        selected_global_ids=[1, 1, 0],
        global_time_totals=[3, 3],
        max_chunk_frames=9,
    )

    assert collection.global_ids.tolist() == [1, 1, 0]
    assert collection.num_motions == 3
    assert collection.is_distributed_shard
    sampled = collection.sample("joint_pos", torch.tensor([0, 1, 2]), torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(sampled[:, 0], torch.tensor([120.0, 120.0, 20.0]))


def test_window_shape_and_parallel_loading_are_deterministic(tmp_path: Path) -> None:
    paths = []
    for index in range(6):
        path = tmp_path / f"motion-{index}.npz"
        _write_motion(path, 3 + index, float(index))
        paths.append(str(path))

    kwargs = dict(
        body_indexes=(0, 1),
        device="cpu",
        storage_device="cpu",
        joint_names=JOINT_OUTPUT_ORDER,
        body_names=BODY_OUTPUT_ORDER,
        world_size=2,
        rank=1,
        max_chunk_frames=8,
    )
    sequential = MotionCollection(paths, max_workers=1, **kwargs)
    parallel = MotionCollection(paths, max_workers=3, **kwargs)
    assert sequential.names == parallel.names
    assert sequential.global_ids.tolist() == [1, 3, 5]
    assert sequential.global_time_totals.tolist() == [3, 4, 5, 6, 7, 8]
    torch.testing.assert_close(sequential.global_time_totals, parallel.global_time_totals)
    motion_ids = torch.tensor([[0, 0], [1, 2]])
    time_steps = torch.tensor([[0, 1], [2, 99]])
    sequential_samples = sequential.sample_many(("body_pos_w", "body_quat_w"), motion_ids, time_steps)
    parallel_samples = parallel.sample_many(("body_pos_w", "body_quat_w"), motion_ids, time_steps)
    for field in sequential_samples:
        assert sequential_samples[field].shape[:2] == (2, 2)
        torch.testing.assert_close(sequential_samples[field], parallel_samples[field])


def test_motion_larger_than_chunk_limit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "long-motion.npz"
    _write_motion(path, 6, 0.0)

    with pytest.raises(ValueError, match="exceeding motion_chunk_frames"):
        MotionCollection(
            [str(path)],
            body_indexes=(0, 1),
            device="cpu",
            storage_device="cpu",
            joint_names=JOINT_OUTPUT_ORDER,
            body_names=BODY_OUTPUT_ORDER,
            max_chunk_frames=5,
        )
