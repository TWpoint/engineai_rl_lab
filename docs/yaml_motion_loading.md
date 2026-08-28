# Scalable YAML motion loading

The tracking tasks keep `commands.motion.motion_file` as their only dataset
interface. It may point to one NPZ file or to the existing recursive YAML
manifest. An optional JSON catalog caches the resolved paths and frame counts;
the original YAML and NPZ files remain the source of truth and no converted
motion format is required.

## Distributed layout

For distributed training, every worker resolves the same ordered YAML list and
loads the deterministic slice `files[rank::world_size]`. The slices are
disjoint and their union is the complete YAML dataset. Motion IDs stored inside
one worker are local for fast indexing; the corresponding global YAML IDs are
retained for diagnostics.

This changes host memory from `world_size * dataset_size` to approximately one
corpus copy across the complete job. With equal environments per worker, DDP
still trains on the union of all rank-local motion distributions.

## In-memory chunks

Each worker reads its original NPZ files into bounded construction chunks. A
chunk stores two contiguous groups:

- `pose`: body position and quaternion;
- `state`: joint position/velocity and body linear/angular velocity.

The loader first reads only NPZ array headers, preallocates exact chunk sizes,
then copies a bounded prefetch queue of decoded clips directly into their final
locations. It never retains every per-file tensor or allocates a second
full-corpus concatenation. At runtime, one row-index calculation feeds all
requested fields in a group. The body-command look-ahead therefore reads only
the pose group, while the per-step command cache reads both groups.

The chunks are an internal, process-local memory layout. They are never written
to disk and do not alter the YAML or NPZ contract.

## Persistent catalog

`motion_catalog_cache` avoids reopening every NPZ merely to discover its frame
count. A cache hit validates the complete recursive YAML dependency graph and
the ordered path/length digest, but deliberately does not `stat` every NPZ. The
referenced NPZ corpus is therefore treated as immutable/versioned: modify a
manifest or delete the catalog when replacing an NPZ in place.

Concurrent writers use an atomic exclusive-create lock file and publish through
`fsync` plus `os.replace`; this also works on the cluster's NFS mounts where
advisory locking is disabled. The V23 launcher warms this shared catalog before
starting `torchrun`, so the first scan is performed once instead of by 24 ranks.

## Adaptive sampling

The legacy command keeps rank-local adaptive tables when files are sharded.
`MotionCommandV1` instead keeps the complete global bin layout on every rank and
all-reduces its sufficient statistics. Each rank then conditions that global
distribution on its resident shard. This preserves global IDs, checkpoint
layout, and corpus-wide evidence, while deliberately giving every rank equal
training capacity even if adaptive probability mass becomes uneven across
shards.

## Relevant configuration

```python
commands.motion.motion_file = "/path/to/t800_v0.yaml"
commands.motion.motion_catalog_cache = "/path/to/t800_v0.motion_catalog.json"
commands.motion.motion_shard_across_ranks = True
commands.motion.motion_load_workers = 4
commands.motion.motion_chunk_frames = 8_388_608
commands.motion.adaptive_max_bins = 5_000_000
```

`motion_shard_across_ranks=False` preserves the old replicated behavior for
debugging. Single-process runs are never divided.

## Measured `t800_v0` scale

The 151,335-file manifest contains 68,985,003 frames. With 32 ranks, workers
receive 4,729 or 4,730 files and 2.084M--2.228M frames (a 1.069 max/min frame
ratio). A rank-0 CPU validation using the 14 V8 command bodies measured:

- 6.7 seconds to resolve the complete YAML with LibYAML;
- 1.895 GiB final motion storage and 2.97 GiB peak process RSS;
- 12.4 seconds after resolution to inspect, validate, decode, and pack 4,730
  original NPZ files;
- 0.29 ms for an 8,192-row all-field CPU `sample_many` call.

The V23 24-rank configuration was also validated on the same corpus: the first
catalog build took 52.3 seconds, a catalog hit took 6.4 seconds including Python
startup and dependency validation, and rank 0 decoded its 6,306 files / 2.846M
frames into 2.46 GiB in 20.1 seconds. The 24 shards are disjoint, contain 6,305
or 6,306 files each, and sum to all 68,985,003 frames.

Storage remains linear in selected frames: 232 float32 values, or 928 bytes,
per frame for the current 25 joints and 14 bodies. At 100 times this corpus,
an eight-rank node owns about 1.5 TiB of resident motion data. That fits when
2 TiB is available per eight-rank node, with room for bounded construction
overhead. If 2 TiB is the capacity of the entire 32-rank cluster instead, one
hundred times the corpus cannot be fully resident and requires additional
nodes or a separately designed rotating working set.
