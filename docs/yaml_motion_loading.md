# Scalable YAML motion loading

The tracking tasks keep `commands.motion.motion_file` as their only dataset
interface. It may point to one NPZ file or to the existing recursive YAML
manifest. Training does not require a converted dataset, generated catalog, or
project-specific on-disk format.

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

## Adaptive sampling

Adaptive statistics are rank-local when motion files are rank-sharded because
different ranks intentionally own different bin tables. Gradients remain
globally reduced by DDP. A cached inverse CDF makes reset-time sampling
`O(batch * log(bins))`; the distribution is rebuilt only when accumulated
statistics are merged. If a rank would exceed the configured bin limit, the
sampler falls back from fixed-size temporal bins to one bucket per local motion.

## Relevant configuration

```python
commands.motion.motion_file = "/path/to/t800_v0.yaml"
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

Storage remains linear in selected frames: 232 float32 values, or 928 bytes,
per frame for the current 25 joints and 14 bodies. At 100 times this corpus,
an eight-rank node owns about 1.5 TiB of resident motion data. That fits when
2 TiB is available per eight-rank node, with room for bounded construction
overhead. If 2 TiB is the capacity of the entire 32-rank cluster instead, one
hundred times the corpus cannot be fully resident and requires additional
nodes or a separately designed rotating working set.
