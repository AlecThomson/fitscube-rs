# Performance

`fitscube-rs` writes the same cubes as
[`fitscube`](https://github.com/AlecThomson/fitscube). The parity tests
compare data, axis cards and beam tables.

The tables below are written by `scripts/update_performance.py`, which the
`Benchmark` workflow runs on a GitHub-hosted runner for each release. Shared
runners are noisy, so trust the ratios more than the absolute times.

<!-- benchmark:machine -->
Measured 2026-11-28 on Intel(R) Xeon(R) 6975P-C (4 cores), 16 GB RAM, Linux. `fitscube-rs` v2.5.0, `fitscube` 2.12.2, `astropy` 8.0.0, `numpy` 2.4.6, `fpack` 4.070 (CFITSIO).
<!-- /benchmark:machine -->

To refresh them locally:

```sh
python scripts/update_performance.py --workdir /scratch
```

## Python API

`scripts/benchmark.py` writes `nchan` single-channel `float32` FITS images,
times `combine_fits` for both implementations on the same inputs, and checks
that the two cubes agree. Best of 5 runs; input generation is not timed.

```sh
python scripts/benchmark.py --nchan 100 --size 256 --repeat 3
```

<!-- benchmark:api -->
| Channels × image | Cube size | `fitscube` (py) | `fitscube-rs` | Speedup |
| ---------------- | --------: | --------------: | ------------: | ------: |
| 50 × 128² | 3 MB | 0.246 s | 0.002 s | 117× |
| 100 × 256² | 26 MB | 0.462 s | 0.010 s | 47× |
| 100 × 512² | 105 MB | 0.474 s | 0.030 s | 16× |
| 200 × 256² | 52 MB | 0.887 s | 0.019 s | 47× |
| 500 × 256² | 131 MB | 2.202 s | 0.044 s | 50× |
| 1000 × 128² | 66 MB | 4.350 s | 0.036 s | 123× |
| 100 × 1024² | 419 MB | 0.568 s | 0.110 s | 5.2× |
<!-- /benchmark:api -->

## Command line

`scripts/benchmark_cli.py` times the two CLIs, and `fpack`, on the same cube:
combine, extract one plane, then compress and decompress the cube with 4
threads. Noise images with a blank top and bottom eighth. Median of 5 runs.
Wall time, then peak RSS; `fpack` is wall time only, and single-threaded.

```sh
python scripts/benchmark_cli.py /scratch --nchan 100 --size 1024
```

100 × 1024² `float32` (419 MB cube):

<!-- benchmark:cli-100x1024 -->
| Command | `fitscube` | `fitscubers` | `fpack` |
| ------- | ---------: | -----------: | ------: |
| `combine` | 1.15 s, 305 MB | 0.13 s, 62 MB |  |
| `combine --bounding-box` | 1.39 s, 521 MB | 0.14 s, 49 MB |  |
| `extract` | 0.59 s, 134 MB | 0.01 s, — |  |
| `compress tile-gzip` | 6.37 s, 188 MB | 2.62 s, 32 MB | 7.78 s |
| `decompress` (tile-gzip) | 2.52 s, 138 MB | 1.40 s, 25 MB |  |
| `compress tile-rice` | 2.47 s, 185 MB | 1.04 s, 31 MB | 2.96 s |
| `decompress` (tile-rice) | 1.98 s, 138 MB | 1.40 s, 24 MB |  |
<!-- /benchmark:cli-100x1024 -->

1000 × 128² `float32` (66 MB cube), where fitscube's per-plane overhead
shows:

<!-- benchmark:cli-1000x128 -->
| Command | `fitscube` | `fitscubers` | `fpack` |
| ------- | ---------: | -----------: | ------: |
| `combine` | 4.76 s, 151 MB | 0.04 s, 10 MB |  |
| `combine --bounding-box` | 5.53 s, 213 MB | 0.05 s, 10 MB |  |
| `extract` | 0.60 s, 128 MB | 0.01 s, — |  |
| `compress tile-gzip` | 1.18 s, 131 MB | 0.30 s, 19 MB | 5.03 s |
| `decompress` (tile-gzip) | 1.33 s, 129 MB | 0.09 s, 9 MB |  |
| `compress tile-rice` | 9.12 s, 131 MB | 0.18 s, 16 MB | 0.52 s |
| `decompress` (tile-rice) | 1.17 s, 129 MB | 0.05 s, 9 MB |  |
<!-- /benchmark:cli-1000x128 -->

On a 415 MB cube with sky structure, `tile-gzip` gives 147 MB in either tool,
and `tile-rice` 44.5 MB against fitscube's 43.0 MB (see
[Algorithm background](algorithms.md)).

## How combine works

- Headers, WCS and the frequency axis are handled in Rust, so there is no
  per-image Python overhead.
- All the header cards combine needs are read in one parallel pass, one open
  per file.
- Each plane is read into its output type, byte-swapped in place and written
  at its offset in a sparse file, so the data unit is written once. With
  `--bounding-box` only the box is read.
- Planes are read and swapped on reader threads (`--max-workers`, default one
  per core) while one thread writes.
- `extract` reads one plane, not the whole cube, and writes the plane file
  directly.

## Memory

`fitscube-rs` streams: at most about three planes per reader thread are in
memory, never the whole cube. `fitscube` memmaps the output cube;
`fitscube-rs` writes each plane with positional writes.
