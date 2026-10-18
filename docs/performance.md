# Performance

`fitscube-rs` writes the same cubes as
[`fitscube`](https://github.com/AlecThomson/fitscube). The parity tests
compare data, axis cards and beam tables.

The tables below are written by `scripts/update_performance.py`, which the
`Benchmark` workflow runs on a GitHub-hosted runner for each release. Shared
runners are noisy, so trust the ratios more than the absolute times.

<!-- benchmark:machine -->
Measured 2026-10-18 on AMD EPYC 7763 64-Core Processor (4 cores), 16 GB RAM, Linux. `fitscube-rs` v2.1.1, `fitscube` 2.11.0, `astropy` 8.0.0, `numpy` 2.4.6, `fpack` 4.070 (CFITSIO).
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
| 50 × 128² | 3 MB | 0.399 s | 0.004 s | 96× |
| 100 × 256² | 26 MB | 0.754 s | 0.014 s | 55× |
| 100 × 512² | 105 MB | 0.755 s | 0.038 s | 20× |
| 200 × 256² | 52 MB | 1.477 s | 0.027 s | 54× |
| 500 × 256² | 131 MB | 3.652 s | 0.062 s | 59× |
| 1000 × 128² | 66 MB | 7.206 s | 0.068 s | 105× |
| 100 × 1024² | 419 MB | 0.818 s | 0.129 s | 6.3× |
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
| `combine` | 1.62 s, 282 MB | 0.16 s, 62 MB |  |
| `combine --bounding-box` | 1.95 s, 517 MB | 0.18 s, 48 MB |  |
| `extract` | 0.81 s, 117 MB | 0.01 s, 10 MB |  |
| `compress tile-gzip` | 7.43 s, 186 MB | 3.44 s, 61 MB | 13.02 s |
| `decompress` (tile-gzip) | 3.25 s, 120 MB | 0.81 s, 26 MB |  |
| `compress tile-rice` | 3.28 s, 159 MB | 1.24 s, 57 MB | 4.00 s |
| `decompress` (tile-rice) | 2.23 s, 121 MB | 0.80 s, 24 MB |  |
<!-- /benchmark:cli-100x1024 -->

1000 × 128² `float32` (66 MB cube), where fitscube's per-plane overhead
shows:

<!-- benchmark:cli-1000x128 -->
| Command | `fitscube` | `fitscubers` | `fpack` |
| ------- | ---------: | -----------: | ------: |
| `combine` | 7.96 s, 133 MB | 0.09 s, 10 MB |  |
| `combine --bounding-box` | 9.26 s, 195 MB | 0.12 s, 10 MB |  |
| `extract` | 0.85 s, 111 MB | 0.01 s, — |  |
| `compress tile-gzip` | 1.57 s, 114 MB | 0.42 s, 19 MB | 8.06 s |
| `decompress` (tile-gzip) | 1.92 s, 112 MB | 0.10 s, 8 MB |  |
| `compress tile-rice` | 14.02 s, 114 MB | 0.21 s, 15 MB | 0.73 s |
| `decompress` (tile-rice) | 1.72 s, 112 MB | 0.07 s, 8 MB |  |
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
