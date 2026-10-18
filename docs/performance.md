# Performance

`fitscube-rs` writes the same cubes as
[`fitscube`](https://github.com/AlecThomson/fitscube). The parity tests
compare data, axis cards and beam tables.

The tables below are written by `scripts/update_performance.py`, which the
`Benchmark` workflow runs on a GitHub-hosted runner for each release. Shared
runners are noisy, so trust the ratios more than the absolute times.

<!-- benchmark:machine -->
Measured 2026-10-18 on Intel(R) Xeon(R) Processor @ 2.10GHz (4 cores), 16 GB RAM, Linux. `fitscube-rs` v2.0.2-2-g22038c1, `fitscube` 2.11.0, `astropy` 8.0.0, `numpy` 2.4.6, `fpack` 4.030 (CFITSIO).
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
| 50 × 128² | 3 MB | 0.555 s | 0.004 s | 140× |
| 100 × 256² | 26 MB | 1.101 s | 0.013 s | 85× |
| 100 × 512² | 105 MB | 1.070 s | 0.048 s | 22× |
| 200 × 256² | 52 MB | 2.074 s | 0.026 s | 80× |
| 500 × 256² | 131 MB | 5.060 s | 0.074 s | 69× |
| 1000 × 128² | 66 MB | 10.170 s | 0.054 s | 187× |
| 100 × 1024² | 419 MB | 1.123 s | 0.375 s | 3.0× |
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
| `combine` | 2.23 s, 265 MB | 0.73 s, 62 MB |  |
| `combine --bounding-box` | 3.00 s, 506 MB | 0.33 s, 49 MB |  |
| `extract` | 0.97 s, 112 MB | 0.01 s, 10 MB |  |
| `compress tile-gzip` | 5.63 s, 183 MB | 3.41 s, 76 MB | 10.88 s |
| `decompress` (tile-gzip) | 4.18 s, 115 MB | 0.86 s, 26 MB |  |
| `compress tile-rice` | 3.72 s, 154 MB | 1.28 s, 60 MB | 3.73 s |
| `decompress` (tile-rice) | 3.31 s, 115 MB | 0.72 s, 24 MB |  |
<!-- /benchmark:cli-100x1024 -->

1000 × 128² `float32` (66 MB cube), where fitscube's per-plane overhead
shows:

<!-- benchmark:cli-1000x128 -->
| Command | `fitscube` | `fitscubers` | `fpack` |
| ------- | ---------: | -----------: | ------: |
| `combine` | 11.14 s, 127 MB | 0.11 s, 11 MB |  |
| `combine --bounding-box` | 13.25 s, 190 MB | 0.14 s, 11 MB |  |
| `extract` | 0.88 s, 106 MB | 0.01 s, — |  |
| `compress tile-gzip` | 1.61 s, 109 MB | 0.42 s, 19 MB | 6.41 s |
| `decompress` (tile-gzip) | 2.11 s, 106 MB | 0.19 s, 9 MB |  |
| `compress tile-rice` | 17.63 s, 109 MB | 0.22 s, 16 MB | 0.67 s |
| `decompress` (tile-rice) | 1.76 s, 106 MB | 0.14 s, 9 MB |  |
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
