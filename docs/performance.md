# Performance

`fitscube-rs` is a drop-in replacement for the original Python
[`fitscube`](https://github.com/AlecThomson/fitscube) that produces
**bit-for-bit identical cubes** (the parity test suite asserts this) while
running faster across the board.

## Benchmark

The benchmark generates `nchan` single-channel `float32` FITS images, then
times `combine_fits` for both implementations on the same inputs. The output
cubes are compared and a speedup is only reported when they agree.

Reproduce it with:

```sh
python scripts/benchmark.py --nchan 100 --size 256 --repeat 3
```

### Results

Best of 3 runs, warm page cache. Times measure `combine_fits` only (input
generation excluded).

Hardware: Apple M4 Pro (12 cores). `fitscube` 2.3.1, `astropy` 8.0.0,
`numpy` 2.4.6.

| Channels × image | Cube size | `fitscube` (py) | `fitscube-rs` | Speedup |
| ---------------- | --------: | --------------: | ------------: | ------: |
| 50 × 128²        |    3 MB   |        0.116 s  |     0.052 s   | **2.2×** |
| 100 × 256²       |   26 MB   |        0.187 s  |     0.113 s   | **1.6×** |
| 100 × 512²       |  105 MB   |        0.199 s  |     0.131 s   | **1.5×** |
| 200 × 256²       |   52 MB   |        0.328 s  |     0.223 s   | **1.5×** |
| 500 × 256²       |  131 MB   |        0.751 s  |     0.555 s   | **1.4×** |
| 1000 × 128²      |   66 MB   |        1.375 s  |     1.060 s   | **1.3×** |
| 100 × 1024²      |  419 MB   |        0.295 s  |     0.243 s   | **1.2×** |

## Why it's faster

- **No per-image Python/`astropy` overhead.** Header parsing, WCS handling,
  and frequency-axis construction run in compiled Rust. The win is largest for
  workloads with **many small images**, where this fixed per-image cost
  dominates — hence 2.2× at 50 × 128² shrinking toward parity as raw pixel I/O
  takes over.
- **Streamed, single-pass writes.** The output cube is created and written
  through one open `cfitsio` handle, avoiding a redundant zero-fill pass over
  the full cube before data is written.
- **Parallelism.** Per-channel work (header reads, plane copies, beam-table
  assembly) is fanned out across cores with `rayon`.

## What it does *not* claim

- **Memory.** Peak RSS is comparable to, and on large cubes higher than, the
  Python implementation. `fitscube-rs` optimizes for wall-clock time, not
  footprint — do not pick it expecting lower memory use.
- **Huge single-plane cubes.** Once a job is bound by raw pixel write
  throughput (large image dimensions, few channels), both implementations
  converge on `cfitsio`/`numpy` I/O and the speedup narrows to ~1.2×.

## Caveats

- Times are best-of-3 with a warm page cache; absolute numbers vary with
  hardware, disk, and thread count. The **relative** ordering and the
  many-images-favor-Rust trend are stable.
- The benchmark uses synthetic Gaussian-noise images with minimal headers.
  Real data with rich headers and per-channel beams increases the per-image
  parsing cost, which favors `fitscube-rs` further.
