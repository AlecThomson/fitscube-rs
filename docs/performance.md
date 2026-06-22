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
- **Streamed, single-pass writes.** The output cube's header is built in memory
  and written once; the data unit is laid down plane-by-plane with raw I/O over
  a sparse file, avoiding cfitsio's redundant zero-fill pass.
- **Parallelism.** Per-channel work (header reads, plane copies, beam-table
  assembly) is fanned out across cores with `rayon`.

## Memory

`fitscube-rs` streams: only a small number of channel planes are resident at
once, never the whole cube. Peak memory is roughly *one plane × in-flight count*
plus the header, independent of channel count.

Peak resident memory (max RSS), combine-only, inputs generated in a separate
process so only the combine is measured:

| Channels × image | Cube size | fitscube (py) | fitscube-rs (CLI) | fitscube-rs (Python) |
| ---------------- | --------: | ------------: | ----------------: | -------------------: |
| 200 × 512²       |  210 MB   |     ~155 MB   |          **58 MB** |              82 MB   |
| 100 × 1024²      |  419 MB   |     ~300 MB   |         **159 MB** |             275 MB   |

The native CLI uses the least memory and is stable run-to-run. The
`fitscube-rs` Python binding pays the interpreter's baseline on top of the
streaming core, so it wins at small/medium cubes and roughly ties `fitscube` at
large ones — use the CLI when footprint matters.

(An earlier `fitscube-rs` build allocated a cube-sized buffer in RAM while
assembling the header — peaking at the full cube size — which has since been
fixed; the numbers above are post-fix.)

### Larger than memory

Both tools stream, so neither needs to hold the whole cube — but *how* they
write the data unit differs, and it matters once the cube exceeds RAM.
`fitscube` memmaps the output and writes through the mapping, which dirties the
entire output in memory; the peak *footprint* (dirty + anonymous pages) grows to
the full cube size. `fitscube-rs` writes each plane with positional
`write_all_at`, so written pages are flushed and reclaimed rather than charged
to the process — its footprint stays bounded by the in-flight planes regardless
of cube size.

Combine of a **26 GB cube on a 24 GB-RAM machine** (420 × 4096², float32),
combine-only:

| Implementation                    |   Time | Peak footprint | Max RSS |
| --------------------------------- | -----: | -------------: | ------: |
| `fitscube` (py)                   |  78 s  |     **23.5 GB**|  1.6 GB |
| `fitscube-rs` (CLI, default)      |  19 s  |       2.6 GB   |  2.6 GB |
| `fitscube-rs` (CLI, max-workers 4)|  16 s  |     **1.2 GB** |    —    |

Once the cube exceeds RAM, `fitscube`'s footprint hits the memory ceiling and it
thrashes — ~4–5× slower here, and it would OOM outright on a cube comfortably
larger than RAM. `fitscube-rs` is unaffected: its footprint is set by *plane
size × in-flight count*, tunable with `--max-workers` (the default over-buffers
for very large planes; `--max-workers 4` was both lighter and faster above).

## What it does *not* claim

- **Huge single-plane cubes (time).** Once a job is bound by raw pixel write
  throughput (large image dimensions, few channels), both implementations
  converge on `cfitsio`/`numpy` I/O and the speedup narrows to ~1.5×.

## Caveats

- Times are best-of-3 with a warm page cache; absolute numbers vary with
  hardware, disk, and thread count. The **relative** ordering and the
  many-images-favor-Rust trend are stable.
- The benchmark uses synthetic Gaussian-noise images with minimal headers.
  Real data with rich headers and per-channel beams increases the per-image
  parsing cost, which favors `fitscube-rs` further.
