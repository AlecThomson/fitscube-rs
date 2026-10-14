# Performance

`fitscube-rs` writes the same cubes as
[`fitscube`](https://github.com/AlecThomson/fitscube) (the parity tests compare
data, axis cards and beam tables) and is faster on every case measured.

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

- **No per-image Python overhead.** Headers, WCS and the frequency axis are
  handled in Rust. This matters most for many small images, hence 2.2x at
  50 x 128² shrinking towards parity as pixel I/O takes over.
- **One header read per input.** All the header cards combine needs are read
  in one parallel pass, one open per file.
- **Less pixel copying.** Each plane is read straight into its output type,
  byte-swapped in place and written with positional writes into a sparse file,
  so the data unit is written once. With `--bounding-box` only the box is read.
- **Parallel reads.** Planes are read and swapped on `rayon` threads
  (`--max-workers` of them) while one thread writes.
- **`extract` reads one plane**, not the whole cube.

### Against the previous release

200 x 1024² float32 images (800 MB cube), 4 cores, median of 4 runs, the CLI
before and after the single header pass and plane-read changes. Combine ran on
tmpfs; extract on disk with a warm cache.

| Command                         | Before | After  |
| ------------------------------- | -----: | -----: |
| `combine`                       | 0.73 s | 0.44 s |
| `combine --bounding-box`        | 1.40 s | 0.80 s |
| `extract --channel-index 100`   | 2.6 s  | 0.02 s |

Overwriting an existing cube adds the time to delete the old one, since the
new cube is written beside it and renamed into place.

## Memory

`fitscube-rs` streams: at most about three planes per reader thread are in
memory, never the whole cube, whatever the channel count.

Peak resident memory (max RSS), combine-only, inputs generated in a separate
process so only the combine is measured:

| Channels × image | Cube size | fitscube (py) | fitscube-rs (CLI) | fitscube-rs (Python) |
| ---------------- | --------: | ------------: | ----------------: | -------------------: |
| 200 × 512²       |  210 MB   |     ~155 MB   |          **58 MB** |              82 MB   |
| 100 × 1024²      |  419 MB   |     ~300 MB   |         **159 MB** |             275 MB   |

The native CLI uses the least memory and is stable run-to-run. The
`fitscube-rs` Python binding pays the interpreter's baseline on top of the
streaming core, so it wins at small/medium cubes and roughly ties `fitscube` at
large ones. Use the CLI when memory matters.

### Larger than memory

Both tools stream, but they write the data unit differently, which matters
once the cube exceeds RAM.
`fitscube` memmaps the output and writes through the mapping, which dirties the
entire output in memory; the peak *footprint* (dirty + anonymous pages) grows to
the full cube size. `fitscube-rs` writes each plane with positional
writes, so written pages are flushed and reclaimed rather than charged to the
process, and its footprint stays bounded by the planes in flight.

Combine of a **26 GB cube on a 24 GB-RAM machine** (420 × 4096², float32),
combine-only:

| Implementation                    |   Time | Peak footprint | Max RSS |
| --------------------------------- | -----: | -------------: | ------: |
| `fitscube` (py)                   |  78 s  |     **23.5 GB**|  1.6 GB |
| `fitscube-rs` (CLI, default)      |  19 s  |       2.6 GB   |  2.6 GB |
| `fitscube-rs` (CLI, max-workers 4)|  16 s  |     **1.2 GB** |    -    |

Once the cube exceeds RAM, `fitscube`'s footprint reaches the memory ceiling
and it thrashes: 4 to 5 times slower here, and it would run out of memory on a
cube well beyond RAM. The footprint of `fitscube-rs` is set by plane size and
the number of reader threads, which `--max-workers` limits. (These figures
predate `--max-workers` setting the thread count; it then only limited the
plane queue.)

## What it does *not* claim

- **Large planes.** When a job is bound by pixel write throughput (large
  images, few channels), both tools are limited by the same I/O and the
  speedup narrows.

## Caveats

- Times are best of 3 with a warm page cache, and vary with hardware, disk and
  thread count. The ordering and the many-small-images trend hold.
- The benchmark uses synthetic Gaussian-noise images with minimal headers.
  Real headers with per-channel beams cost more to parse per image, which
  favours `fitscube-rs` further.
