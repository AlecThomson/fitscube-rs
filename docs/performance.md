# Performance

`fitscube-rs` writes the same cubes as
[`fitscube`](https://github.com/AlecThomson/fitscube). The parity tests
compare data, axis cards and beam tables.

## Benchmark

`scripts/benchmark.py` writes `nchan` single-channel `float32` FITS images,
times `combine_fits` for both implementations on the same inputs, and checks
that the two cubes agree.

```sh
python scripts/benchmark.py --nchan 100 --size 256 --repeat 3
```

Best of 3 runs. Input generation is not timed. Intel Xeon at 2.10 GHz
(4 vCPUs, KVM), 16 GB RAM, Linux, ext4 disk. `fitscube` 2.8.0, `astropy`
8.0.0, `numpy` 2.4.6.

| Channels × image | Cube size | `fitscube` (py) | `fitscube-rs` | Speedup  |
| ---------------- | --------: | --------------: | ------------: | -------: |
| 50 × 128²        |      3 MB |         0.466 s |       0.004 s |     108× |
| 100 × 256²       |     26 MB |         0.906 s |       0.014 s |      66× |
| 100 × 512²       |    105 MB |         0.922 s |       0.050 s |      18× |
| 200 × 256²       |     52 MB |         1.713 s |       0.026 s |      66× |
| 500 × 256²       |    131 MB |         4.251 s |       0.065 s |      65× |
| 1000 × 128²      |     66 MB |         8.535 s |       0.064 s |     133× |
| 100 × 1024²      |    419 MB |         0.975 s |        0.13 s |     7.6× |

The 100 × 1024² row is the median of 5 runs: at this size an occasional run
stalls on disk write-back.

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
- `extract` reads one plane, not the whole cube.

## Against the previous release

200 × 1024² `float32` images (800 MB cube), 4 cores, median of 4 runs. The CLI
on `main` before #9 against #9 (single header pass and plane reads). Combine
ran on tmpfs; extract on disk with a warm cache.

| Command                       | Before | After  |
| ----------------------------- | -----: | -----: |
| `combine`                     | 0.73 s | 0.44 s |
| `combine --bounding-box`      | 1.40 s | 0.80 s |
| `extract --channel-index 100` | 2.6 s  | 0.02 s |

## Memory

`fitscube-rs` streams: at most about three planes per reader thread are in
memory, never the whole cube.

Peak RSS of the combine alone, from `/usr/bin/time -v`, median of 3 runs, on
the hardware above. Inputs were generated in a separate process.

| Channels × image | Cube size | `fitscube` (py) | `fitscube-rs` (CLI) | `fitscube-rs` (Python) |
| ---------------- | --------: | --------------: | ------------------: | ---------------------: |
| 200 × 512²       |    210 MB |          131 MB |               22 MB |                  29 MB |
| 100 × 1024²      |    419 MB |          241 MB |               63 MB |                  70 MB |

`fitscube` memmaps the output cube; `fitscube-rs` writes each plane with
positional writes. On a 26 GB cube (420 × 4096²) on a 24 GB machine, measured
before `fitscube-rs` 2.0 and not re-measured here, `fitscube` took 78 s with a
23.5 GB peak footprint and `fitscube-rs` took 19 s with 2.6 GB. At that
version `--max-workers` only limited the plane queue.
