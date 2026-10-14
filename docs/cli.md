# CLI usage

`fitscubers` combines single-frequency or single-time FITS images into a cube
and extracts planes back out. Install it with:

```sh
cargo install fitscube-rs
```

```{note}
The help text on this page comes from running the binary when the docs are
built.
```

```{command-output} cargo run -q --bin fitscubers -- --help
:cwd: ..
```

## Common workflows

Combine a set of single-channel images into a cube:

```sh
fitscubers combine chan_*.fits cube.fits
```

Overwrite an existing output and trim the all-blank border of every plane to a
common bounding box:

```sh
fitscubers combine --overwrite --bounding-box chan_*.fits cube.fits
```

Build a cube along the time axis instead of frequency:

```sh
fitscubers combine --time-domain epoch_*.fits cube.fits
```

Extract one channel back out of a cube:

```sh
fitscubers extract cube.fits --channel-index 0 --output-path chan0.fits
```

## `fitscubers combine`

```{command-output} cargo run -q --bin fitscubers -- combine --help
:cwd: ..
```

Frequencies come from `--spec-file` or `--specs` if given, otherwise from each
image: `REFFREQ` for a 2D image, or its `FREQ` axis. Times come from `DATE-OBS`.

Evenly spaced values become a linear axis. Uneven values give a plain channel
axis (`CTYPE = CHAN`) with a warning; the values are still written to
`OUT_CUBE.specs_Hz.txt` (`.specs_s.txt` for time), which the CLI writes next to
every cube. Varying beams go into a CASA `BEAMS` table (`CASAMBM = T`). See
[Algorithm background](algorithms.md).

## `fitscubers extract`

```{command-output} cargo run -q --bin fitscubers -- extract --help
:cwd: ..
```

`extract` picks a plane by `--channel-index` (frequency cubes) or
`--time-index` (time cubes) and writes it with the cube's axes, the extracted
axis now of length 1. `--hdu-index` reads a cube stored in an image
extension.
