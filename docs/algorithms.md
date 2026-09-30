# Algorithm background

This page summarises what fitscube-rs does when it combines a set of
single-plane FITS images into a cube and when it extracts a plane back out. It
is a Rust port of the [`fitscube`](https://github.com/AlecThomson/fitscube)
Python package and follows the same conventions, so cubes produced by either
tool are interchangeable.

## Combining images into a cube

The input is a list of FITS images, each holding a single plane: one frequency
channel (the default) or one time step (`--time-domain`). fitscube-rs reads the
2D data from each image and stacks them along a new leading axis, producing a
3D (or 4D, with a degenerate Stokes axis preserved) cube whose plane order
follows the order the files are given.

Before any output is written, every input header is checked against the first:
the same pixel grid (`NAXIS1`/`NAXIS2`), the same axes in the same order (the
`CTYPE` of every axis), and the same Stokes parameter. The cube is labelled from
the first image, so an input that differs is an error rather than a silently
mislabelled or scrambled plane. Each input may carry several planes of its own
(e.g. a Stokes axis) as long as they sit below the combine axis: the
frequency/time axis must be the slowest-varying axis of the cube, since each
channel is written at a fixed offset. The spatial WCS
(`CRVAL1/2`, `CDELT1/2`, projection, …) is taken from the first image and
carried onto the cube unchanged.

## Frequency vs time domain

The new axis can be either spectral or temporal:

- **Frequency** (default): the per-plane coordinate is the spectral value of
  each image, read from its spectral WCS axis (`CTYPE`/`CRVAL`/`CUNIT`) or
  supplied explicitly. The cube gains a `FREQ` axis.
- **Time** (`--time-domain`): the per-plane coordinate is a time stamp and the
  cube gains a time axis instead.

In both cases the per-plane coordinates may be provided out-of-band via a
spec file (`--spec-file`) or an inline list (`--specs`), which overrides what
is read from the headers; `--ignore-spec` skips reading the coordinate
entirely and writes a bare pixel axis.

## Even vs uneven spacing detection

After collecting the per-plane coordinates, fitscube-rs decides how to describe
the new axis in the output WCS:

- **Evenly spaced**: if successive coordinates differ by a constant step (within
  a small tolerance), the axis is written as a linear WCS — a single reference
  value (`CRVALn`) and increment (`CDELTn`). This is the compact, standard form
  most downstream tools expect.
- **Unevenly spaced**: if the step varies, no single `CDELT` can describe the
  axis without losing information. fitscube-rs instead writes the full list of
  per-plane coordinates as an explicit table alongside the cube, so the exact
  frequency (or time) of every plane is recoverable. The user is warned that
  the axis is non-linear.

With `--create-blanks`, fitscube-rs instead builds a regular grid through the
inputs (sorted first, with the step recovered as the greatest common divisor of
the gaps) and fills the grid points without an input with blank (NaN) planes.
If some input would not land on a grid point — an irregular spacing with no
common step, or two inputs at the same frequency — the combine is refused
rather than silently dropping that input.

## Per-channel beams (`BEAMS` table)

Radio images frequently carry a restoring beam (`BMAJ`, `BMIN`, `BPA`) that
differs from plane to plane. fitscube-rs preserves this:

- If all input planes share one beam, the single `BMAJ`/`BMIN`/`BPA` is written
  to the primary header.
- Beams follow the planes: they are sorted with them, and blank channels get a
  NaN beam. Any input with a beam counts, not just the first.
- If the beams differ across planes, fitscube-rs writes a CASA-style `BEAMS`
  binary-table extension — one row per plane with that plane's beam and its
  `CHAN`/`POL`, which are 0-based indices into the cube's frequency (or time)
  and Stokes axes, not FITS Stokes codes — and sets `CASAMBM=T` in the primary header. This is
  the multi-beam convention understood by CASA and `astropy`, so per-channel
  beam information survives the round trip into the cube. Only single-Stokes
  cubes are supported (`POL = 0`, `NPOL = 1`); a multi-Stokes cube with
  varying beams is refused before any data are written.
- A NaN beam is stored as `numpy.finfo(float32).tiny`, the sentinel CASA
  expects, and so is an exactly-zero beam, which is not a valid PSF.

### Zero beams

wsclean writes `BMAJ = BMIN = 0` into an image when no PSF was fitted for that
plane. With `-fit-spectral-pol` the channels that were not imaged directly are
filled with the model image instead, which looks like data but is not
comparable to the rest of the cube. By default such planes are blanked with
NaNs (with a warning naming the channels); `--no-blank-zero-beams`
(`blank_zero_beams=False`) keeps their data.

## Bounding-box trimming

With `--bounding-box`, fitscube-rs trims away the blank border shared by every
plane. It computes, across all input planes, the smallest rectangle that
contains all the valid (non-blank) pixels, then crops every plane to that
common box and updates the spatial reference pixel (`CRPIX1/2`) so the WCS stays
correct. This shrinks cubes that were padded out to a large common canvas
without discarding any real data. `--invalidate-zeros` first treats exact-zero
pixels as blank, so zero-padded borders are trimmed too.

From Python, compute the box once with `get_common_bounding_box` and pass it as
`combine_fits(..., bounding_box=box)` to trim several cubes — e.g. an image
cube and its weights cube — to an identical pixel grid.

## Floating-point precision

`--floating {32,64}` selects the pixel data type of the output cube
(`float32` / BITPIX −32, or `float64` / BITPIX −64). These are the only IEEE
float widths the FITS standard defines, so other values are rejected. The
default follows the inputs; downcasting to `float32` is offered for cubes where
storage matters more than the last bits of precision.

Integer inputs are copied as integers at their own BITPIX, keeping any
`BSCALE`/`BZERO`. Blank planes (from `--create-blanks` or zero beams) and
`--invalidate-zeros` are written as NaNs, which an integer cube cannot hold, so
those need `--floating 32` or `--floating 64`.

## Plane extraction

`extract` is the inverse operation. Given a cube and a plane selector
(`--channel-index` for spectral cubes, `--time-index` for time cubes), it reads
that single plane, rebuilds a 2D image header from the cube WCS (dropping the
combined axis and restoring the per-plane coordinate and beam where available),
and writes a standalone FITS image. `--hdu-index` selects which HDU of the cube
to read from when it is not the primary.
