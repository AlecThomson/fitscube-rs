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

Every input is checked for a consistent shape and pixel grid; a plane that does
not match the others is an error rather than a silent reshape. The spatial WCS
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

## Per-channel beams (`BEAMS` table)

Radio images frequently carry a restoring beam (`BMAJ`, `BMIN`, `BPA`) that
differs from plane to plane. fitscube-rs preserves this:

- If all input planes share one beam, the single `BMAJ`/`BMIN`/`BPA` is written
  to the primary header.
- If the beams differ across planes, fitscube-rs writes a CASA-style `BEAMS`
  binary-table extension — one row per plane with that plane's beam (and its
  channel/Stokes index) — and sets `CASAMBM=T` in the primary header. This is
  the multi-beam convention understood by CASA and `astropy`, so per-channel
  beam information survives the round trip into the cube.

## Bounding-box trimming

With `--bounding-box`, fitscube-rs trims away the blank border shared by every
plane. It computes, across all input planes, the smallest rectangle that
contains all the valid (non-blank) pixels, then crops every plane to that
common box and updates the spatial reference pixel (`CRPIX1/2`) so the WCS stays
correct. This shrinks cubes that were padded out to a large common canvas
without discarding any real data. `--invalidate-zeros` first treats exact-zero
pixels as blank, so zero-padded borders are trimmed too.

## Floating-point precision

`--floating {16,32,64}` selects the pixel data type of the output cube
(`float16`, `float32`, or `float64`). The default follows the inputs;
downcasting is offered for cubes where storage matters more than the last bits
of precision.

## Plane extraction

`extract` is the inverse operation. Given a cube and a plane selector
(`--channel-index` for spectral cubes, `--time-index` for time cubes), it reads
that single plane, rebuilds a 2D image header from the cube WCS (dropping the
combined axis and restoring the per-plane coordinate and beam where available),
and writes a standalone FITS image. `--hdu-index` selects which HDU of the cube
to read from when it is not the primary.
