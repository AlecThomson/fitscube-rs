# Algorithm background

What fitscube-rs does when it combines single-plane FITS images into a cube,
and when it extracts planes back out. It follows the conventions of
[`fitscube`](https://github.com/AlecThomson/fitscube), so cubes from either
tool are interchangeable.

## Combining images into a cube

The input is a list of FITS images, each holding one frequency channel (the
default) or one time step (`--time-domain`). The planes are sorted by
frequency or time and stacked along that axis. A 2D input gains a third axis;
an input that already has a FREQ (or TIME) axis is stacked along it.
`--ignore-spec` keeps the input order instead.

Every input header is checked against the first before anything is written:
the same pixel grid (`NAXIS1`/`NAXIS2`), the same axes in the same order (the
`CTYPE` of every axis) and the same Stokes parameter. The cube header is copied
from the first image, so a differing input is an error rather than a
mislabelled plane. An input may hold several planes of its own (a Stokes axis,
say) as long as they sit below the combine axis: each channel is written at a
fixed offset, so the frequency or time axis must vary slowest.

The cube is written to a hidden file next to the output and renamed into
place at the end, so a failed combine leaves no partial cube.

## Frequency and time

- **Frequency** (default): a 2D image gives its frequency in `REFFREQ`. An
  image with a `FREQ` axis gives the frequency of its first pixel along that
  axis, from `CRVAL`, `CRPIX` and `CDELT` (or `CDi_i`, or `CDELTi * PCi_i`),
  with the FITS defaults for missing cards. `CUNIT` may be Hz, kHz, MHz or GHz;
  other units are an error. The cube gets a `FREQ` axis in Hz.
- **Time** (`--time-domain`): each image's `DATE-OBS`, as MJD seconds. The cube
  gets a `TIME` axis in seconds.

`--spec-file` (one value per line) or `--specs` supply the values instead of
the headers. NaN or infinite values are an error.

## Even and uneven spacing

- **Evenly spaced** values (standard deviation of the steps below 1e-4, or for
  time a running drift under 2% of the mean step) become a linear axis:
  `CRVAL` is the first value and `CDELT` the median step.
- **Uneven** values cannot be described by one `CDELT`. The axis is written as
  `CTYPE = CHAN`, `CRVAL = 1`, `CDELT = 1` with a warning, so the header no
  longer records the frequencies or times. `combine_fits` returns them, and the
  CLI writes them to `OUT_CUBE.specs_Hz.txt` (or `.specs_s.txt`).

With `--create-blanks`, fitscube-rs builds a regular grid through the sorted
values instead, with the step taken as the greatest common divisor of the gaps,
and fills the grid points that have no input with NaN planes. The combine is
refused if an input would fall between grid points, if two inputs share one,
or if the grid would need more than 1000 channels per input.

## Per-channel beams (`BEAMS` table)

Radio images often carry a restoring beam (`BMAJ`, `BMIN`, `BPA`) that changes
from plane to plane.

- If every input has the same beam, it stays in the primary header.
- Otherwise fitscube-rs appends a CASA `BEAMS` binary table with one row per
  channel: `BMAJ` and `BMIN` in arcsec, `BPA` in degrees, and `CHAN`/`POL`,
  0-based indices into the cube's frequency and Stokes axes. It sets
  `CASAMBM = T` and removes the single-beam cards. Only single-Stokes cubes are
  supported; a multi-Stokes cube with varying beams is refused before any data
  are written.
- Beams follow the planes when they are sorted, and blank channels get a NaN
  beam. Any input with a beam counts, not just the first.
- CASA cannot read NaN, so a NaN beam is stored as `numpy.finfo(float32).tiny`,
  as is a beam with a zero axis.

### Zero beams

wsclean writes `BMAJ = BMIN = 0` for a plane with no fitted PSF. With
`-fit-spectral-pol`, channels that were not imaged hold the model image, which
looks like data but is not comparable to the rest of the cube. By default these
planes are blanked with NaN, with a warning naming the channels;
`--no-blank-zero-beams` (`blank_zero_beams=False`) keeps them.

## Bounding-box trimming

`--bounding-box` finds the smallest rectangle holding every finite pixel of
every input, crops all planes to it, and shifts `CRPIX1/2` to match.
`--invalidate-zeros` treats exact zeros as blank, so zero-padded borders are
trimmed too.

From Python, compute the box once with `get_common_bounding_box` and pass it as
`combine_fits(..., bounding_box=box)` to put several cubes (an image cube and
its weights, say) on the same pixel grid.

## Precision

`--floating {32,64}` sets the output to float32 (BITPIX -32) or float64
(BITPIX -64), the only float widths FITS defines. By default the cube keeps
the input BITPIX.

An integer cube copies the stored integers, so every input must share BITPIX,
`BSCALE`, `BZERO` and `BLANK`, which the cube keeps. Blank channels and
`--invalidate-zeros` produce NaN, which an integer cube cannot hold, so they
need `--floating`. A float cube made from integer inputs decodes them, with
`BLANK` pixels as NaN.

## Plane extraction

`extract` reads one plane of a cube (`--channel-index` or `--time-index`) and
writes it as an image with the same axes, the extracted axis now of length 1.
That axis gets `CRPIX = 1` and the plane's own `CRVAL`. Stored values are
copied as they are, so integer cubes extract exactly. If the cube has a
`BEAMS` table, the plane's beam goes into `BMAJ`/`BMIN`/`BPA` and `CASAMBM` is
removed. `--hdu-index` picks the HDU, which may be an image extension.

`extract_planes_from_cube` (Python and Rust) writes every channel to its own
file, reading the beam table once.

fitscube computes the extracted `CRVAL` as `CRVAL + index * CDELT`, ignoring
`CRPIX`; fitscube-rs uses `CRPIX`, so the two differ for cubes whose `CRPIX`
on that axis is not 1.
