---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Python quickstart

This page is an executable notebook: every cell below is re-run on each docs
build, so the outputs are guaranteed to match the current release.

## Building a set of single-channel images

fitscube-rs combines many single-frequency (or single-time) FITS images into a
single FITS cube. To demonstrate, we first write out a handful of synthetic
single-channel images, each with its own frequency in the header.

```{code-cell} ipython3
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

workdir = Path(tempfile.mkdtemp())

freqs_hz = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9])  # evenly spaced in frequency
n = 64

paths = []
for i, freq in enumerate(freqs_hz):
    data = np.full((n, n), float(i), dtype=np.float32)
    header = fits.Header()
    header["CRVAL3"] = freq
    header["CTYPE3"] = "FREQ"
    header["CUNIT3"] = "Hz"
    path = workdir / f"chan_{i:04d}.fits"
    fits.writeto(path, data[np.newaxis, :, :], header, overwrite=True)
    paths.append(path)

sorted(p.name for p in paths)
```

## Combining images into a cube

{func}`~fitscube_rs.combine` stacks the per-channel images along a new spectral
(or time) axis, building the WCS for that axis from the per-image headers. The
resulting cube is written to disk:

```{code-cell} ipython3
from fitscube_rs import combine

out_cube = workdir / "cube.fits"
freqs = combine(
    [str(p) for p in sorted(paths)],
    str(out_cube),
    overwrite=True,
)

with fits.open(out_cube) as hdul:
    print("cube shape:", hdul[0].data.shape)
    print("frequencies (Hz):", freqs)
```

The frequency axis is detected as evenly spaced, so the cube header carries a
linear `CRVAL3`/`CDELT3` description. When the spacing is uneven, fitscube-rs
instead writes an explicit per-plane frequency table so no information is lost
(see [Algorithm background](algorithms.md)).

## Extracting a plane

{func}`~fitscube_rs.extract` pulls a single plane back out of a cube — the
inverse of combining — which is handy for inspecting one channel or feeding a
downstream tool that expects a 2D image:

```{code-cell} ipython3
from fitscube_rs import extract

plane_path = workdir / "chan2.fits"
extract(str(out_cube), channel_index=2, output_path=str(plane_path), overwrite=True)

with fits.open(plane_path) as hdul:
    print("plane shape:", hdul[0].data.shape)
    print("plane value:", float(np.nanmean(hdul[0].data)))
```

## Per-channel beams

Radio images often carry a restoring beam (`BMAJ`/`BMIN`/`BPA`) that varies
from channel to channel. When the input images have differing beams, the
combined cube records them in a CASA-style `BEAMS` binary-table extension and
sets `CASAMBM=T` in the primary header, matching the convention used by
`astropy` and CASA:

```{code-cell} ipython3
with fits.open(out_cube) as hdul:
    has_beams = any(h.name == "BEAMS" for h in hdul)
    print("primary HDUs:", [h.name or "PRIMARY" for h in hdul])
    print("multi-beam table present:", has_beams)
```

## Working with the CLI

For batch jobs (hundreds of channels, large mosaics) prefer the
[`fitscube` CLI](cli.md) — it parallelises the read/stack across workers and
streams planes to disk so peak memory stays bounded:

```sh
fitscube combine chan_*.fits cube.fits --overwrite
fitscube extract cube.fits --channel-index 2 --output-path chan2.fits
```
