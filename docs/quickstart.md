---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Python quickstart

The cells on this page run on every docs build.

## Building a set of single-channel images

First, some synthetic single-channel images, each with its own frequency and
restoring beam in the header.

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
    header["BMAJ"] = header["BMIN"] = (10 + i) / 3600  # degrees
    header["BPA"] = 0.0
    path = workdir / f"chan_{i:04d}.fits"
    fits.writeto(path, data[np.newaxis, :, :], header, overwrite=True)
    paths.append(path)

sorted(p.name for p in paths)
```

## Combining images into a cube

{func}`~fitscube_rs.combine_fits` stacks the images along their frequency axis
and returns the frequencies:

```{code-cell} ipython3
from fitscube_rs import combine_fits

out_cube = workdir / "cube.fits"
freqs = combine_fits(
    [str(p) for p in sorted(paths)],
    str(out_cube),
    overwrite=True,
)

with fits.open(out_cube) as hdul:
    print("cube shape:", hdul[0].data.shape)
    print("frequencies (Hz):", freqs)
```

The frequencies are evenly spaced, so the header describes them with `CRVAL3`
and `CDELT3`. Unevenly spaced frequencies give a plain channel axis instead
(see [Algorithm background](algorithms.md)), and the returned list is then the
only record of them.

## Extracting a plane

{func}`~fitscube_rs.extract_plane_from_cube` pulls one plane back out, with
its own frequency and beam in the header:

```{code-cell} ipython3
from fitscube_rs import extract_plane_from_cube

plane_path = workdir / "chan2.fits"
extract_plane_from_cube(
    str(out_cube), channel_index=2, output_path=str(plane_path), overwrite=True
)

with fits.open(plane_path) as hdul:
    print("plane shape:", hdul[0].data.shape)
    print("plane value:", float(np.nanmean(hdul[0].data)))
    print("frequency (Hz):", hdul[0].header["CRVAL3"])
    print("BMAJ (arcsec):", hdul[0].header["BMAJ"] * 3600)
```

## Per-channel beams

The beams differ from channel to channel, so the cube keeps them in a CASA
`BEAMS` table and sets `CASAMBM = T`:

```{code-cell} ipython3
from astropy.table import Table

with fits.open(out_cube) as hdul:
    print([h.name for h in hdul])
    print("CASAMBM:", hdul[0].header["CASAMBM"])
    print(Table.read(hdul["BEAMS"]))
```

## Working with the CLI

The same from the shell, with the [`fitscubers` CLI](cli.md):

```sh
fitscubers combine chan_*.fits cube.fits --overwrite
fitscubers extract cube.fits --channel-index 2 --output-path chan2.fits
fitscubers compress cube.fits --method tile-rice
```
