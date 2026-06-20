#!/usr/bin/env python3
"""Benchmark fitscube_rs against the original Python `fitscube`.

Generates a set of single-channel FITS images, then times `combine_fits` for
both implementations on the same inputs and reports the wall-clock speedup. The
output cubes are compared so a speedup is only reported when the results agree.

Usage:
    python scripts/benchmark.py --nchan 100 --size 256 --repeat 3
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import fitscube  # the reference implementation
import numpy as np
from astropy.io import fits

import fitscube_rs


def make_images(directory: Path, nchan: int, size: int) -> list[Path]:
    """Write `nchan` single-channel FITS images of shape (size, size)."""
    rng = np.random.default_rng(0)
    files = []
    for i in range(nchan):
        data = rng.standard_normal((size, size)).astype(np.float32)
        header = fits.Header()
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        header["CRPIX1"] = 1.0
        header["CRPIX2"] = 1.0
        header["CRVAL1"] = 0.0
        header["CRVAL2"] = 0.0
        header["CDELT1"] = -1.0 / 3600.0
        header["CDELT2"] = 1.0 / 3600.0
        header["REFFREQ"] = 1.0e9 + i * 1.0e6
        path = directory / f"chan_{i:04d}.fits"
        fits.writeto(path, data, header, overwrite=True)
        files.append(path)
    return files


def _time(fn, repeat: int) -> float:
    """Best-of-`repeat` wall-clock time in seconds."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run_benchmark(nchan: int, size: int, repeat: int) -> dict[str, float]:
    """Time both implementations and return their best wall-clock times."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        files = make_images(d, nchan, size)
        py_cube = d / "py_cube.fits"
        rs_cube = d / "rs_cube.fits"

        py_time = _time(
            lambda: fitscube.combine_fits(
                file_list=files, out_cube=py_cube, overwrite=True
            ),
            repeat,
        )
        rs_time = _time(
            lambda: fitscube_rs.combine_fits(
                [str(f) for f in files], str(rs_cube), overwrite=True
            ),
            repeat,
        )

        # Only a meaningful comparison if the cubes agree.
        with fits.open(py_cube) as a, fits.open(rs_cube) as b:
            np.testing.assert_allclose(
                np.nan_to_num(a[0].data), np.nan_to_num(b[0].data), rtol=1e-6, atol=1e-6
            )

        return {"python": py_time, "rust": rs_time}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nchan", type=int, default=100, help="Number of channels")
    parser.add_argument(
        "--size", type=int, default=256, help="Image edge length (pixels)"
    )
    parser.add_argument("--repeat", type=int, default=3, help="Best-of-N timing runs")
    args = parser.parse_args()

    cube_mb = args.nchan * args.size * args.size * 4 / 1e6
    print(
        f"Combining {args.nchan} x {args.size}x{args.size} images (~{cube_mb:.0f} MB cube)"
    )
    print(f"best of {args.repeat} run(s)\n")

    times = run_benchmark(args.nchan, args.size, args.repeat)
    speedup = times["python"] / times["rust"]

    print(f"{'implementation':<16}{'time [s]':>12}")
    print(f"{'-' * 28}")
    print(f"{'fitscube (py)':<16}{times['python']:>12.3f}")
    print(f"{'fitscube_rs':<16}{times['rust']:>12.3f}")
    print(f"\nfitscube_rs is {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
