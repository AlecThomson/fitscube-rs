#!/usr/bin/env python3
"""Time the `fitscube` and `fitscubers` CLIs, and `fpack`, on the same cube.

Writes `nchan` noise images with a blank border, then times combine (with and
without --bounding-box), extract, compress and decompress. Reports the wall
time, CPU time and peak RSS of the median run of each command. Inputs are kept
in WORKDIR for reuse.

Usage:
    python scripts/benchmark_cli.py WORKDIR --nchan 100 --size 1024
"""

from __future__ import annotations

import argparse
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits


def make_images(directory: Path, nchan: int, size: int, int16: bool) -> list[Path]:
    """Noise images with blank top and bottom eighths, for --bounding-box to trim."""
    if not directory.exists():
        directory.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(nchan):
            data = rng.standard_normal((size, size)).astype(np.float32)
            data[: size // 8] = data[-size // 8 :] = np.nan
            header = fits.Header()
            if int16:
                data = np.nan_to_num(data * 1000).astype(np.int16)
                header["BSCALE"], header["BZERO"] = 1e-3, 0.0
            header["CTYPE1"], header["CTYPE2"] = "RA---SIN", "DEC--SIN"
            header["CRPIX1"], header["CRPIX2"] = size / 2, size / 2
            header["CDELT1"], header["CDELT2"] = -1 / 3600, 1 / 3600
            header["REFFREQ"] = 1.0e9 + i * 1.0e6
            fits.writeto(directory / f"chan_{i:04d}.fits", data, header)
    return sorted(directory.glob("*.fits"))


def run(cmd: list[str]) -> tuple[float, float, float]:
    """Wall time (s), CPU time (s) and peak RSS (MB) of a command.

    The peak is sampled from VmHWM, as a child's rusage keeps this process's
    own size across exec.
    """
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    status = Path(f"/proc/{proc.pid}/status")
    peak = 0
    while proc.poll() is None:
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM"):
                    peak = max(peak, int(line.split()[1]))
        except OSError:
            pass
        time.sleep(0.005)
    wall = time.perf_counter() - start
    if proc.returncode:
        sys.exit(f"{' '.join(cmd)} failed: {proc.communicate()[1].decode()}")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    return wall, cpu, peak / 1e3


def run_cli_benchmark(
    workdir: Path,
    nchan: int,
    size: int,
    repeat: int = 3,
    fitscube: str = "fitscube",
    fitscubers: str = "fitscubers",
    int16: bool = False,
) -> list[tuple]:
    """Time each command and return (case, tool, wall_s, cpu_s, peak_rss_mb) rows."""
    kind = "int16" if int16 else "float32"
    files = make_images(workdir / f"in_{nchan}x{size}_{kind}", nchan, size, int16)
    tools = {"fitscube": fitscube, "fitscubers": fitscubers}
    rows = []

    def record(case: str, tool: str, cmd: list[str], setup=None) -> None:
        """The run with the median wall time, of `repeat`."""
        runs = []
        for _ in range(repeat):
            if setup:
                setup()
            runs.append(run(cmd))
        median = sorted(runs)[len(runs) // 2]
        rows.append((case, tool, *median))
        print(",".join([case, tool, *(f"{v:.2f}" for v in median)]), flush=True)

    print("case,tool,wall_s,cpu_s,peak_rss_mb")
    for tool, exe in tools.items():
        cube = workdir / f"{tool}.fits"
        for extra in ([], ["--bounding-box"]) if not int16 else ([],):
            record(
                " ".join(["combine", *extra]),
                tool,
                [exe, "combine", *map(str, files), str(cube), "--overwrite", *extra],
            )
        plane = workdir / f"{tool}.plane.fits"
        record(
            "extract",
            tool,
            [
                exe,
                "extract",
                str(cube),
                "--channel-index",
                str(nchan // 2),
                "--output-path",
                str(plane),
                "--overwrite",
            ],
        )
        if int16:
            continue
        for method in ("tile-gzip", "tile-rice"):
            record(
                f"compress {method}",
                tool,
                [exe, "compress", str(cube), "--method", method, "--max-workers", "4"],
            )
            compressed = cube.with_name(cube.name + ".fz")
            restored = workdir / f"{tool}.restored.fits"
            record(
                f"decompress {method}",
                tool,
                [
                    exe,
                    "decompress",
                    str(compressed),
                    "-o",
                    str(restored),
                    "--overwrite",
                ],
            )
            compressed.unlink()

    if shutil.which("fpack") and not int16:
        cube = workdir / "fitscubers.fits"
        copy = workdir / "fpack.fits"
        compressed = copy.with_name(copy.name + ".fz")

        def fresh_copy() -> None:
            compressed.unlink(missing_ok=True)
            shutil.copy(cube, copy)

        for method, flags in (
            ("tile-gzip", ["-g2", "-q", "0"]),
            ("tile-rice", ["-q", "4"]),
        ):
            record(
                f"compress {method}",
                "fpack",
                ["fpack", *flags, "-D", "-Y", str(copy)],
                fresh_copy,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--nchan", type=int, default=100)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--int16", action="store_true", help="Scaled int16 inputs")
    parser.add_argument("--repeat", type=int, default=3, help="Median of N runs")
    parser.add_argument("--fitscubers", default="fitscubers")
    parser.add_argument("--fitscube", default="fitscube")
    args = parser.parse_args()
    run_cli_benchmark(
        args.workdir,
        args.nchan,
        args.size,
        args.repeat,
        args.fitscube,
        args.fitscubers,
        args.int16,
    )


if __name__ == "__main__":
    main()
