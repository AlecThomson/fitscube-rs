#!/usr/bin/env python3
"""Re-run the benchmarks and rewrite their tables in docs/performance.md.

Each table sits between `<!-- benchmark:NAME -->` and `<!-- /benchmark:NAME -->`
markers; everything outside the markers is left alone. The Benchmark workflow
runs this and opens a PR with the result.

Usage:
    python scripts/update_performance.py --workdir /scratch
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import re
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

from benchmark import run_benchmark
from benchmark_cli import run_cli_benchmark


def machine() -> str:
    """One line naming the hardware and the versions measured."""
    cpu = platform.processor()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    meminfo = Path("/proc/meminfo").read_text().split()
    ram_gb = int(meminfo[meminfo.index("MemTotal:") + 1]) / 1e6
    rs = subprocess.run(
        ["git", "describe", "--tags", "--always"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    fpack = ""
    if shutil.which("fpack"):
        out = subprocess.run(
            ["fpack", "-V"], capture_output=True, text=True, check=False
        ).stdout
        fpack = f", `fpack` {out.split('CFITSIO version')[-1].strip()} (CFITSIO)"
    return (
        f"Measured {datetime.datetime.now(datetime.timezone.utc).date()} on "
        f"{cpu} ({os.cpu_count()} cores), {ram_gb:.0f} GB RAM, "
        f"{platform.system()}. `fitscube-rs` {rs}, `fitscube` {version('fitscube')}, "
        f"`astropy` {version('astropy')}, `numpy` {version('numpy')}{fpack}."
    )


def api_table(cases: list[tuple[int, int]], repeat: int) -> str:
    """The Python API speedup table, best of `repeat` runs."""
    lines = [
        "| Channels × image | Cube size | `fitscube` (py) | `fitscube-rs` | Speedup |",
        "| ---------------- | --------: | --------------: | ------------: | ------: |",
    ]
    for nchan, size in cases:
        t = run_benchmark(nchan, size, repeat)
        mb = nchan * size * size * 4 / 1e6
        speedup = t["python"] / t["rust"]
        lines.append(
            f"| {nchan} × {size}² | {mb:.0f} MB | {t['python']:.3f} s "
            f"| {t['rust']:.3f} s | {speedup:.{1 if speedup < 10 else 0}f}× |"
        )
    return "\n".join(lines)


def cli_table(rows: list[tuple]) -> str:
    """Wall time and peak RSS per command, one column per tool."""
    tools = ["fitscube", "fitscubers", "fpack"]
    cells: dict[str, dict[str, str]] = {}
    for case, tool, wall, _cpu, rss in rows:
        name, _, method = case.partition(" ")
        label = f"`{name}` ({method})" if name == "decompress" else f"`{case}`"
        # A run too short to sample has no peak RSS.
        memory = f"{rss:.0f} MB" if rss >= 1 else "—"
        cell = f"{wall:.2f} s" if tool == "fpack" else f"{wall:.2f} s, {memory}"
        cells.setdefault(label, {})[tool] = cell
    lines = [
        "| Command | `fitscube` | `fitscubers` | `fpack` |",
        "| ------- | ---------: | -----------: | ------: |",
    ]
    for label, by_tool in cells.items():
        lines.append(
            f"| {label} | " + " | ".join(by_tool.get(t, "") for t in tools) + " |"
        )
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    """Swap the text between the `name` markers for `body`."""
    pattern = re.compile(
        rf"(<!-- benchmark:{name} -->\n).*?(<!-- /benchmark:{name} -->)", re.DOTALL
    )
    if not pattern.search(text):
        msg = f"No benchmark:{name} markers in the doc"
        raise ValueError(msg)
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fitscubers", default="fitscubers")
    parser.add_argument("--doc", type=Path, default=Path("docs/performance.md"))
    parser.add_argument("--summary", type=Path, help="Also append the tables here")
    args = parser.parse_args()

    api_cases = [
        (50, 128),
        (100, 256),
        (100, 512),
        (200, 256),
        (500, 256),
        (1000, 128),
        (100, 1024),
    ]
    cli_cases = [(100, 1024), (1000, 128)]

    # Fail on a missing marker before the long runs, not after.
    text = args.doc.read_text()
    for name in ["machine", "api", *(f"cli-{n}x{s}" for n, s in cli_cases)]:
        replace_block(text, name, "")

    blocks = {
        "machine": machine(),
        "api": api_table(api_cases, args.repeat),
    }
    for nchan, size in cli_cases:
        rows = run_cli_benchmark(
            args.workdir / f"cli_{nchan}x{size}",
            nchan,
            size,
            args.repeat,
            fitscubers=args.fitscubers,
        )
        blocks[f"cli-{nchan}x{size}"] = cli_table(rows)

    for name, body in blocks.items():
        text = replace_block(text, name, body)
    args.doc.write_text(text)

    if args.summary:
        with args.summary.open("a") as f:
            f.write("\n\n".join(blocks.values()) + "\n")


if __name__ == "__main__":
    main()
