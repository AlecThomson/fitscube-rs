# fitscube-rs

[![Docs](https://readthedocs.org/projects/fitscube-rs/badge/?version=latest)](https://fitscube-rs.readthedocs.io/en/latest/)
[![docs.rs](https://img.shields.io/docsrs/fitscube-rs)](https://docs.rs/fitscube-rs)
[![PyPI](https://img.shields.io/pypi/v/fitscube-rs)](https://pypi.org/project/fitscube-rs/)
[![crates.io](https://img.shields.io/crates/v/fitscube-rs)](https://crates.io/crates/fitscube-rs)

Documentation for the Python API, CLI and algorithms is at
[fitscube-rs.readthedocs.io](https://fitscube-rs.readthedocs.io/). The Rust API
is at [docs.rs/fitscube-rs](https://docs.rs/fitscube-rs).

<!-- SPHINX-START -->

A Rust port of the [`fitscube`](https://github.com/AlecThomson/fitscube) Python
package. It stacks single-frequency or single-time FITS images into a cube and
extracts planes from a cube. The frequency or time axis comes from the image
headers. Per-channel beams go into a CASA `BEAMS` table.

> Note: this is an experiment in LLM-assisted coding with Claude. Do not trust this software as far as you can throw it.

## Installation

### Python library

```sh
pip install fitscube-rs
```

### CLI binary

Requires [Rust](https://rustup.rs/) 1.88+.

```sh
cargo install fitscube-rs
```

## Usage

- Python API and quickstart:
  [fitscube-rs.readthedocs.io/en/latest/quickstart.html](https://fitscube-rs.readthedocs.io/en/latest/quickstart.html)
- CLI reference:
  [fitscube-rs.readthedocs.io/en/latest/cli.html](https://fitscube-rs.readthedocs.io/en/latest/cli.html)
- Rust API:
  [docs.rs/fitscube-rs](https://docs.rs/fitscube-rs)

## Development

Install in editable mode:

```sh
uv pip install -e .
```

After changing the Python-facing Rust API in `src/python.rs`, rebuild with the
`stubgen` feature (the default build omits `_generate_stubs`) and regenerate
the type stubs:

```sh
uv run maturin develop --features stubgen
uv run --no-sync python -c "from fitscube_rs._fitscube_rs import _generate_stubs; _generate_stubs()"
```

This rewrites `fitscube_rs/_fitscube_rs/__init__.pyi` from the Rust signatures
and docstrings. Commit it with the API change.

### Running tests

Python tests need the compiled extension and the `test` extra, which
includes [`fitscube`](https://github.com/AlecThomson/fitscube) for the parity
tests. `uv sync` builds the extension into `.venv`. `uvx pytest` does not work,
because it runs without the extension.

```sh
uv sync --extra test --reinstall-package fitscube-rs   # rebuild after Rust changes
uv run --no-sync pytest
```

Rust tests run with `cargo test`:

```sh
cargo test
```

### Pre-commit hooks

[prek](https://github.com/j178/prek) runs the hooks in
`.pre-commit-config.yaml`, the same checks as CI: `ruff`, [`ty`](https://github.com/astral-sh/ty),
`cargo fmt` and `cargo clippy`.

```sh
uv sync --extra dev   # installs prek + ty into the venv
uvx prek install      # install the git hook (runs on every commit)
uvx prek run --all-files   # run all hooks manually
```

## License

fitscube-rs is released under the
[BSD 3-Clause License](https://github.com/alecthomson/fitscube-rs/blob/main/LICENSE).

It is a port of the [`fitscube`](https://github.com/AlecThomson/fitscube) Python
package by Alec Thomson (BSD 3-Clause). The tests compare its output with that
package. See
[NOTICE.md](https://github.com/alecthomson/fitscube-rs/blob/main/NOTICE.md)
for full attributions.
