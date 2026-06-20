# fitscube-rs

[![Docs](https://readthedocs.org/projects/fitscube-rs/badge/?version=latest)](https://fitscube-rs.readthedocs.io/en/latest/)
[![docs.rs](https://img.shields.io/docsrs/fitscube-rs)](https://docs.rs/fitscube-rs)
[![PyPI](https://img.shields.io/pypi/v/fitscube-rs)](https://pypi.org/project/fitscube-rs/)
[![crates.io](https://img.shields.io/crates/v/fitscube-rs)](https://crates.io/crates/fitscube-rs)

**Documentation:** Python API, CLI, and algorithm background at
[fitscube-rs.readthedocs.io](https://fitscube-rs.readthedocs.io/); Rust API at
[docs.rs/fitscube-rs](https://docs.rs/fitscube-rs).

<!-- SPHINX-START -->

A Rust port of the [`fitscube`](https://github.com/AlecThomson/fitscube) Python package. Combines single-frequency/single-time FITS images into a FITS cube — building the spectral (or time) axis from the per-image headers, detecting even vs uneven spacing, preserving per-channel beams in a CASA `BEAMS` table — and extracts individual planes back out.

> **Note:** This is an experiment in LLM-assisted coding with Claude. Do not trust this software as far as you can throw it.

## Installation

### Python library

```sh
pip install fitscube-rs
```

### CLI binary

Requires [Rust](https://rustup.rs/) 1.85+.

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

This overwrites `fitscube_rs/_fitscube_rs.pyi` from the Rust annotations and docstrings. Commit the result alongside any API changes.

### Running tests

**Python** tests need the compiled extension and the `test` extra (`pytest`,
`astropy`), plus the upstream [`fitscube`](https://github.com/AlecThomson/fitscube)
package, which the parity tests validate against. `uv sync` builds the maturin
extension into `.venv`, so a plain `uvx pytest` won't work — it runs in an
isolated env with neither the module nor the deps:

```sh
uv sync --extra test       # builds the extension + installs test deps
uv pip install fitscube    # upstream package, for parity tests
uv run --no-sync pytest
```

**Rust** tests run with `cargo test`:

```sh
cargo test
```

### Pre-commit hooks

Formatters and linters run via [prek](https://github.com/j178/prek), a fast
drop-in [pre-commit](https://pre-commit.com) reimplementation. The hooks
(`.pre-commit-config.yaml`) are the same checks CI enforces: `ruff` lint +
format, [`ty`](https://github.com/astral-sh/ty) type checking, `cargo fmt`, and
`cargo clippy`.

```sh
uv sync --extra dev   # installs prek + ty into the venv
uvx prek install      # install the git hook (runs on every commit)
uvx prek run --all-files   # run all hooks manually
```

## License

fitscube-rs is released under the
[BSD 3-Clause License](https://github.com/alecthomson/fitscube-rs/blob/main/LICENSE).

It is a port of the [`fitscube`](https://github.com/AlecThomson/fitscube) Python
package by Alec Thomson (BSD 3-Clause), and validates its output against that
package in the test suite. See
[NOTICE.md](https://github.com/alecthomson/fitscube-rs/blob/main/NOTICE.md)
for full attributions.
