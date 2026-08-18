# ny-python

Python bindings for `ny`, built with `pyo3` and `maturin`.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip maturin
cd crates/ny-python
maturin develop --release
python -m pytest tests
```

The Python module name is `ny`; the type stub is `ny.pyi`.
