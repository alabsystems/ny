# Contributing

## Build

```bash
cargo build --release -p ny-cli    # → target/release/ny
```

## Test

```bash
cargo test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make PYTHON="$VIRTUAL_ENV/bin/python" test-python-tooling

# Optional Python bindings lane:
python -m pip install maturin
make PYTHON="$VIRTUAL_ENV/bin/python" test-python
```

## Lint

```bash
make lint
```

Changes that can affect soundness (bound propagation, relaxations, certificate
emission, verdict admission) need tests demonstrating the behavior.

Report bugs and request features via GitHub issues.

## Provenance note

`#NNNN` issue numbers and `designs/*.md` paths in comments refer to the private
tracker this project was developed with; they are retained as provenance and
are not resolvable here.
