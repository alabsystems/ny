# Contributing

## Build

```bash
cargo build --release -p ny-cli    # → target/release/ny
```

**Do not drop `--features mip` when building anything you intend to measure.**
Without it, `try_sign_space_falsify` and `try_ste_pgd_falsify` cfg-compile to
`None` and both BNN falsification lanes silently vanish — a row that normally
takes 335 s "reproduces" in 2.9 s because a later lane now runs first. Guard a
measurement by asserting `mip_compiled=true` in a smoke run's flight sidecar.

**"ny builds under Trust" is not "ny's build is verified."** The build line uses
`targo --unverified build`, and Trust has an open false-Safe channel
(`transmute_provenance_repro.rs::provenance_roundtrip_CONTROL_failclosed`
verifies in a mode its own header says must fail closed). Its 1122-test corpus is
currently clean on missed bugs, but the verified-build claim is not one to make.

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
