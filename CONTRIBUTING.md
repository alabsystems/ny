# Contributing

## Build

```bash
targo --unverified build --locked --release -p ny-cli    # → target/release/ny
```

Every build names a Trust lane. `targo --unverified` builds and tests without
a compiler proof. `targo trust check --allow-l0-gaps` runs advisory
verification; unknown obligations are conditional passes, not discharged proofs.
A bare `targo build` is refused. The current release gate runs an unverified
workspace build and the 19-test relaxation-envelope floor; it makes no compiler
proof claim.

### If targo is not on PATH

Where Trust is only a linked rustup toolchain, prefix the command:

```bash
rustup run trust targo --unverified build --locked --release -p ny-cli --features mip
```

Install Trust through [aterm](https://github.com/alabsystems/aterm#install)
with `aterm pkg install trust`, then run `aterm help rust` in the checkout.
The `trust` toolchain is a local rustup link; its compiler version depends on
the host. If `targo` rejects the pinned AY manifests with `unknown field
backstop`, update the managed Trust installation.

The competition builder, `scripts/vnncomp_trust_linux_build.sh`, requires
Ubuntu 24.04 x86_64 and enables `--features mip,cuda`.
Ordinary Linux builds also need a compatible ONNX Runtime: the prebuilt library
for GNU/Linux x86_64 requires glibc 2.39 or newer, as shipped by Ubuntu 24.04.
On older systems, use a 24.04 container or set `ORT_LIB_LOCATION` to a locally
built runtime. `install_tool.sh` can fall back to a source build, which still
requires Trust.

**Build measured binaries with `--features mip`.** Without it, both BNN
falsification lanes are disabled. Confirm `mip_compiled=true` in the
`bnn_falsification_policy` event of a smoke run's flight sidecar
(`<results>.flight.json`).

## Test

```bash
targo --unverified test --locked --workspace --exclude ny-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make PYTHON="$VIRTUAL_ENV/bin/python" test-python-tooling

# Optional Python bindings lane:
python -m pip install maturin
make PYTHON="$VIRTUAL_ENV/bin/python" test-python
```

Workspace tests include GPU controls enabled by `ny-cli`'s dev dependencies and
need a suitable adapter. See `TEST_CONFORMANCE.md` for
individual device, corpus and checker lanes.

Tests may not use ignore or skip directives (`#[ignore]`, `pytest.skip`, and
similar); a source-policy test enforces this. Optional-corpus controls that
return early without assets provide no execution evidence.

## Lint

```bash
make lint
```

## Guidelines

- **Soundness comes first.** A wrong verdict is far worse than no verdict. Any
  change that can affect one (bound propagation, relaxations, certificate
  emission, verdict admission) needs tests that demonstrate the behavior.
- New source files carry the same copyright and `SPDX-License-Identifier:
  Apache-2.0` header as the files around them. Contributions are licensed
  under the Apache License 2.0.
- Report bugs and request features through GitHub issues. Report security
  problems privately; see [`SECURITY.md`](SECURITY.md).

## Provenance note

Some comments cite `#NNNN` issue numbers and `docs/` or `designs/` paths from
the private tracker `ny` was developed with. They are kept as provenance and
do not resolve in the public repository.
