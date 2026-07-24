# Contributing

## Build

```bash
cargo build --release -p ny-cli    # → target/release/ny
```

## Test

```bash
cargo test
make test-python    # Python bindings
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
