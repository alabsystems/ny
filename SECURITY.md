# Security Policy

`ny` is pre-1.0 research software. Public issues are fine for ordinary bugs, but
anything involving leaked credentials or a vulnerability that needs private
coordination should be reported privately first.

## How to report

Email the maintainer directly rather than opening a public issue:

- Andrew Yates — <andrewyates.name@gmail.com>

A useful report includes:

- the affected commit or release
- steps to reproduce
- the impact you expect
- whether any model, benchmark, or credential data involved is sensitive

## What we consider in scope

- verifier unsoundness that can yield false decisive results
- crashes in the parser or loader triggered by untrusted model/property files
- credential exposure committed to the repository or emitted into artifacts
- command-injection or path-traversal in the shipped scripts and harnesses

Out of scope:

- ordinary verification timeouts
- benchmark-score disagreements with no soundness or integrity consequence
- compromise of a local developer machine unrelated to this repository

## Before publishing artifacts

Run a quick pre-publish sweep:

```sh
cargo check --workspace --exclude ny-python
rg -n "token|secret|password|credential" --glob '!target/**'
```
