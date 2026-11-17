# VNN-COMP scripts

These scripts adapt `ny` to the VNN-COMP harness contract. The competition
calls the three entry points at the repository root:

```bash
./install_tool.sh v1
./prepare_instance.sh v1 <category> <onnx> <vnnlib>
./run_instance.sh v1 <category> <onnx> <vnnlib> <results-file> <timeout-seconds>
```

## Packaging a submission

```bash
targo --unverified run -p ny-cli -- vnncomp-submit
```

`vnncomp-submit` validates the harness scripts, runs
`build_submission_binary.sh` (skip with `--no-build`), and writes
`dist/ny-vnncomp-submission.tar.gz` (`--output` to change it; `--dry-run`
validates without writing). Every packaged input must be committed. The
tarball includes `rust-toolchain.toml` and `.cargo/config.toml`. `ay` stays a
Git dependency pinned by `Cargo.lock`; its sources are never copied in.

### Prebuilt binary

If a release builder has produced the triplet `dist/bin/ny-x86_64-linux.xz`,
its `.sha256` sidecar, and `ny-x86_64-linux.provenance.txt`, the tarball
includes it and `install_tool.sh` installs that binary instead of building
from source. A partial, stale, or mismatched triplet is a packaging error,
even under `--dry-run`.

- The provenance manifest records the sources, compiler, builder, lock file,
  gate log, and ONNX Runtime that produced the binary, and the same record is
  sealed inside the executable. It records identities only: the binary is
  built with the Trust toolchain by `targo --unverified build`, so it is not
  verified by Trust.
- `ny_commit` names the source commit that produced the binary. A release
  commit may differ from it only by the three prebuilt files.
- The prebuilt needs Linux x86_64 with GNU glibc 2.39 or newer, which the
  Ubuntu 24.04 evaluation image provides. Build it with
  `scripts/vnncomp_trust_linux_build.sh`, which refuses any host but Ubuntu
  24.04 x86_64: a binary built on Ubuntu 26.04 can require glibc 2.43.
- `install_tool.sh` runs `verify_prebuilt.py` (checksum, manifest, embedded
  seal, ELF architecture, source commit), then checks glibc and sanity-runs the
  staged binary before it replaces `target/release/ny`. A missing sidecar or
  failed verification is an error; an old or undetected glibc, or a failed
  sanity run, falls back to a source build.

A source build needs a rustup toolchain linked as `trust` (`targo` and
`trustc`), network access to crates.io and the ONNX Runtime archive, read
access to the pinned `ay` revision, and native build tools. It fails unless
the full `mip,cuda` feature set builds. `NY_ALLOW_DEGRADED_BUILD=1` allows a
reduced build, for development only.

### Release artifact (`dist/bin/<target>/`)

`scripts/release/make_submission.sh` builds the binary from a clean checkout at
one commit, with the Trust toolchain in the unverified lane, and writes
`dist/bin/<target>/{MANIFEST,ny.xz,ny.sha256}`. `MANIFEST` declares the build
lane (`native` or `cross-build-std`) and binds the binary's bytes, its
`ny --build-id` line, its ELF link record and the embedded CUDA images to the
source commit. When the sealed triplet above is absent, `install_tool.sh`
installs this artifact. It first runs `verify_release.py`, then requires the
binary to report MANIFEST's exact `--build-id` and `--build-stamp`, then requires
correct answers on both `selftest/` instances, and then binds a `release`
receipt. Verification, identity and self-test failures refuse the install.
An older or undetected glibc instead falls back to a source build.
`MANIFEST` also records `ny_commit`'s tree id and its raw commit object, which
must hash to `ny_commit` and name that tree. The verifier and the receipt bind
the checkout by content: `HEAD`'s tree, minus exactly the release files, must
be that tree. So a shallow `git clone --depth 1` of the release commit
installs, and so does a clone owned by another user, because git is run with
`safe.directory` set to that checkout only. The runbook, the MANIFEST schema
and the official-contract smoke (`scripts/release/smoke_submission.sh`) are in
`docs/RELEASE.md`.

### Readiness receipts

Every installed `target/release/ny` has a `target/release/ny.receipt` sidecar
that binds the executable's SHA-256, the source commit and checkout state, the
`Cargo.lock` and `ay` identities, the feature set, and the compiler.
`run_instance.sh` refuses a binary whose receipt is missing, stale, malformed,
or does not match the source. Submission tarballs include
`.ny-vnncomp-source.txt`, so the check also works in an extracted tarball
without `.git`.

## Local use

```bash
./vnncomp_scripts/build_submission_binary.sh
./vnncomp_scripts/prepare_instance.sh v1 <category> <onnx> <vnnlib>
./vnncomp_scripts/run_instance.sh v1 <category> <onnx> <vnnlib> <results-file> <timeout-seconds>
```

`prepare_instance.sh` validates its arguments and checks that a binary exists;
it never analyzes the instance. `run_instance.sh` uses `target/release/ny`, or
`target/debug/ny` when no release binary exists, and refuses either without a
valid receipt. It execs `ny vnncomp` under an OS `timeout` of the budget plus
10 s, with `OMP_NUM_THREADS=1`.

- `NY_BIN=/path/to/ny` skips the receipt check, for mock binaries and A/B
  runs; the wrapper logs the override and fails if the path is not an
  executable file.
- `NY_EXPECT_BUILD_ID=<ny --build-stamp line>` makes the wrapper refuse any
  other binary, so a multi-row measurement cannot switch binaries mid-run.

The root `run_instance.sh` is the hermetic competition entry point: it unsets
every `AY_*`, `NY_*`, and `MIMALLOC_*` variable plus `RAYON_NUM_THREADS` and
`RAYON_RS_NUM_CPUS`, re-exports only `NY_BIN` and `NY_EXPECT_BUILD_ID`, and
`vnncomp_scripts/run_instance.sh` unsets those two before starting `ny`.
`vnncomp_scripts/run_instance.sh` and direct `ny vnncomp` calls keep the
caller's environment, for developer A/B and query-capture runs.
