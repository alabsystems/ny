# VNN-COMP Scripts

These scripts adapt `ny` to the VNN-COMP harness contract.

```bash
./install_tool.sh v1
./prepare_instance.sh v1 <category> <onnx> <vnnlib>
./run_instance.sh v1 <category> <onnx> <vnnlib> <results-file> <timeout-seconds>
```

The native packaging command validates the harness and writes a tarball suitable
for upload through the VNN-COMP evaluation website:

```bash
cargo run -p ny-cli -- vnncomp-benchmarks --year 2026
cargo run -p ny-cli -- vnncomp-submit --output dist/ny-vnncomp-submission.tar.gz
```

The package always includes `rust-toolchain.toml` and `.cargo/config.toml`, but
does not copy internal AY sources into NY. AY remains an exact Git dependency
pinned by `Cargo.lock`. If a release builder has produced
`dist/bin/ny-x86_64-linux.xz`, its `.sha256` sidecar, and the strict
`ny-x86_64-linux.provenance.txt` gate manifest, that complete triplet is
included and `install_tool.sh` uses the binary as the offline fast path. A
partial, stale, unproven, wrong-architecture, or source-mismatched triplet is a
hard packaging error, including under `--dry-run`.
The provenance manifest is also a canonical Trust gate receipt: its NY, AY,
Trust compiler, executed-builder, dependency-lock, gate-log, and static ONNX
Runtime identities are bound into one sealed record inside the executable. The
packager validates the exact record without running a foreign-architecture
binary, snapshots the validated triplet, and verifies the bytes captured in the
tarball. The fresh-machine builder quarantines an earlier public triplet before
work and publishes provenance last, so an interrupted or failed rebuild cannot
silently reuse an older complete result.
The manifest's `ny_commit` names the source commit that produced the executable;
it does not attempt to name a commit containing its own artifact bytes. A Git
release may be a descendant of that source commit only when the complete tree
delta is exactly the three prebuilt files. This artifact-only rule makes a
clone-installable release possible without weakening source binding.
The prebuilt is dynamically linked and requires Linux x86_64 with GNU glibc
2.39 or newer (the official Ubuntu 24.04 target provides glibc 2.39). The
installer's standard-library verifier checks the checksum, strict manifest and
Trust receipt, Cargo/AY/builder identities, embedded binary seal, architecture,
and (when installed from a Git clone) the artifact-only source relationship
before checking host compatibility. It exclusively stages and sanity-runs the
validated binary before replacing `target/release/ny`; a malformed present
triplet is a hard error, while an incompatible host may use the source fallback.
Do not publish an ordinary Ubuntu 26.04 local build: that host's glibc 2.43 can
produce a binary importing `GLIBC_2.43`, which is not compatible with the
Ubuntu 24.04/glibc 2.39 evaluator. Use the Ubuntu-24.04-locked Trust builder.
Without a compatible artifact, installation performs a networked source build
requiring authenticated read access to the pinned AY revision, crates.io and
the ORT archive; native build prerequisites must also be available.
Credential-free evaluation therefore requires the validated prebuilt triplet.
Competition installation fails closed if the full `mip,cuda` tier cannot be
built. `NY_ALLOW_DEGRADED_BUILD=1` is available only for an explicit
non-competition development build.

Every installed or locally published `target/release/ny` has a
`target/release/ny.receipt` readiness sidecar. It binds the executable SHA-256,
the exact NY commit and tracked checkout state, `Cargo.lock` and AY identities,
the compiled feature tier, and the compiler identity. The validated prebuilt
path additionally binds the complete sealed Trust provenance manifest and
publishes the receipt last. `run_instance.sh` refuses automatic release/debug
selection when that receipt is missing, stale, malformed, or source-mismatched.
Submission tarballs inject `.ny-vnncomp-source.txt`, so the same exact-commit
check works after extraction without `.git`, including source-build fallback.

For local helper-script work:

```bash
./vnncomp_scripts/build_submission_binary.sh
./vnncomp_scripts/prepare_instance.sh v1 <category> <onnx> <vnnlib>
./vnncomp_scripts/run_instance.sh v1 <category> <onnx> <vnnlib> <results-file> <timeout-seconds>
```

The scripts prefer a receipted `target/release/ny` and fall back only to a
receipted debug binary for local dry runs. An explicit `NY_BIN=/path/to/ny`
remains a receipt-free developer override for mock binaries and controlled A/B
runs; the wrapper logs that bypass and never silently falls back when the
explicit path is invalid. The root `run_instance.sh` is the
competition-hermetic entrypoint and removes `AY_MILP_SMT` and
`AY_DUMP_QUERY_DIR` before starting NY. The shared
`vnncomp_scripts/run_instance.sh` helper and direct `ny vnncomp` invocations
preserve both variables for intentional developer A/B and query-capture runs.
