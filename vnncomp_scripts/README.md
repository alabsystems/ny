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
The prebuilt is dynamically linked and requires Linux x86_64 with GNU glibc
2.38 or newer (the official Ubuntu 24.04 target provides glibc 2.39). The
installer verifies its checksum before checking host compatibility, stages and
sanity-runs it before replacing `target/release/ny`, and otherwise falls back to
the source build. Without a compatible artifact, installation performs a
networked source build requiring authenticated read access to the pinned AY
revision, crates.io and the ORT archive; native build prerequisites must also be
available. Credential-free evaluation therefore requires the validated
prebuilt triplet. Competition installation fails closed if the full `mip,cuda`
tier cannot be built. `NY_ALLOW_DEGRADED_BUILD=1` is available only for an
explicit non-competition development build.

For local helper-script work:

```bash
./vnncomp_scripts/build_submission_binary.sh
./vnncomp_scripts/prepare_instance.sh v1 <category> <onnx> <vnnlib>
./vnncomp_scripts/run_instance.sh v1 <category> <onnx> <vnnlib> <results-file> <timeout-seconds>
```

The scripts prefer `target/release/ny` and fall back to debug builds for local
dry runs. The root `run_instance.sh` is the competition-hermetic entrypoint and
removes `AY_MILP_SMT` and `AY_DUMP_QUERY_DIR` before starting NY. The shared
`vnncomp_scripts/run_instance.sh` helper and direct `ny vnncomp` invocations
preserve both variables for intentional developer A/B and query-capture runs.
