// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use ny_test_utils::workspace_root;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::Duration;
use tempfile::{tempdir, TempDir};

fn copy_build_script(temp_repo: &Path) -> PathBuf {
    let script_source = workspace_root().join("vnncomp_scripts/build_submission_binary.sh");
    let script_target = temp_repo.join("vnncomp_scripts/build_submission_binary.sh");
    fs::create_dir_all(script_target.parent().expect("script parent"))
        .expect("failed to create script directory");
    // fs::copy preserves the source permission bits, so the copy is executable
    // exactly when the checked-in script is — the same direct-exec contract
    // install_tool.sh relies on. Re-adding the exec bit here would let these
    // tests pass against a committed non-executable script.
    fs::copy(&script_source, &script_target).expect("failed to copy build_submission_binary.sh");
    fs::copy(
        workspace_root().join("vnncomp_scripts/submission_binary_receipt.sh"),
        temp_repo.join("vnncomp_scripts/submission_binary_receipt.sh"),
    )
    .expect("failed to copy submission_binary_receipt.sh");
    script_target
}

fn write_archive_source_marker(temp_repo: &Path) {
    let lock_path = temp_repo.join("Cargo.lock");
    if !lock_path.is_file() {
        fs::write(&lock_path, "version = 4\n").expect("failed to write fixture Cargo.lock");
    }
    let lock = fs::read(&lock_path).expect("failed to read fixture Cargo.lock");
    let lock_sha256 = format!("{:x}", Sha256::digest(lock));
    fs::write(
        temp_repo.join(".ny-vnncomp-source.txt"),
        format!(
            "schema=ny-vnncomp-source-v1\n\
             ny_commit=0123456789abcdef0123456789abcdef01234567\n\
             cargo_lock_sha256={lock_sha256}\n"
        ),
    )
    .expect("failed to write fixture archive source marker");
}

// `${...}` below is intentional shell parameter expansion in a raw fixture.
#[allow(clippy::literal_string_with_formatting_args)]
fn write_fake_cargo(temp_repo: &Path) -> PathBuf {
    let fake_cargo = temp_repo.join("bin/cargo");
    fs::create_dir_all(fake_cargo.parent().expect("fake cargo parent"))
        .expect("failed to create fake cargo directory");
    fs::write(
        &fake_cargo,
        r#"#!/bin/bash
set -euo pipefail

printf '%s\n' "$@" > "$PWD/cargo-args.txt"
{
    printf 'OPENSSL_DIR=%s\n' "${OPENSSL_DIR:-}"
    printf 'OPENSSL_LIB_DIR=%s\n' "${OPENSSL_LIB_DIR:-}"
    printf 'OPENSSL_INCLUDE_DIR=%s\n' "${OPENSSL_INCLUDE_DIR:-}"
    printf 'CFLAGS=%s\n' "${CFLAGS:-}"
    printf 'RUSTFLAGS=%s\n' "${RUSTFLAGS:-}"
    printf 'CARGO_TARGET_DIR=%s\n' "${CARGO_TARGET_DIR:-}"
    printf 'CARGO_BUILD_TARGET_DIR=%s\n' "${CARGO_BUILD_TARGET_DIR:-}"
    printf 'CARGO_BUILD_JOBS=%s\n' "${CARGO_BUILD_JOBS:-}"
} > "$PWD/cargo-env.txt"
: "${CARGO_TARGET_DIR:?build script must select an explicit Cargo target directory}"
artifact_relative="${FAKE_CARGO_ARTIFACT_RELATIVE:-release/ny}"
artifact="${CARGO_TARGET_DIR}/${artifact_relative}"
mkdir -p "$(dirname "${artifact}")"

case "${FAKE_CARGO_LAYOUT:?missing FAKE_CARGO_LAYOUT}" in
    root-release)
        cat > "${artifact}" <<'EOF'
#!/bin/bash
echo root-release
EOF
        chmod +x "${artifact}"
        ;;
    worker-release)
        : "${AI_WORKER_ID:?missing AI_WORKER_ID}"
        cat > "${artifact}" <<'EOF'
#!/bin/bash
echo worker-release
EOF
        chmod +x "${artifact}"
        ;;
    *)
        echo "unexpected FAKE_CARGO_LAYOUT: ${FAKE_CARGO_LAYOUT}" >&2
        exit 1
        ;;
esac

if [ "${FAKE_CARGO_PAD_MIB:-0}" != "0" ]; then
    dd if=/dev/zero bs=1048576 count="${FAKE_CARGO_PAD_MIB}" status=none >> "${artifact}"
fi

emit_artifact() {
    local executable="$1"
    local fresh="${2:-false}"
    printf '{"reason":"compiler-artifact","manifest_path":"%s","target":{"kind":["bin"],"crate_types":["bin"],"name":"ny"},"filenames":["%s"],"executable":"%s","fresh":%s}\n' \
        "$PWD/crates/ny-cli/Cargo.toml" "${executable}" "${executable}" "${fresh}"
}

case "${FAKE_CARGO_OUTPUT_MODE:-success}" in
    success)
        case "${FAKE_CARGO_ARTIFACT_COUNT:-1}" in
            0) ;;
            1) emit_artifact "${artifact}" ;;
            2)
                second_artifact="${artifact}.second"
                cp "${artifact}" "${second_artifact}"
                emit_artifact "${artifact}"
                emit_artifact "${second_artifact}"
                ;;
            *)
                echo "invalid FAKE_CARGO_ARTIFACT_COUNT" >&2
                exit 1
                ;;
        esac
        printf '{"reason":"build-finished","success":true}\n'
        ;;
    malformed)
        printf '{this-is-not-cargo-json\n'
        printf '{"reason":"build-finished","success":true}\n'
        ;;
    failed)
        emit_artifact "${artifact}"
        printf '{"reason":"build-finished","success":false}\n'
        exit 42
        ;;
    stale)
        emit_artifact "${artifact}" true
        printf '{"reason":"build-finished","success":true}\n'
        ;;
    *) echo "invalid FAKE_CARGO_OUTPUT_MODE" >&2; exit 1 ;;
esac
"#,
    )
    .expect("failed to write fake cargo");
    make_executable(&fake_cargo);
    fake_cargo
}

fn write_failing_pkg_config(temp_repo: &Path) {
    let fake_pkg_config = temp_repo.join("bin/pkg-config");
    fs::write(
        &fake_pkg_config,
        "#!/bin/bash\n: > \"$PWD/pkg-config-invoked\"\nexit 1\n",
    )
    .expect("failed to write fake pkg-config");
    make_executable(&fake_pkg_config);
}

fn write_forbidden_package_manager_shims(temp_repo: &Path) {
    for tool in ["apt-get", "dnf", "sudo"] {
        let shim = temp_repo.join("bin").join(tool);
        fs::write(
            &shim,
            format!("#!/bin/bash\n: > \"$PWD/{tool}-invoked\"\nexit 91\n"),
        )
        .unwrap_or_else(|err| panic!("failed to write fake {tool}: {err}"));
        make_executable(&shim);
    }
}

fn make_executable(path: &Path) {
    let mut permissions = fs::metadata(path)
        .unwrap_or_else(|err| panic!("failed to stat {}: {err}", path.display()))
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)
        .unwrap_or_else(|err| panic!("failed to chmod {}: {err}", path.display()));
}

fn fake_repo() -> (TempDir, PathBuf) {
    let temp_repo = tempdir().expect("failed to create temp repo");
    let script = copy_build_script(temp_repo.path());
    write_archive_source_marker(temp_repo.path());
    write_fake_cargo(temp_repo.path());
    write_failing_pkg_config(temp_repo.path());
    write_forbidden_package_manager_shims(temp_repo.path());
    (temp_repo, script)
}

fn real_cargo_repo() -> (TempDir, PathBuf) {
    let temp_repo = tempdir().expect("failed to create real-Cargo test repo");
    let script = copy_build_script(temp_repo.path());
    let crate_root = temp_repo.path().join("crates/ny-cli");
    fs::create_dir_all(crate_root.join("src")).expect("failed to create tiny ny-cli crate");
    fs::write(
        temp_repo.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/ny-cli\"]\nresolver = \"2\"\n",
    )
    .expect("failed to write tiny workspace manifest");
    fs::write(
        crate_root.join("Cargo.toml"),
        concat!(
            "[package]\n",
            "name = \"ny-cli\"\n",
            "version = \"0.0.0\"\n",
            "edition = \"2021\"\n\n",
            "[[bin]]\n",
            "name = \"ny\"\n",
            "path = \"src/main.rs\"\n\n",
            "[features]\n",
            "default = []\n",
            "mip = []\n",
            "cuda = []\n",
        ),
    )
    .expect("failed to write tiny ny-cli manifest");
    fs::write(
        crate_root.join("src/main.rs"),
        "fn main() { println!(\"fresh-json-artifact\"); }\n",
    )
    .expect("failed to write tiny ny binary");
    let lock = Command::new("cargo")
        .args(["generate-lockfile", "--quiet"])
        .current_dir(temp_repo.path())
        .output()
        .expect("failed to generate tiny Cargo.lock");
    assert_success(&lock, "tiny cargo generate-lockfile");
    write_archive_source_marker(temp_repo.path());
    (temp_repo, script)
}

fn rust_host() -> String {
    let output = Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("failed to query rustc host");
    assert_success(&output, "rustc -vV");
    let metadata = String::from_utf8(output.stdout).expect("rustc metadata should be UTF-8");
    metadata
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .filter(|host| !host.is_empty())
        .expect("rustc host must be present")
        .to_owned()
}

fn native_fp16_injection_expected() -> bool {
    if rust_host() != "aarch64-unknown-linux-gnu" {
        return false;
    }
    let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") else {
        return false;
    };
    let feature_lines: Vec<_> = cpuinfo
        .lines()
        .filter(|line| line.starts_with("Features"))
        .collect();
    if feature_lines.is_empty()
        || !feature_lines
            .iter()
            .all(|line| line.split_whitespace().any(|feature| feature == "asimdhp"))
    {
        return false;
    }
    let cfg = Command::new("rustc")
        .args(["--print", "cfg"])
        .output()
        .expect("failed to query rustc target features");
    assert_success(&cfg, "rustc --print cfg");
    !String::from_utf8_lossy(&cfg.stdout)
        .lines()
        .any(|line| line == "target_feature=\"fp16\"")
}

fn write_stale_alias(temp_repo: &Path) -> PathBuf {
    let alias = temp_repo.join("target/release/ny");
    fs::create_dir_all(alias.parent().expect("alias parent"))
        .expect("failed to create stale alias directory");
    fs::write(&alias, "#!/bin/bash\necho stale-root-artifact\n")
        .expect("failed to write stale alias");
    make_executable(&alias);
    alias
}

fn assert_no_staging_directories(parent: &Path) {
    if !parent.is_dir() {
        return;
    }
    for entry in fs::read_dir(parent).expect("failed to inspect staging parent") {
        let entry = entry.expect("failed to inspect staging entry");
        assert!(
            !entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(".ny-submission-build.")),
            "staging directory leaked after build: {}",
            entry.path().display()
        );
    }
}

fn run_build_script(
    script: &Path,
    temp_repo: &Path,
    layout: &str,
    worker_id: Option<&str>,
) -> Output {
    run_build_script_with_target(script, temp_repo, layout, worker_id, None)
}

fn run_build_script_with_target(
    script: &Path,
    temp_repo: &Path,
    layout: &str,
    worker_id: Option<&str>,
    build_target: Option<&str>,
) -> Output {
    run_build_script_with_options(
        script,
        temp_repo,
        layout,
        worker_id,
        build_target,
        None,
        None,
    )
}

fn run_build_script_with_options(
    script: &Path,
    temp_repo: &Path,
    layout: &str,
    worker_id: Option<&str>,
    build_target: Option<&str>,
    artifact_relative: Option<&str>,
    artifact_count: Option<&str>,
) -> Output {
    run_build_script_with_scenario(
        script,
        temp_repo,
        layout,
        worker_id,
        build_target,
        artifact_relative,
        artifact_count,
        None,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_build_script_with_scenario(
    script: &Path,
    temp_repo: &Path,
    layout: &str,
    worker_id: Option<&str>,
    build_target: Option<&str>,
    artifact_relative: Option<&str>,
    artifact_count: Option<&str>,
    output_mode: Option<&str>,
    pad_mib: Option<&str>,
    cargo_target_dir: Option<&Path>,
    cargo_build_target_dir: Option<&Path>,
) -> Output {
    let mut path_entries = vec![temp_repo.join("bin")];
    let inherited_path = env::var_os("PATH").expect("PATH should be set");
    path_entries.extend(env::split_paths(&inherited_path));
    let joined_path = env::join_paths(path_entries).expect("failed to join PATH entries");
    let inherited_rustup_home = env::var_os("RUSTUP_HOME").or_else(|| {
        env::var_os("HOME").map(|home| PathBuf::from(home).join(".rustup").into_os_string())
    });

    let mut command = Command::new(script);
    command
        .current_dir(temp_repo)
        .env("PATH", joined_path)
        .env("HOME", temp_repo.join("home"))
        .env("CARGO_HOME", temp_repo.join("home/.cargo"))
        .env("FAKE_CARGO_LAYOUT", layout)
        .env_remove("OPENSSL_DIR")
        .env_remove("OPENSSL_LIB_DIR")
        .env_remove("OPENSSL_INCLUDE_DIR")
        .env_remove("CFLAGS")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR");
    if let Some(rustup_home) = inherited_rustup_home {
        // HOME is intentionally replaced for fixture isolation. Keep rustup's
        // real toolchain root so `rustc -vV` still reports a host and the fp16
        // assertions are not vacuous.
        command.env("RUSTUP_HOME", rustup_home);
    }

    if let Some(worker_id) = worker_id {
        command.env("AI_WORKER_ID", worker_id);
    } else {
        command.env_remove("AI_WORKER_ID");
    }
    if let Some(build_target) = build_target {
        command.env("CARGO_BUILD_TARGET", build_target);
    } else {
        command.env_remove("CARGO_BUILD_TARGET");
    }
    if let Some(artifact_relative) = artifact_relative {
        command.env("FAKE_CARGO_ARTIFACT_RELATIVE", artifact_relative);
    } else {
        command.env_remove("FAKE_CARGO_ARTIFACT_RELATIVE");
    }
    if let Some(artifact_count) = artifact_count {
        command.env("FAKE_CARGO_ARTIFACT_COUNT", artifact_count);
    } else {
        command.env_remove("FAKE_CARGO_ARTIFACT_COUNT");
    }
    if let Some(output_mode) = output_mode {
        command.env("FAKE_CARGO_OUTPUT_MODE", output_mode);
    } else {
        command.env_remove("FAKE_CARGO_OUTPUT_MODE");
    }
    if let Some(pad_mib) = pad_mib {
        command.env("FAKE_CARGO_PAD_MIB", pad_mib);
    } else {
        command.env_remove("FAKE_CARGO_PAD_MIB");
    }
    if let Some(target_dir) = cargo_target_dir {
        command.env("CARGO_TARGET_DIR", target_dir);
    }
    if let Some(target_dir) = cargo_build_target_dir {
        command.env("CARGO_BUILD_TARGET_DIR", target_dir);
    }

    output_retrying_transient_text_file_busy(&mut command, script)
}

fn run_real_build_script(script: &Path, temp_repo: &Path, build_target: Option<&str>) -> Output {
    let inherited_rustup_home = env::var_os("RUSTUP_HOME").or_else(|| {
        env::var_os("HOME").map(|home| PathBuf::from(home).join(".rustup").into_os_string())
    });
    let mut command = Command::new(script);
    command
        .current_dir(temp_repo)
        .env("HOME", temp_repo.join("home"))
        .env("CARGO_HOME", temp_repo.join("home/.cargo"))
        .env_remove("OPENSSL_DIR")
        .env_remove("OPENSSL_LIB_DIR")
        .env_remove("OPENSSL_INCLUDE_DIR")
        .env_remove("CFLAGS")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR")
        .env_remove("AI_WORKER_ID")
        .env_remove("FAKE_CARGO_LAYOUT")
        .env_remove("FAKE_CARGO_ARTIFACT_RELATIVE")
        .env_remove("FAKE_CARGO_ARTIFACT_COUNT")
        .env_remove("FAKE_CARGO_OUTPUT_MODE")
        .env_remove("FAKE_CARGO_PAD_MIB");
    if let Some(rustup_home) = inherited_rustup_home {
        command.env("RUSTUP_HOME", rustup_home);
    }
    if let Some(build_target) = build_target {
        command.env("CARGO_BUILD_TARGET", build_target);
    } else {
        command.env_remove("CARGO_BUILD_TARGET");
    }
    output_retrying_transient_text_file_busy(&mut command, script)
}

fn output_retrying_transient_text_file_busy(command: &mut Command, script: &Path) -> Output {
    // These tests deliberately preserve and exercise the script's direct-exec
    // contract. On Linux, many concurrent copies into tmpfs can briefly make a
    // just-closed executable report ETXTBSY. Retrying that one transient spawn
    // error retains direct execution while keeping the parallel suite stable.
    const MAX_ATTEMPTS: u32 = 8;

    for attempt in 0..MAX_ATTEMPTS {
        match command.output() {
            Ok(output) => return output,
            Err(err)
                if err.raw_os_error() == Some(rustix::io::Errno::TXTBSY.raw_os_error())
                    && attempt + 1 < MAX_ATTEMPTS =>
            {
                thread::sleep(Duration::from_millis(1_u64 << attempt.min(5)));
            }
            Err(err) => panic!("failed to run {}: {err}", script.display()),
        }
    }
    unreachable!("the bounded spawn loop always returns or panics")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn validate_receipt(temp_repo: &Path, binary: &Path) -> Output {
    Command::new("bash")
        .arg(temp_repo.join("vnncomp_scripts/submission_binary_receipt.sh"))
        .arg("validate")
        .arg(binary)
        .arg(temp_repo)
        .output()
        .expect("failed to run receipt validator")
}

fn assert_full_tier_uses_private_staging(cargo_args: &str, expected_parent: &Path) -> PathBuf {
    let arguments: Vec<_> = cargo_args.lines().collect();
    assert_eq!(
        arguments.len(),
        10,
        "unexpected Cargo arguments: {cargo_args}"
    );
    assert_eq!(
        &arguments[..6],
        [
            "build",
            "--locked",
            "--release",
            "-p",
            "ny-cli",
            "--target-dir"
        ]
    );
    assert_eq!(arguments[7], "--message-format=json-render-diagnostics");
    assert_eq!(&arguments[8..], ["--features", "mip,cuda"]);
    let staging = PathBuf::from(arguments[6]);
    let expected_parent = expected_parent
        .canonicalize()
        .expect("failed to canonicalize the requested staging parent");
    assert_eq!(
        staging.parent(),
        Some(expected_parent.as_path()),
        "staging target escaped its requested parent: {}",
        staging.display()
    );
    assert!(
        staging
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".ny-submission-build.")),
        "target is not invocation-scoped staging: {}",
        staging.display()
    );
    staging
}

#[test]
fn test_committed_harness_scripts_are_executable() {
    // install_tool.sh execs build_submission_binary.sh directly, and the
    // VNN-COMP harness and matrix runner exec prepare_instance.sh /
    // run_instance.sh the same way: without the exec bit the submission
    // install path dies with "Permission denied" even though every
    // content-level test passes.
    for script in [
        "install_tool.sh",
        "vnncomp_scripts/build_submission_binary.sh",
        "vnncomp_scripts/prepare_instance.sh",
        "vnncomp_scripts/run_instance.sh",
        "vnncomp_scripts/submission_binary_receipt.sh",
    ] {
        let path = workspace_root().join(script);
        let mode = fs::metadata(&path)
            .unwrap_or_else(|err| panic!("failed to stat {}: {err}", path.display()))
            .permissions()
            .mode();
        assert!(
            mode & 0o111 != 0,
            "{script} must be executable (mode {mode:o}): the harness execs it directly"
        );
    }

    let cargo_config = fs::read_to_string(workspace_root().join(".cargo/config.toml"))
        .expect("read repository Cargo config");
    assert!(
        !cargo_config
            .lines()
            .any(|line| line.trim_start().starts_with("jobs")),
        "a developer host's build-job cap must not throttle every checkout"
    );
}

#[test]
fn test_build_submission_binary_copies_root_release_binary() {
    let (temp_repo, script) = fake_repo();

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary root-release");

    let cargo_args = fs::read_to_string(temp_repo.path().join("cargo-args.txt"))
        .expect("failed to read fake cargo args");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    let jobs = cargo_env
        .lines()
        .find_map(|line| line.strip_prefix("CARGO_BUILD_JOBS="))
        .expect("build-job policy reaches Cargo")
        .parse::<usize>()
        .expect("build-job policy is numeric");
    assert!(jobs > 0, "build-job policy must retain at least one worker");
    let staging =
        assert_full_tier_uses_private_staging(&cargo_args, &temp_repo.path().join("target"));
    assert!(
        !staging.exists(),
        "successful staging directory was not cleaned"
    );

    let alias_path = temp_repo.path().join("target/release/ny");
    assert!(
        alias_path.is_file(),
        "alias binary missing: {}",
        alias_path.display()
    );

    let alias_output = Command::new(&alias_path)
        .output()
        .expect("failed to execute alias binary");
    assert_success(&alias_output, "ny alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "root-release"
    );
    let receipt = alias_path.with_extension("receipt");
    assert!(receipt.is_file(), "published binary receipt is missing");
    let receipt_contents =
        fs::read_to_string(&receipt).expect("failed to read published binary receipt");
    assert!(receipt_contents.contains("schema=ny-submission-binary-receipt-v1\n"));
    assert!(receipt_contents.contains("source_commit=0123456789abcdef0123456789abcdef01234567\n"));
    assert!(receipt_contents.contains("features=mip,cuda\n"));
    assert!(receipt_contents.contains("toolchain_kind=rustc-vv\n"));
    assert_success(
        &validate_receipt(temp_repo.path(), &alias_path),
        "fresh binary receipt",
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("source:") && stdout.contains(".ny-submission-build."),
        "expected staged release source in stdout, got: {stdout}"
    );
    assert!(
        stdout.contains("alias:") && stdout.contains("target/release/ny"),
        "expected stable alias path in stdout, got: {stdout}"
    );
    assert!(
        stdout.contains("receipt:") && stdout.contains("target/release/ny.receipt"),
        "expected stable receipt path in stdout, got: {stdout}"
    );
}

#[test]
fn test_receipt_rejects_binary_bytes_changed_after_publication() {
    let (temp_repo, script) = fake_repo();
    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "receipt fixture build");
    let alias = temp_repo.path().join("target/release/ny");

    fs::write(&alias, "#!/bin/bash\necho replaced-after-build\n")
        .expect("failed to replace published fixture bytes");
    make_executable(&alias);
    let validation = validate_receipt(temp_repo.path(), &alias);
    assert!(!validation.status.success());
    assert!(
        String::from_utf8_lossy(&validation.stderr).contains("stale/mismatched binary"),
        "missing binary mismatch diagnostic: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
}

#[test]
fn test_receipt_rejects_archive_source_identity_changed_after_build() {
    let (temp_repo, script) = fake_repo();
    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "receipt fixture build");
    let alias = temp_repo.path().join("target/release/ny");
    let marker = temp_repo.path().join(".ny-vnncomp-source.txt");
    let marker_contents =
        fs::read_to_string(&marker).expect("failed to read fixture source marker");
    fs::write(
        &marker,
        marker_contents.replace(
            "0123456789abcdef0123456789abcdef01234567",
            "89abcdef0123456789abcdef0123456789abcdef",
        ),
    )
    .expect("failed to change fixture source marker");

    let validation = validate_receipt(temp_repo.path(), &alias);
    assert!(!validation.status.success());
    assert!(
        String::from_utf8_lossy(&validation.stderr).contains("stale source identity"),
        "missing source mismatch diagnostic: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
}

#[test]
fn test_build_submission_binary_copies_worker_release_binary() {
    let (temp_repo, script) = fake_repo();

    let output = run_build_script(&script, temp_repo.path(), "worker-release", Some("17"));
    assert_success(&output, "build_submission_binary worker-release");

    let cargo_args = fs::read_to_string(temp_repo.path().join("cargo-args.txt"))
        .expect("failed to read fake cargo args");
    let staging = assert_full_tier_uses_private_staging(
        &cargo_args,
        &temp_repo.path().join("target/worker_17"),
    );
    assert!(
        !staging.exists(),
        "worker staging directory was not cleaned"
    );

    let alias_path = temp_repo.path().join("target/release/ny");
    assert!(
        alias_path.is_file(),
        "alias binary missing: {}",
        alias_path.display()
    );

    let alias_output = Command::new(&alias_path)
        .output()
        .expect("failed to execute alias binary");
    assert_success(&alias_output, "ny alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "worker-release"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("source:") && stdout.contains("target/worker_17/.ny-submission-build."),
        "expected worker staged source in stdout, got: {stdout}"
    );
    assert!(
        stdout.contains("alias:") && stdout.contains("target/release/ny"),
        "expected stable alias path in stdout, got: {stdout}"
    );
}

#[test]
fn test_explicit_target_artifact_replaces_stale_root_alias() {
    let (temp_repo, script) = fake_repo();
    let host = rust_host();
    let artifact_relative = format!("{host}/release/ny");
    let alias = write_stale_alias(temp_repo.path());

    let output = run_build_script_with_options(
        &script,
        temp_repo.path(),
        "root-release",
        None,
        Some(&host),
        Some(&artifact_relative),
        None,
    );
    assert_success(&output, "build_submission_binary targeted artifact");

    let alias_output = Command::new(&alias)
        .output()
        .expect("failed to execute replaced alias");
    assert_success(&alias_output, "replaced ny alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "root-release"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(".ny-submission-build.") && stdout.contains(&format!("/{host}/release/ny")),
        "builder did not report the targeted Cargo artifact: {stdout}"
    );
}

#[test]
fn test_explicit_target_dir_variables_become_private_staging_parents() {
    for variable in ["CARGO_TARGET_DIR", "CARGO_BUILD_TARGET_DIR"] {
        let (temp_repo, script) = fake_repo();
        let requested_parent = temp_repo.path().join(format!("requested-{variable}"));
        let (cargo_target_dir, cargo_build_target_dir) = if variable == "CARGO_TARGET_DIR" {
            (Some(requested_parent.as_path()), None)
        } else {
            (None, Some(requested_parent.as_path()))
        };
        let output = run_build_script_with_scenario(
            &script,
            temp_repo.path(),
            "root-release",
            None,
            None,
            None,
            None,
            None,
            None,
            cargo_target_dir,
            cargo_build_target_dir,
        );
        assert_success(&output, &format!("explicit {variable} staging"));
        let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
            .expect("failed to read explicit target-dir environment");
        let target_dir = cargo_env
            .lines()
            .find_map(|line| line.strip_prefix("CARGO_TARGET_DIR="))
            .expect("fake Cargo did not record CARGO_TARGET_DIR");
        let build_target_dir = cargo_env
            .lines()
            .find_map(|line| line.strip_prefix("CARGO_BUILD_TARGET_DIR="))
            .expect("fake Cargo did not record CARGO_BUILD_TARGET_DIR");
        assert_eq!(target_dir, build_target_dir);
        let staging = Path::new(target_dir);
        let canonical_parent = requested_parent
            .canonicalize()
            .expect("failed to canonicalize the explicit staging parent");
        assert_eq!(staging.parent(), Some(canonical_parent.as_path()));
        assert!(staging
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".ny-submission-build.")));
        assert!(
            !staging.exists(),
            "explicit staging directory was not cleaned"
        );
        assert_no_staging_directories(&requested_parent);

        let alias = temp_repo.path().join("target/release/ny");
        let alias_output = Command::new(&alias)
            .output()
            .expect("failed to execute explicitly staged alias");
        assert_success(&alias_output, "explicitly staged alias");
        assert_eq!(
            String::from_utf8_lossy(&alias_output.stdout).trim(),
            "root-release"
        );
    }
}

#[test]
fn test_default_layout_rejections_preserve_prior_alias_byte_for_byte() {
    for (label, artifact_count, output_mode, expected_status, diagnostic) in [
        (
            "zero",
            Some("0"),
            Some("success"),
            1,
            "expected exactly one ny executable, got 0",
        ),
        (
            "multiple",
            Some("2"),
            Some("success"),
            1,
            "expected exactly one ny executable, got 2",
        ),
        (
            "malformed",
            Some("1"),
            Some("malformed"),
            1,
            "is not valid Cargo JSON",
        ),
        (
            "failed Cargo",
            Some("1"),
            Some("failed"),
            42,
            "required competition feature tier 'mip,cuda' failed",
        ),
        (
            "stale event",
            Some("1"),
            Some("stale"),
            1,
            "was not freshly built in this invocation",
        ),
    ] {
        let (temp_repo, script) = fake_repo();
        let alias = write_stale_alias(temp_repo.path());
        let prior_bytes = fs::read(&alias).expect("failed to read prior alias");
        let prior_mode = fs::metadata(&alias)
            .expect("failed to stat prior alias")
            .permissions()
            .mode();
        let output = run_build_script_with_scenario(
            &script,
            temp_repo.path(),
            "root-release",
            None,
            None,
            None,
            artifact_count,
            output_mode,
            None,
            None,
            None,
        );
        assert_eq!(
            output.status.code(),
            Some(expected_status),
            "{label} rejection returned the wrong status\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(diagnostic),
            "missing {label} rejection diagnostic in: {stderr}"
        );
        assert_eq!(
            fs::read(&alias).expect("failed to reread preserved alias"),
            prior_bytes,
            "{label} rejection changed the prior alias bytes"
        );
        assert_eq!(
            fs::metadata(&alias)
                .expect("failed to restat prior alias")
                .permissions()
                .mode(),
            prior_mode,
            "{label} rejection changed the prior alias mode"
        );
        assert_no_staging_directories(&temp_repo.path().join("target"));
    }
}

#[test]
fn test_publication_replaces_destination_symlink_without_touching_victim() {
    let (temp_repo, script) = fake_repo();
    let victim = temp_repo.path().join("victim");
    fs::write(&victim, "#!/bin/bash\necho untouched-victim\n")
        .expect("failed to write symlink victim");
    make_executable(&victim);
    let victim_bytes = fs::read(&victim).expect("failed to read victim");
    let alias = temp_repo.path().join("target/release/ny");
    fs::create_dir_all(alias.parent().expect("alias parent"))
        .expect("failed to create alias directory");
    symlink(&victim, &alias).expect("failed to create destination symlink");

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "symlink-safe publication");
    assert_eq!(
        fs::read(&victim).expect("failed to reread victim"),
        victim_bytes,
        "publication wrote through the destination symlink"
    );
    assert!(
        fs::symlink_metadata(&alias)
            .expect("failed to lstat published alias")
            .file_type()
            .is_file(),
        "published alias remained a symlink"
    );
    let alias_output = Command::new(&alias)
        .output()
        .expect("failed to execute symlink-safe alias");
    assert_success(&alias_output, "symlink-safe alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "root-release"
    );
}

#[test]
fn test_publication_atomically_replaces_inode_without_temp_leaks() {
    let (temp_repo, script) = fake_repo();
    let alias = write_stale_alias(temp_repo.path());
    let prior_bytes = fs::read(&alias).expect("failed to read old alias");
    let prior_inode = fs::metadata(&alias)
        .expect("failed to stat old alias")
        .ino();
    let mut published_bytes = b"#!/bin/bash\necho root-release\n".to_vec();
    published_bytes.resize(published_bytes.len() + 32 * 1024 * 1024, 0);

    let thread_script = script;
    let thread_repo = temp_repo.path().to_owned();
    let build = thread::spawn(move || {
        run_build_script_with_scenario(
            &thread_script,
            &thread_repo,
            "root-release",
            None,
            None,
            None,
            None,
            None,
            Some("32"),
            None,
            None,
        )
    });

    while !build.is_finished() {
        let observed = fs::read(&alias).expect("alias disappeared during publication");
        assert!(
            observed == prior_bytes || observed == published_bytes,
            "a reader observed a partially published alias ({} bytes)",
            observed.len()
        );
        thread::yield_now();
    }

    let output = build.join().expect("build thread panicked");
    assert_success(&output, "atomic regular-file publication");
    assert_eq!(
        fs::read(&alias).expect("failed to read published alias"),
        published_bytes,
        "atomic publication installed unexpected bytes"
    );
    let published = fs::metadata(&alias).expect("failed to stat published alias");
    assert_ne!(
        published.ino(),
        prior_inode,
        "publication modified the old inode instead of atomically replacing it"
    );
    for entry in fs::read_dir(alias.parent().expect("alias parent"))
        .expect("failed to inspect publication directory")
    {
        let entry = entry.expect("failed to inspect publication entry");
        assert!(
            !entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(".ny-publish-")),
            "publication temp file leaked: {}",
            entry.path().display()
        );
    }
}

#[test]
fn test_real_cargo_configured_native_target_replaces_stale_alias() {
    let (temp_repo, script) = real_cargo_repo();
    let host = rust_host();
    let cargo_dir = temp_repo.path().join(".cargo");
    fs::create_dir_all(&cargo_dir).expect("failed to create real Cargo config directory");
    fs::write(
        cargo_dir.join("config.toml"),
        format!("[build]\ntarget = \"{host}\"\n"),
    )
    .expect("failed to write explicit native Cargo target");
    let alias = write_stale_alias(temp_repo.path());

    let output = run_real_build_script(&script, temp_repo.path(), None);
    assert_success(&output, "real Cargo configured native target");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(".ny-submission-build.") && stdout.contains(&format!("/{host}/release/ny")),
        "real Cargo did not report its staged explicit-target artifact: {stdout}"
    );
    let alias_output = Command::new(&alias)
        .output()
        .expect("failed to execute real-Cargo alias");
    assert_success(&alias_output, "real-Cargo alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "fresh-json-artifact"
    );
}

#[test]
fn test_real_cargo_environment_native_target_replaces_stale_alias() {
    let (temp_repo, script) = real_cargo_repo();
    let host = rust_host();
    let alias = write_stale_alias(temp_repo.path());

    let output = run_real_build_script(&script, temp_repo.path(), Some(&host));
    assert_success(&output, "real Cargo CARGO_BUILD_TARGET native target");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(".ny-submission-build.") && stdout.contains(&format!("/{host}/release/ny")),
        "real Cargo did not report its staged CARGO_BUILD_TARGET artifact: {stdout}"
    );
    let alias_output = Command::new(&alias)
        .output()
        .expect("failed to execute environment-target alias");
    assert_success(&alias_output, "environment-target alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "fresh-json-artifact"
    );
}

#[test]
fn test_real_cargo_host_tuple_uses_targeted_artifact() {
    let (temp_repo, script) = real_cargo_repo();
    let host = rust_host();
    let cargo_dir = temp_repo.path().join(".cargo");
    fs::create_dir_all(&cargo_dir).expect("failed to create host-tuple Cargo directory");
    fs::write(
        cargo_dir.join("config.toml"),
        "[build]\ntarget = \"host-tuple\"\n",
    )
    .expect("failed to write real Cargo host-tuple target");
    let alias = write_stale_alias(temp_repo.path());

    let output = run_real_build_script(&script, temp_repo.path(), None);
    assert_success(&output, "real Cargo host-tuple target");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(".ny-submission-build.") && stdout.contains(&format!("/{host}/release/ny")),
        "Cargo's host-tuple target did not report a staged target artifact: {stdout}"
    );
    let alias_output = Command::new(&alias)
        .output()
        .expect("failed to execute host-tuple alias");
    assert_success(&alias_output, "host-tuple alias");
    assert_eq!(
        String::from_utf8_lossy(&alias_output.stdout).trim(),
        "fresh-json-artifact"
    );
}

#[test]
fn test_real_cargo_failure_preserves_diagnostics_status_and_stale_alias() {
    let (temp_repo, script) = real_cargo_repo();
    fs::write(
        temp_repo.path().join("crates/ny-cli/src/main.rs"),
        "fn main( {\n",
    )
    .expect("failed to write deliberate compiler error");
    let alias = write_stale_alias(temp_repo.path());

    let output = run_real_build_script(&script, temp_repo.path(), None);
    assert_eq!(
        output.status.code(),
        Some(101),
        "builder did not preserve Cargo's failure status\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("error:") && stderr.contains("could not compile `ny-cli`"),
        "Cargo's rendered compiler diagnostics were lost: {stderr}"
    );
    let stale_output = Command::new(&alias)
        .output()
        .expect("failed to execute stale alias after Cargo failure");
    assert_success(&stale_output, "stale alias after Cargo failure");
    assert_eq!(
        String::from_utf8_lossy(&stale_output.stdout).trim(),
        "stale-root-artifact"
    );
}

#[test]
fn test_host_tuple_is_normalized_for_native_fp16_and_artifact_selection() {
    let (temp_repo, script) = fake_repo();
    let host = rust_host();
    let artifact_relative = format!("{host}/release/ny");
    let cargo_dir = temp_repo.path().join(".cargo");
    fs::create_dir_all(&cargo_dir).expect("failed to create host-tuple config directory");
    fs::write(
        cargo_dir.join("config.toml"),
        "[build]\ntarget = \"host-tuple\"\n",
    )
    .expect("failed to write host-tuple Cargo target");

    let output = run_build_script_with_options(
        &script,
        temp_repo.path(),
        "root-release",
        None,
        None,
        Some(&artifact_relative),
        None,
    );
    assert_success(&output, "build_submission_binary host-tuple");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read host-tuple fake Cargo environment");
    if native_fp16_injection_expected() {
        assert!(
            cargo_env.contains("RUSTFLAGS=-C target-feature=+fp16\n"),
            "host-tuple did not receive native fp16: {cargo_env}"
        );
    } else {
        assert!(
            cargo_env.contains("RUSTFLAGS=\n"),
            "non-capable host unexpectedly received fp16: {cargo_env}"
        );
    }
}

#[test]
fn test_build_submission_binary_does_not_probe_openssl_or_install_packages() {
    let (temp_repo, script) = fake_repo();

    let build_output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&build_output, "build_submission_binary rustls-only build");

    for marker in [
        "pkg-config-invoked",
        "apt-get-invoked",
        "dnf-invoked",
        "sudo-invoked",
    ] {
        assert!(
            !temp_repo.path().join(marker).exists(),
            "submission builder unexpectedly invoked {marker}"
        );
    }
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(cargo_env.contains("OPENSSL_DIR=\n"));
    assert!(cargo_env.contains("OPENSSL_LIB_DIR=\n"));
    assert!(cargo_env.contains("OPENSSL_INCLUDE_DIR=\n"));
    assert!(cargo_env.contains("CFLAGS=\n"));
}

#[test]
fn test_cross_target_never_inherits_host_aarch64_fp16() {
    let (temp_repo, script) = fake_repo();
    let output = run_build_script_with_target(
        &script,
        temp_repo.path(),
        "root-release",
        None,
        Some("x86_64-unknown-linux-gnu"),
    );
    assert_success(&output, "build_submission_binary cross target");

    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=\n"),
        "cross target inherited host-only flags: {cargo_env}"
    );
}

#[test]
fn test_configured_cross_target_never_inherits_host_aarch64_fp16() {
    let (temp_repo, script) = fake_repo();
    let cargo_config = temp_repo.path().join(".cargo/config.toml");
    fs::create_dir_all(cargo_config.parent().expect("Cargo config parent"))
        .expect("failed to create Cargo config directory");
    fs::write(
        &cargo_config,
        "[build]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write Cargo cross-target config");

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary configured cross target");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=\n"),
        "configured cross target inherited host-only flags: {cargo_env}"
    );
}

#[test]
fn test_toml_equivalent_cross_targets_never_inherit_host_aarch64_fp16() {
    let (temp_repo, script) = fake_repo();
    let cargo_config = temp_repo.path().join(".cargo/config.toml");
    fs::create_dir_all(cargo_config.parent().expect("Cargo config parent"))
        .expect("failed to create Cargo config directory");

    for (label, contents) in [
        (
            "quoted table",
            "[\"build\"]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
        ),
        (
            "spaced table",
            "[ build ]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
        ),
        (
            "quoted dotted key",
            "\"build\".target = \"x86_64-unknown-linux-gnu\"\n",
        ),
    ] {
        fs::write(&cargo_config, contents).expect("failed to write Cargo cross-target config");
        let output = run_build_script(&script, temp_repo.path(), "root-release", None);
        assert_success(
            &output,
            &format!("build_submission_binary {label} cross target"),
        );
        let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
            .expect("failed to read fake cargo environment");
        assert!(
            cargo_env.contains("RUSTFLAGS=\n"),
            "{label} cross target inherited host-only flags: {cargo_env}"
        );
    }
}

#[test]
fn test_quoted_single_key_does_not_override_lower_priority_cross_target() {
    let (temp_repo, script) = fake_repo();
    let cargo_home_config = temp_repo.path().join("home/.cargo/config.toml");
    fs::create_dir_all(cargo_home_config.parent().expect("Cargo home parent"))
        .expect("failed to create Cargo home");
    fs::write(
        &cargo_home_config,
        "[build]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write lower-priority Cargo cross target");

    let repository_config = temp_repo.path().join(".cargo/config.toml");
    fs::create_dir_all(
        repository_config
            .parent()
            .expect("repository config parent"),
    )
    .expect("failed to create repository Cargo config directory");
    // Cargo 1.95 treats this as one unrelated top-level key.  It is not the
    // dotted `"build".target`, so the lower-priority x86 target remains live.
    fs::write(
        &repository_config,
        "\"build.target\" = \"aarch64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write quoted single-key Cargo config");

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary quoted single key");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=\n"),
        "quoted single key overrode Cargo's actual x86 target: {cargo_env}"
    );
}

#[test]
fn test_quoted_nonbuild_table_does_not_override_lower_priority_cross_target() {
    let (temp_repo, script) = fake_repo();
    let cargo_home_config = temp_repo.path().join("home/.cargo/config.toml");
    fs::create_dir_all(cargo_home_config.parent().expect("Cargo home parent"))
        .expect("failed to create Cargo home");
    fs::write(
        &cargo_home_config,
        "[build]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write lower-priority Cargo cross target");

    let repository_config = temp_repo.path().join(".cargo/config.toml");
    fs::create_dir_all(
        repository_config
            .parent()
            .expect("repository config parent"),
    )
    .expect("failed to create repository Cargo config directory");
    // Whitespace inside a quoted TOML key is data, not insignificant syntax.
    fs::write(
        &repository_config,
        "[\"b u i l d\"]\ntarget = \"aarch64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write quoted non-build Cargo table");

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary quoted non-build table");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=\n"),
        "quoted non-build table overrode Cargo's actual x86 target: {cargo_env}"
    );
}

#[test]
#[cfg(all(
    feature = "native-arm-conformance",
    target_arch = "aarch64",
    target_os = "linux"
))]
fn test_capable_native_aarch64_fixture_reaches_fp16_injection() {
    assert!(
        native_fp16_injection_expected(),
        "native FP16 conformance requires aarch64-unknown-linux-gnu, \
         /proc/cpuinfo asimdhp support, and no pre-existing rustc fp16 cfg"
    );

    let (temp_repo, script) = fake_repo();
    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary native fp16 exercise");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=-C target-feature=+fp16\n"),
        "capable native AArch64 fixture did not exercise fp16 injection: {cargo_env}"
    );
}

#[test]
fn test_legacy_config_wins_over_config_toml_for_cross_target() {
    let (temp_repo, script) = fake_repo();
    let cargo_dir = temp_repo.path().join(".cargo");
    fs::create_dir_all(&cargo_dir).expect("failed to create Cargo config directory");
    fs::write(
        cargo_dir.join("config.toml"),
        "[build]\ntarget = \"aarch64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write lower-precedence Cargo config.toml");
    fs::write(
        cargo_dir.join("config"),
        "[build]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write authoritative legacy Cargo config");

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary legacy Cargo config");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=\n"),
        "legacy cross-target config inherited host-only flags: {cargo_env}"
    );
}

#[test]
fn test_unresolved_included_config_never_gets_host_aarch64_fp16() {
    let (temp_repo, script) = fake_repo();
    let cargo_dir = temp_repo.path().join(".cargo");
    fs::create_dir_all(&cargo_dir).expect("failed to create Cargo config directory");
    fs::write(
        cargo_dir.join("config.toml"),
        "include = [\"cross.toml\"]\n",
    )
    .expect("failed to write including Cargo config");
    fs::write(
        cargo_dir.join("cross.toml"),
        "[build]\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    )
    .expect("failed to write included cross-target config");

    let output = run_build_script(&script, temp_repo.path(), "root-release", None);
    assert_success(&output, "build_submission_binary included Cargo config");
    let cargo_env = fs::read_to_string(temp_repo.path().join("cargo-env.txt"))
        .expect("failed to read fake cargo environment");
    assert!(
        cargo_env.contains("RUSTFLAGS=\n"),
        "unresolved included config inherited host-only flags: {cargo_env}"
    );
}
