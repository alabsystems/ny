// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cargo-owned black-box contracts for the VNN-COMP release pipeline
//! (docs/RELEASE.md):
//!
//! * `vnncomp_scripts/verify_release.py` admits a well-formed
//!   `dist/bin/<target>/` artifact (at its commit, or on an artifact-only
//!   descendant, in a full or a SHALLOW clone) and refuses every tampered
//!   field, each with its own reason;
//! * the release path of `install_tool.sh` verifies, makes the binary name
//!   itself, runs both self-test instances, binds a `release` receipt and only
//!   then installs -- and fails closed, without a source-build fallback, on a
//!   verifier refusal, an identity mismatch or a wrong self-test answer;
//! * `submission_binary_receipt.sh validate` accepts that receipt and refuses it
//!   once its MANIFEST changes;
//! * the scored `run_instance.sh` hands the verifier no `NY_*`/`AY_*` variable
//!   (with a positive control showing the canary is otherwise visible);
//! * a shallow clone (`git clone --depth 1`, no ny_commit object) is bound by
//!   content -- HEAD's tree minus the release files is MANIFEST's
//!   `ny_commit_tree`, and MANIFEST's `ny_commit_object` hashes to ny_commit
//!   and names that tree -- by the verifier and by the receipt alike, and a
//!   clone owned by another user (simulated with git's own
//!   `GIT_TEST_ASSUME_DIFFERENT_OWNER`) is read through a `safe.directory`
//!   naming that clone and nothing else.
//!
//! Python and bash are the programs under test; the fixtures (a tiny ELF64
//! image, a stub binary, a git repository) are synthesized here. Scratch lives
//! under Cargo's per-target tmp directory and is never recursively deleted.

use std::fs;
use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const AY: &str = "0123456789abcdef0123456789abcdef01234567";
const TRUSTC: &str = "1183ee6ed47da889033c1a6c47221abc4e57c4be";
const X86: &str = "x86_64-unknown-linux-gnu";
const UNSEALED: &str = "ny.vnncomp.unsealed.v1|status=unsealed|";
const SEALED: &str = "ny.vnncomp.build.v1|status=sealed|";
const FD_FATBIN: &[u8] = b"\x50\xed\x55\xba fixture fd_search fatbin bytes \x00\x01\x02";
const WAVE_FATBIN: &[u8] = b"\x50\xed\x55\xba fixture wave_search fatbin bytes \x03\x04";
const FIXTURE_ID: &str = "NY_BUILD_ID_V1 fixture-build-id";
const FIXTURE_STAMP: &str = "NY_BUILD_STAMP_V1 fixture-build-stamp";
const FIXTURE_PRESET: &str = "configs/vnncomp26/fixture.yaml";
const AARCH64: &str = "aarch64-unknown-linux-gnu";
/// MANIFEST `release_paths`: every path a release commit may add, all targets.
const RELEASE_PATHS_CSV: &str = "dist/bin/aarch64-unknown-linux-gnu/MANIFEST,\
dist/bin/aarch64-unknown-linux-gnu/ny.sha256,dist/bin/aarch64-unknown-linux-gnu/ny.xz,\
dist/bin/x86_64-unknown-linux-gnu/MANIFEST,dist/bin/x86_64-unknown-linux-gnu/ny.sha256,\
dist/bin/x86_64-unknown-linux-gnu/ny.xz";
/// git's own switch for "this repository is owned by someone else" (setup.c
/// `ensure_valid_ownership`): the different-owner refusal, without sudo.
const DIFFERENT_OWNER: (&str, &str) = ("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1");
/// The tree-rule refusal both the verifier and the receipt print.
const TREE_REFUSAL: &str = "tree without the release files is";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scratch(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time follows the Unix epoch")
        .as_nanos();
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "release-pipeline-{label}-{}-{nonce}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
    path
}

fn write(path: &Path, contents: &[u8]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("mkdir {}: {error}", parent.display()));
    }
    fs::write(path, contents).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
}

fn write_executable(path: &Path, contents: &str) {
    write(path, contents.as_bytes());
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .unwrap_or_else(|error| panic!("chmod {}: {error}", path.display()));
}

fn copy_repository_file(relative: &str, root: &Path) {
    let source = workspace_root().join(relative);
    let bytes =
        fs::read(&source).unwrap_or_else(|error| panic!("read {}: {error}", source.display()));
    write(&root.join(relative), &bytes);
    let mode = fs::metadata(&source)
        .expect("stat repository file")
        .permissions()
        .mode();
    fs::set_permissions(root.join(relative), fs::Permissions::from_mode(mode)).expect("chmod copy");
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n")
}

fn describe(output: &Output) -> String {
    format!(
        "status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        text(&output.stdout),
        text(&output.stderr)
    )
}

fn run(command: &mut Command) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("spawn {command:?}: {error}"))
}

fn piped(program: &str, arguments: &[&str], input: &[u8]) -> Vec<u8> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("spawn {program}: {error}"));
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(input)
        .expect("feed stdin");
    let output = child.wait_with_output().expect("wait");
    assert!(output.status.success(), "{program} {arguments:?} failed");
    output.stdout
}

fn sha256(bytes: &[u8]) -> String {
    text(&piped("sha256sum", &["-"], bytes))
        .split_whitespace()
        .next()
        .expect("digest")
        .to_string()
}

fn xz(bytes: &[u8]) -> Vec<u8> {
    piped("xz", &["-c", "-6"], bytes)
}

fn git(root: &Path, arguments: &[&str]) -> String {
    let output = run(Command::new("git").arg("-C").arg(root).args(arguments));
    assert!(
        output.status.success(),
        "git {arguments:?}: {}",
        describe(&output)
    );
    text(&output.stdout).trim().to_string()
}

fn commit_all(root: &Path, message: &str) -> String {
    git(root, &["add", "-A"]);
    git(
        root,
        &[
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
    );
    git(root, &["rev-parse", "HEAD"])
}

/// MANIFEST `ny_commit_object`: COMMIT's raw commit object, base64 on one line.
fn commit_object(root: &Path, commit: &str) -> String {
    let output = run(Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["cat-file", "commit", commit]));
    assert!(output.status.success(), "{}", describe(&output));
    text(&piped("base64", &["-w0"], &output.stdout))
        .trim()
        .to_string()
}

/// ENCODED (a `ny_commit_object`) with its `tree` header replaced by TREE: the
/// object a forger would need, which no longer hashes to its commit.
fn forge_tree_header(encoded: &str, tree: &str) -> String {
    let raw = piped("base64", &["-d"], encoded.as_bytes());
    let newline = raw
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("a commit object has a header line");
    let mut forged = format!("tree {tree}").into_bytes();
    forged.extend_from_slice(&raw[newline..]);
    text(&piped("base64", &["-w0"], &forged)).trim().to_string()
}

/// A minimal ELF64-LE image: section headers for .dynstr, .dynamic (DT_NEEDED)
/// and .gnu.version_r, followed by `payload`. Not executable; parseable.
fn fake_elf(machine: u16, needed: &[&str], versions: &[&str], payload: &[u8]) -> Vec<u8> {
    let mut dynstr = vec![0_u8];
    let mut add = |name: &str| -> u32 {
        let offset = u32::try_from(dynstr.len()).expect("small string table");
        dynstr.extend_from_slice(name.as_bytes());
        dynstr.push(0);
        offset
    };
    let needed_offsets: Vec<u32> = needed.iter().map(|name| add(name)).collect();
    let version_offsets: Vec<u32> = versions.iter().map(|name| add(name)).collect();
    while dynstr.len() % 8 != 0 {
        dynstr.push(0);
    }
    let mut dynamic = Vec::new();
    for offset in &needed_offsets {
        dynamic.extend_from_slice(&1_i64.to_le_bytes());
        dynamic.extend_from_slice(&u64::from(*offset).to_le_bytes());
    }
    dynamic.extend_from_slice(&[0_u8; 16]);
    // One Verneed (against the first NEEDED) carrying every version as a Vernaux.
    let mut verneed = Vec::new();
    verneed.extend_from_slice(&1_u16.to_le_bytes());
    verneed.extend_from_slice(
        &u16::try_from(versions.len())
            .expect("few versions")
            .to_le_bytes(),
    );
    verneed.extend_from_slice(&needed_offsets[0].to_le_bytes());
    verneed.extend_from_slice(&16_u32.to_le_bytes());
    verneed.extend_from_slice(&0_u32.to_le_bytes());
    for (index, offset) in version_offsets.iter().enumerate() {
        verneed.extend_from_slice(&0_u32.to_le_bytes());
        verneed.extend_from_slice(&0_u16.to_le_bytes());
        verneed.extend_from_slice(&0_u16.to_le_bytes());
        verneed.extend_from_slice(&offset.to_le_bytes());
        let next: u32 = if index + 1 == version_offsets.len() {
            0
        } else {
            16
        };
        verneed.extend_from_slice(&next.to_le_bytes());
    }
    let dynstr_offset = 64_u64;
    let dynamic_offset = dynstr_offset + dynstr.len() as u64;
    let verneed_offset = dynamic_offset + dynamic.len() as u64;
    let mut body = Vec::new();
    body.extend_from_slice(&dynstr);
    body.extend_from_slice(&dynamic);
    body.extend_from_slice(&verneed);
    body.extend_from_slice(payload);
    while body.len() % 8 != 0 {
        body.push(0);
    }
    let shoff = 64 + body.len() as u64;
    let mut image = vec![0_u8; 64];
    image[..8].copy_from_slice(b"\x7fELF\x02\x01\x01\x00");
    image[16..18].copy_from_slice(&3_u16.to_le_bytes());
    image[18..20].copy_from_slice(&machine.to_le_bytes());
    image[20..24].copy_from_slice(&1_u32.to_le_bytes());
    image[40..48].copy_from_slice(&shoff.to_le_bytes());
    image[52..54].copy_from_slice(&64_u16.to_le_bytes());
    image[54..56].copy_from_slice(&56_u16.to_le_bytes());
    image[58..60].copy_from_slice(&64_u16.to_le_bytes());
    image[60..62].copy_from_slice(&4_u16.to_le_bytes());
    image.extend_from_slice(&body);
    let section = |kind: u32, offset: u64, size: u64, link: u32, info: u32, entsize: u64| {
        let mut header = Vec::with_capacity(64);
        header.extend_from_slice(&0_u32.to_le_bytes());
        header.extend_from_slice(&kind.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.extend_from_slice(&offset.to_le_bytes());
        header.extend_from_slice(&size.to_le_bytes());
        header.extend_from_slice(&link.to_le_bytes());
        header.extend_from_slice(&info.to_le_bytes());
        header.extend_from_slice(&8_u64.to_le_bytes());
        header.extend_from_slice(&entsize.to_le_bytes());
        header
    };
    image.extend_from_slice(&[0_u8; 64]);
    image.extend(section(3, dynstr_offset, dynstr.len() as u64, 0, 0, 0));
    image.extend(section(6, dynamic_offset, dynamic.len() as u64, 1, 0, 16));
    image.extend(section(
        0x6fff_fffe,
        verneed_offset,
        verneed.len() as u64,
        1,
        1,
        0,
    ));
    image
}

/// The mutable description of one release artifact; `write_artifact` renders it.
#[derive(Clone)]
struct Artifact {
    target: String,
    binary: Vec<u8>,
    manifest: Vec<(String, String)>,
    /// When false, `package_sha256` is left as is after the binary changes.
    reseal_package: bool,
    /// When false, `ny.sha256` keeps its previous contents.
    reseal_sidecar: bool,
}

impl Artifact {
    fn set(&mut self, key: &str, value: &str) {
        let entry = self
            .manifest
            .iter_mut()
            .find(|(name, _)| name == key)
            .expect("known manifest key");
        entry.1 = value.to_string();
    }

    fn get(&self, key: &str) -> String {
        self.manifest
            .iter()
            .find(|(name, _)| name == key)
            .expect("known manifest key")
            .1
            .clone()
    }
}

fn payload(commit: &str, ay: &str, fatbins: bool, sealed: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    if fatbins {
        bytes.extend_from_slice(FD_FATBIN);
        bytes.extend_from_slice(b"..");
        bytes.extend_from_slice(WAVE_FATBIN);
    }
    bytes.extend_from_slice(
        format!("..{UNSEALED}..commit={commit}..build.commit={ay}..").as_bytes(),
    );
    if sealed {
        bytes.extend_from_slice(SEALED.as_bytes());
    }
    bytes
}

struct Fixture {
    root: PathBuf,
    commit: String,
    artifact: Artifact,
}

/// A committed source tree plus a self-consistent x86_64 cross-lane artifact.
fn verifier_fixture(label: &str) -> Fixture {
    verifier_fixture_with_message(label, "fixture source")
}

/// `verifier_fixture`, with ny_commit committed under MESSAGE.
fn verifier_fixture_with_message(label: &str, message: &str) -> Fixture {
    let root = scratch(label).join("repo");
    write(&root.join(".gitignore"), b"dist/\ntarget/\n");
    write(
        &root.join("Cargo.lock"),
        format!(
            "version = 4\n\n[[package]]\nname = \"ay-milp\"\nversion = \"0.1.0\"\n\
             source = \"git+https://github.com/alabsystems/ay.git?rev={AY}#{AY}\"\n"
        )
        .as_bytes(),
    );
    let kernels = root.join("crates/ny-cuda/kernels");
    write(&kernels.join("fd_search.fatbin"), FD_FATBIN);
    write(&kernels.join("wave_search.fatbin"), WAVE_FATBIN);
    write(
        &kernels.join("search_kernels.manifest"),
        format!(
            "# fixture\nformat=2\nimage=fd_search.fatbin\nimage_sha256={}\nwave_image=wave_search.fatbin\n\
             wave_image_sha256={}\n",
            sha256(FD_FATBIN),
            sha256(WAVE_FATBIN)
        )
        .as_bytes(),
    );
    write(
        &root.join("crates/ny-core/src/lib.rs"),
        b"// fixture source\n",
    );
    copy_repository_file("scripts/release/make_submission.sh", &root);
    git(&root, &["init", "-q"]);
    let commit = commit_all(&root, message);

    let binary = fake_elf(
        62,
        &["libstdc++.so.6", "libc.so.6"],
        &["GLIBC_2.34", "GLIBC_2.39", "GLIBCXX_3.4.31"],
        &payload(&commit, AY, true, false),
    );
    let stamp = "a".repeat(64);
    let common = format!(
        "version=0.5.1 features=mip,cuda backend=cuda+wgpu markers=fixture-v9 commit={commit} dirty=false \
         revision=1 source=clean profile=release target={X86} rustc=1.99.0-dev-{} stamp={stamp}",
        &TRUSTC[..9]
    );
    let digest = |seed: char| seed.to_string().repeat(64);
    let manifest: Vec<(String, String)> = [
        ("schema", "ny-release-manifest-v1".to_string()),
        ("target", X86.to_string()),
        ("build_lane", "cross-build-std".to_string()),
        ("build_host", "aarch64-unknown-linux-gnu".to_string()),
        ("features", "mip,cuda".to_string()),
        ("ny_commit", commit.clone()),
        ("ny_tree", "clean".to_string()),
        (
            "ny_commit_tree",
            git(&root, &["rev-parse", &format!("{commit}^{{tree}}")]),
        ),
        ("ny_commit_object", commit_object(&root, &commit)),
        ("release_paths", RELEASE_PATHS_CSV.to_string()),
        (
            "cargo_lock_sha256",
            sha256(&fs::read(root.join("Cargo.lock")).expect("lock")),
        ),
        ("ay_lock_commit", AY.to_string()),
        ("trustc_commit", TRUSTC.to_string()),
        ("trustc_version_sha256", digest('b')),
        ("trustc_sha256", digest('c')),
        ("targo_version_sha256", digest('d')),
        ("std_source", format!("build-std:{}", digest('e'))),
        ("cross_sysroot_sha256", digest('f')),
        ("linker_sha256", digest('1')),
        ("onnxruntime_static_sha256", digest('2')),
        ("binary_sha256", String::new()),
        ("binary_size", String::new()),
        ("package", "ny.xz".to_string()),
        ("package_sha256", String::new()),
        (
            "build_id",
            format!("NY_BUILD_ID_V1 {common} exe=BINARY id={}", digest('3')),
        ),
        ("build_stamp", format!("NY_BUILD_STAMP_V1 {common}")),
        ("vnncomp_build_provenance", UNSEALED.to_string()),
        ("needed", "libstdc++.so.6,libc.so.6".to_string()),
        ("max_glibc", "2.39".to_string()),
        ("max_glibcxx", "3.4.31".to_string()),
        ("cuda_runtime", "dlopen".to_string()),
        (
            "cuda_fatbin_manifest_sha256",
            sha256(&fs::read(kernels.join("search_kernels.manifest")).expect("kernel manifest")),
        ),
        (
            "cuda_fatbins",
            format!(
                "fd_search.fatbin:{},wave_search.fatbin:{}",
                sha256(FD_FATBIN),
                sha256(WAVE_FATBIN)
            ),
        ),
        (
            "self_identified_by",
            "qemu-user:qemu-x86_64:Haswell-v4".to_string(),
        ),
        (
            "make_submission_sha256",
            sha256(&fs::read(root.join("scripts/release/make_submission.sh")).expect("builder")),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect();
    let artifact = Artifact {
        target: X86.to_string(),
        binary,
        manifest,
        reseal_package: true,
        reseal_sidecar: true,
    };
    Fixture {
        root,
        commit,
        artifact,
    }
}

/// Render the artifact into dist/bin/<target>/. Binary-derived fields are
/// recomputed unless the artifact says otherwise.
fn write_artifact(root: &Path, artifact: &mut Artifact) {
    let directory = root.join("dist/bin").join(&artifact.target);
    fs::create_dir_all(&directory).expect("artifact dir");
    let binary_sha = sha256(&artifact.binary);
    if artifact.get("binary_sha256").is_empty() || artifact.reseal_package {
        let id = artifact.get("build_id");
        if id.contains("exe=BINARY") || artifact.reseal_package {
            let exe_start = id.find(" exe=").expect("exe field");
            let tail = &id[exe_start + 5..];
            let after = tail.find(' ').map_or("", |end| &tail[end..]);
            artifact.set(
                "build_id",
                &format!("{} exe={binary_sha}{after}", &id[..exe_start]),
            );
        }
        artifact.set("binary_sha256", &binary_sha);
        artifact.set("binary_size", &artifact.binary.len().to_string());
    }
    let package = xz(&artifact.binary);
    if artifact.get("package_sha256").is_empty() || artifact.reseal_package {
        artifact.set("package_sha256", &sha256(&package));
    }
    write(&directory.join("ny.xz"), &package);
    if artifact.reseal_sidecar || !directory.join("ny.sha256").exists() {
        write(
            &directory.join("ny.sha256"),
            format!("{}  ny\n", artifact.get("binary_sha256")).as_bytes(),
        );
    }
    let mut manifest = String::new();
    for (key, value) in &artifact.manifest {
        manifest.push_str(key);
        manifest.push('=');
        manifest.push_str(value);
        manifest.push('\n');
    }
    write(&directory.join("MANIFEST"), manifest.as_bytes());
}

fn verify(root: &Path, target: &str) -> (Output, PathBuf) {
    verify_with(root, target, &[])
}

fn verify_with(root: &Path, target: &str, settings: &[(&str, &str)]) -> (Output, PathBuf) {
    let output_path = root.join("target").join(format!(
        "verified-{}",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(output_path.parent().expect("parent")).expect("target dir");
    let mut command = Command::new("python3");
    command
        .args(["-I", "-B"])
        .arg(workspace_root().join("vnncomp_scripts/verify_release.py"))
        .arg("--repo-root")
        .arg(root)
        .args(["--target", target])
        .arg("--output")
        .arg(&output_path);
    for (key, value) in settings {
        command.env(key, value);
    }
    (run(&mut command), output_path)
}

/// Force-add whichever of `dist/bin/<target>/{MANIFEST,ny.xz,ny.sha256}`
/// exist and commit them: the artifact-only descendant of docs/RELEASE.md
/// step 5.
fn commit_release_files(root: &Path, target: &str, message: &str) -> String {
    for name in ["MANIFEST", "ny.xz", "ny.sha256"] {
        let relative = format!("dist/bin/{target}/{name}");
        if root.join(&relative).exists() {
            git(root, &["add", "-f", &relative]);
        }
    }
    git(
        root,
        &[
            "-c",
            "user.name=f",
            "-c",
            "user.email=f@invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
    );
    git(root, &["rev-parse", "HEAD"])
}

/// `git clone` of SOURCE's HEAD, `--depth 1` when SHALLOW (over file://: a
/// plain local path ignores --depth).
fn clone_of(source: &Path, label: &str, shallow: bool) -> PathBuf {
    let destination = scratch(label).join("clone");
    let mut command = Command::new("git");
    command.args(["clone", "-q"]);
    if shallow {
        command.args(["--depth", "1"]);
    }
    command
        .arg(format!("file://{}", source.display()))
        .arg(&destination);
    let output = run(&mut command);
    assert!(output.status.success(), "{}", describe(&output));
    destination
}

/// The shallow path is exercised only if the clone really lacks ny_commit.
fn assert_shallow_without(clone: &Path, ny_commit: &str) {
    assert_eq!(
        git(clone, &["rev-parse", "--is-shallow-repository"]),
        "true"
    );
    let probe = run(Command::new("git").arg("-C").arg(clone).args([
        "cat-file",
        "-e",
        &format!("{ny_commit}^{{commit}}"),
    ]));
    assert!(
        !probe.status.success(),
        "the shallow clone unexpectedly holds ny_commit {ny_commit}"
    );
    assert_ne!(git(clone, &["rev-parse", "HEAD"]), ny_commit);
}

/// A verifier fixture whose release files are committed on top of ny_commit:
/// first the x86_64 files, then (a second release commit) an aarch64 MANIFEST,
/// so a depth-1 clone is TWO commits away from ny_commit.
fn released_verifier_fixture(label: &str) -> Fixture {
    let mut fixture = verifier_fixture(label);
    write_artifact(&fixture.root, &mut fixture.artifact);
    commit_release_files(&fixture.root, X86, "release: x86_64 files");
    write(
        &fixture.root.join(format!("dist/bin/{AARCH64}/MANIFEST")),
        b"schema=ny-release-manifest-v1\n",
    );
    commit_release_files(&fixture.root, AARCH64, "release: aarch64 files");
    fixture
}

#[test]
fn release_verifier_accepts_a_well_formed_artifact() {
    let mut fixture = verifier_fixture("accept");
    write_artifact(&fixture.root, &mut fixture.artifact);
    let (output, staged) = verify(&fixture.root, X86);
    assert!(output.status.success(), "{}", describe(&output));
    let stdout = text(&output.stdout);
    assert!(
        stdout.contains(&format!(
            "NY_RELEASE_VERIFY_OK target={X86} lane=cross-build-std"
        )),
        "{stdout}"
    );
    assert!(stdout.contains("source_binding=head"), "{stdout}");
    assert_eq!(
        fs::read(&staged).expect("staged binary"),
        fixture.artifact.binary
    );
}

#[test]
fn release_verifier_accepts_an_artifact_only_descendant_commit() {
    let mut fixture = verifier_fixture("descendant");
    write_artifact(&fixture.root, &mut fixture.artifact);
    let directory = format!("dist/bin/{X86}");
    for name in ["MANIFEST", "ny.xz", "ny.sha256"] {
        git(
            &fixture.root,
            &["add", "-f", &format!("{directory}/{name}")],
        );
    }
    git(
        &fixture.root,
        &[
            "-c",
            "user.name=f",
            "-c",
            "user.email=f@invalid",
            "commit",
            "-q",
            "-m",
            "release files",
        ],
    );
    assert_ne!(git(&fixture.root, &["rev-parse", "HEAD"]), fixture.commit);
    let (output, _) = verify(&fixture.root, X86);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(text(&output.stdout).contains("source_binding=artifact-only-descendant"));

    // ... and a descendant that also changes source is refused.
    write(
        &fixture.root.join("crates/ny-core/src/lib.rs"),
        b"// changed after the build\n",
    );
    commit_all(&fixture.root, "source change on top of the release");
    let (output, _) = verify(&fixture.root, X86);
    assert!(
        !output.status.success(),
        "a source change after ny_commit must be refused"
    );
    assert!(
        text(&output.stderr).contains("outside dist/bin/<target>/ release files"),
        "{}",
        describe(&output)
    );
}

type Tamper = fn(&mut Fixture);

#[test]
fn release_verifier_refuses_every_tampered_field() {
    let cases: &[(&str, Tamper, &str)] = &[
        (
            "binary flipped, package resealed",
            |f| {
                f.artifact.reseal_package = false;
                let last = f.artifact.binary.len() - 1;
                f.artifact.binary[last] ^= 0xff;
                f.artifact
                    .set("package_sha256", &sha256(&xz(&f.artifact.binary)));
            },
            "does not match binary_sha256",
        ),
        (
            "package not resealed",
            |f| {
                f.artifact.reseal_package = false;
                f.artifact.set("package_sha256", &"0".repeat(64));
            },
            "ny.xz does not match package_sha256",
        ),
        (
            "stale sidecar",
            |f| {
                write(
                    &f.root.join(format!("dist/bin/{X86}/ny.sha256")),
                    format!("{}  ny\n", "9".repeat(64)).as_bytes(),
                );
                f.artifact.reseal_sidecar = false;
            },
            "ny.sha256 is not the exact sha256sum record",
        ),
        (
            "unknown key",
            |f| {
                f.artifact
                    .manifest
                    .push(("relabelled".into(), "yes".into()))
            },
            "unknown key",
        ),
        (
            "duplicate key",
            |f| f.artifact.manifest.push(("target".into(), X86.into())),
            "duplicate key",
        ),
        (
            "missing key",
            |f| f.artifact.manifest.retain(|(key, _)| key != "needed"),
            "missing keys",
        ),
        (
            "sealed record smuggled in",
            |f| {
                f.artifact.binary = fake_elf(
                    62,
                    &["libstdc++.so.6", "libc.so.6"],
                    &["GLIBC_2.34", "GLIBC_2.39", "GLIBCXX_3.4.31"],
                    &payload(&f.commit, AY, true, true),
                );
            },
            "SEALED",
        ),
        (
            "aarch64 bytes under the x86_64 name",
            |f| {
                f.artifact.binary = fake_elf(
                    183,
                    &["libstdc++.so.6", "libc.so.6"],
                    &["GLIBC_2.34", "GLIBC_2.39", "GLIBCXX_3.4.31"],
                    &payload(&f.commit, AY, true, false),
                );
            },
            "e_machine=183",
        ),
        (
            "dirty build",
            |f| {
                let id = f
                    .artifact
                    .get("build_id")
                    .replace("dirty=false", "dirty=true");
                let stamp = f
                    .artifact
                    .get("build_stamp")
                    .replace("dirty=false", "dirty=true");
                f.artifact.set("build_id", &id);
                f.artifact.set("build_stamp", &stamp);
            },
            "dirty='true'",
        ),
        (
            "build_stamp disagrees with build_id",
            |f| {
                let stamp = f
                    .artifact
                    .get("build_stamp")
                    .replace("revision=1", "revision=2");
                f.artifact.set("build_stamp", &stamp);
            },
            "disagrees with build_id",
        ),
        (
            "fatbins not embedded",
            |f| {
                f.artifact.binary = fake_elf(
                    62,
                    &["libstdc++.so.6", "libc.so.6"],
                    &["GLIBC_2.34", "GLIBC_2.39", "GLIBCXX_3.4.31"],
                    &payload(&f.commit, AY, false, false),
                );
            },
            "does not embed the committed fd_search.fatbin",
        ),
        (
            "feature tier",
            |f| f.artifact.set("features", "mip"),
            "'features' mismatch",
        ),
        (
            "native lane claimed for a cross build",
            |f| f.artifact.set("build_lane", "native"),
            "native build_lane",
        ),
        (
            "CUDA linked at load time",
            |f| {
                f.artifact.binary = fake_elf(
                    62,
                    &["libcuda.so.1", "libc.so.6"],
                    &["GLIBC_2.34", "GLIBC_2.39", "GLIBCXX_3.4.31"],
                    &payload(&f.commit, AY, true, false),
                );
                f.artifact.set("needed", "libcuda.so.1,libc.so.6");
            },
            "dlopen'd",
        ),
        (
            "glibc floor understated",
            |f| f.artifact.set("max_glibc", "2.35"),
            "'max_glibc' mismatch",
        ),
        (
            "AY build commit",
            |f| {
                f.artifact.binary = fake_elf(
                    62,
                    &["libstdc++.so.6", "libc.so.6"],
                    &["GLIBC_2.34", "GLIBC_2.39", "GLIBCXX_3.4.31"],
                    &payload(&f.commit, &"7".repeat(40), true, false),
                );
            },
            "AY build.commit",
        ),
        (
            "Cargo.lock changed",
            |f| {
                let mut lock = fs::read(f.root.join("Cargo.lock")).expect("lock");
                lock.extend_from_slice(b"\n");
                write(&f.root.join("Cargo.lock"), &lock);
            },
            "'cargo_lock_sha256' mismatch",
        ),
        (
            "tracked source modified",
            |f| write(&f.root.join("crates/ny-core/src/lib.rs"), b"// edited\n"),
            "tracked files are modified",
        ),
        (
            "untracked source",
            |f| write(&f.root.join("crates/ny-core/src/extra.rs"), b"// new\n"),
            "untracked files under package roots",
        ),
        (
            "builder replaced",
            |f| f.artifact.set("make_submission_sha256", &"4".repeat(64)),
            "'make_submission_sha256' mismatch",
        ),
        (
            "ny_commit_tree is not ny_commit's tree",
            |f| f.artifact.set("ny_commit_tree", &"5".repeat(40)),
            "is not the tree of ny_commit",
        ),
        (
            "ny_commit_object is another object",
            |f| f.artifact.set("ny_commit_object", "dHJlZSAwCg=="),
            "MANIFEST ny_commit_object hashes to",
        ),
        (
            "ny_commit_object is not base64",
            |f| f.artifact.set("ny_commit_object", "not*base64"),
            "'ny_commit_object' is not base64",
        ),
        (
            // "tree 0\n" is canonically dHJlZSAwCg==; the same bytes spelled
            // with a non-zero padding bit.
            "ny_commit_object is non-canonical base64",
            |f| f.artifact.set("ny_commit_object", "dHJlZSAwCh=="),
            "'ny_commit_object' is not canonical base64",
        ),
        (
            "release_paths widened to cover a source file",
            |f| {
                f.artifact.set(
                    "release_paths",
                    &format!("{RELEASE_PATHS_CSV},crates/ny-core/src/lib.rs"),
                )
            },
            "'release_paths' mismatch",
        ),
        (
            "unsealed provenance claim changed",
            |f| f.artifact.set("vnncomp_build_provenance", SEALED),
            "'vnncomp_build_provenance' mismatch",
        ),
    ];
    for (name, tamper, expected) in cases {
        let mut fixture = verifier_fixture("tamper");
        write_artifact(&fixture.root, &mut fixture.artifact);
        tamper(&mut fixture);
        let mut artifact = fixture.artifact.clone();
        write_artifact(&fixture.root, &mut artifact);
        let (output, staged) = verify(&fixture.root, X86);
        assert!(
            !output.status.success(),
            "{name}: must be refused\n{}",
            describe(&output)
        );
        assert!(
            text(&output.stderr).contains(expected),
            "{name}: expected '{expected}'\n{}",
            describe(&output)
        );
        assert!(
            !staged.exists(),
            "{name}: a refused artifact must leave no staged binary"
        );
    }

    // A symlinked artifact directory is refused before anything is read.
    let mut fixture = verifier_fixture("symlink");
    write_artifact(&fixture.root, &mut fixture.artifact);
    let real = fixture.root.join("dist/bin").join(format!("{X86}.real"));
    fs::rename(fixture.root.join("dist/bin").join(X86), &real).expect("move artifact");
    std::os::unix::fs::symlink(&real, fixture.root.join("dist/bin").join(X86)).expect("symlink");
    let (output, _) = verify(&fixture.root, X86);
    assert!(!output.status.success());
    assert!(
        text(&output.stderr).contains("must be a real directory"),
        "{}",
        describe(&output)
    );
}

// ---------------------------------------------------------------------------
// install_tool.sh's release path, with a stub verifier and a stub binary.
// ---------------------------------------------------------------------------

const STUB_VERIFIER: &str = r#"#!/usr/bin/env python3
import argparse, lzma, os, sys
parser = argparse.ArgumentParser()
for flag in ("--repo-root", "--target", "--output"):
    parser.add_argument(flag, required=True)
arguments = parser.parse_args()
with open(os.environ["FAKE_COMMAND_LOG"], "a") as log:
    log.write(f"verify-release {arguments.target}\n")
code = int(os.environ.get("FAKE_VERIFY_EXIT", "0"))
if code:
    print("ERROR: release artifact verification failed: stub refusal", file=sys.stderr)
    raise SystemExit(code)
archive = os.path.join(arguments.repo_root, "dist", "bin", arguments.target, "ny.xz")
with open(arguments.output, "wb") as staged:
    staged.write(lzma.decompress(open(archive, "rb").read()))
print("stub verifier: ok")
"#;

const STUB_BINARY: &str = r#"#!/bin/bash
echo "exec $*" >> "${FAKE_COMMAND_LOG:-/dev/null}"
case "$1" in
    --version) echo "ny 0.5.1-fixture" ;;
    --build-id) echo "${FAKE_BUILD_ID}" ;;
    --build-stamp) echo "${FAKE_BUILD_STAMP}" ;;
    vnncomp)
        env | cut -d= -f1 | grep -E '^(NY_|AY_|MIMALLOC_)' > "$6.env"
        verdict=sat
        case "$5" in *_unsat.vnnlib) verdict=unsat ;; esac
        if [ -n "${FAKE_WRONG:-}" ]; then
            if [ "${verdict}" = sat ]; then verdict=unsat; else verdict=sat; fi
        fi
        echo "${verdict}" > "$6"
        ;;
    *) exit 3 ;;
esac
"#;

struct InstallFixture {
    root: PathBuf,
    log: PathBuf,
    fake_bin: PathBuf,
}

fn install_fixture(label: &str, sealed_triplet_too: bool) -> InstallFixture {
    let base = scratch(label);
    let root = base.join("tool");
    for relative in [
        "install_tool.sh",
        "run_instance.sh",
        "vnncomp_scripts/run_instance.sh",
        "vnncomp_scripts/submission_binary_receipt.sh",
        "vnncomp_scripts/selftest/abs_relu.onnx",
        "vnncomp_scripts/selftest/abs_relu_unsat.vnnlib",
        "vnncomp_scripts/selftest/abs_relu_sat.vnnlib",
    ] {
        copy_repository_file(relative, &root);
    }
    write_executable(
        &root.join("vnncomp_scripts/verify_release.py"),
        STUB_VERIFIER,
    );
    write_executable(
        &root.join("vnncomp_scripts/verify_prebuilt.py"),
        "#!/usr/bin/env python3\nimport os\nopen(os.environ['FAKE_COMMAND_LOG'], 'a').write('verify-prebuilt\\n')\nraise SystemExit(1)\n",
    );
    write_executable(
        &root.join("vnncomp_scripts/build_submission_binary.sh"),
        "#!/bin/bash\necho source-fallback >> \"${FAKE_COMMAND_LOG}\"\n",
    );
    write(&root.join(".gitignore"), b"dist/\ntarget/\n");
    write(
        &root.join("Cargo.lock"),
        format!("version = 4\n\n[[package]]\nname = \"ay-milp\"\nsource = \"git+https://github.com/alabsystems/ay.git?rev={AY}#{AY}\"\n")
            .as_bytes(),
    );
    // A tracked preset: ny reads configs/ at run time, so the receipt must
    // refuse once it no longer matches ny_commit.
    write(&root.join(FIXTURE_PRESET), b"bab: {timeout: 60}\n");
    git(&root, &["init", "-q"]);
    let commit = commit_all(&root, "install fixture");

    let binary = STUB_BINARY.as_bytes();
    let directory = root.join("dist/bin").join(X86);
    write(&directory.join("ny.xz"), &xz(binary));
    write(
        &directory.join("ny.sha256"),
        format!("{}  ny\n", sha256(binary)).as_bytes(),
    );
    write(
        &directory.join("MANIFEST"),
        format!(
            "schema=ny-release-manifest-v1\ntarget={X86}\nfeatures=mip,cuda\nny_commit={commit}\n\
             ny_commit_tree={}\nny_commit_object={}\nrelease_paths={RELEASE_PATHS_CSV}\n\
             cargo_lock_sha256={}\nay_lock_commit={AY}\nbinary_sha256={}\ntrustc_sha256={}\n\
             max_glibc=2.39\nbuild_id={FIXTURE_ID}\nbuild_stamp={FIXTURE_STAMP}\n",
            git(&root, &["rev-parse", "HEAD^{tree}"]),
            commit_object(&root, &commit),
            sha256(&fs::read(root.join("Cargo.lock")).expect("lock")),
            sha256(binary),
            "c".repeat(64)
        )
        .as_bytes(),
    );
    if sealed_triplet_too {
        for name in [
            "ny-x86_64-linux.xz",
            "ny-x86_64-linux.xz.sha256",
            "ny-x86_64-linux.provenance.txt",
        ] {
            write(
                &root.join("dist/bin").join(name),
                b"sealed-triplet-fixture\n",
            );
        }
    }
    write(&root.join("target/release/ny"), b"existing-binary\n");

    let fake_bin = base.join("fake-bin");
    write_executable(
        &fake_bin.join("uname"),
        "#!/bin/bash\ncase \"$1\" in -s) echo Linux ;; -m) echo x86_64 ;; *) exit 2 ;; esac\n",
    );
    write_executable(
        &fake_bin.join("getconf"),
        "#!/bin/bash\nversion=\"$FAKE_GLIBC\"\n[ -n \"$version\" ] || version=2.39\necho \"glibc $version\"\n",
    );
    write_executable(
        &fake_bin.join("apt-get"),
        "#!/bin/bash\necho apt-get >> \"${FAKE_COMMAND_LOG}\"\n",
    );
    write_executable(&fake_bin.join("id"), "#!/bin/bash\necho 0\n");
    write_executable(&fake_bin.join("cargo"), "#!/bin/bash\nexit 99\n");
    write_executable(&fake_bin.join("rustup"), "#!/bin/bash\nexit 0\n");
    InstallFixture {
        root,
        log: base.join("commands.log"),
        fake_bin,
    }
}

fn install(fixture: &InstallFixture, settings: &[(&str, &str)]) -> Output {
    let mut command = Command::new(fixture.root.join("install_tool.sh"));
    command
        .arg("v1")
        .current_dir(&fixture.root)
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", fixture.fake_bin.display()),
        )
        .env("HOME", fixture.root.join("home"))
        .env("FAKE_COMMAND_LOG", &fixture.log)
        .env("FAKE_BUILD_ID", FIXTURE_ID)
        .env("FAKE_BUILD_STAMP", FIXTURE_STAMP);
    for (key, value) in settings {
        command.env(key, value);
    }
    run(&mut command)
}

fn command_log(fixture: &InstallFixture) -> String {
    fs::read_to_string(&fixture.log).unwrap_or_default()
}

fn assert_existing_binary_kept(fixture: &InstallFixture) {
    assert_eq!(
        fs::read(fixture.root.join("target/release/ny")).expect("binary"),
        b"existing-binary\n"
    );
    assert!(
        !fixture.root.join("target/release/ny.receipt").exists(),
        "no receipt may be published"
    );
    assert!(
        !command_log(fixture).contains("apt-get"),
        "no source-build fallback may run"
    );
}

#[test]
fn release_install_verifies_selftests_and_receipts_the_binary() {
    let fixture = install_fixture("install-ok", false);
    let output = install(&fixture, &[]);
    assert!(output.status.success(), "{}", describe(&output));
    let stdout = text(&output.stdout);
    assert!(
        stdout.contains(&format!("NY_RELEASE_INSTALL_OK target={X86} ")),
        "{stdout}"
    );
    assert!(
        stdout.contains("Release self-test abs_relu_unsat: unsat"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Release self-test abs_relu_sat: sat"),
        "{stdout}"
    );
    assert_eq!(
        fs::read(fixture.root.join("target/release/ny")).expect("installed"),
        STUB_BINARY.as_bytes()
    );
    let receipt =
        fs::read_to_string(fixture.root.join("target/release/ny.receipt")).expect("receipt");
    assert!(receipt.contains("source_kind=release\n"), "{receipt}");
    assert!(
        receipt.contains("toolchain_kind=trust-unverified\n"),
        "{receipt}"
    );
    let log = command_log(&fixture);
    assert!(log.contains(&format!("verify-release {X86}")), "{log}");
    assert!(
        log.contains("exec --build-id") && log.contains("exec --build-stamp"),
        "{log}"
    );
    assert_eq!(log.matches("exec vnncomp v1 test ").count(), 2, "{log}");
    assert!(
        !log.contains("apt-get") && !log.contains("source-fallback"),
        "{log}"
    );
    assert!(
        fs::read_dir(fixture.root.join("target/release"))
            .expect("ls")
            .all(|entry| {
                !entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".ny-release")
            }),
        "the staging directory must be cleaned up"
    );

    // The receipt validates while its MANIFEST is untouched ...
    let receipt_helper = fixture
        .root
        .join("vnncomp_scripts/submission_binary_receipt.sh");
    let validate = || {
        run(Command::new("bash")
            .arg(&receipt_helper)
            .arg("validate")
            .arg(fixture.root.join("target/release/ny"))
            .arg(&fixture.root))
    };
    let ok = validate();
    assert!(ok.status.success(), "{}", describe(&ok));
    assert!(
        text(&ok.stdout).contains("source=release@"),
        "{}",
        describe(&ok)
    );

    // ... and the scored entry point hands the binary no NY_/AY_/MIMALLOC_ name,
    // while the same caller environment does reach a direct invocation.
    let onnx = fixture.root.join("vnncomp_scripts/selftest/abs_relu.onnx");
    let vnnlib = fixture
        .root
        .join("vnncomp_scripts/selftest/abs_relu_unsat.vnnlib");
    let scored = fixture.root.join("scored.res");
    let direct = fixture.root.join("direct.res");
    let caller = "NY_RELEASE_CANARY=1 AY_RELEASE_CANARY=1 MIMALLOC_RELEASE_CANARY=1 exec \"$@\"";
    let entry = run(Command::new("bash")
        .args(["-c", caller, "caller"])
        .arg(fixture.root.join("run_instance.sh"))
        .args(["v1", "test"])
        .arg(&onnx)
        .arg(&vnnlib)
        .arg(&scored)
        .arg("60")
        .env("FAKE_COMMAND_LOG", &fixture.log));
    assert!(entry.status.success(), "{}", describe(&entry));
    assert_eq!(fs::read_to_string(&scored).expect("verdict"), "unsat\n");
    let scored_env = fs::read_to_string(fixture.root.join("scored.res.env")).expect("scored env");
    assert_eq!(scored_env, "", "run_instance.sh leaked {scored_env}");
    let control = run(Command::new("bash")
        .args(["-c", caller, "caller"])
        .arg(fixture.root.join("target/release/ny"))
        .args(["vnncomp", "v1", "test"])
        .arg(&onnx)
        .arg(&vnnlib)
        .arg(&direct)
        .arg("60"));
    assert!(control.status.success(), "{}", describe(&control));
    let direct_env = fs::read_to_string(fixture.root.join("direct.res.env")).expect("direct env");
    for name in [
        "NY_RELEASE_CANARY\n",
        "AY_RELEASE_CANARY\n",
        "MIMALLOC_RELEASE_CANARY\n",
    ] {
        assert!(
            direct_env.contains(name),
            "positive control lost {name}: {direct_env}"
        );
    }

    // A changed MANIFEST invalidates the receipt.
    let manifest = fixture.root.join("dist/bin").join(X86).join("MANIFEST");
    let mut bytes = fs::read(&manifest).expect("manifest");
    bytes.extend_from_slice(b"needed=libc.so.6\n");
    write(&manifest, &bytes);
    let refused = validate();
    assert!(
        !refused.status.success(),
        "a changed MANIFEST must invalidate the receipt"
    );
    assert!(
        text(&refused.stderr).contains("gone or has changed"),
        "{}",
        describe(&refused)
    );
}

#[test]
fn release_install_refuses_a_wrong_selftest_answer_without_fallback() {
    let fixture = install_fixture("install-wrong", false);
    let output = install(&fixture, &[("FAKE_WRONG", "1")]);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("self-test WRONG ANSWER: abs_relu_unsat answered sat"),
        "{}",
        describe(&output)
    );
    assert_existing_binary_kept(&fixture);
}

#[test]
fn release_install_refuses_a_binary_that_names_itself_differently() {
    let fixture = install_fixture("install-identity", false);
    let output = install(
        &fixture,
        &[("FAKE_BUILD_ID", "NY_BUILD_ID_V1 some-other-build")],
    );
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("identifies itself as 'NY_BUILD_ID_V1 some-other-build'"),
        "{}",
        describe(&output)
    );
    assert!(
        !command_log(&fixture).contains("exec vnncomp"),
        "no self-test may run after an identity mismatch"
    );
    assert_existing_binary_kept(&fixture);
}

#[test]
fn release_install_refuses_when_the_verifier_refuses() {
    let fixture = install_fixture("install-refused", false);
    let output = install(&fixture, &[("FAKE_VERIFY_EXIT", "1")]);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("verify_release.py refused it"),
        "{}",
        describe(&output)
    );
    assert!(
        !command_log(&fixture).contains("exec "),
        "nothing may execute before verification passes"
    );
    assert_existing_binary_kept(&fixture);
}

#[test]
fn release_install_defers_to_the_sealed_triplet() {
    let fixture = install_fixture("install-sealed", true);
    let output = install(&fixture, &[]);
    assert!(
        !output.status.success(),
        "the stub sealed verifier refuses: {}",
        describe(&output)
    );
    assert!(
        text(&output.stderr).contains("is ignored: the sealed"),
        "{}",
        describe(&output)
    );
    let log = command_log(&fixture);
    assert!(log.contains("verify-prebuilt"), "{log}");
    assert!(
        !log.contains("verify-release"),
        "the release path must not run beside a sealed triplet: {log}"
    );
    assert_existing_binary_kept(&fixture);
}

#[test]
fn release_install_falls_back_to_source_only_for_an_older_glibc() {
    let fixture = install_fixture("install-glibc", false);
    let output = install(&fixture, &[("FAKE_GLIBC", "2.35")]);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("needs GNU glibc >= 2.39; detected '2.35'"),
        "{}",
        describe(&output)
    );
    let log = command_log(&fixture);
    assert!(log.contains(&format!("verify-release {X86}")), "{log}");
    assert!(
        !log.contains("exec "),
        "an incompatible host must not run the binary: {log}"
    );
    assert!(log.contains("source-fallback"), "{log}");
    assert_eq!(
        fs::read(fixture.root.join("target/release/ny")).expect("binary"),
        b"existing-binary\n"
    );
}

// ---------------------------------------------------------------------------
// Review (#ship-pipeline): the runtime receipt binds an UNMODIFIED checkout,
// and the release command lets nothing from outside the checkout steer the
// build.
// ---------------------------------------------------------------------------

fn validate_receipt(fixture: &InstallFixture) -> Output {
    validate_receipt_with(fixture, &[])
}

fn validate_receipt_with(fixture: &InstallFixture, settings: &[(&str, &str)]) -> Output {
    let mut command = Command::new("bash");
    command
        .arg(
            fixture
                .root
                .join("vnncomp_scripts/submission_binary_receipt.sh"),
        )
        .arg("validate")
        .arg(fixture.root.join("target/release/ny"))
        .arg(&fixture.root);
    for (key, value) in settings {
        command.env(key, value);
    }
    run(&mut command)
}

#[test]
fn release_receipt_refuses_a_checkout_edited_after_install() {
    let fixture = install_fixture("install-edit", false);
    let output = install(&fixture, &[]);
    assert!(output.status.success(), "{}", describe(&output));
    let ok = validate_receipt(&fixture);
    assert!(ok.status.success(), "{}", describe(&ok));

    // A tracked preset edited after the install: every scored run would read
    // it, so the receipt must refuse rather than score a different policy.
    let preset = fixture.root.join(FIXTURE_PRESET);
    write(&preset, b"bab: {timeout: 1}\n");
    let refused = validate_receipt(&fixture);
    assert!(
        !refused.status.success(),
        "an edited tracked preset must invalidate the release receipt"
    );
    assert!(
        text(&refused.stderr).contains("tracked files are modified relative to HEAD"),
        "{}",
        describe(&refused)
    );
    git(&fixture.root, &["checkout", "--", FIXTURE_PRESET]);
    let restored = validate_receipt(&fixture);
    assert!(restored.status.success(), "{}", describe(&restored));

    // An untracked file under a package root is an input ny_commit does not
    // cover (the same rule verify_release.py applies at install time) ...
    let extra = fixture.root.join("configs/vnncomp26/extra.yaml");
    write(&extra, b"bab: {timeout: 1}\n");
    let refused = validate_receipt(&fixture);
    assert!(
        !refused.status.success(),
        "an untracked preset must invalidate the release receipt"
    );
    assert!(
        text(&refused.stderr).contains("untracked files under the release package roots"),
        "{}",
        describe(&refused)
    );
    fs::remove_file(&extra).expect("remove the untracked preset");

    // ... while a file outside every package root (an organizer's log next to
    // the clone) is not.
    write(&fixture.root.join("install.log"), b"organizer log\n");
    let unaffected = validate_receipt(&fixture);
    assert!(unaffected.status.success(), "{}", describe(&unaffected));
}

fn quoted_strings(block: &str) -> Vec<String> {
    block
        .split('"')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

fn delimited<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let begin = source
        .find(start)
        .unwrap_or_else(|| panic!("marker {start:?} missing"))
        + start.len();
    let length = source[begin..]
        .find(end)
        .unwrap_or_else(|| panic!("end {end:?} after {start:?} missing"));
    &source[begin..begin + length]
}

#[test]
fn release_package_roots_agree_between_verifier_and_receipt() {
    let verifier = fs::read_to_string(workspace_root().join("vnncomp_scripts/verify_release.py"))
        .expect("verify_release.py");
    let receipt =
        fs::read_to_string(workspace_root().join("vnncomp_scripts/submission_binary_receipt.sh"))
            .expect("submission_binary_receipt.sh");
    let python = quoted_strings(delimited(&verifier, "\nPACKAGE_ROOTS = (", "\n)"));
    let bash = quoted_strings(delimited(&receipt, "\nRELEASE_PACKAGE_ROOTS=(", "\n)"));
    assert!(
        python.iter().any(|root| root == "configs") && python.iter().any(|root| root == "crates"),
        "{python:?}"
    );
    assert_eq!(
        python, bash,
        "install-time (verify_release.py) and run-time (receipt) package roots diverged"
    );
}

/// Exports every override cargo, cc-rs, ort-sys, AY or the scored closure
/// would read, plus the names that must survive, then runs the scrub.
const SCRUB_PROBE: &str = r#"
overrides='CARGO_BUILD_RUSTC_WRAPPER CARGO_BUILD_RUSTC CARGO_PROFILE_RELEASE_OPT_LEVEL
  CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS CARGO_ENCODED_RUSTFLAGS
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER CARGO_TARGET_DIR __CARGO_TESTS_ONLY_SRC_ROOT
  RUSTFLAGS RUSTC_WRAPPER RUSTUP_TOOLCHAIN CC CFLAGS CXXFLAGS_x86_64_unknown_linux_gnu
  TARGET_CC ORT_PREFER_DYNAMIC_LINK ORT_LIB_LOCATION AY_TEST_BUILD_IDENTITY
  AY_SOURCE_GIT_DIRTY NY_VNNCOMP_BUILD_PROVENANCE_V1 NY_BUILD_COMMIT
  MIMALLOC_ARENA_EAGER_COMMIT RAYON_NUM_THREADS RUSTC_BOOTSTRAP'
keepers='CARGO_HOME NY_BOX_WAIT_LOCK RUSTUP_HOME HOME'
for name in $overrides $keepers; do export "$name=probe-value"; done
scrub_build_environment
overrides_checked=0
keepers_checked=0
for name in $overrides; do
    overrides_checked=$((overrides_checked + 1))
    if printenv "$name" >/dev/null; then echo "SCRUB_LEAK $name"; fi
done
for name in $keepers; do
    keepers_checked=$((keepers_checked + 1))
    printenv "$name" >/dev/null || echo "SCRUB_LOST $name"
done
echo "SCRUB_CHECKED overrides=$overrides_checked keepers=$keepers_checked"
"#;

/// The `# BEGIN/END release-build-hygiene` block of make_submission.sh: shell
/// functions only, so it can be sourced and exercised on its own.
fn hygiene_script(base: &Path) -> PathBuf {
    let script = fs::read_to_string(workspace_root().join("scripts/release/make_submission.sh"))
        .expect("make_submission.sh");
    // From the start of the BEGIN line (a comment) to the start of the END
    // line: whole lines only, so the block sources as written.
    let begin = script
        .find("# BEGIN release-build-hygiene")
        .expect("BEGIN release-build-hygiene marker");
    let end = script
        .find("# END release-build-hygiene")
        .expect("END release-build-hygiene marker");
    assert!(begin < end, "hygiene markers out of order");
    let block = &script[begin..end];
    assert!(
        block.contains("scrub_build_environment()")
            && block.contains("external_cargo_configs()")
            && block.contains("git_dependency_checkout_state()")
            && block.contains("release_commit_tree()")
            && block.contains("\nRELEASE_PATHS=("),
        "the hygiene block lost a function or the release-path list"
    );
    let path = base.join("hygiene.sh");
    write(&path, block.as_bytes());
    path
}

fn hygiene(script: &Path, body: &str, settings: &[(&str, &str)]) -> Output {
    let mut command = Command::new("bash");
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .args([
            "-c",
            &format!(
                "set -uo pipefail\nsource \"$1\" || exit 96\n\
                 declare -F scrub_build_environment external_cargo_configs \
                 git_dependency_checkout_state release_commit_tree >/dev/null || exit 97\n{body}"
            ),
            "hygiene",
        ])
        .arg(script);
    for (key, value) in settings {
        command.env(key, value);
    }
    run(&mut command)
}

#[test]
fn release_build_hygiene_scrubs_overrides_and_refuses_outside_inputs() {
    let base = scratch("hygiene");
    let script = hygiene_script(&base);

    // 1. Every caller-side build, profile, compiler, flag or policy override
    //    is removed; CARGO_HOME, the box's own knobs and unrelated names
    //    survive. The probe is a shell fixture: it exports each name, runs the
    //    scrub, and reports any override still set (SCRUB_LEAK) or keeper lost
    //    (SCRUB_LOST), so the variable names live in shell text only.
    let output = hygiene(&script, SCRUB_PROBE, &[("HOME", "/kept/home")]);
    assert!(output.status.success(), "{}", describe(&output));
    let report = text(&output.stdout);
    assert!(
        !report.contains("SCRUB_LEAK") && !report.contains("SCRUB_LOST"),
        "the release build scrub is wrong:\n{report}"
    );
    assert!(
        report.contains("SCRUB_CHECKED overrides=24 keepers=4"),
        "the probe did not run every check:\n{}",
        describe(&output)
    );

    // 2. Cargo configuration outside the checkout (CARGO_HOME, any ancestor)
    //    is reported; the checkout's own .cargo/config.toml is not.
    let checkout = base.join("outer/checkout");
    write(&checkout.join(".cargo/config.toml"), b"[net]\n");
    let cargo_home = base.join("cargo-home");
    fs::create_dir_all(&cargo_home).expect("cargo home");
    let body = format!("external_cargo_configs '{}'", checkout.display());
    let home = base.join("home").display().to_string();
    let clean = hygiene(
        &script,
        &body,
        &[
            ("CARGO_HOME", &cargo_home.display().to_string()),
            ("HOME", &home),
        ],
    );
    assert!(clean.status.success(), "{}", describe(&clean));
    let clean_list = text(&clean.stdout);
    assert!(
        !clean_list.contains(&base.display().to_string()),
        "nothing under the fixture is outside the checkout yet: {clean_list}"
    );
    let ancestor = base.join("outer/.cargo/config.toml");
    write(
        &ancestor,
        b"[build]\nrustflags = [\"-Ctarget-cpu=native\"]\n",
    );
    let home_config = cargo_home.join("config.toml");
    write(&home_config, b"[profile.release]\nopt-level = 0\n");
    let dirty = hygiene(
        &script,
        &body,
        &[
            ("CARGO_HOME", &cargo_home.display().to_string()),
            ("HOME", &home),
        ],
    );
    let dirty_list = text(&dirty.stdout);
    for expected in [&ancestor, &home_config] {
        assert!(
            dirty_list
                .lines()
                .any(|line| line == expected.display().to_string()),
            "{} must be reported: {dirty_list}",
            expected.display()
        );
    }
    assert!(
        !dirty_list.contains(&checkout.join(".cargo").display().to_string()),
        "the checkout's own config is part of ny_commit: {dirty_list}"
    );
    // Without CARGO_HOME, cargo reads $HOME/.cargo (legacy `config` name too).
    let legacy = base.join("home/.cargo/config");
    write(&legacy, b"[build]\n");
    let defaulted = hygiene(&script, &body, &[("HOME", &home)]);
    assert!(
        text(&defaulted.stdout)
            .lines()
            .any(|line| line == legacy.display().to_string()),
        "{}",
        describe(&defaulted)
    );

    // 3. A git dependency's checkout must be exactly the locked commit with no
    //    tracked edit (cargo's `.cargo-ok` marker is untracked and allowed).
    let dependency = base.join("dep");
    write(
        &dependency.join("crates/dep/src/lib.rs"),
        b"pub fn f() {}\n",
    );
    git(&dependency, &["init", "-q"]);
    let locked = commit_all(&dependency, "dependency");
    write(&dependency.join(".cargo-ok"), b"{\"v\":1}\n");
    let package = dependency.join("crates/dep").display().to_string();
    let state = |commit: &str| {
        hygiene(
            &script,
            &format!("git_dependency_checkout_state '{package}' '{commit}'"),
            &[("HOME", &home)],
        )
    };
    let ok = state(&locked);
    assert!(ok.status.success(), "{}", describe(&ok));
    assert_eq!(
        text(&ok.stdout).trim(),
        fs::canonicalize(&dependency)
            .expect("canonical dependency")
            .display()
            .to_string()
    );
    let wrong = state(&"0".repeat(40));
    assert!(!wrong.status.success(), "{}", describe(&wrong));
    assert!(
        text(&wrong.stdout).contains("not the locked"),
        "{}",
        describe(&wrong)
    );
    write(
        &dependency.join("crates/dep/src/lib.rs"),
        b"pub fn f() { /* edited in the cargo cache */ }\n",
    );
    let edited = state(&locked);
    assert!(!edited.status.success(), "{}", describe(&edited));
    assert!(
        text(&edited.stdout).contains("tracked files modified"),
        "{}",
        describe(&edited)
    );
    // A package directory outside every git checkout (the filesystem root;
    // the fixture itself sits inside this worktree) is refused, not skipped.
    let outside = hygiene(
        &script,
        &format!("git_dependency_checkout_state / '{locked}'"),
        &[("HOME", &home)],
    );
    assert!(!outside.status.success(), "{}", describe(&outside));
    assert!(
        text(&outside.stdout).contains("is not inside a git checkout"),
        "{}",
        describe(&outside)
    );
}

// ---------------------------------------------------------------------------
// #ship-shallow: a SHALLOW clone of the release commit (no ny_commit object) is
// bound by content, and a clone owned by another user is read, not refused --
// by the installer's verifier and by the runtime receipt alike.
// ---------------------------------------------------------------------------

#[test]
fn release_verifier_admits_a_shallow_clone_of_an_untampered_release() {
    let fixture = released_verifier_fixture("shallow-ok");
    let clone = clone_of(&fixture.root, "shallow-ok", true);
    assert_shallow_without(&clone, &fixture.commit);
    let (output, staged) = verify(&clone, X86);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stdout).contains("source_binding=artifact-only-tree"),
        "{}",
        describe(&output)
    );
    assert_eq!(
        fs::read(&staged).expect("staged binary"),
        fixture.artifact.binary
    );

    // Control: a full clone of the same commit binds through ny_commit's
    // history, and the tree rule agrees with it.
    let full = clone_of(&fixture.root, "shallow-ok-full", false);
    assert_eq!(
        git(&full, &["rev-parse", "--is-shallow-repository"]),
        "false"
    );
    let (output, _) = verify(&full, X86);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stdout).contains("source_binding=artifact-only-descendant"),
        "{}",
        describe(&output)
    );
}

type RepoEdit = fn(&Path);

#[test]
fn release_verifier_refuses_a_shallow_clone_with_any_source_change() {
    // Committed on top of the release commits, then cloned --depth 1: the
    // clone is clean and shallow, so only the tree rule can refuse these.
    let committed: &[(&str, RepoEdit)] = &[
        ("tracked source edited", |root| {
            write(&root.join("crates/ny-core/src/lib.rs"), b"// edited\n")
        }),
        ("preset added under a package root", |root| {
            write(&root.join("configs/vnncomp26/extra.yaml"), b"bab: {}\n")
        }),
        ("file added outside every package root", |root| {
            write(&root.join("NOTES.txt"), b"not part of ny_commit\n")
        }),
        ("tracked source deleted", |root| {
            fs::remove_file(root.join("crates/ny-core/src/lib.rs")).expect("delete source")
        }),
        ("tracked source made executable", |root| {
            fs::set_permissions(
                root.join("crates/ny-core/src/lib.rs"),
                fs::Permissions::from_mode(0o755),
            )
            .expect("chmod source")
        }),
        ("raw binary committed beside the release files", |root| {
            write(&root.join(format!("dist/bin/{X86}/ny")), b"raw\n");
            git(root, &["add", "-f", &format!("dist/bin/{X86}/ny")]);
        }),
        ("symlink at a release path", |root| {
            let link = root.join(format!("dist/bin/{AARCH64}/ny.xz"));
            std::os::unix::fs::symlink(format!("../{X86}/ny.xz"), link).expect("symlink");
            git(root, &["add", "-f", &format!("dist/bin/{AARCH64}/ny.xz")]);
        }),
    ];
    for (name, edit) in committed {
        let fixture = released_verifier_fixture("shallow-tamper");
        edit(&fixture.root);
        commit_all(&fixture.root, name);
        let clone = clone_of(&fixture.root, "shallow-tamper", true);
        assert_shallow_without(&clone, &fixture.commit);
        let (output, staged) = verify(&clone, X86);
        assert!(
            !output.status.success(),
            "{name}: must be refused\n{}",
            describe(&output)
        );
        assert!(
            text(&output.stderr).contains(TREE_REFUSAL),
            "{name}: expected the tree refusal\n{}",
            describe(&output)
        );
        assert!(!staged.exists(), "{name}: a refused clone left a binary");
    }

    // A MANIFEST whose ny_commit_tree is not ny_commit's tree: unprovable in
    // the shallow clone, so the tree rule must refuse it.
    let mut fixture = verifier_fixture("shallow-wrong-tree");
    fixture.artifact.set("ny_commit_tree", &"6".repeat(40));
    write_artifact(&fixture.root, &mut fixture.artifact);
    commit_release_files(&fixture.root, X86, "release files");
    let clone = clone_of(&fixture.root, "shallow-wrong-tree", true);
    assert_shallow_without(&clone, &fixture.commit);
    let (output, _) = verify(&clone, X86);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains(TREE_REFUSAL),
        "{}",
        describe(&output)
    );

    // Uncommitted changes in the shallow clone itself keep their own refusals.
    let uncommitted: &[(&str, RepoEdit, &str)] = &[
        (
            "tracked edit in the clone",
            |root| write(&root.join("crates/ny-core/src/lib.rs"), b"// edited\n"),
            "tracked files are modified",
        ),
        (
            "untracked preset in the clone",
            |root| write(&root.join("configs/vnncomp26/extra.yaml"), b"bab: {}\n"),
            "untracked files under package roots",
        ),
    ];
    for (name, edit, expected) in uncommitted {
        let fixture = released_verifier_fixture("shallow-dirty");
        let clone = clone_of(&fixture.root, "shallow-dirty", true);
        assert_shallow_without(&clone, &fixture.commit);
        edit(&clone);
        let (output, _) = verify(&clone, X86);
        assert!(
            !output.status.success(),
            "{name}: must be refused\n{}",
            describe(&output)
        );
        assert!(
            text(&output.stderr).contains(expected),
            "{name}: expected '{expected}'\n{}",
            describe(&output)
        );
    }
}

/// The install fixture with its release files committed on top of ny_commit,
/// cloned --depth 1. The clone shares the fixture's command log and fake bin.
fn shallow_install_fixture(label: &str) -> (InstallFixture, String) {
    let fixture = install_fixture(label, false);
    let ny_commit = git(&fixture.root, &["rev-parse", "HEAD"]);
    commit_release_files(&fixture.root, X86, "release files");
    let clone = clone_of(&fixture.root, label, true);
    assert_shallow_without(&clone, &ny_commit);
    (
        InstallFixture {
            root: clone,
            log: fixture.log,
            fake_bin: fixture.fake_bin,
        },
        ny_commit,
    )
}

#[test]
fn release_install_and_receipt_bind_a_shallow_clone() {
    let (clone, ny_commit) = shallow_install_fixture("install-shallow");
    let output = install(&clone, &[]);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stdout).contains(&format!("NY_RELEASE_INSTALL_OK target={X86} ")),
        "{}",
        describe(&output)
    );
    let receipt =
        fs::read_to_string(clone.root.join("target/release/ny.receipt")).expect("receipt");
    assert!(
        receipt.contains(&format!("source_commit={ny_commit}\n")),
        "{receipt}"
    );
    let ok = validate_receipt(&clone);
    assert!(ok.status.success(), "{}", describe(&ok));

    // A tracked edit after the install is refused as before ...
    let preset = clone.root.join(FIXTURE_PRESET);
    write(&preset, b"bab: {timeout: 1}\n");
    let refused = validate_receipt(&clone);
    assert!(!refused.status.success(), "{}", describe(&refused));
    assert!(
        text(&refused.stderr).contains("tracked files are modified relative to HEAD"),
        "{}",
        describe(&refused)
    );
    // ... and once COMMITTED in the shallow clone (no history to diff
    // against), the tree rule refuses it.
    commit_all(&clone.root, "edit the preset after the install");
    let refused = validate_receipt(&clone);
    assert!(!refused.status.success(), "{}", describe(&refused));
    assert!(
        text(&refused.stderr).contains(TREE_REFUSAL),
        "{}",
        describe(&refused)
    );
}

/// A `git` on PATH that records each invocation's arguments (joined by 0x1f,
/// one call per line) and then runs the real git.
fn git_argument_recorder(directory: &Path, log: &Path) {
    let real = text(&run(Command::new("bash").args(["-c", "command -v git"])).stdout)
        .trim()
        .to_string();
    assert!(real.starts_with('/'), "real git not found: {real:?}");
    write_executable(
        &directory.join("git"),
        &format!(
            "#!/bin/bash\n(IFS=$'\\x1f'; printf '%s\\n' \"$*\") >> '{}'\nexec '{real}' \"$@\"\n",
            log.display()
        ),
    );
}

/// Every recorded call names the canonical ROOT -- and only it -- as
/// safe.directory, read-only, before the subcommand. Returns the subcommands.
fn assert_git_calls_scoped_to(log: &Path, root: &Path) -> Vec<String> {
    let canonical = fs::canonicalize(root)
        .expect("canonical root")
        .display()
        .to_string();
    let safe_directory = format!("safe.directory={canonical}");
    let recorded = fs::read_to_string(log).unwrap_or_default();
    let mut subcommands = Vec::new();
    for line in recorded.lines() {
        let call: Vec<&str> = line.split('\u{1f}').collect();
        assert!(
            call.len() > 5
                && call[0] == "-c"
                && call[1] == safe_directory
                && call[2] == "--no-optional-locks"
                && call[3] == "-C"
                && call[4] == canonical,
            "a git call is not scoped to the checkout {canonical}: {call:?}"
        );
        subcommands.push(call[5..].join(" "));
    }
    assert!(
        !recorded.contains("safe.directory=*"),
        "a wildcard safe.directory: {recorded}"
    );
    subcommands
}

#[test]
fn release_verifier_and_receipt_read_a_clone_owned_by_another_user() {
    // The simulation is effective: plain git refuses the clone outright.
    let fixture = released_verifier_fixture("owner");
    let clone = clone_of(&fixture.root, "owner", true);
    let probe = run(Command::new("git")
        .arg("-C")
        .arg(&clone)
        .args(["rev-parse", "--show-toplevel"])
        .env(DIFFERENT_OWNER.0, DIFFERENT_OWNER.1));
    assert!(
        !probe.status.success() && text(&probe.stderr).contains("dubious ownership"),
        "the different-owner simulation did not take effect: {}",
        describe(&probe)
    );

    // The verifier reads it (shallow, so the tree rule runs too) ...
    let (output, _) = verify_with(&clone, X86, &[DIFFERENT_OWNER]);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stdout).contains("source_binding=artifact-only-tree"),
        "{}",
        describe(&output)
    );
    // ... and the binding is not weakened: an edit is still refused.
    write(&clone.join("crates/ny-core/src/lib.rs"), b"// edited\n");
    let (output, _) = verify_with(&clone, X86, &[DIFFERENT_OWNER]);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("tracked files are modified"),
        "{}",
        describe(&output)
    );

    // The install path and every run's receipt check, on a shallow clone owned
    // by "someone else" -- the organizers' likely layout.
    let (clone, _) = shallow_install_fixture("install-owner");
    let output = install(&clone, &[DIFFERENT_OWNER]);
    assert!(output.status.success(), "{}", describe(&output));
    let ok = validate_receipt_with(&clone, &[DIFFERENT_OWNER]);
    assert!(ok.status.success(), "{}", describe(&ok));
    write(&clone.root.join(FIXTURE_PRESET), b"bab: {timeout: 1}\n");
    let refused = validate_receipt_with(&clone, &[DIFFERENT_OWNER]);
    assert!(!refused.status.success(), "{}", describe(&refused));
    assert!(
        text(&refused.stderr).contains("tracked files are modified relative to HEAD"),
        "{}",
        describe(&refused)
    );
}

#[test]
fn release_git_calls_name_only_the_checkout_as_safe_directory() {
    // The verifier, on a shallow clone (every branch of the binding runs).
    let fixture = released_verifier_fixture("argv");
    let clone = clone_of(&fixture.root, "argv", true);
    let recorder = scratch("argv-bin");
    let log = recorder.join("git-calls.log");
    git_argument_recorder(&recorder, &log);
    // The install fixture's own PATH shape, not the caller's: the NY source
    // frontier counts every environment read site, NY_* or not.
    let path = format!("{}:/usr/bin:/bin", recorder.display());
    let (output, _) = verify_with(&clone, X86, &[("PATH", &path), DIFFERENT_OWNER]);
    assert!(output.status.success(), "{}", describe(&output));
    let calls = assert_git_calls_scoped_to(&log, &clone);
    for expected in [
        "rev-parse --show-toplevel",
        "diff --quiet HEAD --",
        "rev-parse --is-shallow-repository",
        "ls-tree --full-tree -z",
    ] {
        assert!(
            calls.iter().any(|call| call.starts_with(expected)),
            "the verifier never ran `git {expected}`: {calls:?}"
        );
    }

    // The receipt (install-time create-release and run-time validate).
    let (clone, _) = shallow_install_fixture("argv-install");
    let log = clone.fake_bin.join("git-calls.log");
    git_argument_recorder(&clone.fake_bin, &log);
    let output = install(&clone, &[DIFFERENT_OWNER]);
    assert!(output.status.success(), "{}", describe(&output));
    let installed = assert_git_calls_scoped_to(&log, &clone.root);
    for expected in [
        "rev-parse --show-toplevel",
        "rev-parse --is-shallow-repository",
        "ls-tree --full-tree -z",
        "hash-object -t tree --stdin",
        "hash-object --literally -t commit --stdin",
    ] {
        assert!(
            installed.iter().any(|call| call.starts_with(expected)),
            "the receipt never ran `git {expected}`: {installed:?}"
        );
    }
}

#[test]
fn release_builder_records_ny_commit_tree_and_refuses_tracked_release_files() {
    let base = scratch("commit-tree");
    let script = hygiene_script(&base);
    let repo = base.join("repo");
    write(&repo.join(".gitignore"), b"dist/\n");
    write(&repo.join("crates/a/src/lib.rs"), b"// a\n");
    git(&repo, &["init", "-q"]);
    let commit = commit_all(&repo, "source");
    let home = base.join("home").display().to_string();
    let tree_of = |commit: &str| {
        hygiene(
            &script,
            &format!(
                "cd '{}' || exit 95\nrelease_commit_tree '{commit}' || exit 1\n\
                 printf 'PATHS=%s\\n' \"$(IFS=,; printf '%s' \"${{RELEASE_PATHS[*]}}\")\"",
                repo.display()
            ),
            &[("HOME", &home)],
        )
    };
    let ok = tree_of(&commit);
    assert!(ok.status.success(), "{}", describe(&ok));
    let stdout = text(&ok.stdout);
    assert_eq!(
        stdout.lines().next(),
        Some(git(&repo, &["rev-parse", &format!("{commit}^{{tree}}")]).as_str()),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("PATHS={RELEASE_PATHS_CSV}\n")),
        "{stdout}"
    );

    // A commit that already tracks a release file cannot be bound by the
    // tree rule (it would be stripped from HEAD but not from the record).
    write(
        &repo.join(format!("dist/bin/{X86}/MANIFEST")),
        b"stale release\n",
    );
    git(&repo, &["add", "-f", &format!("dist/bin/{X86}/MANIFEST")]);
    let tracked = commit_all(&repo, "a stale release file");
    let refused = tree_of(&tracked);
    assert!(!refused.status.success(), "{}", describe(&refused));
    assert!(
        text(&refused.stdout).contains(&format!(
            "already tracks release files: dist/bin/{X86}/MANIFEST"
        )),
        "{}",
        describe(&refused)
    );
}

#[test]
fn release_manifest_contract_agrees_between_builder_verifier_and_receipt() {
    let read = |relative: &str| {
        fs::read_to_string(workspace_root().join(relative))
            .unwrap_or_else(|error| panic!("{relative}: {error}"))
    };
    let verifier = read("vnncomp_scripts/verify_release.py");
    let receipt = read("vnncomp_scripts/submission_binary_receipt.sh");
    let builder = read("scripts/release/make_submission.sh");

    // One release-path list in three places, and it is what MANIFEST records.
    let python = quoted_strings(delimited(&verifier, "\nRELEASE_PATHS = (", "\n)"));
    let bash = quoted_strings(delimited(&receipt, "\nRELEASE_PATHS=(", "\n)"));
    let recorded = quoted_strings(delimited(&builder, "\nRELEASE_PATHS=(", "\n)"));
    let mut expected: Vec<String> = [AARCH64, X86]
        .iter()
        .flat_map(|target| {
            ["MANIFEST", "ny.sha256", "ny.xz"]
                .iter()
                .map(move |name| format!("dist/bin/{target}/{name}"))
        })
        .collect();
    expected.sort();
    assert_eq!(python, expected, "verify_release.py RELEASE_PATHS");
    assert_eq!(bash, expected, "submission_binary_receipt.sh RELEASE_PATHS");
    assert_eq!(recorded, expected, "make_submission.sh RELEASE_PATHS");
    assert_eq!(expected.join(","), RELEASE_PATHS_CSV);

    // The builder writes exactly the verifier's MANIFEST keys, in order.
    let keys = quoted_strings(delimited(&verifier, "\nMANIFEST_KEYS = (", "\n)"));
    let end = builder
        .find("} >\"${stage}/MANIFEST\"")
        .expect("the builder's MANIFEST writer");
    let start = builder[..end]
        .rfind("\n    {\n")
        .expect("the start of the MANIFEST writer");
    let written: Vec<String> = builder[start..end]
        .lines()
        .filter_map(|line| line.trim().strip_prefix("printf '"))
        .map(|rest| rest.split('=').next().expect("key").to_string())
        .collect();
    assert!(
        keys.iter().any(|key| key == "ny_commit_tree")
            && keys.iter().any(|key| key == "ny_commit_object")
            && keys.iter().any(|key| key == "release_paths"),
        "{keys:?}"
    );
    assert_eq!(
        written, keys,
        "make_submission.sh's MANIFEST writer and verify_release.py MANIFEST_KEYS diverged"
    );
}

/// Replace MANIFEST's `key=` line in dist/bin/<x86_64>/MANIFEST.
fn rewrite_manifest(root: &Path, key: &str, value: &str) {
    let path = root.join("dist/bin").join(X86).join("MANIFEST");
    let manifest = fs::read_to_string(&path).expect("manifest");
    let prefix = format!("{key}=");
    assert_eq!(
        manifest
            .lines()
            .filter(|line| line.starts_with(&prefix))
            .count(),
        1,
        "{key} in {manifest}"
    );
    let mut rewritten = String::new();
    for line in manifest.lines() {
        if line.starts_with(&prefix) {
            rewritten.push_str(&prefix);
            rewritten.push_str(value);
        } else {
            rewritten.push_str(line);
        }
        rewritten.push('\n');
    }
    write(&path, rewritten.as_bytes());
}

#[test]
fn release_receipt_refuses_a_manifest_or_tree_that_does_not_bind() {
    // A widened release_paths would let the tree rule strip a source file.
    let fixture = install_fixture("receipt-paths", false);
    rewrite_manifest(
        &fixture.root,
        "release_paths",
        &format!("{RELEASE_PATHS_CSV},{FIXTURE_PRESET}"),
    );
    let output = install(&fixture, &[]);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("release_paths is not exactly the release files"),
        "{}",
        describe(&output)
    );
    assert_existing_binary_kept(&fixture);

    // With ny_commit present, ny_commit_tree must be its tree.
    let fixture = install_fixture("receipt-tree", false);
    rewrite_manifest(&fixture.root, "ny_commit_tree", &"7".repeat(40));
    let output = install(&fixture, &[]);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("is not the tree of ny_commit"),
        "{}",
        describe(&output)
    );
    assert_existing_binary_kept(&fixture);

    // In a shallow clone, a symlink committed at a release path is not a
    // release file: it stays in the tree, so the tree rule refuses it.
    let fixture = install_fixture("receipt-symlink", false);
    let ny_commit = git(&fixture.root, &["rev-parse", "HEAD"]);
    let link = format!("dist/bin/{AARCH64}/ny.xz");
    fs::create_dir_all(fixture.root.join(format!("dist/bin/{AARCH64}"))).expect("aarch64 dir");
    std::os::unix::fs::symlink(format!("../{X86}/ny.xz"), fixture.root.join(&link))
        .expect("symlink");
    git(&fixture.root, &["add", "-f", &link]);
    commit_release_files(&fixture.root, X86, "release files and a symlink");
    let clone = clone_of(&fixture.root, "receipt-symlink", true);
    assert_shallow_without(&clone, &ny_commit);
    let clone = InstallFixture {
        root: clone,
        log: fixture.log,
        fake_bin: fixture.fake_bin,
    };
    let output = install(&clone, &[]);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains(TREE_REFUSAL),
        "{}",
        describe(&output)
    );
    assert!(
        !clone.root.join("target/release/ny.receipt").exists(),
        "no receipt may be published"
    );
}

// ---------------------------------------------------------------------------
// review(ship-shallow): in a shallow clone the tree rule alone trusted MANIFEST
// for `ny_commit_tree` -- a source edit committed on top of the release PLUS a
// re-pointed `ny_commit_tree` passed there while a full clone refused it.
// MANIFEST's `ny_commit_object` (ny_commit's raw commit object) closes that
// without history: it must hash to ny_commit and name `ny_commit_tree`.
// ---------------------------------------------------------------------------

/// Stage EDIT in ROOT's index and return the tree it makes: the "source tree"
/// a forger would re-point `ny_commit_tree` at.
fn staged_tree_after(root: &Path, edit: impl FnOnce(&Path)) -> String {
    edit(root);
    git(root, &["add", "-A"]);
    git(root, &["write-tree"])
}

#[test]
fn release_verifier_binds_ny_commit_tree_to_ny_commit_in_a_shallow_clone() {
    // (forge the object too?, expected refusal)
    let cases: &[(bool, &str)] = &[
        (false, "not MANIFEST ny_commit_tree"),
        (true, "MANIFEST ny_commit_object hashes to"),
    ];
    for (forge_object, expected) in cases {
        let mut fixture = verifier_fixture("bind-object");
        let edited = staged_tree_after(&fixture.root, |root| {
            write(
                &root.join("crates/ny-core/src/lib.rs"),
                b"// edited after the build\n",
            )
        });
        assert_ne!(edited, fixture.artifact.get("ny_commit_tree"));
        fixture.artifact.set("ny_commit_tree", &edited);
        if *forge_object {
            let forged = forge_tree_header(&fixture.artifact.get("ny_commit_object"), &edited);
            fixture.artifact.set("ny_commit_object", &forged);
        }
        write_artifact(&fixture.root, &mut fixture.artifact);
        // The edit and the release files in one commit on top of ny_commit.
        commit_release_files(&fixture.root, X86, "release files and a source edit");

        // The tree rule alone cannot see it: HEAD's tree minus the release
        // files IS the re-pointed tree. Only ny_commit_object refuses it.
        let clone = clone_of(&fixture.root, "bind-object", true);
        assert_shallow_without(&clone, &fixture.commit);
        let (output, staged) = verify(&clone, X86);
        assert!(
            !output.status.success(),
            "forge_object={forge_object}: a re-pointed ny_commit_tree must be refused\n{}",
            describe(&output)
        );
        assert!(
            text(&output.stderr).contains(expected),
            "forge_object={forge_object}: expected '{expected}'\n{}",
            describe(&output)
        );
        assert!(!staged.exists(), "a refused clone left a binary");

        // Control: a full clone refuses the same tamper through ny_commit.
        let full = clone_of(&fixture.root, "bind-object-full", false);
        let (output, _) = verify(&full, X86);
        assert!(!output.status.success(), "{}", describe(&output));
        assert!(
            text(&output.stderr).contains("is not the tree of ny_commit"),
            "{}",
            describe(&output)
        );
    }

    // A long ny_commit message still fits MANIFEST: a 12 KB message (the
    // longest of 3000 recent ny commits is 10.7 KB) installs from a shallow
    // clone.
    let message = format!(
        "release candidate\n\n{}\n",
        "long commit body line.\n".repeat(530)
    );
    assert!(message.len() > 12_000);
    let mut fixture = verifier_fixture_with_message("long-message", &message);
    write_artifact(&fixture.root, &mut fixture.artifact);
    commit_release_files(&fixture.root, X86, "release files");
    let clone = clone_of(&fixture.root, "long-message", true);
    assert_shallow_without(&clone, &fixture.commit);
    let (output, _) = verify(&clone, X86);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stdout).contains("source_binding=artifact-only-tree"),
        "{}",
        describe(&output)
    );

    // ... while a commit object over the 32 KiB bound make_submission.sh
    // enforces is refused even though MANIFEST itself still fits.
    let message = format!(
        "release candidate\n\n{}\n",
        "over-long commit body line.\n".repeat(1250)
    );
    let mut fixture = verifier_fixture_with_message("huge-message", &message);
    assert!(
        piped(
            "base64",
            &["-d"],
            fixture.artifact.get("ny_commit_object").as_bytes()
        )
        .len()
            > 32 * 1024
    );
    write_artifact(&fixture.root, &mut fixture.artifact);
    let (output, _) = verify(&fixture.root, X86);
    assert!(!output.status.success(), "{}", describe(&output));
    assert!(
        text(&output.stderr).contains("'ny_commit_object' is larger than 32768 bytes"),
        "{}",
        describe(&output)
    );
}

#[test]
fn release_receipt_binds_ny_commit_tree_to_ny_commit_in_a_shallow_clone() {
    // install_tool.sh's verifier is a stub here, so the receipt is the only
    // check: a preset edit committed on top of the release with
    // ny_commit_tree re-pointed at it must not install from a shallow clone.
    let cases: &[(bool, &str)] = &[
        (false, "not release ny_commit_tree"),
        (true, "release MANIFEST ny_commit_object hashes to"),
    ];
    for (forge_object, expected) in cases {
        let fixture = install_fixture("receipt-object", false);
        let ny_commit = git(&fixture.root, &["rev-parse", "HEAD"]);
        let edited = staged_tree_after(&fixture.root, |root| {
            write(&root.join(FIXTURE_PRESET), b"bab: {timeout: 1}\n")
        });
        rewrite_manifest(&fixture.root, "ny_commit_tree", &edited);
        if *forge_object {
            let manifest =
                fs::read_to_string(fixture.root.join("dist/bin").join(X86).join("MANIFEST"))
                    .expect("manifest");
            let honest = manifest
                .lines()
                .find_map(|line| line.strip_prefix("ny_commit_object="))
                .expect("ny_commit_object line")
                .to_string();
            rewrite_manifest(
                &fixture.root,
                "ny_commit_object",
                &forge_tree_header(&honest, &edited),
            );
        }
        commit_release_files(&fixture.root, X86, "release files and a preset edit");
        let clone = clone_of(&fixture.root, "receipt-object", true);
        assert_shallow_without(&clone, &ny_commit);
        let clone = InstallFixture {
            root: clone,
            log: fixture.log,
            fake_bin: fixture.fake_bin,
        };
        let output = install(&clone, &[]);
        assert!(
            !output.status.success(),
            "forge_object={forge_object}: must be refused\n{}",
            describe(&output)
        );
        assert!(
            text(&output.stderr).contains(expected),
            "forge_object={forge_object}: expected '{expected}'\n{}",
            describe(&output)
        );
        assert!(
            !clone.root.join("target/release/ny.receipt").exists(),
            "no receipt may be published"
        );
    }

    // A missing or malformed ny_commit_object is refused when MANIFEST is
    // parsed, and an over-bound one before it is hashed.
    let oversized = text(&piped("base64", &["-w0"], &vec![b'a'; 33_000]))
        .trim()
        .to_string();
    let malformed: &[(&str, &str)] = &[
        (
            "not*base64",
            "release MANIFEST ny_commit_object is missing or not base64",
        ),
        (
            &oversized,
            "release MANIFEST ny_commit_object is larger than 32768 bytes",
        ),
    ];
    for (value, expected) in malformed {
        let fixture = install_fixture("receipt-object-malformed", false);
        rewrite_manifest(&fixture.root, "ny_commit_object", value);
        let output = install(&fixture, &[]);
        assert!(!output.status.success(), "{}", describe(&output));
        assert!(
            text(&output.stderr).contains(expected),
            "expected '{expected}'\n{}",
            describe(&output)
        );
        assert_existing_binary_kept(&fixture);
    }
}

#[test]
fn release_builder_records_ny_commit_object_within_the_manifest_bound() {
    let base = scratch("commit-object");
    let script = hygiene_script(&base);
    let repo = base.join("repo");
    write(&repo.join(".gitignore"), b"dist/\n");
    write(&repo.join("crates/a/src/lib.rs"), b"// a\n");
    git(&repo, &["init", "-q"]);
    let commit = commit_all(&repo, "source");
    let home = base.join("home").display().to_string();
    let object_of = |commit: &str| {
        hygiene(
            &script,
            &format!(
                "declare -F release_commit_object >/dev/null || exit 98\n\
                 cd '{}' || exit 95\nrelease_commit_object '{commit}' || exit 1",
                repo.display()
            ),
            &[("HOME", &home)],
        )
    };
    let ok = object_of(&commit);
    assert!(ok.status.success(), "{}", describe(&ok));
    let recorded = text(&ok.stdout);
    let recorded = recorded.trim_end_matches('\n');
    assert!(
        !recorded.contains('\n') && !recorded.is_empty(),
        "one base64 line: {recorded:?}"
    );
    assert_eq!(recorded, commit_object(&repo, &commit));
    let raw = piped("base64", &["-d"], recorded.as_bytes());
    let hashed = text(&piped(
        "git",
        &["hash-object", "--literally", "-t", "commit", "--stdin"],
        &raw,
    ))
    .trim()
    .to_string();
    assert_eq!(hashed, commit, "the recorded object hashes to ny_commit");
    assert!(
        raw.starts_with(format!("tree {}\n", git(&repo, &["rev-parse", "HEAD^{tree}"])).as_bytes()),
        "the recorded object names ny_commit's tree"
    );

    // A commit object over the verifier's 32 KiB bound is refused before the
    // build, not at install time.
    write(&repo.join("crates/a/src/lib.rs"), b"// b\n");
    let huge = commit_all(&repo, &"an over-long release message line\n".repeat(1100));
    let refused = object_of(&huge);
    assert!(!refused.status.success(), "{}", describe(&refused));
    assert!(
        text(&refused.stdout).contains("over the 32768-byte MANIFEST bound"),
        "{}",
        describe(&refused)
    );
}
