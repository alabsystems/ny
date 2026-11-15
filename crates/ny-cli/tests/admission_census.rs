// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #admission-census on the REAL binary: `ny vnncomp census` over a hermetic
//! corpus of tracked models (`tests/models/`) and properties written into a
//! temporary directory. No `benchmarks/` dependency, no verification run.
//!
//! What is pinned: one `CENSUS_DONE rc=0` line; the binding names THIS
//! binary's sha256 and the manifest's; the trailer verifies; the census is
//! byte-identical on a second run; the same files under category-like names
//! and other row ids give the same decisions; an existing census is never
//! overwritten.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ny_test_utils::workspace_root;
use sha2::{Digest, Sha256};

const HALFSPACE: &str = "(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (<= Y_0 -1000000.0))
";

const ROBUSTNESS: &str = "(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -1.0))
(assert (<= X_1 1.0))
(assert (or
    (and (>= Y_1 Y_0))
))
";

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

fn census(manifest: &Path, out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ny"))
        .current_dir(workspace_root())
        .args([
            "vnncomp",
            "census",
            "--manifest",
            manifest.to_str().expect("manifest path UTF-8"),
            "--out",
            out.to_str().expect("out path UTF-8"),
            "--router-target-host",
            "m5",
        ])
        .output()
        .expect("run ny vnncomp census")
}

fn manifest(dir: &Path, name: &str, rows: &[(&str, &Path, &Path)]) -> PathBuf {
    let mut text = String::from("# ny-admission-census-corpus/v1\n");
    for (id, model, property) in rows {
        text.push_str(&format!(
            "{id}\t{}\t{}\t60\n",
            model.display(),
            property.display()
        ));
    }
    let path = dir.join(name);
    fs::write(&path, text).expect("write the corpus manifest");
    path
}

fn copy(dir: &Path, from: &Path, rel: &str) -> PathBuf {
    let to = dir.join(rel);
    fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
    fs::copy(from, &to).expect("copy fixture");
    to
}

/// The body rows without the echoed row id.
fn decisions(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_once('\t').expect("row id").1.to_owned())
        .collect()
}

#[ntest::timeout(240000)]
#[test]
fn census_command_writes_a_bound_deterministic_name_blind_census() {
    let dir = tempfile::tempdir().expect("temporary census root");
    let root = workspace_root();
    let mlp = root.join("tests/models/simple_mlp.onnx");
    let conv = root.join("tests/models/conv_relu.onnx");
    let half = dir.path().join("half.vnnlib");
    fs::write(&half, HALFSPACE).expect("property");
    let robust = dir.path().join("robust.vnnlib");
    fs::write(&robust, ROBUSTNESS).expect("property");
    let missing = dir.path().join("absent.onnx");

    let m1 = manifest(
        dir.path(),
        "a.corpus.tsv",
        &[
            ("r1", &mlp, &half),
            ("r2", &mlp, &robust),
            ("r3", &conv, &robust),
            ("r4", &missing, &robust),
        ],
    );
    let out1 = dir.path().join("a.census.tsv");
    let run = census(&m1, &out1);
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        run.status.success(),
        "census failed: {}\nstdout={stdout}\nstderr={}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let done: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("CENSUS_DONE "))
        .collect();
    assert_eq!(done.len(), 1, "{stdout}");
    assert!(
        done[0].starts_with("CENSUS_DONE rc=0 rows=4 caps=10 "),
        "{}",
        done[0]
    );
    assert_eq!(done[0].matches(" rc=").count(), 1, "exactly one rc=");

    let text = fs::read_to_string(&out1).expect("census written");
    let binary = fs::read(env!("CARGO_BIN_EXE_ny")).expect("read the binary");
    let corpus = fs::read(&m1).expect("read the manifest");
    let bound = |key: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("# {key}\t")))
            .unwrap_or_else(|| panic!("binding `{key}` missing"))
            .to_owned()
    };
    assert_eq!(bound("binary_sha256"), sha256_hex(&binary), "THIS binary");
    assert_eq!(bound("corpus_sha256"), sha256_hex(&corpus), "THIS corpus");
    assert_eq!(bound("corpus_rows"), "4");
    assert!(bound("host").starts_with("class=m5 "), "{}", bound("host"));
    // The trailer is the sha256 of every byte above it.
    let (above, trailer) = text
        .trim_end_matches('\n')
        .rsplit_once('\n')
        .expect("trailer");
    let want = trailer
        .strip_prefix("# census_sha256\t")
        .expect("trailer line");
    assert_eq!(sha256_hex(format!("{above}\n").as_bytes()), want);
    assert!(done[0].contains(&format!("census_sha256={want}")));
    let rows = decisions(&text);
    assert_eq!(
        rows.len(),
        4 * 10,
        "one row per (corpus row, routable capability; #kfsb-production: + kfsb-single-shot; \
         #prepared-root-route: + prepared-root)"
    );
    assert!(text
        .lines()
        .any(|l| l.starts_with("r4\tfrozen-dual-bab\t0\t0\t-\tno-router:model-missing\t")));
    // The MLP is outside the margin-row family: where the router alone would
    // route the frozen-dual tier, the lane's own admission refuses it.
    let fd_r2 = text
        .lines()
        .find(|l| l.starts_with("r2\tfrozen-dual-bab\t"))
        .expect("r2 frozen-dual row");
    assert!(
        fd_r2.starts_with("r2\tfrozen-dual-bab\t0\t1\tdecline:model-outside-twin-family\t"),
        "{fd_r2}"
    );

    // Deterministic: a second run writes the same bytes.
    let out2 = dir.path().join("b.census.tsv");
    assert!(census(&m1, &out2).status.success());
    assert_eq!(fs::read(&out2).expect("second census"), text.as_bytes());

    // Name-blind: the same files under category-like names, other row ids.
    let m3 = manifest(
        dir.path(),
        "c.corpus.tsv",
        &[
            (
                "y26_cifar100_2024_0001",
                &copy(
                    dir.path(),
                    &mlp,
                    "cifar100_2024/onnx/CIFAR100_resnet_large.onnx",
                ),
                &copy(dir.path(), &half, "cifar100_2024/vnnlib/x.vnnlib"),
            ),
            (
                "q",
                &copy(dir.path(), &mlp, "metaroom_2023/zz.onnx"),
                &copy(dir.path(), &robust, "metaroom_2023/p_7.vnnlib"),
            ),
            (
                "z9",
                &copy(dir.path(), &conv, "acasxu/ACASXU_run2a_1_1_batch_2000.onnx"),
                &copy(dir.path(), &robust, "acasxu/prop_1.vnnlib"),
            ),
            ("w", &dir.path().join("cora_2024/none.onnx"), &robust),
        ],
    );
    let out3 = dir.path().join("c.census.tsv");
    assert!(census(&m3, &out3).status.success());
    assert_eq!(
        decisions(&fs::read_to_string(&out3).expect("renamed census")),
        rows,
        "decisions differ by name"
    );

    // A census is written once.
    let again = census(&m1, &out1);
    assert!(
        !again.status.success(),
        "an existing census was overwritten"
    );
    assert_eq!(fs::read_to_string(&out1).expect("still there"), text);
}
