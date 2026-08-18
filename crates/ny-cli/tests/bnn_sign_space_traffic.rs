// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Acceptance test for the sign-space falsification lane, against the REAL
// staged `traffic_signs_recognition_2023` corpus and through the REAL scored
// entry point (`ny vnncomp v1`).
//
// WHY IT GOES THROUGH THE BINARY. `ny-cli` has no library target, so an
// integration test cannot reach `commands::beta_crown::sign_space_falsify`
// directly. That is a feature here rather than a limitation: driving the scored
// CLI is the only way to test the thing that actually matters — that the
// CATEGORY PRESET arms the lane with NO environment variable set (the only
// arming route a scored competition run has), that doing so turns a
// `timeout`/`error` row into a `sat`, that the `sat` carries a witness the
// ORGANIZER's own check would accept, and that `NY_BNN_SIGN_SPACE` still
// overrides the preset in both directions.
//
// WHAT THE UNIT TESTS ALREADY COVER, and therefore what is NOT re-litigated
// here: the exact-`"1"` arming rule, the "core never constructed while dark"
// code path, and the call-site rule that only an upheld `sat` terminates an
// instance all live in `sign_space_falsify.rs` and `vnncomp.rs`'s own `mod
// tests`. This file covers the two claims those cannot make:
//
//   1. the extraction really matches the shipped ONNX bytes and the shipped
//      VNN-LIB shape — a front end that refuses everything would pass every
//      unit test and be worthless; and
//   2. any witness the lane publishes is INDEPENDENTLY authenticated here, by
//      re-forwarding it through `ny_onnx::diff::OrtForward` (a real ONNX Runtime
//      session on the ORIGINAL model bytes) and re-checking box membership and
//      the property against the parsed VNN-LIB — NOT by trusting the verdict
//      token ny wrote.
//
// This is deliberately behind `external-vnncomp`: it needs the downloaded
// corpus, it runs the real 480 s scored budget, and it hard-fails rather than
// skipping when the corpus is absent.
//
// RUN IT IN RELEASE:
//
//   cargo test --release -p ny-cli --features mip,external-vnncomp \
//       --test bnn_sign_space_traffic -- --test-threads=1
//
// `ny_binary()` resolves `CARGO_BIN_EXE_ny`, which cargo points at the profile
// the TEST was built in, so a debug run drives the debug `ny`. That is not a
// slower version of the same experiment — it is a different one. The scored
// budget here is a real wall clock, the realizability LP is the hot loop, and
// under `--release` the three rows land in 60-132 s of a 480 s budget. Unoptimized
// they do not land at all, and the test reports `timeout` for a lane that works.
// `--test-threads=1` for the same reason: three concurrent scored runs on one
// machine are not three scored runs.

#![cfg(all(feature = "external-vnncomp", feature = "mip"))]

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/vnncomp.rs"]
mod vnncomp_support;
use vnncomp_support::{ny_binary, require_benchmark_file};

const CATEGORY: &str = "traffic_signs_recognition_2023";
const MODEL_30: &str = "3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx";
const MODEL_48: &str =
    "3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx";
const MODEL_64: &str = "3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx";

/// The three `model_30` eps=1 rows the method is documented to capture
/// (`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md`), with the pre-Softmax
/// logit margin each banked witness reached. The margin is a LOWER bar here,
/// not an equality: the search is deterministic but the lane's budget is not
/// the generator script's, so a different (still violating) witness is fine.
const BANKED_EPS1_ROWS: &[(&str, usize)] = &[
    ("model_30_idx_1703_eps_1.00000.vnnlib", 25),
    ("model_30_idx_6371_eps_1.00000.vnnlib", 17),
    ("model_30_idx_178_eps_1.00000.vnnlib", 13),
];

/// Official per-instance budget for this family on the 2026 board.
const SCORED_BUDGET_SECS: &str = "480";

// ---------------------------------------------------------------------------
// Corpus resolution (gz-aware), mirroring
// `ny-onnx/tests/traffic_terminal_softmax_strip.rs`.
// ---------------------------------------------------------------------------

/// The staged 2026 benchmark tree. One declared-lever read, never a raw
/// `NY_*` process read (the `ny-levers` ratchet forbids adding one), then an
/// ancestor walk so a git worktree still finds the primary checkout's untracked
/// benchmark tree. NEVER silently skips: a vacuous pass is exactly the hole an
/// acceptance test cannot afford.
fn benchmark_root() -> PathBuf {
    let inputs = ny_levers::RawLeverInputs::capture(ny_levers::all());
    if let Some(root) = inputs.get(&ny_levers::decls::onnx::BENCH_ROOT_2026) {
        return PathBuf::from(root);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest.ancestors() {
        let candidate = ancestor.join("benchmarks/vnncomp2026/benchmarks");
        if candidate.is_dir() {
            return candidate;
        }
    }
    panic!(
        "Benchmark tree missing: no `benchmarks/vnncomp2026/benchmarks` in any ancestor of {}. \
         Run benchmarks/download_benchmarks.sh first, or set {}.",
        manifest.display(),
        ny_levers::decls::onnx::BENCH_ROOT_2026.name,
    );
}

fn traffic_root() -> PathBuf {
    benchmark_root().join(format!("{CATEGORY}/2.0"))
}

/// Prefer the plain file, fall back to the `.gz` the download script may leave.
fn gz_aware(dir: &str, name: &str) -> PathBuf {
    let plain = traffic_root().join(dir).join(name);
    if plain.is_file() {
        return plain;
    }
    let gz = traffic_root().join(dir).join(format!("{name}.gz"));
    require_benchmark_file(&gz);
    gz
}

fn model_path(name: &str) -> PathBuf {
    gz_aware("onnx", name)
}

fn spec_path(name: &str) -> PathBuf {
    gz_aware("vnnlib", name)
}

// ---------------------------------------------------------------------------
// Driving the scored entry point
// ---------------------------------------------------------------------------

struct ScoredRun {
    result: String,
    witness: Option<String>,
    stderr: String,
}

/// Run one instance through the REAL scored entry point.
///
/// `lever` is the exact `NY_BNN_SIGN_SPACE` value to export, or `None` to
/// REMOVE it from the child's environment; everything else — preset, timeout
/// tier, backend, attack selection — is left to the binary's own AUTO defaults,
/// because the point is to observe the scored path, not a configuration of it.
///
/// `None` is the SCORED shape: a competition run exports no `NY_*`, so it is
/// the arm in which the category preset's `attack.bnn_sign_space` is the only
/// thing that can arm the lane.
fn run_scored(onnx: &Path, vnnlib: &Path, budget: &str, lever: Option<&str>) -> ScoredRun {
    let dir = std::env::temp_dir().join(format!(
        "ny-sign-space-{}-{}-{}",
        std::process::id(),
        vnnlib.file_stem().unwrap_or_default().to_string_lossy(),
        lever.unwrap_or("absent")
    ));
    std::fs::create_dir_all(&dir).expect("temp results dir");
    let results = dir.join("results.txt");
    let _ = std::fs::remove_file(&results);

    let mut command = Command::new(ny_binary());
    command.args([
        "vnncomp",
        "v1",
        CATEGORY,
        &onnx.to_string_lossy(),
        &vnnlib.to_string_lossy(),
        &results.to_string_lossy(),
        budget,
    ]);
    // `remove_env` on the no-lever arm is not paranoia: an ambient value in the
    // test runner's environment would silently invalidate the claim that the
    // PRESET is what armed (or did not arm) the lane.
    match lever {
        Some(value) => command.env("NY_BNN_SIGN_SPACE", value),
        None => command.env_remove("NY_BNN_SIGN_SPACE"),
    };
    let output = command.output().expect("failed to execute ny binary");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let body = std::fs::read_to_string(&results).unwrap_or_else(|error| {
        panic!(
            "RESULTS_FILE {} unreadable after `ny vnncomp v1` ({error})\nstderr:\n{stderr}",
            results.display()
        )
    });
    let mut lines = body.lines();
    let result = lines.next().unwrap_or_default().trim().to_string();
    let rest: Vec<&str> = lines.collect();
    let witness = (!rest.is_empty()).then(|| rest.join("\n"));
    ScoredRun {
        result,
        witness,
        stderr,
    }
}

// ---------------------------------------------------------------------------
// Independent witness authentication
// ---------------------------------------------------------------------------

/// Parse the input assignment out of a published witness.
///
/// Both shipped shapes are accepted, because which one `ny` writes is decided by
/// the property's declared VNN-LIB version and is not this lane's business:
///
/// * the VNN-COMP 2023+ TENSOR TEXT form — `X float32 [1, 30, 30, 3]` followed
///   by one decimal per line, then a `Y` block (which is deliberately IGNORED
///   here: this test re-derives the outputs from ONNX Runtime rather than
///   trusting the ones ny wrote); and
/// * the SMT-LIB `(X_i v)` pair form.
fn parse_witness_inputs(witness: &str, num_inputs: usize) -> Vec<f64> {
    if let Some(rest) = witness.split_once("X float32").map(|(_, rest)| rest) {
        // Skip the shape suffix on the header line, then read decimals until the
        // `Y` block starts.
        let body = rest.split_once('\n').map(|(_, body)| body).unwrap_or("");
        let values: Vec<f64> = body
            .lines()
            .map(str::trim)
            .take_while(|line| !line.starts_with('Y'))
            .filter(|line| !line.is_empty())
            .map(|line| {
                line.parse()
                    .unwrap_or_else(|_| panic!("unparseable witness value {line:?}"))
            })
            .collect();
        assert_eq!(
            values.len(),
            num_inputs,
            "tensor-text witness assigns {} values, the property declares {num_inputs}",
            values.len()
        );
        return values;
    }
    let mut values = vec![f64::NAN; num_inputs];
    let mut seen = vec![false; num_inputs];
    for token in witness.split(['(', ')']) {
        let mut parts = token.split_whitespace();
        let (Some(name), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Some(index) = name.strip_prefix("X_") else {
            continue;
        };
        let index: usize = index.parse().unwrap_or_else(|_| panic!("bad index {name}"));
        assert!(index < num_inputs, "witness names {name} out of range");
        values[index] = value
            .parse()
            .unwrap_or_else(|_| panic!("witness {name} has unparseable value {value}"));
        seen[index] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "witness does not assign every one of the {num_inputs} inputs"
    );
    values
}

/// Authenticate a published witness the way the ORGANIZER would, and WITHOUT
/// consulting anything ny computed.
///
/// * box membership is checked on the `f64` decimals AS WRITTEN, against the
///   bounds parsed straight out of the VNN-LIB — this is the `#witness-f64-
///   membership` rule, and it is what makes "in box" mean the same thing here
///   and at the organizer;
/// * the forward is a real ONNX Runtime session on the ORIGINAL model bytes
///   (`ny_onnx::diff::OrtForward`, gzip-aware), on the `f32` cast of that input;
/// * the property is re-evaluated at that trusted output with a STRICT `>`
///   margin, so a boundary point is NOT accepted.
///
/// Returns the violated challenger and its post-Softmax margin over the target.
fn authenticate_witness(onnx: &Path, vnnlib: &Path, witness: &str) -> (usize, usize, f64) {
    let spec = ny_onnx::vnnlib::load_vnnlib(vnnlib).expect("re-parse the property");
    let input = parse_witness_inputs(witness, spec.num_inputs);

    for (index, (&value, &(lo, hi))) in input.iter().zip(spec.input_bounds.iter()).enumerate() {
        assert!(
            value >= lo && value <= hi,
            "witness X_{index} = {value} is OUTSIDE the declared box [{lo}, {hi}]"
        );
    }

    let input_f32: Vec<f32> = input.iter().map(|&v| v as f32).collect();
    let mut forward = ny_onnx::diff::OrtForward::from_path(onnx, input_f32.len())
        .expect("build an ONNX Runtime session on the ORIGINAL model");
    let output = forward.run(&input_f32).expect("ORT forward on the witness");
    assert_eq!(
        output.len(),
        spec.num_outputs,
        "ORT returned {} outputs, the property declares {}",
        output.len(),
        spec.num_outputs
    );

    // The property is the argmax complement: SOME clause `Y_c >= Y_t` must hold.
    // Demand a STRICT margin so a tie cannot pass as a violation.
    let mut violated: Option<(usize, usize, f64)> = None;
    for clause in &spec.output_constraint_clauses {
        let [ny_onnx::vnnlib::OutputConstraint::GreaterEq(challenger, target)] = clause[..] else {
            panic!("staged traffic clause is not a single `Y_i >= Y_t` atom: {clause:?}");
        };
        let margin = f64::from(output[challenger]) - f64::from(output[target]);
        if margin > 0.0 {
            violated = Some((challenger, target, margin));
            break;
        }
    }
    violated.unwrap_or_else(|| {
        panic!(
            "ONNX Runtime on the ORIGINAL model does NOT violate the property at the \
             published witness — outputs {output:?}"
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// THE ACCEPTANCE TEST, and it runs with NO ENVIRONMENT AT ALL.
///
/// The lane must take the three `model_30` eps=1 rows — open in every result
/// bank inspected for this investigation, and the whole reason the lane exists
/// — armed solely by the shipped category preset's `attack.bnn_sign_space`,
/// which is the only arming route a scored competition run can use. Every
/// witness it publishes must independently authenticate under ONNX Runtime.
///
/// This is the test that would catch a front end that extracts the right shape
/// but the wrong LAYOUT: a transposed flatten produces arithmetically plausible
/// logits, a candidate ny's own search believes, and an ORT re-forward that does
/// not violate. The assertion is on the re-forward, not on the verdict token.
#[test]
fn preset_armed_lane_captures_the_three_open_eps1_rows_with_authenticated_witnesses() {
    let onnx = model_path(MODEL_30);
    require_benchmark_file(&onnx);
    for (spec_name, expected_target) in BANKED_EPS1_ROWS {
        let vnnlib = spec_path(spec_name);
        let run = run_scored(&onnx, &vnnlib, SCORED_BUDGET_SECS, None);
        assert!(
            run.stderr.contains("Sign-space falsification"),
            "{spec_name}: the preset must arm the lane with no NY_BNN_SIGN_SPACE set\n\
             stderr:\n{}",
            run.stderr
        );
        assert_eq!(
            run.result, "sat",
            "{spec_name}: preset-armed lane produced `{}`, not `sat`\nstderr:\n{}",
            run.result, run.stderr
        );
        let witness = run
            .witness
            .as_deref()
            .unwrap_or_else(|| panic!("{spec_name}: a `sat` must carry a witness"));
        let (challenger, target, margin) = authenticate_witness(&onnx, &vnnlib, witness);
        assert_eq!(
            target, *expected_target,
            "{spec_name}: the property's true class moved"
        );
        assert!(
            challenger != target && margin > 0.0,
            "{spec_name}: witness does not strictly violate"
        );
    }
}

/// THE ENVIRONMENT STILL OVERRIDES, IN THE DISARMING DIRECTION.
///
/// `NY_BNN_SIGN_SPACE=0` must kill a lane the category preset armed, leaving no
/// trace on stderr at all — the dark arm this file used to get from the absence
/// of the variable. Keeping this reachable is what makes the preset default
/// A/B-able: the unarmed baseline (27/45) is still one environment variable
/// away, on the shipped configuration.
///
/// The verdict assertion is deliberately narrow — it does not claim the
/// ordinary path times out, only that nothing here came from the sign-space
/// lane — because pinning a specific dark-arm verdict would make this test a
/// regression alarm for every unrelated change to the attack schedule.
#[test]
fn an_explicit_zero_disarms_the_preset_and_leaves_no_trace() {
    let onnx = model_path(MODEL_30);
    require_benchmark_file(&onnx);
    // One row is enough for the trace claim and keeps the dark arm cheap; the
    // budget is short on purpose, since the point is the absence of the lane.
    let vnnlib = spec_path(BANKED_EPS1_ROWS[0].0);
    let run = run_scored(&onnx, &vnnlib, "120", Some("0"));
    assert!(
        !run.stderr.contains("Sign-space falsification"),
        "NY_BNN_SIGN_SPACE=0 must disarm the preset: the lane must not run, log, or \
         otherwise appear:\n{}",
        run.stderr
    );
    assert_ne!(
        run.result, "error",
        "the disarmed arm must not error:\nstderr:\n{}",
        run.stderr
    );
}

/// EXACT-`"1"` SEMANTICS, THROUGH THE REAL PROCESS ENVIRONMENT.
///
/// `sign_space_falsify_armed_from`'s unit test pins the parser; this pins that
/// the parser is what the SHIPPED BINARY actually consults. A near-miss token
/// must leave the lane completely dark — not "armed but declining" — so the
/// evidence is the ABSENCE of the lane's stderr line on a net it WOULD have
/// spoken about had it been armed (the 48x48 net, which the armed run above
/// declines out loud).
///
/// This got STRICTER when the category preset started arming the lane, and the
/// test body did not have to change to get there: the preset now says `true`
/// for exactly this category, so every token below has to actively SUPPRESS a
/// preset that asked for the lane rather than merely fail to arm a dark one.
/// That is `ny_levers`' documented layering — a present-but-inadmissible value
/// resolves to the DECLARATION default, never to the config layer — and this
/// is where it is checked against the shipped binary.
#[test]
fn near_miss_lever_values_leave_the_lane_dark_in_the_real_binary() {
    let onnx = model_path(MODEL_48);
    require_benchmark_file(&onnx);
    let vnnlib = spec_path("model_48_idx_1703_eps_1.00000.vnnlib");
    for off in ["", "0", "01", "true", "on", " 1"] {
        let dir = std::env::temp_dir().join(format!("ny-sign-space-off-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp results dir");
        let results = dir.join("results.txt");
        let output = Command::new(ny_binary())
            .args([
                "vnncomp",
                "v1",
                CATEGORY,
                &onnx.to_string_lossy(),
                &vnnlib.to_string_lossy(),
                &results.to_string_lossy(),
                "120",
            ])
            .env("NY_BNN_SIGN_SPACE", off)
            .output()
            .expect("failed to execute ny binary");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("Sign-space falsification"),
            "NY_BNN_SIGN_SPACE={off:?} armed the lane; only the single character \"1\" may:\n\
             {stderr}"
        );
    }
}

/// THE DEEPER NETS ARE NOW INSIDE THE FRAGMENT.
///
/// `model_48` (31 nodes: MaxPool, BatchNorm, a folded-BN third convolution and
/// a dense head) and `model_64` (33 nodes, with a third MaxPool) used to be
/// declined by the graph walk with "node 2 is MaxPool, expected Transpose".
/// They are admitted now, which is the whole point of the widening — so the
/// evidence is that the lane REACHES THE CORE on them: the stderr line reports
/// a search outcome (`candidate` / `exhausted` / `core refused`), not
/// `not admitted`.
///
/// This is deliberately NOT an assertion that either row is captured. Whether a
/// given eps lands is a measurement, banked in
/// `reports/measured-2026/traffic_signs_recognition_2023.csv`; what this file
/// owns is that the extraction matches the shipped bytes.
#[test]
fn the_deeper_nets_reach_the_search_core_instead_of_being_declined() {
    for (model, spec) in [
        (MODEL_48, "model_48_idx_1703_eps_3.00000.vnnlib"),
        (MODEL_64, "model_64_idx_1703_eps_3.00000.vnnlib"),
    ] {
        let onnx = model_path(model);
        require_benchmark_file(&onnx);
        let vnnlib = spec_path(spec);
        let run = run_scored(&onnx, &vnnlib, "120", None);
        assert!(
            run.stderr.contains("Sign-space falsification"),
            "{spec}: the preset must arm the lane\nstderr:\n{}",
            run.stderr
        );
        assert!(
            !run.stderr.contains("not admitted"),
            "{spec}: the widened graph walk still declines this net\nstderr:\n{}",
            run.stderr
        );
        // A published `sat` must still authenticate under ONNX Runtime — the
        // widening did not touch the gate, and this proves the extraction is
        // the RIGHT network and not merely a plausible one.
        if run.result == "sat" {
            let witness = run
                .witness
                .as_deref()
                .unwrap_or_else(|| panic!("{spec}: a `sat` must carry a witness"));
            let (challenger, target, margin) = authenticate_witness(&onnx, &vnnlib, witness);
            assert!(
                challenger != target && margin > 0.0,
                "{spec}: witness does not strictly violate under ORT"
            );
        }
    }
}

/// A NET OUTSIDE THE ADMITTED FRAGMENT FALLS THROUGH WITH THE VERDICT
/// UNCHANGED, AND CHEAPLY.
///
/// The armed arm here is the SHIPPED configuration — no environment at all —
/// driven at a FOREIGN category's instance (the traffic-signs preset is what
/// arms the lane; the bytes it is pointed at are `relusplitter_2026`'s, whose
/// input is `[1, 784]` and therefore not rank-4 NHWC). That decline must be
/// verdict-NEUTRAL and must not cost real time, because "the admission
/// predicate refuses cheaply" is the entire argument that arming by default
/// does not starve anything.
#[test]
fn a_net_outside_the_admitted_fragment_falls_through_to_the_same_verdict() {
    let root = benchmark_root().join("relusplitter_2026/2.0");
    let onnx = root.join("onnx/model_2_2.onnx");
    let vnnlib = root.join("vnnlib/d2_eps_0.04_sample_15_label_6.vnnlib");
    require_benchmark_file(&onnx);
    require_benchmark_file(&vnnlib);
    let armed = run_scored(&onnx, &vnnlib, "120", None);
    let disarmed = run_scored(&onnx, &vnnlib, "120", Some("0"));
    assert!(
        armed
            .stderr
            .contains("Sign-space falsification: not admitted"),
        "the preset-armed lane must DECLINE this net rather than search it:\n{}",
        armed.stderr
    );
    assert!(
        !disarmed.stderr.contains("Sign-space falsification"),
        "NY_BNN_SIGN_SPACE=0 must keep the lane silent:\n{}",
        disarmed.stderr
    );
    assert_eq!(
        armed.result, disarmed.result,
        "a lane decline changed the scored verdict ({} armed vs {} disarmed)\n\
         armed stderr:\n{}\ndisarmed stderr:\n{}",
        armed.result, disarmed.result, armed.stderr, disarmed.stderr
    );
    let refusal_secs = lane_seconds(&armed.stderr)
        .expect("the lane's stderr line must carry its own elapsed time");
    assert!(
        refusal_secs < 10.0,
        "a structural refusal cost {refusal_secs:.2}s of the scored budget; the default-armed \
         scope is only defensible while this is noise against the 480s budget\n{}",
        armed.stderr
    );
}

/// Seconds from the lane's `... [N.NNs]` stderr suffix.
fn lane_seconds(stderr: &str) -> Option<f64> {
    let line = stderr
        .lines()
        .find(|line| line.starts_with("Sign-space falsification:"))?;
    let start = line.rfind('[')?;
    let end = line.rfind("s]")?;
    line.get(start + 1..end)?.parse().ok()
}
