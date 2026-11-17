// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use tempfile::TempDir;

fn write(root: &TempDir, relative: &str, body: &str) -> PathBuf {
    let path = root.path().join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, body).unwrap();
    path
}

fn row(category: &str, property: &str, verdict: &str, seconds: &str) -> String {
    format!(
        "{category},corpus/vnncomp2026/benchmarks/{category}/2.0/onnx/m.onnx,\
             corpus/vnncomp2026/benchmarks/{category}/2.0/vnnlib/{property}.vnnlib,\
             0,{verdict},{seconds}\n"
    )
}

fn harness(seconds: &str) -> String {
    "test,corpus/vnncomp2026/benchmarks/test/2.0/onnx/test_nano.onnx,\
     corpus/vnncomp2026/benchmarks/test/2.0/vnnlib/test_nano.vnnlib,0,unsat,"
        .to_owned()
        + seconds
        + "\n"
}

fn budgets(field: &OfficialField, seconds: u64) -> BudgetTable {
    field
        .values()
        .flat_map(|rows| rows.keys())
        .map(|key| (key.semantic_key(), seconds))
        .collect()
}

#[test]
fn field_startup_minimum_includes_decided_harness_rows_without_scoring_them() {
    let root = TempDir::new().unwrap();
    let body = harness("2")
        + &row("acasxu_2023", "at", "sat", "12")
        + &row("acasxu_2023", "late", "sat", "12.125");
    let source = write(&root, "tool/results.csv", &body);
    let mut field = load_official_field(root.path()).unwrap();
    assert_eq!(field["tool"].len(), 2);
    let before = field["tool"]
        .iter()
        .find(|(key, _)| key.vnnlib.ends_with("late.vnnlib"))
        .unwrap()
        .1
        .organizer_row_identity
        .clone();
    let table = budgets(&field, 10);
    let report = normalize(root.path(), &mut field, &table).unwrap();
    assert_eq!(report.tools["tool"].startup_overhead_seconds, 2.);
    assert_eq!(report.forced_timeouts, 1);
    let at = field["tool"]
        .iter()
        .find(|(key, _)| key.vnnlib.ends_with("at.vnnlib"))
        .unwrap()
        .1;
    let late = field["tool"]
        .iter()
        .find(|(key, _)| key.vnnlib.ends_with("late.vnnlib"))
        .unwrap()
        .1;
    assert_eq!(at.verdict, Verdict::Sat);
    assert_eq!(late.verdict, Verdict::Undecided);
    assert_eq!(late.seconds, 12.125);
    assert_eq!(late.organizer_row_identity, before);
    assert_eq!(std::fs::read_to_string(source).unwrap(), body);
}

#[test]
fn field_startup_uses_unselected_track_decisions_and_is_independent_per_tool() {
    let root = TempDir::new().unwrap();
    let extended = "adaptive_cruise_control_non_linear_2026";
    write(
        &root,
        "first/results.csv",
        &(harness("1")
            + &row(extended, "fast", "unsat", "0.5")
            + &row("acasxu_2023", "p", "unsat", "10.5")),
    );
    write(
        &root,
        "second/results.csv",
        &(harness("0.25") + &row("acasxu_2023", "p", "unsat", "10.5")),
    );
    let mut field = load_official_field(root.path()).unwrap();
    let table = budgets(&field, 10);
    let report = normalize(root.path(), &mut field, &table).unwrap();
    assert_eq!(report.tools["first"].startup_overhead_seconds, 0.5);
    assert_eq!(report.tools["second"].startup_overhead_seconds, 0.25);
    assert!(field["first"]
        .keys()
        .any(|key| key.benchmark == "acasxu_2023"));
    assert!(!field.contains_key("second"));
    assert_eq!(
        report.tools["second"].removed_categories,
        vec!["acasxu_2023".to_owned()]
    );
}

#[test]
fn field_timeouts_use_original_instance_identity_instead_of_result_position() {
    let root = TempDir::new().unwrap();
    write(
        &root,
        "tool/results.csv",
        &(harness("2")
            + &row("nn4sys", "long", "unsat", "12")
            + &row("nn4sys", "short", "unsat", "12")),
    );
    let mut field = load_official_field(root.path()).unwrap();
    let table = field["tool"]
        .keys()
        .map(|key| {
            (
                key.semantic_key(),
                if key.vnnlib.ends_with("long.vnnlib") {
                    20
                } else {
                    1
                },
            )
        })
        .collect();
    normalize(root.path(), &mut field, &table).unwrap();
    assert_eq!(
        field["tool"]
            .iter()
            .find(|(key, _)| key.vnnlib.ends_with("long.vnnlib"))
            .unwrap()
            .1
            .verdict,
        Verdict::Unsat
    );
    assert_eq!(
        field["tool"]
            .iter()
            .find(|(key, _)| key.vnnlib.ends_with("short.vnnlib"))
            .unwrap()
            .1
            .verdict,
        Verdict::Undecided
    );
}

#[test]
fn late_field_strict_witness_loses_authority_before_shared_denominator_scoring() {
    let root = TempDir::new().unwrap();
    write(
        &root,
        "late/results.csv",
        &(harness("2")
            + &row("acasxu_2023", "contested", "sat", "13")
            + &row("acasxu_2023", "retained", "unsat", "4")),
    );
    write(
        &root,
        "holds/results.csv",
        &(harness("0.5") + &row("acasxu_2023", "contested", "unsat", "1")),
    );
    let mut field = load_official_field(root.path()).unwrap();
    for entry in field.get_mut("late").unwrap().values_mut() {
        if entry.verdict == Verdict::Sat {
            entry.counterexample = Some(CounterexampleCheck::Strict);
        }
    }
    let table = budgets(&field, 10);
    normalize(root.path(), &mut field, &table).unwrap();
    let late = field["late"]
        .values()
        .find(|entry| entry.seconds == 13.)
        .unwrap();
    assert_eq!(late.verdict, Verdict::Undecided);
    assert_eq!(late.counterexample, None);
    let modeled = modeled_official_score(
        "acasxu_2023",
        &ResultRows::new(),
        &field,
        None,
        ScoreYear::Y2026,
        ScoreTrack::Regular,
        &BTreeSet::new(),
        Witnesses::Adjudicated,
        None,
    )
    .unwrap();
    assert_eq!(modeled.field_scores["holds"].raw, 10);
    assert_eq!(modeled.field_scores["late"].raw, 10);
    assert_eq!(modeled.field_best_raw, 10);
}

#[test]
fn original_field_cohort_survives_all_late_incumbents_for_a_valid_ny_entrant() {
    let root = TempDir::new().unwrap();
    write(
        &root,
        "late/results.csv",
        &(harness("2") + &row("acasxu_2023", "original", "sat", "13")),
    );
    let mut field = load_official_field(root.path()).unwrap();
    let cohort = field
        .values()
        .flat_map(|rows| rows.keys())
        .cloned()
        .collect();
    let table = budgets(&field, 10);
    normalize(root.path(), &mut field, &table).unwrap();
    assert!(field.is_empty());
    let ny = write(
        &root,
        "ny.csv",
        &row("acasxu_2023", "original", "unsat", "5"),
    );
    let current = read_results(&ny).unwrap();
    let evaluate = |current: &ResultRows, excluded: &BTreeSet<InstanceKey>| {
        modeled_official_score(
            "acasxu_2023",
            current,
            &field,
            Some(&cohort),
            ScoreYear::Y2026,
            ScoreTrack::Regular,
            excluded,
            Witnesses::Adjudicated,
            None,
        )
    };
    let score = evaluate(&current, &BTreeSet::new()).unwrap();
    assert_eq!(
        (score.current_raw, score.field_best_raw, score.normalized),
        (10, 10, 100.)
    );
    assert!(
        score.field_scores.is_empty(),
        "no zero-point incumbent is invented"
    );
    let excluded = current.keys().cloned().collect();
    assert!(
        evaluate(&current, &excluded).is_none(),
        "late NY cannot activate the category"
    );
    let other = write(
        &root,
        "outside.csv",
        &row("acasxu_2023", "outside", "unsat", "1"),
    );
    assert!(
        evaluate(&read_results(&other).unwrap(), &BTreeSet::new()).is_none(),
        "a different NY row cannot invent an original-field occurrence"
    );
}

#[test]
fn field_without_decisions_uses_zero_overhead_and_drops_undecided_categories() {
    let root = TempDir::new().unwrap();
    write(
        &root,
        "tool/results.csv",
        &row("acasxu_2023", "p", "unknown", "50"),
    );
    let mut field = load_official_field(root.path()).unwrap();
    let table = budgets(&field, 10);
    let report = normalize(root.path(), &mut field, &table).unwrap();
    assert_eq!(report.tools["tool"].startup_overhead_seconds, 0.);
    assert_eq!(report.forced_timeouts, 0);
    assert_eq!(report.removed_categories, 1);
    assert!(field.is_empty());
}

#[test]
fn field_normalization_refuses_partial_aggregate_unknown_budget_and_version_mismatch() {
    let root = TempDir::new().unwrap();
    write(
        &root,
        "tool/results.csv",
        &row("acasxu_2023", "p", "sat", "1"),
    );
    let mut field = load_official_field(root.path()).unwrap();
    assert!(normalize(root.path(), &mut field, &BudgetTable::new())
        .unwrap_err()
        .to_string()
        .contains("no original instance budget"));
    let mut field = load_official_field(root.path()).unwrap();
    field
        .get_mut("tool")
        .unwrap()
        .values_mut()
        .next()
        .unwrap()
        .seconds = 2.;
    let table = budgets(&field, 10);
    assert!(normalize(root.path(), &mut field, &table)
        .unwrap_err()
        .to_string()
        .contains("differs from its complete organizer aggregate"));
    let body = row("acasxu_2023", "p", "sat", "1").replace("/2.0/vnnlib/", "/1.0/vnnlib/");
    write(&root, "tool/results.csv", &body);
    let mut field = load_official_field(root.path()).unwrap();
    let table = budgets(&field, 10);
    assert!(normalize(root.path(), &mut field, &table)
        .unwrap_err()
        .to_string()
        .contains("versions disagree"));
}

#[test]
fn field_normalization_requires_organizer_aggregate_layout_and_six_column_source() {
    let root = TempDir::new().unwrap();
    write(
        &root,
        "tool/2026_acasxu_2023/results.csv",
        &row("acasxu_2023", "p", "sat", "1"),
    );
    let mut field = load_official_field(root.path()).unwrap();
    let table = budgets(&field, 10);
    assert!(normalize(root.path(), &mut field, &table)
        .unwrap_err()
        .to_string()
        .contains("complete organizer aggregate"));
    let mut fields = parse_csv_line(row("acasxu_2023", "p", "sat", "1").trim()).unwrap();
    fields.remove(3);
    write(&root, "tool/results.csv", &(fields.join(",") + "\n"));
    // Exercise the aggregate scan directly: legacy five-column inputs may
    // still be compared locally, but cannot claim this organizer policy.
    assert!(overhead(&root.path().join("tool/results.csv"))
        .unwrap_err()
        .to_string()
        .contains("six-column organizer rows"));
}
