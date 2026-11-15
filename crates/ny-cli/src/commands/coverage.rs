// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny coverage`: surface the trace-bridge soundness-coverage manifest.
//!
//! An exhaustive match classifies each `TraceOp` as `exact`, `sound`,
//! `sound_but_loose` or `unsupported`. These describe possible relaxations,
//! not current `translate()` coverage or the numeric soundness of a run.
//! Classified but unimplemented operators still fail with `UnsupportedOp`.

use std::fmt::Write as _;

use anyhow::Result;
use ny_trace_bridge::coverage::{coverage, CoverageReport};
use serde_json::{json, Value};

/// Handle the `ny coverage` command.
pub(crate) fn handle_coverage_command(json_output: bool) -> Result<()> {
    let report = coverage();
    if json_output {
        println!("{}", serde_json::to_string_pretty(&coverage_json(&report))?);
    } else {
        print!("{}", coverage_text(&report));
    }
    Ok(())
}

/// Build the machine-readable manifest: bucket counts plus the op names in each
/// soundness class. Keys are the stable soundness-class labels (the `loose`
/// field is surfaced under its full `sound_but_loose` name for consumers).
fn coverage_json(report: &CoverageReport) -> Value {
    json!({
        "total": report.total(),
        "translatable": report.translatable(),
        "unsupported": report.unsupported.len(),
        "counts": {
            "exact": report.exact.len(),
            "sound": report.sound.len(),
            "sound_but_loose": report.loose.len(),
            "unsupported": report.unsupported.len(),
        },
        "ops": {
            "exact": report.exact,
            "sound": report.sound,
            "sound_but_loose": report.loose,
            "unsupported": report.unsupported,
        },
        // The taxonomy caveat travels WITH the machine-readable data (not only in
        // the text footer) so a consumer piping this JSON into a report cannot
        // over-read "translatable" as "implemented + numerically sound today".
        "note": "Soundness TAXONOMY, not current translate() implementation status: \
                 a 'translatable' op may still be refused today with an UnsupportedOp \
                 error (fail-closed). The exact/sound/sound_but_loose classes describe \
                 the abstract bound-propagation relaxation and are independent of the \
                 numeric (f32 vs directed-rounding) soundness of any given run.",
    })
}

/// Build the human-readable manifest report as a single string (so the exact
/// output is unit-testable without capturing stdout).
fn coverage_text(report: &CoverageReport) -> String {
    let mut out = String::new();
    // These pushes are infallible (writing into a String never errors); the
    // `let _ =` keeps clippy quiet without masking a real fallible write.
    let _ = writeln!(out, "NY trace-bridge soundness-coverage manifest");
    let _ = writeln!(out, "===========================================");
    let _ = writeln!(out, "Total TraceOp variants classified: {}", report.total());
    let _ = writeln!(
        out,
        "  translatable (a sound lowering exists in principle): {}",
        report.translatable()
    );
    let _ = writeln!(out, "    exact            : {}", report.exact.len());
    let _ = writeln!(out, "    sound            : {}", report.sound.len());
    let _ = writeln!(out, "    sound_but_loose  : {}", report.loose.len());
    let _ = writeln!(
        out,
        "  unsupported (data-dependent shape/routing — refused, sound by construction): {}",
        report.unsupported.len()
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "Soundness classes:");
    let _ = writeln!(
        out,
        "  exact            bounds reproduce the op exactly (reshape / reindex / sign-aware affine)"
    );
    let _ = writeln!(
        out,
        "  sound            correct, practically-tight over-approximation (ReLU family, clamps, monotone activations)"
    );
    let _ = writeln!(
        out,
        "  sound_but_loose  correct but knowingly loose (normalisation, softmax, attention, reductions, transcendentals; Custom is the vacuous ±inf extreme)"
    );
    let _ = writeln!(
        out,
        "  unsupported      output shape/routing is data-dependent — the bridge REFUSES these (sound by construction)"
    );
    let _ = writeln!(out);

    push_bucket(&mut out, "exact", &report.exact);
    push_bucket(&mut out, "sound", &report.sound);
    push_bucket(&mut out, "sound_but_loose", &report.loose);
    push_bucket(&mut out, "unsupported", &report.unsupported);

    let _ = writeln!(
        out,
        "Note: this is the soundness TAXONOMY (which ops admit a sound lowering), not"
    );
    let _ = writeln!(
        out,
        "current translate() implementation coverage. translate() refuses any not-yet-"
    );
    let _ = writeln!(
        out,
        "ported op with an explicit UnsupportedOp error (fail-closed). The exact/sound"
    );
    let _ = writeln!(
        out,
        "classes describe the abstract relaxation and are independent of the numeric"
    );
    let _ = writeln!(
        out,
        "(f32 vs directed-rounding) soundness of any given verification run."
    );
    out
}

/// Append one soundness bucket: a `label (count):` header followed by the op
/// names wrapped to ~78 columns for terminal readability.
fn push_bucket(out: &mut String, label: &str, ops: &[String]) {
    let _ = writeln!(out, "{label} ({}):", ops.len());
    if ops.is_empty() {
        let _ = writeln!(out, "  (none)");
        let _ = writeln!(out);
        return;
    }
    let mut line = String::new();
    for op in ops {
        if !line.is_empty() && line.len() + op.len() + 2 > 78 {
            let _ = writeln!(out, "  {line}");
            line.clear();
        }
        if line.is_empty() {
            line.push_str(op);
        } else {
            line.push_str(", ");
            line.push_str(op);
        }
    }
    if !line.is_empty() {
        let _ = writeln!(out, "  {line}");
    }
    let _ = writeln!(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The number of `TraceOp` variants, pinned in the bridge's own tests. If
    /// this changes, the bridge catalogue changed and this manifest tracks it.
    const EXPECTED_TRACE_OP_VARIANTS: usize = 123;

    #[test]
    fn json_manifest_partitions_all_variants() {
        let report = coverage();
        let v = coverage_json(&report);
        assert_eq!(v["total"], EXPECTED_TRACE_OP_VARIANTS);
        // translatable + unsupported == total (the four buckets partition the set).
        assert_eq!(
            v["translatable"].as_u64().unwrap() + v["unsupported"].as_u64().unwrap(),
            EXPECTED_TRACE_OP_VARIANTS as u64
        );
        // Bucket counts sum to the total.
        let counts = &v["counts"];
        let sum = counts["exact"].as_u64().unwrap()
            + counts["sound"].as_u64().unwrap()
            + counts["sound_but_loose"].as_u64().unwrap()
            + counts["unsupported"].as_u64().unwrap();
        assert_eq!(sum, EXPECTED_TRACE_OP_VARIANTS as u64);
        // The op-name arrays are present and non-empty for the populated buckets.
        assert_ne!((v["ops"]["exact"].as_array().unwrap()).len(), 0);
        assert_ne!((v["ops"]["unsupported"].as_array().unwrap()).len(), 0);
        // The taxonomy caveat travels with the machine-readable payload.
        let note = v["note"].as_str().expect("note field present");
        assert!(note.contains("TAXONOMY") && note.contains("UnsupportedOp"));
    }

    #[test]
    fn json_unsupported_lists_the_data_dependent_ops() {
        let report = coverage();
        let v = coverage_json(&report);
        let unsupported: Vec<String> = v["ops"]["unsupported"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect();
        // Spot-check the must-refuse ops appear in the manifest.
        //
        // Trimmed as the bridge grew real lowerings — the manifest is a
        // taxonomy, so an op leaves this list the moment a sound lowering
        // exists, and pinning a stale name here would misreport coverage:
        //   - `Argmax` / `Argmin` / `ArgSort` are now Sound (exact index
        //     under interval separation, [0, n-1] hull otherwise) and `Sort`
        //     Exact (monotone, tie-agnostic order statistics).
        //   - `WhereCond` is now sound_but_loose (union-hull select).
        //   - `Topk` left the list at e4abe1110: the bridge lowers it now that
        //     nn serializes the largest/sorted flags (SoundButLoose for
        //     `sorted: false`), so pinning it here misreported coverage — this
        //     stale pin held every stream's `--bin ny` gate red for a day.
        //   - `Custom` moved to sound_but_loose at INC-FINAL: the explicit
        //     opaque escape hatch lowers to the vacuous-but-sound ±inf
        //     OpaqueSkip.
        for op in ["Scatter", "MoeGating", "GridSample"] {
            assert!(
                unsupported.iter().any(|o| o == op),
                "{op} must be listed as unsupported in the manifest"
            );
        }
        let loose: Vec<String> = v["ops"]["sound_but_loose"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect();
        assert!(
            loose.iter().any(|o| o == "Custom"),
            "Custom must be listed sound_but_loose (vacuous ±inf OpaqueSkip)"
        );
        assert!(
            loose.iter().any(|o| o == "WhereCond"),
            "WhereCond must be listed sound_but_loose (union-hull select)"
        );
        let sound: Vec<String> = v["ops"]["sound"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect();
        for op in ["Argmax", "Argmin", "ArgSort"] {
            assert!(
                sound.iter().any(|o| o == op),
                "{op} must be listed sound (index layers), not unsupported"
            );
        }
    }

    #[test]
    fn text_manifest_is_nonempty_and_labels_every_bucket() {
        let report = coverage();
        let text = coverage_text(&report);
        assert!(text.contains("soundness-coverage manifest"));
        for label in ["exact (", "sound (", "sound_but_loose (", "unsupported ("] {
            assert!(
                text.contains(label),
                "text manifest must label bucket {label}"
            );
        }
        // The honest-limits note must be present (taxonomy, not impl status).
        assert!(text.contains("soundness TAXONOMY"));
    }

    #[test]
    fn handler_runs_in_both_modes() {
        // Exercises the full handler wiring (serialization + formatting) end to
        // end; both modes must succeed.
        handle_coverage_command(true).expect("json coverage handler runs");
        handle_coverage_command(false).expect("text coverage handler runs");
    }
}
