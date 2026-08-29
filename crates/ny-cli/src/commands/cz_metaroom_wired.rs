// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed production boundary for the independently certified MetaRoom
//! constrained-zonotope lane.
//!
//! The qualification implementation intentionally contains sealed structural
//! assertions. This boundary supplies the two missing production contracts:
//! exact typed-preset admission and panic containment. A mismatch, missing
//! asset, deadline, or failed certificate always declines to the ordinary
//! verifier pipeline; only a report that independently closes every challenger
//! may publish `unsat`.

use std::ffi::OsStr;
use std::path::Path;
use std::time::Instant;

use super::cz_metaroom_unwired::qualification::{
    run_wired_metaroom_cz_hybrid, MetaroomAssets, MetaroomCzWiredReport,
};
use crate::preset::PresetConfig;

const RECEIPT_SCHEMA: &str = "ny-cli/cz-hybrid-lane/v1";

#[derive(Debug)]
pub(crate) enum CzHybridLaneOutcome {
    /// The exact typed allowlist did not select this instance.
    Skipped { reason: &'static str },
    /// The lane was selected but failed closed without a verdict.
    Declined {
        reason: &'static str,
        report: Option<MetaroomCzWiredReport>,
    },
    /// Every sealed challenger margin has an independent positive certificate.
    ProvedUnsat { report: MetaroomCzWiredReport },
}

impl CzHybridLaneOutcome {
    #[must_use]
    pub(crate) fn engaged(&self) -> bool {
        !matches!(self, Self::Skipped { .. })
    }

    #[must_use]
    pub(crate) fn receipt(&self) -> String {
        let value = match self {
            Self::Skipped { reason } => serde_json::json!({
                "schema": RECEIPT_SCHEMA,
                "status": "skipped",
                "reason": reason,
            }),
            Self::Declined { reason, report } => serde_json::json!({
                "schema": RECEIPT_SCHEMA,
                "status": "declined",
                "reason": reason,
                "report": report.as_ref().map(report_receipt),
            }),
            Self::ProvedUnsat { report } => serde_json::json!({
                "schema": RECEIPT_SCHEMA,
                "status": "proved_unsat",
                "report": report_receipt(report),
            }),
        };
        value.to_string()
    }
}

fn report_receipt(report: &MetaroomCzWiredReport) -> serde_json::Value {
    serde_json::json!({
        "proved_unsat": report.proved_unsat,
        "trunk_relu_rows": report.hybrid_relu_rows.len(),
        "relu5_cz_unstable": report.relu5_cz_unstable,
        "relu5_box_unstable": report.relu5_box_unstable,
        "certificates": report.certificates.len(),
        "strictly_positive": report
            .certificates
            .iter()
            .filter(|certificate| certificate.strictly_positive)
            .count(),
        "unresolved": report.unresolved,
        "trunk_secs": report.trunk_elapsed.as_secs_f64(),
        "tail_secs": report.tail_elapsed.as_secs_f64(),
    })
}

fn admitted(preset: Option<&PresetConfig>, onnx: &Path, vnnlib: &Path) -> bool {
    let Some(preset) = preset else { return false };
    let (Some(expected_onnx), Some(expected_vnnlib)) = (
        preset.cz_hybrid.only_onnx.as_deref(),
        preset.cz_hybrid.only_vnnlib.as_deref(),
    ) else {
        return false;
    };
    onnx.file_name() == Some(OsStr::new(expected_onnx))
        && vnnlib.file_name() == Some(OsStr::new(expected_vnnlib))
}

#[must_use]
pub(crate) fn evaluate(
    preset: Option<&PresetConfig>,
    onnx: &Path,
    vnnlib: &Path,
    deadline: Instant,
) -> CzHybridLaneOutcome {
    if !admitted(preset, onnx, vnnlib) {
        return CzHybridLaneOutcome::Skipped {
            reason: "exact_instance_allowlist_miss",
        };
    }
    if Instant::now() >= deadline {
        return CzHybridLaneOutcome::Declined {
            reason: "deadline_reached_before_start",
            report: None,
        };
    }

    let assets = MetaroomAssets {
        onnx: onnx.to_path_buf(),
        vnnlib: vnnlib.to_path_buf(),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_wired_metaroom_cz_hybrid(&assets, deadline)
    }));
    match result {
        Ok(Some(report)) if report.proved_unsat => CzHybridLaneOutcome::ProvedUnsat { report },
        Ok(Some(report)) => CzHybridLaneOutcome::Declined {
            reason: "challengers_not_all_certified",
            report: Some(report),
        },
        Ok(None) => CzHybridLaneOutcome::Declined {
            reason: "qualification_assets_unavailable",
            report: None,
        },
        Err(_) => CzHybridLaneOutcome::Declined {
            reason: "sealed_qualification_assertion",
            report: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::admitted;
    use crate::preset::PresetConfig;
    use std::path::Path;

    fn selected() -> PresetConfig {
        serde_yaml::from_str(
            "cz_hybrid:\n  only_onnx: model.onnx\n  only_vnnlib: property.vnnlib\n",
        )
        .expect("typed CZ admission preset")
    }

    #[test]
    fn exact_two_basename_allowlist_is_required() {
        let preset = selected();
        assert!(admitted(
            Some(&preset),
            Path::new("/models/model.onnx"),
            Path::new("/properties/property.vnnlib")
        ));
        assert!(!admitted(
            Some(&preset),
            Path::new("/models/other.onnx"),
            Path::new("/properties/property.vnnlib")
        ));
        assert!(!admitted(
            None,
            Path::new("model.onnx"),
            Path::new("property.vnnlib")
        ));
    }

    #[test]
    fn partial_allowlists_fail_closed() {
        let preset: PresetConfig = serde_yaml::from_str("cz_hybrid:\n  only_onnx: model.onnx\n")
            .expect("partial preset still parses");
        assert!(!admitted(
            Some(&preset),
            Path::new("model.onnx"),
            Path::new("property.vnnlib")
        ));
    }
}
