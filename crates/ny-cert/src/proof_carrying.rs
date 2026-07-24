// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-carrying certificate composer — the **cite-discharge at the composition
//! level** (Trust = Clean fusion, v1).
//!
//! tRustc verifies the checker's L0 safety and *captures* its L1 soundness
//! postcondition, but ay alone cannot discharge that postcondition over the opaque
//! bignum `Rat` type (see `SPEC.md` / task #24). The soundness link *is* proved — in
//! Clean, by `farkas_premise_combination` — and [`crate::cite_check`] machine-verifies
//! that theorem is declared and sorry-free in the exact pinned Clean dependency.
//!
//! This module **composes** those two streams of *already-verified* evidence — the
//! tRustc per-function result and the cite_check grounding — into the honest
//! per-function [`ProofCarryingStatus`]. Crucially it does **not** modify the verifier,
//! so it cannot introduce unsoundness: it can only *report* a status that the inputs
//! already justify, and it is **fail-closed** (a missing or ungrounded citation never
//! yields a "certified" status). The planned `trust_verify` integration (task #24) is
//! a v2 that emits the same `CertifiedModuloCite` status from inside the verifier.

use crate::cite_check::{citation_status, CitationStatus};
use std::collections::HashMap;
use std::path::Path;

/// The honest proof-carrying status of one function, composed from the tRustc
/// verification result and the Clean-grounding verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofCarryingStatus {
    /// L0 safety proved by tRustc **and** the L1 postcondition discharged by the
    /// solver alone — sound to the kernel's axioms with no extra assumption.
    CertifiedToAxioms,
    /// L0 safety proved; the L1 postcondition is captured but not solver-discharged,
    /// and is grounded in a Clean-kernel-checked theorem (which is therefore in the
    /// result's assumption closure). The honest "modulo a named, kernel-checked
    /// theorem" status — the realized cite-discharge.
    CertifiedModuloCite { theorem: String },
    /// L0 safety proved; the L1 postcondition is open and not cited (no grounding).
    L0OnlyL1Open,
    /// L0 safety not fully proved, or a cited theorem failed to resolve/ground.
    Incomplete { reason: String },
}

/// The tRustc-derived facts about one function (extracted from a `targo trust survey`
/// result): whether its L0 safety obligations all proved, and whether its L1
/// `#[ensures]` postcondition was captured and/or solver-discharged.
#[derive(Debug, Clone, Copy)]
pub struct FunctionFacts {
    /// All L0 safety obligations (panic/overflow/index) proved.
    pub l0_proved: bool,
    /// An `#[ensures]` postcondition obligation exists for the function.
    pub l1_postcond_captured: bool,
    /// The L1 postcondition was discharged by the solver (ay) itself.
    pub l1_postcond_discharged: bool,
}

/// Compose the honest proof-carrying status for one function. Fail-closed: a captured-
/// but-undischarged postcondition only reaches `CertifiedModuloCite` when the cited
/// theorem resolves to a declared, sorry-free theorem in the corpus at `corpus_root`.
#[must_use]
pub fn classify(
    facts: FunctionFacts,
    cite: Option<&str>,
    corpus_root: &Path,
) -> ProofCarryingStatus {
    // Manual ASCII reason construction (`String::from` + `push_str` of literal
    // parts, no `format!`/`str::to_string`): extern `core::fmt` dispatch is
    // opaque to the verifier. Byte-identical to the previous `format!` output.
    // total: exhaustive over the fieldless `CitationStatus` variants — each
    // string is exactly the variant's derived-`Debug` rendering, and a future
    // variant is a compile error here rather than a silent mismatch.
    fn status_name(s: &CitationStatus) -> &'static str {
        match s {
            CitationStatus::Grounded => "Grounded",
            CitationStatus::NotFound => "NotFound",
            CitationStatus::HasSorry => "HasSorry",
        }
    }
    if !facts.l0_proved {
        return ProofCarryingStatus::Incomplete {
            reason: String::from("L0 safety obligations not all proved"),
        };
    }
    if facts.l1_postcond_discharged {
        return ProofCarryingStatus::CertifiedToAxioms;
    }
    if facts.l1_postcond_captured {
        if let Some(theorem) = cite {
            return match citation_status(corpus_root, theorem) {
                Ok(CitationStatus::Grounded) => ProofCarryingStatus::CertifiedModuloCite {
                    theorem: String::from(theorem),
                },
                Ok(other) => {
                    let mut reason = String::from("cited theorem `");
                    reason.push_str(theorem);
                    reason.push_str("` not grounded (");
                    reason.push_str(status_name(&other));
                    reason.push(')');
                    ProofCarryingStatus::Incomplete { reason }
                }
                Err(e) => {
                    // The literal/theorem parts are manual pushes; the error text
                    // itself (`std::io::Error`'s `Display` — platform-dependent
                    // OS strings) is not manually reconstructible, so this one
                    // `to_string` dispatch remains (byte-identical to `{e}`).
                    let mut reason = String::from("corpus read error resolving `");
                    reason.push_str(theorem);
                    reason.push_str("`: ");
                    reason.push_str(&e.to_string());
                    ProofCarryingStatus::Incomplete { reason }
                }
            };
        }
    }
    ProofCarryingStatus::L0OnlyL1Open
}

/// Extract `function -> verdict` (e.g. "Verified" / "HasViolations") from a
/// `targo trust survey` JSON value.
#[must_use]
pub fn function_verdicts(survey: &serde_json::Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Some(fns) = survey["functions"].as_array() {
        for f in fns {
            if let (Some(name), Some(verdict)) =
                (f["function"].as_str(), f["summary"]["verdict"].as_str())
            {
                out.insert(name.to_string(), verdict.to_string());
            }
        }
    }
    out
}

/// Compose the proof-carrying certificate from two `targo trust survey` results and
/// the cite-map. The clean L0/L1 separation: the **structural** survey (no
/// `--contracts`) verifies L0 safety only; the **contracts** survey adds the L1
/// postcondition, so a function `Verified` structurally is L0-safe, and `Verified`
/// under contracts means the postcondition discharged too. A function L0-safe but
/// only-failing-under-contracts has exactly its postcondition open — discharged
/// modulo its cited theorem (fail-closed via [`classify`]).
#[must_use]
pub fn compose_from_surveys(
    structural: &serde_json::Value,
    contracts: &serde_json::Value,
    cite_map: &[(String, String)],
    corpus_root: &Path,
) -> Vec<(String, ProofCarryingStatus)> {
    let l0 = function_verdicts(structural);
    let l1 = function_verdicts(contracts);
    // Explicit loop + `match` booleans instead of `.map(closure).collect()` with
    // inner `is_some_and(|v| …)` closures: behaviour-identical, but the inlined
    // body carries no closure-bundling obligation and no absent `map`/`collect`
    // adapter (a directly-invoked closure inside an absent std iterator adapter
    // is otherwise opaque to the verifier).
    let mut out: Vec<(String, ProofCarryingStatus)> = Vec::new();
    for (func, theorem) in cite_map {
        let l0_proved = match l0.get(func) {
            Some(v) => v == "Verified",
            None => false,
        };
        let l1_postcond_discharged = match l1.get(func) {
            Some(v) => v == "Verified",
            None => false,
        };
        let facts = FunctionFacts {
            l0_proved,
            l1_postcond_captured: l1.contains_key(func),
            l1_postcond_discharged,
        };
        out.push((
            func.clone(),
            classify(facts, Some(theorem.as_str()), corpus_root),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cite_check::clean_corpus_root;

    const PROVED_CAPTURED: FunctionFacts = FunctionFacts {
        l0_proved: true,
        l1_postcond_captured: true,
        l1_postcond_discharged: false,
    };

    #[test]
    fn captured_postcondition_with_grounded_cite_is_modulo_cite() {
        let s = classify(
            PROVED_CAPTURED,
            Some("farkas_premise_combination"),
            &clean_corpus_root(),
        );
        assert_eq!(
            s,
            ProofCarryingStatus::CertifiedModuloCite {
                theorem: "farkas_premise_combination".to_string()
            }
        );
    }

    #[test]
    fn fail_closed_on_missing_cite() {
        // No citation → never "certified", just honestly L0-only/L1-open.
        let s = classify(PROVED_CAPTURED, None, &clean_corpus_root());
        assert_eq!(s, ProofCarryingStatus::L0OnlyL1Open);
    }

    #[test]
    fn fail_closed_on_ungrounded_cite() {
        // A citation to a non-existent theorem must NOT yield certified-modulo.
        let s = classify(
            PROVED_CAPTURED,
            Some("no_such_theorem_xyz"),
            &clean_corpus_root(),
        );
        assert!(
            matches!(s, ProofCarryingStatus::Incomplete { .. }),
            "got {s:?}"
        );
    }

    #[test]
    fn solver_discharged_is_certified_to_axioms() {
        let facts = FunctionFacts {
            l1_postcond_discharged: true,
            ..PROVED_CAPTURED
        };
        assert_eq!(
            classify(
                facts,
                Some("farkas_premise_combination"),
                &clean_corpus_root()
            ),
            ProofCarryingStatus::CertifiedToAxioms
        );
    }

    #[test]
    fn unproved_l0_is_incomplete_regardless_of_cite() {
        let facts = FunctionFacts {
            l0_proved: false,
            ..PROVED_CAPTURED
        };
        assert!(matches!(
            classify(
                facts,
                Some("farkas_premise_combination"),
                &clean_corpus_root()
            ),
            ProofCarryingStatus::Incomplete { .. }
        ));
    }

    #[test]
    fn compose_from_two_surveys_yields_modulo_cite() {
        // Structural survey (L0 only): check_farkas L0-safe → Verified.
        let structural = serde_json::json!({
            "functions": [{ "function": "selfcheck::check_farkas", "summary": { "verdict": "Verified" } }]
        });
        // Contracts survey (L0 + L1): the postcondition fails → HasViolations.
        let contracts = serde_json::json!({
            "functions": [{ "function": "selfcheck::check_farkas", "summary": { "verdict": "HasViolations" } }]
        });
        let cite_map = vec![(
            "selfcheck::check_farkas".to_string(),
            "farkas_premise_combination".to_string(),
        )];
        let cert = compose_from_surveys(&structural, &contracts, &cite_map, &clean_corpus_root());
        assert_eq!(
            cert,
            vec![(
                "selfcheck::check_farkas".to_string(),
                ProofCarryingStatus::CertifiedModuloCite {
                    theorem: "farkas_premise_combination".to_string()
                }
            )]
        );
    }

    #[test]
    fn compose_fail_closed_when_l0_unproved() {
        // If L0 safety is not Verified structurally, never certify — even with a cite.
        let structural = serde_json::json!({
            "functions": [{ "function": "selfcheck::check_farkas", "summary": { "verdict": "HasViolations" } }]
        });
        let contracts = structural.clone();
        let cite_map = vec![(
            "selfcheck::check_farkas".to_string(),
            "farkas_premise_combination".to_string(),
        )];
        let cert = compose_from_surveys(&structural, &contracts, &cite_map, &clean_corpus_root());
        assert!(
            matches!(cert[0].1, ProofCarryingStatus::Incomplete { .. }),
            "got {:?}",
            cert[0].1
        );
    }

    /// The current, honest proof-carrying status of `check_farkas`, composed from the
    /// live-survey facts (L0 proved; L1 captured but ay-undischarged) and the verified
    /// grounding: `CertifiedModuloCite(farkas_premise_combination)`.
    #[test]
    fn check_farkas_current_status_is_modulo_cite() {
        // Facts as established by `targo trust survey ny-cert --contracts` (2026-06-27):
        // L0 verified, the #[ensures] postcondition captured but not ay-discharged.
        let s = classify(
            PROVED_CAPTURED,
            Some("farkas_premise_combination"),
            &clean_corpus_root(),
        );
        assert_eq!(
            s,
            ProofCarryingStatus::CertifiedModuloCite {
                theorem: "farkas_premise_combination".to_string()
            },
            "check_farkas should be sound modulo its cited, kernel-checked theorem",
        );
    }
}
