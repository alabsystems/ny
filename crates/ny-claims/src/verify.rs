// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification: re-derive, then classify.
//!
//! **The ORDER of the checks in [`evaluate`] is the mechanism.** Any other
//! order reproduces one of the five reversals. It is spelled out at the top of
//! that function and pinned by tests in `tests/lattice.rs`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::claim::{ArtifactRole, Claim, ClaimKind};
use crate::extract::{Extracted, Value};
use crate::predicate::Judgement;
use crate::world::World;

/// The five-valued lattice. See the crate docs for why three is not enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Status {
    True,
    /// Re-derived and the predicate fails. The claim is wrong.
    False,
    /// Intact evidence, but the world moved along a declared-sensitive
    /// dimension. Re-measure.
    Stale,
    /// Intact evidence, but the engagement observable is absent or failing.
    /// The measurement never observed its own subject. Fix the HARNESS; do not
    /// re-run it.
    Vacuous,
    /// A cited artifact is missing, unreadable, or the record is malformed.
    /// "I cannot check this" — never rounded to anything else.
    Ungrounded,
}

impl Status {
    pub const fn label(self) -> &'static str {
        match self {
            Self::True => "TRUE",
            Self::False => "FALSE",
            Self::Stale => "STALE",
            Self::Vacuous => "VACUOUS",
            Self::Ungrounded => "UNGROUNDED",
        }
    }

    /// Whether this status should fail a gate. Only `True` passes: a STALE or
    /// VACUOUS claim is not a warning, it is a claim that must not be cited.
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::True)
    }
}

/// A named reason the world no longer matches the claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Drift {
    /// The bytes of a cited artifact changed. Unconditional: this is not a
    /// declared sensitivity, it is the evidence itself.
    Artifact {
        path: String,
        pinned: String,
        found: String,
    },
    /// The verifier binary moved and this claim declared `sensitivity.binary`.
    Binary {
        pinned: String,
        found: String,
        source: String,
    },
    /// The claim is binary-sensitive but never recorded which binary produced
    /// it. Reported, not assumed benign — an unrecorded provenance is the
    /// state every one of the five reversals was measured in.
    BinaryUnrecorded,
    /// The claim is binary-sensitive and the CURRENT binary could not identify
    /// itself, so whether it moved is unknown.
    ///
    /// Unknown is drift, not agreement. Reversal 5's A/B was read as green on
    /// exactly this pattern — an observable that was never emitted was scored
    /// as an observable that was fine — and a checker that treats "I could not
    /// tell" as "it matches" has rebuilt the disease inside the cure.
    BinaryUnknown {
        pinned: String,
    },
    Host {
        pinned: String,
        found: String,
    },
    Route {
        pinned: String,
        found: String,
    },
    Lever {
        name: String,
        pinned: String,
        found: String,
    },
    Preset {
        pinned: String,
        found: String,
    },
}

impl Drift {
    pub fn render(&self) -> String {
        match self {
            Self::Artifact {
                path,
                pinned,
                found,
            } => format!(
                "artifact {path} changed: pinned {} != found {}",
                short(pinned),
                short(found)
            ),
            Self::Binary {
                pinned,
                found,
                source,
            } => format!(
                "binary moved: measured on stamp {}, verifying against {} (stamp {})",
                short(pinned),
                source,
                short(found)
            ),
            Self::BinaryUnrecorded => {
                "binary-sensitive claim records no binary stamp: unverifiable provenance"
                    .to_string()
            }
            Self::BinaryUnknown { pinned } => format!(
                "binary-sensitive claim measured on stamp {}, but no current verifier could \
                 identify itself; unknown is not agreement",
                short(pinned)
            ),
            Self::Host { pinned, found } => {
                format!("host moved: measured on {pinned}, verifying on {found}")
            }
            Self::Route { pinned, found } => {
                format!("route moved: claim assumes {pinned}, receipt says {found}")
            }
            Self::Lever {
                name,
                pinned,
                found,
            } => format!("lever {name} moved: claim assumes {pinned}, receipt says {found}"),
            Self::Preset { pinned, found } => {
                format!("preset moved: claim assumes {pinned}, found {found}")
            }
        }
    }
}

fn short(digest: &str) -> String {
    digest.chars().take(16).collect()
}

/// The result of re-checking one engagement witness.
#[derive(Debug, Clone, Serialize)]
pub struct WitnessOutcome {
    pub name: String,
    pub holds: bool,
    pub observed: String,
    pub citation: String,
    pub rendered: String,
    pub why: String,
}

/// The full verification receipt for one claim.
#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub id: String,
    pub status: Status,
    pub assertion: String,
    pub kind: ClaimKind,
    /// One line saying WHY this status, in the imperative where there is
    /// something to do.
    pub headline: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    pub witnesses: Vec<WitnessOutcome>,
    pub drifts: Vec<Drift>,
    /// `intrinsic` when the binary provenance was read out of the evidence
    /// artifact itself, `adjacent` when it came from a sibling file, `none`
    /// when it was not recorded. Reversal 3 was an `adjacent` fact read as an
    /// `intrinsic` one.
    pub provenance_grade: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retracted: Option<String>,
}

/// Re-derive a claim against the world as it is now and classify it.
///
/// # The order, and what each step is for
///
/// 1. **Ground.** Every cited artifact must exist and be readable, and every
///    extractor must run. Anything else is [`Status::Ungrounded`]: the honest
///    "I could not check this".
/// 2. **Artifact integrity.** Re-hash every cited artifact against its pin.
///    Moved bytes make everything downstream meaningless, so this short-
///    circuits to [`Status::Stale`] before anything is read into a conclusion.
/// 3. **Engagement.** Run the witnesses. A witness that is absent or failing
///    is [`Status::Vacuous`].
///
///    This is BEFORE the predicate, and that is the whole point. Reversal 5's
///    A/B had a predicate that genuinely held — "verdicts matched the bank" on
///    9 of 9 runs — and it held *because the device lane never executed*, so
///    those were the CPU bounds. Checking the predicate first prints TRUE.
///
///    It is also ABOVE staleness, because staleness says "re-run it" and
///    re-running a vacuous experiment reproduces the vacuum. Vacuity is a
///    defect of the harness and must be reported as one.
/// 4. **Sensitivity drift.** Compare recorded provenance to the current world,
///    but only on the dimensions this claim declared itself sensitive to.
///    [`Status::Stale`].
/// 5. **Predicate.** Only now. [`Status::True`] / [`Status::False`].
pub fn evaluate(claim: &Claim, world: &World) -> Outcome {
    let mut outcome = Outcome {
        id: claim.id.clone(),
        status: Status::Ungrounded,
        assertion: claim.assertion.clone(),
        kind: claim.kind,
        headline: String::new(),
        observed: None,
        citation: None,
        predicate: None,
        witnesses: Vec::new(),
        drifts: Vec::new(),
        provenance_grade: provenance_grade(claim),
        retracted: claim.retracted.clone(),
    };

    // --- 1. Ground -------------------------------------------------------
    // A claim whose extractor reads a file it never cited is not grounded
    // either: the pin would not cover the bytes the number came from.
    let mut cited: Vec<&str> = claim.artifacts.iter().map(|a| a.path.as_str()).collect();
    cited.sort_unstable();
    let mut used: Vec<&str> = std::iter::once(claim.extract.artifact())
        .chain(claim.engagement.iter().map(|w| w.extract.artifact()))
        .collect();
    used.sort_unstable();
    used.dedup();
    for artifact in &used {
        if !cited.contains(artifact) {
            outcome.status = Status::Ungrounded;
            outcome.headline = format!(
                "extractor reads {artifact} but the claim does not cite it, so its bytes are unpinned"
            );
            return outcome;
        }
    }

    let observed = match claim.extract.run(&world.root) {
        Ok(extracted) => extracted,
        Err(error) => {
            outcome.status = Status::Ungrounded;
            outcome.headline = format!("{error:#}");
            return outcome;
        }
    };
    record_observation(&mut outcome, &observed, claim);

    // --- 2. Artifact integrity -------------------------------------------
    for artifact in &claim.artifacts {
        match hash_file(&world.root.join(&artifact.path)) {
            Err(error) => {
                outcome.status = Status::Ungrounded;
                outcome.headline = format!("cited artifact {}: {error:#}", artifact.path);
                return outcome;
            }
            Ok(found) if found != artifact.sha256 => {
                outcome.drifts.push(Drift::Artifact {
                    path: artifact.path.clone(),
                    pinned: artifact.sha256.clone(),
                    found,
                });
            }
            Ok(_) => {}
        }
    }
    if !outcome.drifts.is_empty() {
        outcome.status = Status::Stale;
        outcome.headline =
            "the cited evidence is not the evidence this claim was made from".to_string();
        return outcome;
    }

    // --- 3. Engagement ----------------------------------------------------
    for witness in &claim.engagement {
        let (observed, judgement) = match witness.extract.run(&world.root) {
            Ok(extracted) => {
                let judgement = witness.require.test(&extracted.value);
                (extracted, judgement)
            }
            Err(error) => {
                outcome.status = Status::Ungrounded;
                outcome.headline = format!("engagement witness {}: {error:#}", witness.name);
                return outcome;
            }
        };
        outcome.witnesses.push(WitnessOutcome {
            name: witness.name.clone(),
            holds: judgement.holds,
            observed: observed.value.render(),
            citation: observed.citation.render(),
            rendered: judgement.rendered,
            why: witness.why.clone(),
        });
    }
    if let Some(failed) = outcome.witnesses.iter().find(|w| !w.holds) {
        outcome.status = Status::Vacuous;
        outcome.headline = format!(
            "engagement witness `{}` does not hold ({} at {}) — the measurement never observed \
             its own subject; fix the harness, do not re-run it",
            failed.name, failed.rendered, failed.citation
        );
        return outcome;
    }

    // --- 4. Sensitivity drift --------------------------------------------
    outcome.drifts = drifts(claim, world);
    if !outcome.drifts.is_empty() {
        outcome.status = Status::Stale;
        outcome.headline = outcome
            .drifts
            .iter()
            .map(Drift::render)
            .collect::<Vec<_>>()
            .join("; ");
        return outcome;
    }

    // --- 5. Predicate -----------------------------------------------------
    let judgement: Judgement = claim.predicate.test(&observed.value);
    outcome.status = if judgement.holds {
        Status::True
    } else {
        Status::False
    };
    outcome.headline = judgement.rendered;
    outcome
}

fn record_observation(outcome: &mut Outcome, observed: &Extracted, claim: &Claim) {
    outcome.observed = Some(observed.value.render());
    outcome.citation = Some(format!(
        "{} ({})",
        observed.citation.render(),
        observed.citation.detail
    ));
    outcome.predicate = Some(claim.predicate.test(&observed.value).rendered);
}

fn provenance_grade(claim: &Claim) -> &'static str {
    match claim.provenance.binary_evidence {
        Some(ArtifactRole::Evidence) => "intrinsic",
        Some(ArtifactRole::Provenance) => "adjacent",
        None => "none",
    }
}

fn drifts(claim: &Claim, world: &World) -> Vec<Drift> {
    let mut found = Vec::new();
    let sensitivity = claim.sensitivity;

    if sensitivity.binary {
        match (&claim.provenance.binary_stamp, &world.binary) {
            (None, _) => found.push(Drift::BinaryUnrecorded),
            (Some(pinned), Some(current)) if *pinned != current.stamp => {
                found.push(Drift::Binary {
                    pinned: pinned.clone(),
                    found: current.stamp.clone(),
                    source: current.source.clone(),
                });
            }
            (Some(pinned), None) => found.push(Drift::BinaryUnknown {
                pinned: pinned.clone(),
            }),
            (Some(_), Some(_)) => {}
        }
    }

    if sensitivity.host {
        if let Some(pinned) = &claim.provenance.host {
            if *pinned != world.host {
                found.push(Drift::Host {
                    pinned: pinned.clone(),
                    found: world.host.clone(),
                });
            }
        }
    }

    if sensitivity.route {
        if let (Some(pinned), Some(current)) =
            (&claim.provenance.route, current_route(claim, world))
        {
            if *pinned != current {
                found.push(Drift::Route {
                    pinned: pinned.clone(),
                    found: current,
                });
            }
        }
    }

    if sensitivity.levers {
        for (name, pinned) in &claim.provenance.levers {
            if let Some(current) = current_lever(claim, world, name) {
                if *pinned != current {
                    found.push(Drift::Lever {
                        name: name.clone(),
                        pinned: pinned.clone(),
                        found: current,
                    });
                }
            }
        }
    }

    if sensitivity.preset {
        if let (Some(pinned), Some(current)) =
            (&claim.provenance.preset, current_preset(claim, world))
        {
            if *pinned != current {
                found.push(Drift::Preset {
                    pinned: pinned.clone(),
                    found: current,
                });
            }
        }
    }

    found
}

/// Re-read the route from the first cited flight receipt, if there is one.
/// Route drift is checked against the EVIDENCE, not against a belief.
fn current_route(claim: &Claim, world: &World) -> Option<String> {
    for artifact in &claim.artifacts {
        if !artifact.path.ends_with(".flight.json") {
            continue;
        }
        let extractor = crate::extract::Extractor::FlightEventField {
            artifact: artifact.path.clone(),
            method: "internal_timeout_route".to_string(),
            key: "route".to_string(),
        };
        if let Ok(extracted) = extractor.run(&world.root) {
            if let Value::Str(route) = extracted.value {
                return Some(route);
            }
        }
    }
    None
}

fn current_lever(claim: &Claim, world: &World, name: &str) -> Option<String> {
    for artifact in &claim.artifacts {
        if !artifact.path.ends_with(".flight.json") {
            continue;
        }
        let extractor = crate::extract::Extractor::JsonPointer {
            artifact: artifact.path.clone(),
            pointer: format!("/ambient_env/{name}"),
        };
        match extractor.run(&world.root).map(|extracted| extracted.value) {
            Ok(Value::Absent) => return Some("<unset>".to_string()),
            Ok(value) => return Some(value.render().trim_matches('"').to_string()),
            Err(_) => continue,
        }
    }
    None
}

/// The effective preset is recorded as the `preset` flight event's whole
/// `reason` (a path), not as a `key=` token, so it is read directly rather
/// than through the token op.
fn current_preset(claim: &Claim, world: &World) -> Option<String> {
    for artifact in &claim.artifacts {
        if !artifact.path.ends_with(".flight.json") {
            continue;
        }
        let text = std::fs::read_to_string(world.root.join(&artifact.path)).ok()?;
        let document: serde_json::Value = serde_json::from_str(&text).ok()?;
        let events = document.get("events")?.as_array()?;
        let preset = events.iter().find(|event| {
            event.get("method").and_then(serde_json::Value::as_str) == Some("preset")
        })?;
        return preset
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
    }
    None
}

pub(crate) fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes =
        std::fs::read(path).map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hex(&hasher.finalize()))
}

pub(crate) fn hex(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

/// Summary counts for a whole ledger run.
pub fn tally(outcomes: &[Outcome]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for outcome in outcomes {
        *counts.entry(outcome.status.label()).or_insert(0) += 1;
    }
    counts
}
