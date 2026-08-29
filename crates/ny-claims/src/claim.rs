// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The claim record: what is stored in `claims/<id>.claim.json`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::extract::Extractor;
use crate::predicate::Predicate;

/// Greppable schema tag. Bump when a field is removed or changes meaning.
pub const SCHEMA: &str = "NY_CLAIM_V1";

/// What sort of thing is being asserted. This is not decoration: it decides
/// whether an engagement witness is OPTIONAL or MANDATORY (see
/// [`crate::lint_claim`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    /// A statement about what an artifact records — a bank verdict, a runtime,
    /// a census. Nothing was intervened on, so there is no arm to engage. A
    /// witness is still encouraged (the ROW-IDENTITY witness below is the
    /// executable form of "verify the log's own INSTANCE line"), but not
    /// required.
    Observational,
    /// A statement that an intervention changed (or did not change) an
    /// outcome: any A/B, any lever, any "we turned X on and Y happened".
    ///
    /// **MUST declare at least one engagement witness.** This single typing
    /// rule is what makes a vacuous null structurally impossible: reversals 4
    /// and 5 were both interventional claims whose intervention never
    /// occurred, and both would have failed to lint.
    Interventional,
    /// A statement about which route/lane a benchmark's scored path takes.
    /// Separated from `Observational` because it is the claim class that a CI
    /// test iterates: changing a lane must break a build, not silently
    /// invalidate a program of work.
    LaneMembership,
}

/// Which dimensions of the world this claim's truth depends on.
///
/// Required — see the crate docs for why there is no `Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sensitivity {
    /// The measurement would change if the verifier binary changed.
    pub binary: bool,
    /// The measurement would change on different hardware. The campaign has
    /// already lost a session to a figure ("97/99") that held on one host and
    /// not on the host it was being acted on from.
    pub host: bool,
    /// The measurement depends on which route/lane the scored path took.
    pub route: bool,
    /// The measurement depends on the resolved lever set.
    pub levers: bool,
    /// The measurement depends on the category preset.
    pub preset: bool,
}

/// Whether an artifact carries the EVIDENCE or merely the PROVENANCE.
///
/// The distinction exists because of reversal 3: a claim was made about which
/// binary produced a set of rows, using a manifest written by a *different*
/// process than the one that produced them. A provenance fact read out of the
/// evidence file itself is `Intrinsic`; a provenance fact read out of a
/// sibling file is `Adjacent`, and [`crate::verify::evaluate`] says so in the
/// receipt rather than letting the two look alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRole {
    /// The bytes the extractor reads the asserted number out of.
    Evidence,
    /// A sidecar that says who produced the evidence. Weaker: nothing but
    /// convention binds it to the evidence file.
    Provenance,
}

/// One cited artifact, pinned by content.
///
/// `sha256` is required and is re-checked on every verify. A cited file whose
/// bytes moved is not the file the claim was made from, whatever its path
/// says — the same content-addressing argument `crate::build_id` makes for the
/// executable, applied to the evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    /// Repo-relative path. Absolute paths are rejected by the linter: a claim
    /// that cites `/tmp` cannot be re-checked by anyone else, or by you next
    /// week.
    pub path: String,
    /// Lowercase hex sha256 of the bytes at assertion time. `ny claims pin`
    /// fills this in.
    pub sha256: String,
    pub role: ArtifactRole,
    #[serde(default)]
    pub note: String,
}

/// The conditions of the run that produced the evidence.
///
/// Every field is optional because a claim is only as good as the artifact it
/// came from, and some artifacts predate the provenance they should carry —
/// but an ABSENT field that the sensitivity set says matters is reported, not
/// assumed benign (see [`crate::verify::evaluate`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimProvenance {
    /// `stamp=` digest from `NY_BUILD_STAMP_V1` / `NY_BUILD_ID_V1`.
    #[serde(default)]
    pub binary_stamp: Option<String>,
    /// `id=` from `NY_BUILD_ID_V1` — the content-addressed identity.
    #[serde(default)]
    pub binary_id: Option<String>,
    /// WHICH artifact this binary fact was read out of, and how directly.
    /// `Adjacent` here is the reversal-3 shape and is surfaced in the receipt.
    #[serde(default)]
    pub binary_evidence: Option<ArtifactRole>,
    /// Hostname of the measuring machine.
    #[serde(default)]
    pub host: Option<String>,
    /// The `route=` marker of the scored path.
    #[serde(default)]
    pub route: Option<String>,
    /// Resolved lever name -> value, as the run actually saw them.
    #[serde(default)]
    pub levers: BTreeMap<String, String>,
    /// Digest or path of the effective preset.
    #[serde(default)]
    pub preset: Option<String>,
}

/// An ENGAGEMENT OBSERVABLE: proof that the thing being claimed about was
/// actually exercised.
///
/// This is R9 ("demand the lever's own observable") turned from a sentence
/// into a field that a program refuses to skip. The witness is evaluated
/// BEFORE the claim's own predicate, and a claim whose witness is absent is
/// [`crate::Status::Vacuous`] no matter how well the predicate holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Witness {
    /// Short name printed in the receipt, e.g. `gpu_backward_lane_executed`.
    pub name: String,
    /// How to find the observable in the artifacts.
    pub extract: Extractor,
    /// What the observable must satisfy for the claim to be about anything.
    pub require: Predicate,
    /// One line: what this witness proves, for the human reading a VACUOUS
    /// verdict at 2am.
    #[serde(default)]
    pub why: String,
}

/// One empirical claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    /// Must equal [`SCHEMA`].
    pub schema: String,
    /// Stable id. Must equal the file stem, and is what prose cites as
    /// `[[claim:<id>]]`.
    pub id: String,
    /// The assertion in English, verbatim as it would appear in a doc. This is
    /// the only field a human is expected to read first, and it is deliberately
    /// stored NEXT TO the extractor so the two cannot drift apart in different
    /// files.
    pub assertion: String,
    pub kind: ClaimKind,
    #[serde(default)]
    pub asserted_on: String,
    #[serde(default)]
    pub asserted_by: String,
    /// Cited artifacts. Non-empty; the linter rejects a claim with none,
    /// because a claim with no artifact is prose with extra steps.
    pub artifacts: Vec<ArtifactRef>,
    /// See [`Witness`]. Mandatory for [`ClaimKind::Interventional`].
    #[serde(default)]
    pub engagement: Vec<Witness>,
    pub extract: Extractor,
    pub predicate: Predicate,
    #[serde(default)]
    pub provenance: ClaimProvenance,
    /// Required. See the crate docs.
    pub sensitivity: Sensitivity,
    /// What observation would make this FALSE. Not machine-checked; it is here
    /// because a claim whose author cannot name its refutation is not a claim,
    /// and the linter does require the field to be non-empty.
    #[serde(default)]
    pub refutation: String,
    #[serde(default)]
    pub notes: String,
    /// Claim ids this one replaces. `ny claims verify` prints a superseded
    /// claim's successor so a reader chasing an old doc reference lands on the
    /// current record instead of the retracted one.
    #[serde(default)]
    pub supersedes: Vec<String>,
    /// Set when a claim has been retracted in place rather than superseded.
    /// A retracted claim is still verified — a retraction that turns out to be
    /// wrong is itself a reversal.
    #[serde(default)]
    pub retracted: Option<String>,
}

impl Claim {
    /// Look up a cited artifact by path.
    pub fn artifact(&self, path: &str) -> Option<&ArtifactRef> {
        self.artifacts.iter().find(|entry| entry.path == path)
    }
}
