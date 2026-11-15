// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `UnsatToken`, class covers, the guard report, and `authorize_unsat`: the
//! aggregation step (U6/U7 of ENGINE_PROGRAM §1.6).

use std::fmt;

use super::cover::TreeCover;
use super::exact::ExactEmptiness;
use super::value::{closed, Certified};

/// Who may consume a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TokenScope {
    /// The whole instance: every adversarial class on the declared input box.
    /// The ONLY publishable scope.
    Instance,
    /// Some of the instance's classes on the whole box (one class tree of the
    /// classwise schedule). Consumed by the instance-level aggregation.
    ClassPart,
    /// A sub-domain frame's root region (a nested epoch subtree). Consumed as
    /// one leaf of the enclosing tree's cover. `Default`, so a driver that
    /// forgets to scope itself fails closed: it can never publish.
    #[default]
    Subdomain,
}

impl TokenScope {
    /// Stable token for observables.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Instance => "instance",
            Self::ClassPart => "class_part",
            Self::Subdomain => "subdomain",
        }
    }
}

/// A LEGACY-UNCOVERED `unsat` door: a pre-M2 lane whose own machinery decides
/// `unsat` but which produces no [`ClassCover`] — no path-checked cover, no
/// certified root, no exact certificate the facade can re-check — so its token
/// rests on the lane's word alone. Every call site of [`legacy_uncovered_unsat`]
/// names one of these; `ny-cli`'s `unsat_receipt` ratchet freezes the sites per
/// file and per authority and holds their production count under a ceiling that
/// may only go DOWN. Routing a lane through a cover deletes its variant.
///
/// Each variant says what certifies the lane's `unsat` TODAY and why that is not
/// yet a cover (#spine-legacy-lanes census, 2026-09-29).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LegacyUnsatAuthority {
    /// The in-process β-CROWN verifier's `verified` status, read back from its
    /// competition JSON. Certified today by each BaB driver's own closure rule
    /// (graph ReLU-split / multi-objective / input-split leaves with `lb > thr`
    /// on the graph lane's bounds, the lowdim prover's rigorous f64 input-split
    /// tree, MIP with `check_farkas` replay, ...). Uncovered because the verdict
    /// crosses the JSON boundary as a status string, the drivers record no leaf
    /// PATHS, and the graph lane's bounds are not `Certified` values.
    BetaCrownVerdict,
    /// The CZ hybrid metaroom lane's `ProvedUnsat`: every challenger margin of
    /// a constrained-zonotope certificate replayed strictly positive. The lane
    /// and its replay live in `ny-cli`/`ny-mip`, outside the facade.
    CzHybridLane,
    /// A relational difference-network proof carrying `RelationalUnsatAuth`:
    /// an α-CROWN emptiness `Verified` on the difference network (graph lane,
    /// not `Certified`) plus per-(clause, ¬E) Farkas certificates re-checked by
    /// `ny_cert::check_farkas`. Only the implication half is exact.
    RelationalImplication,
    /// The non-linear VNN-LIB interval branch-and-bound refutation
    /// (`try_nonlinear_unsat`): every input sub-box refuted by outward f64
    /// interval evaluation of the unrelaxed formula. Leaves are evaluated in
    /// `ny-cli` and the split tree is not recorded as paths.
    NonlinearIntervalBab,
    /// The cctsdb sub-box refutation: both sub-boxes of an authenticated proto
    /// rewrite refuted by exhaustive cell enumeration with a per-cell ONNX
    /// Runtime equivalence check. Enumerated in `ny-cli`.
    CctsdbSubbox,
    /// The TLL structural decoder's exact refutation over the authenticated raw
    /// model. Decided in `ny-cli`.
    TllStructure,
    /// Unit-test fixtures only. Production use is a ratchet failure.
    TestFixture,
}

impl LegacyUnsatAuthority {
    /// Every legacy-uncovered authority. `ny-cli`'s ratchet checks that each
    /// production one still has a door and that no door names anything else.
    pub const ALL: [Self; 7] = [
        Self::BetaCrownVerdict,
        Self::CzHybridLane,
        Self::RelationalImplication,
        Self::NonlinearIntervalBab,
        Self::CctsdbSubbox,
        Self::TllStructure,
        Self::TestFixture,
    ];

    /// Stable token for observables and the ratchet manifest.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::BetaCrownVerdict => "beta_crown_verdict",
            Self::CzHybridLane => "cz_hybrid_lane",
            Self::RelationalImplication => "relational_implication",
            Self::NonlinearIntervalBab => "nonlinear_interval_bab",
            Self::CctsdbSubbox => "cctsdb_subbox",
            Self::TllStructure => "tll_structure",
            Self::TestFixture => "test_fixture",
        }
    }
}

/// Where a token's authority comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnsatSource {
    /// [`authorize_unsat`]: every required class covered by a certified root
    /// close, a frozen-dual ledger, a path-checked tree cover, an exact
    /// specification-emptiness certificate, or a class-part token, with every
    /// guard clear.
    Authorized,
    /// [`legacy_uncovered_unsat`]: the lane's word, no cover.
    LegacyUncovered(LegacyUnsatAuthority),
}

impl UnsatSource {
    /// Stable token for observables.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Authorized => "authorized",
            Self::LegacyUncovered(a) => a.token(),
        }
    }
}

/// The only thing a published `unsat` can be built from.
///
/// Exactly one constructor, [`UnsatToken::mint`], private to this file. It is
/// called by [`authorize_unsat`] and [`legacy_uncovered_unsat`] and nowhere else
/// (`unsat_token_has_one_constructor` pins the call count).
#[derive(Debug, Clone, PartialEq)]
pub struct UnsatToken {
    source: UnsatSource,
    scope: TokenScope,
    classes: Box<[usize]>,
    min_margin: f64,
}

impl UnsatToken {
    /// THE constructor. Private: see the type docs.
    fn mint(
        source: UnsatSource,
        scope: TokenScope,
        classes: Box<[usize]>,
        min_margin: f64,
    ) -> Self {
        Self {
            source,
            scope,
            classes,
            min_margin,
        }
    }

    /// Where the authority came from.
    #[must_use]
    pub fn source(&self) -> UnsatSource {
        self.source
    }

    /// Who may consume it.
    #[must_use]
    pub fn scope(&self) -> TokenScope {
        self.scope
    }

    /// Whether this token may reach RESULTS_FILE.
    #[must_use]
    pub fn publishable(&self) -> bool {
        self.scope == TokenScope::Instance
    }

    /// The classes it covers (empty for a legacy-uncovered lane, which claims
    /// the whole property by its own construction).
    #[must_use]
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Whether every class in `classes` is covered.
    #[must_use]
    pub fn covers_all(&self, classes: &[usize]) -> bool {
        // Both private mint sites produce sorted unique class IDs (the legacy
        // site produces the empty list). Ordered queries need one merge scan;
        // arbitrary caller order still uses the same exact membership test.
        if classes.windows(2).all(|pair| pair[0] <= pair[1]) {
            let mut position = 0;
            for class in classes {
                while self
                    .classes
                    .get(position)
                    .is_some_and(|covered| covered < class)
                {
                    position += 1;
                }
                if self.classes.get(position) != Some(class) {
                    return false;
                }
            }
            true
        } else {
            classes
                .iter()
                .all(|class| self.classes.binary_search(class).is_ok())
        }
    }

    /// The smallest certified margin the proof rests on (G5). `NaN` when the
    /// source does not report one (legacy-uncovered lanes).
    #[must_use]
    pub fn min_margin(&self) -> f64 {
        self.min_margin
    }
}

/// One class's discharge. Every payload is unforgeable outside `certify`
/// (`Certified`, `TreeCover`, `UnsatToken` and `ExactEmptiness` have private
/// constructors), so the variants may be public. No public function builds a
/// payload from a number its caller chose (section-1.7 ratchet).
#[derive(Debug, Clone, PartialEq)]
pub enum ClassCover {
    /// The class's certified root bound is strictly positive on the whole box.
    Root {
        /// Adversarial class.
        class: usize,
        /// Its certified root bound.
        bound: Certified<f64>,
    },
    /// The frozen-dual tier closed the class: the search's level ledger was
    /// re-checked (its enumeration is trusted — `zono_bab`), and `cert` is the
    /// smallest certified value the closure rests on.
    FrozenDual {
        /// Adversarial class.
        class: usize,
        /// Root certificate, or the smallest certified prune.
        cert: Certified<f64>,
    },
    /// A path-checked tree cover for its classes.
    Tree(TreeCover),
    /// A class-part token (one classwise class tree).
    Part(UnsatToken),
    /// The class's unsafe region is empty as a FORMULA: a pair of its scalar
    /// atoms contradicts, checked by `certify::exact` (no network bound).
    Exact {
        /// The unsafe-region disjunct (class) the certificate empties.
        class: usize,
        /// The checked certificate.
        proof: ExactEmptiness,
    },
}

impl ClassCover {
    fn covers(&self, class: usize) -> Option<f64> {
        match self {
            Self::Root { class: k, bound } if *k == class && closed(bound) => Some(bound.value()),
            Self::FrozenDual { class: k, cert } if *k == class && closed(cert) => {
                Some(cert.value())
            }
            Self::Tree(tree) if tree.classes().contains(&class) => Some(tree.min_margin()),
            Self::Part(token)
                if token.scope == TokenScope::ClassPart && token.classes.contains(&class) =>
            {
                Some(token.min_margin)
            }
            Self::Exact { class: k, proof } if *k == class => Some(proof.slack()),
            _ => None,
        }
    }
}

/// The downgrade-only guards (ENGINE_PROGRAM §1.7 (a′)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuardId {
    /// G2: certified boxes and rows contain the f64 reference at sampled points.
    Containment,
    /// G3: the engine's objective rows agree with the sat gate's own spec
    /// evaluator at sampled points.
    SpecAgreement,
    /// G4: an optimised certified kernel matches its scalar reference.
    CrossCheck,
    /// A certified state's content hash changed after it was built.
    StateHash,
}

impl GuardId {
    /// Stable token for observables.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Containment => "g2_containment",
            Self::SpecAgreement => "g3_spec_agreement",
            Self::CrossCheck => "g4_crosscheck",
            Self::StateHash => "state_hash",
        }
    }
}

/// One tripped guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardTrip {
    /// Which guard.
    pub guard: GuardId,
    /// What it saw (log-ready).
    pub detail: String,
}

/// The guards' verdict for one frame. Downgrade-only: a trip refuses a token;
/// nothing here can create one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GuardReport {
    trips: Vec<GuardTrip>,
}

impl GuardReport {
    /// An empty (clear) report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a trip.
    pub fn trip(&mut self, guard: GuardId, detail: impl Into<String>) {
        self.trips.push(GuardTrip {
            guard,
            detail: detail.into(),
        });
    }

    /// Fold another report's trips into this one.
    pub fn absorb(&mut self, other: GuardReport) {
        self.trips.extend(other.trips);
    }

    /// No guard tripped.
    #[must_use]
    pub fn is_clear(&self) -> bool {
        self.trips.is_empty()
    }

    /// Every trip, in order.
    #[must_use]
    pub fn trips(&self) -> &[GuardTrip] {
        &self.trips
    }
}

/// Why [`authorize_unsat`] (or [`ratify`]) refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizeRefusal {
    /// No class was required: a vacuous proof is not a proof.
    NoClasses,
    /// A required class has no valid cover.
    Uncovered {
        /// The class.
        class: usize,
    },
    /// A guard tripped.
    GuardTripped(GuardTrip),
}

impl fmt::Display for AuthorizeRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoClasses => f.write_str("no_classes"),
            Self::Uncovered { class } => write!(f, "uncovered_class={class}"),
            Self::GuardTripped(t) => write!(f, "guard={} ({})", t.guard.token(), t.detail),
        }
    }
}

/// U6 + U7: every required class is covered and every guard is clear, or no
/// token. The same function runs in both proof profiles; it has no profile
/// parameter and cannot learn one.
pub fn authorize_unsat(
    scope: TokenScope,
    required: &[usize],
    covers: &[ClassCover],
    guards: &GuardReport,
) -> Result<UnsatToken, AuthorizeRefusal> {
    if let Some(trip) = guards.trips().first() {
        return Err(AuthorizeRefusal::GuardTripped(trip.clone()));
    }
    if required.is_empty() {
        return Err(AuthorizeRefusal::NoClasses);
    }
    let mut min_margin = f64::INFINITY;
    let canonical_tree = canonical_single_tree_cover(required, covers);
    if canonical_tree {
        // One checked tree discharges exactly the same sorted original IDs.
        // The old per-class contains/maximum/minimum reduces to this identical
        // value (including the generic all-empty +inf representation).
        let ClassCover::Tree(tree) = &covers[0] else {
            unreachable!("canonical tree eligibility")
        };
        min_margin = min_margin.min(tree.min_margin());
    } else if aligned_single_class_covers(required, covers) {
        // Each original ID has exactly its one matching discharge. This is
        // the same maximum-over-covers as below, without a quadratic scan for
        // a complete vocabulary of independent certified root classes.
        for (&class, cover) in required.iter().zip(covers) {
            match cover.covers(class) {
                Some(m) => min_margin = min_margin.min(m),
                None => return Err(AuthorizeRefusal::Uncovered { class }),
            }
        }
    } else {
        for &class in required {
            let best = covers
                .iter()
                .filter_map(|cover| cover.covers(class))
                .fold(None, |acc: Option<f64>, m| {
                    Some(acc.map_or(m, |a| a.max(m)))
                });
            match best {
                Some(m) => min_margin = min_margin.min(m),
                None => return Err(AuthorizeRefusal::Uncovered { class }),
            }
        }
    }
    let mut classes: Vec<usize> = required.to_vec();
    if !canonical_tree {
        classes.sort_unstable();
        classes.dedup();
    }
    Ok(UnsatToken::mint(
        UnsatSource::Authorized,
        scope,
        classes.into_boxed_slice(),
        min_margin,
    ))
}

/// Exact eligibility for the linear complete-tree aggregation. A checked
/// tree's private class IDs are sorted unique; equality and the caller's
/// strict order exclude omissions, duplicates, extra covers or relabeling.
/// This is a lookup-cost classifier, never an authority constructor.
pub(crate) fn canonical_single_tree_cover(required: &[usize], covers: &[ClassCover]) -> bool {
    !required.is_empty()
        && required.windows(2).all(|pair| pair[0] < pair[1])
        && matches!(covers,[ClassCover::Tree(tree)] if tree.classes()==required)
}

/// Exact eligibility for the constant-space single-class aggregation. This
/// changes only lookup cost, never a cover's numerical or scope authority.
pub(crate) fn aligned_single_class_covers(required: &[usize], covers: &[ClassCover]) -> bool {
    required.len() == covers.len()
        && required.windows(2).all(|pair| pair[0] < pair[1])
        && required
            .iter()
            .zip(covers)
            .all(|(&required, cover)| match cover {
                ClassCover::Root { class, .. }
                | ClassCover::FrozenDual { class, .. }
                | ClassCover::Exact { class, .. } => *class == required,
                ClassCover::Tree(_) | ClassCover::Part(_) => false,
            })
}

/// A LEGACY-UNCOVERED lane's `unsat` (see [`LegacyUnsatAuthority`]): the
/// lane's word, with no cover behind the token. Instance scope: the lane claims
/// the whole property by its own construction. Every production call site is
/// counted by `ny-cli`'s `unsat_receipt` ratchet, whose ceiling may only fall.
#[must_use]
pub fn legacy_uncovered_unsat(authority: LegacyUnsatAuthority) -> UnsatToken {
    UnsatToken::mint(
        UnsatSource::LegacyUncovered(authority),
        TokenScope::Instance,
        Box::new([]),
        f64::NAN,
    )
}

/// Pass a token through guards that could only run after it was minted (G3 at
/// the CLI, where the full VNN-LIB evaluator lives). Downgrade-only.
pub fn ratify(token: UnsatToken, guards: &GuardReport) -> Result<UnsatToken, AuthorizeRefusal> {
    match guards.trips().first() {
        Some(trip) => Err(AuthorizeRefusal::GuardTripped(trip.clone())),
        None => Ok(token),
    }
}

#[cfg(test)]
#[path = "token_aggregation_tests.rs"]
mod aggregation_tests;
