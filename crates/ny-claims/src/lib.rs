// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! `ny-claims` — an empirical claim as a first-class, machine-checkable object.
//!
//! # The failure this closes
//!
//! In one session five claims about the running system were asserted and then
//! reversed. Every one had the same shape: **a claim about the running system
//! derived from something that is not the running system.**
//!
//! | # | the claim | what stood in for the system | the check that would have caught it |
//! |---|-----------|------------------------------|-------------------------------------|
//! | 1 | "the cifar100 root gap is pre-alpha" | a root bound quoted from a doc, 3 days stale | re-measure the row |
//! | 2 | "the scored tier is `--features mip`" | reading build-script control flow | `ny --build-id` |
//! | 3 | "an agent ran a swapped binary" | a manifest written by a DIFFERENT agent | ask which process produced the artifact |
//! | 4 | "the blocked-conv fix is the top lever" | ONNX op counts as a proxy for lane membership | grep `route=` in that benchmark's own log |
//! | 5 | "M2 authorizes the M3 flip", and the A/B "passed" | a harness that never exported the arming variable | grep the lever's own telemetry line |
//!
//! The campaign already had rules for all of this — R7 "confirm the arm
//! engaged", R8 "verify binary provenance", R9 "demand the lever's own
//! observable". All three were violated on the same day by the person who
//! wrote them. **They are documentation, and documentation does not execute.**
//!
//! The structural diagnosis is not "insufficient discipline". It is that
//! claims and evidence live in different places: claims live in prose (the
//! ledger, the memory dir, commit messages), evidence lives in logs and
//! receipts, *nothing binds them, and nothing re-checks a claim when the
//! system moves underneath it*.
//!
//! # The object
//!
//! A [`Claim`] is a derivation node, not a sentence. It carries:
//!
//! * the human `assertion` (for reading), and next to it
//! * the `artifacts` it was derived from, each **content-pinned by sha256**,
//! * an [`Extractor`] that RE-DERIVES the number from those artifacts today,
//! * a [`Predicate`] the re-derived number must satisfy,
//! * a set of **engagement witnesses** — observables that must be PRESENT for
//!   the number to mean anything at all,
//! * the `provenance` of the run that produced it (binary stamp, host, route,
//!   levers, preset digest), and
//! * a **sensitivity set** saying which of those dimensions this claim's truth
//!   actually depends on.
//!
//! [`verify::evaluate`] re-runs the extractor and classifies. It never runs
//! the verifier: it reads artifacts. That is what makes it cheap enough to run
//! on every commit, and a check that is cheap enough to always run is the only
//! kind that survives contact with a campaign.
//!
//! # The five-valued lattice, and why three is not enough
//!
//! The obvious design classifies TRUE / STALE / FALSE. That design **prints
//! TRUE for reversal 5**, which is the single most expensive failure in the
//! table: the M3 A/B's predicate ("verdicts match the bank") genuinely held on
//! 9/9 runs. It held because the device lane never ran and those were the CPU
//! bounds. The number was real; it measured nothing.
//!
//! So the lattice has five values and the ordering of the checks is the
//! mechanism, not an implementation detail (see [`verify::evaluate`]):
//!
//! * [`Status::Ungrounded`] — a cited artifact is missing or unreadable. "I
//!   cannot check this" is a distinct and valuable answer; it must never be
//!   silently rounded to FALSE (nor, worse, to TRUE).
//! * [`Status::Stale`] — the artifacts are intact but the world moved along a
//!   dimension this claim declared itself sensitive to. The number may still
//!   be right; it is no longer *supported*. Reversal 1 dies here.
//! * [`Status::Vacuous`] — the artifacts are intact and an engagement witness
//!   is ABSENT or failing. The measurement never observed the thing it claims
//!   to be about. Reversals 4 and 5 die here.
//! * [`Status::False`] — grounded, current, engaged, and the predicate fails.
//! * [`Status::True`] — grounded, current, engaged, predicate holds.
//!
//! **Vacuity is checked before the predicate, and outranks staleness.** Before
//! the predicate, because a satisfied predicate over an unobserved lane is the
//! exact trap of reversal 5. Above staleness, because staleness invites "just
//! re-run it" — and re-running a vacuous experiment reproduces the vacuum.
//! Vacuity is a defect of the *harness*, and it must be reported as one.
//!
//! # Why sensitivity is declared per claim
//!
//! A staleness rule of "any change invalidates everything" is equivalent to no
//! rule: after the first rebuild the whole ledger is STALE, and a reader
//! learns to ignore the column. Each claim therefore declares
//! [`Sensitivity`] — whether its truth depends on the binary, the host, the
//! route, the lever set, the preset. A bank lookup ("this row is banked
//! unsat") is not binary-sensitive and stays TRUE across rebuilds. A measured
//! root bound is binary- AND host-sensitive (the campaign already burned a
//! session on a "97/99" figure that was one host's result) and goes STALE the
//! moment either moves. Artifact bytes are sensitive unconditionally: they are
//! the evidence.
//!
//! # Why `sensitivity` has no default
//!
//! It is a required field. A forgotten sensitivity set is exactly the omission
//! this crate exists to prevent, so a claim that does not state one does not
//! parse. `ny claims lint` reports the parse failure by file.

mod claim;
mod extract;
mod ledger;
mod predicate;
mod verify;
mod world;

pub mod cli;

pub use claim::{
    ArtifactRef, ArtifactRole, Claim, ClaimKind, ClaimProvenance, Sensitivity, Witness, SCHEMA,
};
pub use extract::{Citation, Extracted, Extractor, Occurrence, Value};
pub use ledger::{lint_all, lint_claim, load_dir, LintFinding, LoadedClaim};
pub use predicate::{Judgement, Predicate};
pub use verify::{evaluate, Drift, Outcome, Status, WitnessOutcome};
pub use world::{BinaryFacts, World};
