// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #genbab-selector-convention: always-on engagement telemetry for the GenBaB
//! selector.
//!
//! Two stderr lines, both `eprintln!` (never `tracing`, because
//! `ny benchmarks run` forces `RUST_LOG=error`):
//!
//! * `[genbab-select] armed convention=pre-activation ...` — printed ONCE per
//!   process, the first time the selector runs. Its presence proves the binary
//!   carries the input-space convention; its absence on a GenBaB row means the
//!   selector never ran (root-verified, or a lane that ignores GenBaB), and a
//!   null from such a run is VOID, not negative.
//! * `[genbab-select] cumulative selections=.. relu=.. unary=.. binary=.. norm=..
//!   relu_candidates=.. rejected_non_partition=.. empty_children=..` — printed
//!   after every `[bab-stop]` line once at least one selection has happened in
//!   the process. Counters are process-cumulative (ml4acopf runs one BaB per OR
//!   clause).
//!
//! What to read off it. Before the fix `relu` AND `relu_candidates` were
//! structurally 0 (post-activation bounds of a ReLU never straddle 0) and
//! `empty_children` counted every no-op split whose output-space point missed
//! the input interval. After it, `empty_children` must stay 0 — a non-zero
//! value is a convention mismatch between the selector and
//! `with_general_split`, i.e. a regression of this fix — and `relu`/`unary`
//! show which nonlinearities GenBaB actually branches.
//!
//! `relu`/`unary`/`binary`/`norm` count only the TOP decision of each
//! selection, so on a net whose MulBinary or smooth-activation scores win every
//! time `relu` can legitimately stay 0 with the fix working. The ENGAGEMENT
//! proof for the ReLU half of the fix is `relu_candidates > 0` on a ReLU-bearing
//! row that ran BaB: it counts unstable ReLU/Sign neurons that reached the
//! scored candidate list at all, before the sort and truncation.
//!
//! Pure telemetry: nothing reads these counters back, so they cannot move a
//! bound or a verdict. No env read, no preset key: no ratchet debt.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;

use crate::layers::Layer;

use super::BranchingDecision;

static SELECTIONS: AtomicU64 = AtomicU64::new(0);
static CHOSEN_ZERO_THRESHOLD: AtomicU64 = AtomicU64::new(0);
static CHOSEN_UNARY: AtomicU64 = AtomicU64::new(0);
static CHOSEN_BINARY_OPERAND: AtomicU64 = AtomicU64::new(0);
static CHOSEN_NORM: AtomicU64 = AtomicU64::new(0);
static ZERO_THRESHOLD_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static REJECTED_NON_PARTITION: AtomicU64 = AtomicU64::new(0);
static EMPTY_CHILDREN: AtomicU64 = AtomicU64::new(0);
static ARMED_LINE: Once = Once::new();

/// Print the once-per-process armed line.
pub(super) fn note_selector_armed() {
    ARMED_LINE.call_once(|| {
        eprintln!(
            "[genbab-select] armed convention=pre-activation source=node.inputs[0] \
             partition_guard=strict (#genbab-selector-convention)"
        );
    });
}

/// Record the kind of the TOP decision of one selection (callers branch on
/// `decisions[0]`). `layer` is the chosen node's layer.
pub(super) fn record_selection(top: &BranchingDecision, layer: Option<&Layer>) {
    SELECTIONS.fetch_add(1, Ordering::Relaxed);
    let bucket = if top.norm_inv_rms.is_some() {
        &CHOSEN_NORM
    } else if top.input_index.is_some() {
        &CHOSEN_BINARY_OPERAND
    } else if matches!(layer, Some(Layer::ReLU(_) | Layer::Sign(_))) {
        &CHOSEN_ZERO_THRESHOLD
    } else {
        &CHOSEN_UNARY
    };
    bucket.fetch_add(1, Ordering::Relaxed);
}

/// Count the unstable zero-threshold (ReLU/Sign) neurons that reached ONE
/// selection's scored candidate list, BEFORE the sort and truncation. This is
/// the ReLU-half engagement observable: `relu=` counts only top picks and can
/// stay 0 with the fix working when other kinds out-score every ReLU.
pub(super) fn record_zero_threshold_candidates(count: usize) {
    if count > 0 {
        ZERO_THRESHOLD_CANDIDATES.fetch_add(count as u64, Ordering::Relaxed);
    }
}

/// A candidate whose points do not strictly partition its interval was dropped.
pub(super) fn record_rejected_non_partition() {
    REJECTED_NON_PARTITION.fetch_add(1, Ordering::Relaxed);
}

/// A GenBaB split produced an EMPTY child (`with_general_split` -> `Ok(None)`).
///
/// When it comes from the feasibility check (`effective_l > effective_u`) the
/// dropped region really is empty and dropping it is sound, but it is a
/// PROGRESS failure: its sibling is the unchanged parent. `with_general_split`
/// ALSO returns `Ok(None)` for structural problems (missing pre-bounds, bad
/// input/neuron index, `LayerRef::Index`), where the dropped region is NOT
/// empty; those are unreachable from the fixed selector, which reads the same
/// map. Either way this must stay 0 after the convention fix, and a non-zero
/// value is a stop-and-look, not noise.
pub(crate) fn record_genbab_empty_child() {
    EMPTY_CHILDREN.fetch_add(1, Ordering::Relaxed);
}

/// The cumulative summary line, or `None` if no GenBaB selection has run in
/// this process (so non-GenBaB categories print nothing extra).
pub(crate) fn genbab_select_summary_line() -> Option<String> {
    let selections = SELECTIONS.load(Ordering::Relaxed);
    if selections == 0 {
        return None;
    }
    Some(format!(
        "[genbab-select] cumulative selections={} relu={} unary={} binary={} norm={} \
         relu_candidates={} rejected_non_partition={} empty_children={}",
        selections,
        CHOSEN_ZERO_THRESHOLD.load(Ordering::Relaxed),
        CHOSEN_UNARY.load(Ordering::Relaxed),
        CHOSEN_BINARY_OPERAND.load(Ordering::Relaxed),
        CHOSEN_NORM.load(Ordering::Relaxed),
        ZERO_THRESHOLD_CANDIDATES.load(Ordering::Relaxed),
        REJECTED_NON_PARTITION.load(Ordering::Relaxed),
        EMPTY_CHILDREN.load(Ordering::Relaxed),
    ))
}
