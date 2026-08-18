// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dark, print-only trace of WHERE a CROWN carrier stops being
//! [`CrownBounds::Patches`](crate::bounds::patches::CrownBounds) (#patches-drop).
//!
//! `NY_PATCHES_CARRIER_TRACE=1` enables; the declared `false` default emits no
//! output.
//!
//! WHY. docs/REGRESSION_FC_UNSAT_LOST_2026-08-14.md, last section: on
//! `cifar_bias_field_46` the conv backward runs DENSE at HEAD where
//! `97fb4bd6a` ran it in Patches form — 27% of the profile in the col2im
//! scatter and ~21% in scalar certified-error folds, against 59.6% in NEON
//! microkernels at the good commit. Arming `NY_PATCHES_FINITE_EXPIRY` restores
//! the WORK (collection 0.85 s -> 37.2 s) but not the REPRESENTATION: the dense
//! frames still dominate and alpha iter-0 `lower_sum` stays at −341,063,072.
//! Six individual guard flips plus one seven-site coherent flip all failed to
//! name the densifying site by argument; a throwaway carrier trace named it in
//! one run (`patches_step.rs`'s ReLU door, since repaired). This is that
//! question asked permanently, and pointed at the two blockers the repair
//! exposed. One of them, the dense materialization that misses the CPU budget
//! by 0.041%, IS this funnel and prints as `outcome=refused-memory`. The other,
//! `/layers.6/Relu`'s per-node expiry, is raised inside the conv kernel
//! (`layers/convolution/conv2d/ops_transpose_gemm.rs`) and the funnel never
//! sees it; what the trace contributes there is whether that node's carrier was
//! still Patches when the walk reached it.
//!
//! Two line kinds, both one stderr line:
//!
//! ```text
//! [patches-drop] scope=<s> node=<n> site=<file:line> purpose=<p>
//!     outcome=<ok|refused-deadline|refused-memory|refused-semantic>
//!     deadline=<none|live|expired> rows=<n> unstable=<0|1> coeff_err=<0|1>
//! [patches-carrier] scope=<s> node=<n> op=<layer> repr_in=<patches|dense>
//!     allow_patches=<0|1> hard=<0|1> handled=<0|1>
//!     deadline=<none|live|expired> expiry_lever=<0|1>
//! ```
//!
//! `deadline=` is the STATE, not merely presence — the regression family this
//! instrument serves is guards that tested presence where expiry was meant. It
//! is classified BEFORE the materialization runs, so it describes the authority
//! the work STARTED under; the materializer's own poll may expire a `live`
//! deadline mid-flight, and that shows up as `outcome=refused-deadline`, not as
//! a retroactively rewritten `deadline=`.
//! `expiry_lever=` (the `NY_PATCHES_FINITE_EXPIRY` latch) appears only on the
//! decision line: it is a run-level constant, and the drop-line emitter cannot
//! reach that latch without duplicating its parser.
//!
//! `[patches-drop]` is emitted from the ONE materialization funnel
//! (`PatchesLinearBounds::to_dense_with_deadline_and_resident_for_purpose`):
//! `CrownBounds::into_dense*`, `CrownBounds::ensure_dense*` and every
//! `to_dense*` wrapper bottom out there, and all of them are `#[track_caller]`,
//! so `site=` names the caller that densified rather than the materializer.
//! Row-range materialization (`to_dense_rows*`) and the sparse/chunked
//! concretizers are deliberately NOT traced: they read a Patches relation
//! without replacing the carrier, so they are not carrier transitions.
//!
//! ONLY `outcome=ok` IS A CARRIER DROP. The line is emitted from the funnel's
//! OUTCOME arms, never from its entry, because a refusal does not drop the
//! carrier: both `Err` consumers are transactional.
//! `CrownBounds::ensure_dense_with_deadline_for_purpose` leaves the Patches
//! carrier untouched on refusal, and
//! `graph_crown::propagation::prepare_plain_dense_boundary_for_purpose` maps
//! `DeadlineExceeded`/`CpuMemoryExceeded` to a CROWN→IBP fallback with the
//! carrier STILL Patches. An entry-emitted line would therefore name a
//! non-densifying site in exactly the deadline-expiry regime this probe exists
//! for. Read `outcome=ok` lines to find the densifier; read
//! `outcome=refused-*` lines as attempted-and-declined pressure on the funnel
//! (`refused-memory` is where the 0.041%-over dense budget of
//! `REGRESSION_FC_UNSAT_LOST_2026-08-14.md` shows up). The vocabulary is the
//! funnel's own:
//! `execution_telemetry`'s attempt / success / refusal counters classify the
//! same three refusal kinds with [`PatchesMaterializationRefusal`].
//!
//! One residual, stated rather than tidied away: `outcome=ok` is NECESSARY for
//! a carrier drop but not quite sufficient. `ensure_dense_with_deadline_for_purpose`
//! re-checks expiry AFTER a successful materialization and, if the clock has
//! passed, returns `DeadlineExceeded` without installing the dense relation —
//! the funnel already returned `Ok` and cannot see that. So a very small number
//! of `outcome=ok` lines can correspond to a carrier that stayed Patches. The
//! signature is an `outcome=ok deadline=live` line at a site whose walk then
//! takes the IBP fallback anyway; the trace does not attempt to detect it.
//!
//! ONE-HOP CAVEAT, stated because it changes how the output is read. A
//! densification routed through a forwarding helper that is NOT itself
//! `#[track_caller]` — `graph_crown::propagation::prepare_plain_dense_boundary`
//! is the one that shows up most — resolves to that helper's own line, and the
//! walk node that reached it is one hop above. `node=` disambiguates it, and
//! the helper is deliberately left alone: adding `#[track_caller]` to production
//! forwarders to sharpen a trace would put an instrument into codegen, which is
//! exactly what this module refuses to do. The α-CROWN target walk's own
//! densification (`target_backward.rs`, the `into_dense_with_deadline` after
//! the patches step declines) is a direct call and attributes exactly.
//!
//! `[patches-carrier]` is the DECISION line of the α-CROWN target walk — the
//! `was_patches / allow_patches / hard / handled` tuple this document's
//! investigation previously reconstructed by hand. `repr_in=` is named for what
//! it is: the carrier representation ON ENTRY to the step, sampled before
//! `try_patches_target_step_core` runs. It is an INPUT to the decision, never
//! the outgoing representation — a step with `repr_in=patches handled=0` goes
//! on to densify.
//!
//! `scope`/`node` come from a THREAD-LOCAL set at each backward walk's node
//! head. Walk and materialization are synchronous on the same thread, so the
//! pair is exact for the four instrumented walks (α-CROWN target, DAG α-CROWN,
//! Graph-CROWN, Spec-guided CROWN). A conversion reached from anywhere else —
//! another walk, or a rayon worker — prints the last node entered on THAT
//! thread, which is STALE, or `scope= node=` with both fields EMPTY if that
//! thread never entered an instrumented walk. `site=` is the authoritative
//! field and is never stale.
//!
//! FILTER ON `node=`, NOT `scope=`. The two fields are not the same kind of
//! thing: three of the four call sites publish a WALK-KIND LITERAL as `scope`
//! (`dag-alpha`, `graph-crown`, `spec-crown`) and the fourth publishes the
//! α-CROWN TARGET node, which is not the walked node either. `node` is the only
//! field that names the walked node on both line kinds, so `grep scope=<name>`
//! matches at most the target walk and usually nothing.
//!
//! Print-only: no line here feeds a bound, a verdict, or a schedule decision,
//! and no value, lifetime, or drop order changes. The gate is checked FIRST at
//! every site, so with the lever unset each site is one latched-string compare
//! — no clock read, no formatting, no allocation, no `Location::caller()`.
//! Armed-vs-unarmed deadline/verdict parity is NOT claimed: an armed run reads
//! clocks and writes stderr inside deadline-sensitive walks.

use std::cell::RefCell;
use std::panic::Location;
use std::sync::OnceLock;
use std::time::Instant;

use crate::bounds::patches::{PatchesLinearBounds, PatchesMaterializationPurpose};
use crate::execution_telemetry::PatchesMaterializationRefusal;

/// Pure gate predicate: exactly `"1"` enables (same idiom as
/// `phase_telemetry::gate_on` and `iter0_parity_trace::gate_on`).
fn gate_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Uncached env read through the ny-levers chokepoint's raw view — the cache
/// initializer, and the deterministic seam the gate test drives.
fn raw_uncached() -> Option<String> {
    ny_levers::read_raw(&ny_levers::decls::telemetry::PATCHES_CARRIER_TRACE)
}

/// Latched RAW env string. The gate is checked per backward-walk node and per
/// materialization — hot — so the STRING is latched once and the decision is
/// derived per call by [`gate_on`]. Process-wide, like its two sibling traces;
/// Phase 2 must replace it with an injected per-run `LeverSet`.
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(raw_uncached).as_deref()
}

/// Process-wide gate over the latched raw string. One latched-string compare
/// when unset — check this FIRST at every site.
pub(crate) fn enabled() -> bool {
    gate_on(env_raw())
}

thread_local! {
    /// Walk position: `(scope, node)`, overwritten at each instrumented walk's
    /// node head. Thread-local rather than global because the walks are
    /// synchronous per thread and several walks run concurrently under rayon;
    /// a shared buffer would attribute one walk's conversions to another's
    /// node (the same defect `PATCHES_TO_DENSE_CALL_SITES` was moved off a
    /// process-global `Mutex` to avoid). `const` init so the slot costs
    /// nothing when the gate is off.
    static WALK_POSITION: RefCell<(String, String)> =
        const { RefCell::new((String::new(), String::new())) };
}

/// Record the walk position for subsequent `[patches-drop]` lines. Callers
/// check [`enabled`] first; this reuses the thread-local's existing capacity,
/// so an armed walk allocates at most once per thread per string.
///
/// Never panics: a poisoned/destructing TLS slot or a re-entrant borrow leaves
/// the position stale rather than unwinding through a backward walk.
pub(crate) fn enter_node(scope: &str, node: &str) {
    let _ = WALK_POSITION.try_with(|position| {
        if let Ok(mut position) = position.try_borrow_mut() {
            position.0.clear();
            position.0.push_str(scope);
            position.1.clear();
            position.1.push_str(node);
        }
    });
}

/// Read the walk position. Both halves are EMPTY when the slot is unset or
/// unreadable, so a conversion reached outside the four instrumented walks
/// prints `scope= node=` — `walk_position_round_trips_and_defaults_empty`
/// pins that. `site=` is the field to read in that case.
fn walk_position() -> (String, String) {
    WALK_POSITION
        .try_with(|position| {
            position
                .try_borrow()
                .map(|position| (position.0.clone(), position.1.clone()))
                .unwrap_or_default()
        })
        .unwrap_or_default()
}

/// Deadline STATE, not merely presence — the whole regression this instrument
/// serves is a family of guards that tested presence where expiry was meant.
///
/// Callers classify BEFORE the guarded work, so the reported state is the
/// authority the work started under and never a clock read taken after it.
pub(crate) fn deadline_state(deadline: Option<Instant>) -> &'static str {
    match deadline {
        None => "none",
        Some(limit) if Instant::now() >= limit => "expired",
        Some(_) => "live",
    }
}

/// `outcome=` for a `[patches-drop]` line, in the funnel's own attempt /
/// success / refusal vocabulary: `None` is the SUCCESS (the only case that
/// actually replaced the carrier), `Some(_)` is the typed refusal
/// `record_patches_materialization_refusal` counts on the same path.
fn outcome_label(outcome: Option<PatchesMaterializationRefusal>) -> &'static str {
    match outcome {
        None => "ok",
        Some(PatchesMaterializationRefusal::Deadline) => "refused-deadline",
        Some(PatchesMaterializationRefusal::Memory) => "refused-memory",
        Some(PatchesMaterializationRefusal::Semantic) => "refused-semantic",
    }
}

fn flag(value: bool) -> u8 {
    u8::from(value)
}

/// Pure formatting core for a `[patches-drop]` line, I/O-free and clock-free
/// (`deadline` and `outcome` both arrive already classified) so the unit test
/// can assert the exact output without touching the process clock or stderr.
#[allow(clippy::too_many_arguments)] // one field per record column; that IS the record
fn drop_line(
    scope: &str,
    node: &str,
    site: &str,
    purpose: PatchesMaterializationPurpose,
    outcome: &str,
    deadline: &str,
    rows: usize,
    unstable: bool,
    coeff_err: bool,
) -> String {
    format!(
        "[patches-drop] scope={scope} node={node} site={site} purpose={purpose:?} \
         outcome={outcome} deadline={deadline} rows={rows} unstable={} coeff_err={}",
        flag(unstable),
        flag(coeff_err),
    )
}

/// Emit one `[patches-drop]` line for one full Patches -> Dense carrier
/// materialization ATTEMPT. Callers check [`enabled`] first; `site` must come
/// from a `Location::caller()` taken inside the `#[track_caller]` funnel, so it
/// names the densifying caller and not the materializer.
///
/// Called from the funnel's OUTCOME arms, not its entry: `outcome` is `None`
/// for the success that actually replaced the carrier and `Some(_)` for a
/// typed refusal, which leaves the carrier Patches at every consumer. `deadline`
/// is the state classified BEFORE the work, passed in for exactly that reason.
pub(crate) fn record_densify(
    site: &Location<'_>,
    bounds: &PatchesLinearBounds,
    deadline: &str,
    purpose: PatchesMaterializationPurpose,
    outcome: Option<PatchesMaterializationRefusal>,
) {
    let (scope, node) = walk_position();
    let coeff_err = bounds.lower_a.coeff_err.is_some() || bounds.upper_a.coeff_err.is_some();
    let unstable = bounds.lower_a.unstable_idx.is_some() || bounds.upper_a.unstable_idx.is_some();
    eprintln!(
        "{}",
        drop_line(
            &scope,
            &node,
            &format!("{}:{}", site.file(), site.line()),
            purpose,
            outcome_label(outcome),
            deadline,
            bounds.row_count,
            unstable,
            coeff_err,
        )
    );
}

/// Pure formatting core for a `[patches-carrier]` decision line. Same split as
/// [`drop_line`]: no clock, no I/O.
///
/// `was_patches` renders as `repr_in=`, not `repr=`: it is the carrier
/// representation ON ENTRY to the step, an INPUT to the decision, and the
/// outgoing representation is not known here.
#[allow(clippy::too_many_arguments)] // one field per decision input; that IS the record
fn carrier_line(
    scope: &str,
    node: &str,
    op: &str,
    was_patches: bool,
    allow_patches: bool,
    hard: bool,
    handled: bool,
    deadline: &str,
    expiry_lever: bool,
) -> String {
    format!(
        "[patches-carrier] scope={scope} node={node} op={op} repr_in={} allow_patches={} \
         hard={} handled={} deadline={deadline} expiry_lever={}",
        if was_patches { "patches" } else { "dense" },
        flag(allow_patches),
        flag(hard),
        flag(handled),
        flag(expiry_lever),
    )
}

/// Emit one `[patches-carrier]` decision line for a backward step. Callers
/// check [`enabled`] first.
///
/// `expiry_lever` is supplied by the caller rather than read here: the
/// `NY_PATCHES_FINITE_EXPIRY` latch lives inside the private `network::core`
/// tree, and re-reading its declaration from this module would duplicate a
/// lever parser that is part of that lever's contract.
#[allow(clippy::too_many_arguments)] // one field per decision input; that IS the record
pub(crate) fn record_step_decision(
    scope: &str,
    node: &str,
    op: &str,
    was_patches: bool,
    allow_patches: bool,
    hard: bool,
    handled: bool,
    deadline: Option<Instant>,
    expiry_lever: bool,
) {
    eprintln!(
        "{}",
        carrier_line(
            scope,
            node,
            op,
            was_patches,
            allow_patches,
            hard,
            handled,
            deadline_state(deadline),
            expiry_lever,
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uncached decision, rebuilt from the raw chokepoint view on every call —
    /// the deterministic seam the env-gate test drives (the production path
    /// latches the string in [`env_raw`], which another test in this process
    /// may already have initialized).
    fn enabled_uncached() -> bool {
        gate_on(raw_uncached().as_deref())
    }

    /// The parser contract, pinned: exactly `"1"` arms. `"true"`, `"0"`, `""`
    /// and absence are all OFF — a widened parser would silently arm an
    /// instrument inside deadline-sensitive walks.
    #[test]
    fn gate_requires_exactly_one() {
        assert!(gate_on(Some("1")));
        assert!(!gate_on(Some("true")));
        assert!(!gate_on(Some("0")));
        assert!(!gate_on(Some("")));
        assert!(!gate_on(Some(" 1")));
        assert!(!gate_on(None));

        crate::tests::with_serialized_env_vars_removed(&["NY_PATCHES_CARRIER_TRACE"], || {
            assert!(!enabled_uncached(), "unset must be OFF (silent)");
        });
        crate::tests::with_serialized_env_vars(&[("NY_PATCHES_CARRIER_TRACE", "1")], || {
            assert!(enabled_uncached(), "\"1\" must be ON");
        });
        crate::tests::with_serialized_env_vars(&[("NY_PATCHES_CARRIER_TRACE", "true")], || {
            assert!(!enabled_uncached(), "non-\"1\" must be OFF (silent)");
        });
    }

    #[test]
    fn drop_line_names_the_site_the_outcome_and_the_deadline_state() {
        // `scope` is a WALK-KIND literal at three of the four call sites; using a
        // node name here pinned an example no production site can emit.
        let line = drop_line(
            "graph-crown",
            "/layers.3/Conv",
            "crates/ny-propagate/src/x.rs:42",
            PatchesMaterializationPurpose::Other,
            outcome_label(None),
            "live",
            512,
            false,
            true,
        );
        assert_eq!(
            line,
            "[patches-drop] scope=graph-crown node=/layers.3/Conv \
             site=crates/ny-propagate/src/x.rs:42 purpose=Other outcome=ok deadline=live \
             rows=512 unstable=0 coeff_err=1"
        );
    }

    /// `outcome=` is what separates a real carrier drop from an attempt the
    /// funnel refused. Both `Err` consumers are transactional and leave the
    /// carrier Patches, so only `ok` names a densifying site; an entry-emitted
    /// line could not tell the two apart.
    #[test]
    fn outcome_label_separates_the_drop_from_the_refusals() {
        assert_eq!(outcome_label(None), "ok");
        assert_eq!(
            outcome_label(Some(PatchesMaterializationRefusal::Deadline)),
            "refused-deadline"
        );
        assert_eq!(
            outcome_label(Some(PatchesMaterializationRefusal::Memory)),
            "refused-memory"
        );
        assert_eq!(
            outcome_label(Some(PatchesMaterializationRefusal::Semantic)),
            "refused-semantic"
        );
    }

    #[test]
    fn carrier_line_records_the_full_decision_tuple() {
        let line = carrier_line(
            "/layers.3/Conv",
            "/layers.3/Conv",
            "Conv2d",
            true,
            true,
            true,
            false,
            "live",
            false,
        );
        assert_eq!(
            line,
            "[patches-carrier] scope=/layers.3/Conv node=/layers.3/Conv op=Conv2d \
             repr_in=patches allow_patches=1 hard=1 handled=0 deadline=live expiry_lever=0"
        );
    }

    /// The walk position is per-thread and reusable; an unset slot reads as
    /// empty rather than panicking.
    #[test]
    fn walk_position_round_trips_and_defaults_empty() {
        assert_eq!(walk_position(), (String::new(), String::new()));
        enter_node("spec-crown", "/layers.3/Conv");
        assert_eq!(
            walk_position(),
            ("spec-crown".to_string(), "/layers.3/Conv".to_string())
        );
        enter_node("dag-alpha", "/layers.1/Relu");
        assert_eq!(
            walk_position(),
            ("dag-alpha".to_string(), "/layers.1/Relu".to_string())
        );
    }

    #[test]
    fn deadline_state_distinguishes_expiry_from_presence() {
        assert_eq!(deadline_state(None), "none");
        assert_eq!(deadline_state(Some(Instant::now())), "expired");
        assert_eq!(
            deadline_state(Some(Instant::now() + std::time::Duration::from_mins(10))),
            "live"
        );
    }
}
