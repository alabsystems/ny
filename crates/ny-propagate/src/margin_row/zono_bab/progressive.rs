// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Try a cheap original-row certificate before paying for global ranking.
//! If it stays open, build the original canonical ranked state. An exactly
//! matching prefix may retain its frontier; a changed row order discards all
//! prior cover/warm/proof state and restarts with only the REMAINING allowance.
//! Every route retains the original source evaluator and absolute deadline.

use std::time::Instant;

use super::super::zono_state::build::progressive::ProgressiveClass;
use super::{
    admitted_bytes, fd_proof_table_bytes, fd_proof_table_cap, past, search_growing, ClassRun,
    FdProofSeed, FdTable, FrozenDualFrame, FrozenDualOptions, FrozenDualProblem,
    FrozenDualStateMode, OpenReason, SearchOptions, SearchVerdict, TierPolicy, TwinNet,
    ZonoRefusal, FROZEN_DUAL_CHUNK, FROZEN_DUAL_DEFAULT_EVALUATOR, FROZEN_DUAL_MAX_DEPTH,
};

/// Work accounting only. The original certified node evaluator and ONE
/// complete search ledger retain closure authority.
#[derive(Debug, Clone, Default)]
pub struct ProgressiveStats {
    /// Complete original state plans prepared for the shared class budget.
    pub stages: usize,
    /// Canonical whole-box restarts after discarding an incompatible prefix.
    pub restarts: usize,
    /// False if a refusing canonical build did not return column counters.
    pub columns_complete: bool,
    /// Original rows in the last complete producer batch.
    pub rows_built: usize,
    /// Known attempted reverse-pass columns; see `columns_complete`.
    pub attempted_columns: usize,
    /// Nodes charged across the cheap and canonical plans, counted once.
    pub nodes_charged: u64,
    /// Search seconds across all plans, excluding state construction/checking.
    pub search_secs: f64,
    /// Retained for receipts; the continuous search always returns its count.
    pub unobserved_node_allowance: u64,
    /// The original deadline stopped construction/search/proof replay.
    pub deadline_cut: bool,
    /// Diagnostic stop reason; never a certificate.
    pub stop: &'static str,
}

fn table(
    prepared: &ProgressiveClass<'_>,
    net: &TwinNet,
    frame: &FrozenDualFrame,
    deadline: Instant,
    budget: u64,
) -> FdTable {
    if !crate::certify::hint_capture_enabled() {
        return FdTable::Refused("capture_off");
    }
    let estimate = fd_proof_table_bytes(frame);
    if estimate > fd_proof_table_cap()
        || estimate
            .saturating_add(prepared.resident_bytes())
            .saturating_add(prepared.state().bytes())
            > admitted_bytes(budget)
    {
        return FdTable::Refused("table_cap");
    }
    let mut relax = prepared.compact_relax();
    if relax.heap_bytes() > fd_proof_table_cap() {
        return FdTable::Refused("table_cap");
    }
    match relax.verify(net, frame, prepared.state(), Some(deadline)) {
        Ok(()) => FdTable::Compact(relax),
        Err(why) => FdTable::Refused(why),
    }
}

pub(super) fn run(
    net: &TwinNet,
    frame: &FrozenDualFrame,
    class: usize,
    opts: &FrozenDualOptions,
    deadline: Instant,
    policy: &TierPolicy,
    max_rows: usize,
) -> Result<ClassRun, ZonoRefusal> {
    eprintln!("[frozen-dual-progressive] class={class} admitted=1 original_rows={} max_rows={} original_node_cap={} owner=lazy_canonical authority=original_fd_eval",frame.n_unstable,max_rows.min(frame.n_unstable),policy.effective_node_cap());
    let mut prepared = ProgressiveClass::new(net, frame, class, opts, max_rows)?;
    if prepared.state().root_bound() <= 0.0 && !prepared.advance()? {
        return Err(ZonoRefusal::Internal(
            "progressive chart has no split rows".into(),
        ));
    }
    let rows = prepared.state().num_unstable();
    let problem = FrozenDualProblem::new(prepared.state(), &(0..rows).collect::<Vec<_>>())?;
    let capture = crate::certify::hint_capture_enabled();
    let max_depth = max_rows.min(frame.n_unstable).min(FROZEN_DUAL_MAX_DEPTH);
    // Reserve the producer, overlapping old/new problems and reverse/scoring
    // workspace WHILE the frontier lives, under the original fixed envelope.
    let build_peak = prepared.state().stats.estimate_bytes;
    let budget = admitted_bytes(opts.max_state_bytes.min(policy.state_byte_cap()));
    if build_peak >= budget {
        return Err(ZonoRefusal::Memory {
            estimate_bytes: build_peak,
            budget_bytes: budget,
        });
    }
    // Optional trace capacity shrinks to the spare envelope, rather than
    // refusing an otherwise admissible class. The remaining three quarters
    // stay available to its ordinary frontier. Missing records make only the
    // trace incomplete; they never supply a bound or stop numerical search.
    let table_peak = if capture {
        fd_proof_table_bytes(frame).saturating_mul(3)
    } else {
        0
    };
    let per_record_peak = (96 + 17 * max_depth as u64).saturating_mul(4);
    let record_bytes = budget.saturating_sub(build_peak).saturating_sub(table_peak) / 4;
    let affordable = usize::try_from(record_bytes / per_record_peak.max(1)).unwrap_or(usize::MAX);
    let record_cap = if capture && problem.root_bound() <= 0.0 {
        super::fd_proof_record_cap(max_depth).min(affordable)
    } else {
        0
    };
    let records = (record_cap as u64).min(policy.effective_node_cap());
    let proof_resident = table_peak.saturating_add(records.saturating_mul(per_record_peak));
    let peak = build_peak.saturating_add(proof_resident);
    if peak >= budget {
        return Err(ZonoRefusal::Memory {
            estimate_bytes: peak,
            budget_bytes: budget,
        });
    }
    let search_opts = SearchOptions {
        automatic_refine: policy.automatic(),
        refine_steps: policy.refine_steps.min(32),
        deadline: Some(deadline),
        node_cap: policy.effective_node_cap(),
        max_frontier_bytes: budget.saturating_sub(peak),
        chunk: FROZEN_DUAL_CHUNK,
        record_prunes: record_cap,
        evaluator: FROZEN_DUAL_DEFAULT_EVALUATOR,
        crosscheck_every: policy.crosscheck_every,
    };
    let mut stages = 1usize;
    let mut extra_build_ms = 0.0;
    let mut pending_restart = None;
    let mut append = |prior: &FrozenDualProblem| {
        let began = Instant::now();
        let producer_before = prepared.build_ms();
        let result = (|| {
            if !prepared.canonical_ranked()? {
                return Ok(None);
            }
            let rows = prepared.state().num_unstable();
            let next = FrozenDualProblem::new(prepared.state(), &(0..rows).collect::<Vec<_>>())?;
            stages += 1;
            if prior.check_append(&next, Some(deadline)).is_ok() {
                eprintln!("[frozen-dual-progressive] class={class} append={stages} rows={rows} prior_rows={} ranking=canonical authority=exact_prior_problem",prior.max_depth());
                Ok(Some(next))
            } else {
                // The canonical ranking (or independently rounded row batch)
                // changed the old numerical prefix. Transfer NOTHING from it.
                // Its returned node/time counts still consume this objective's
                // one original allowance, even if no restart time remains.
                eprintln!("[frozen-dual-progressive] class={class} transition=canonical_restart rows={rows} prior_rows={} transfer=none authority=original_fd_eval",prior.max_depth());
                pending_restart = Some(next);
                Ok(None)
            }
        })();
        extra_build_ms += (began.elapsed().as_secs_f64() * 1e3
            - (prepared.build_ms() - producer_before))
            .max(0.0);
        result
    };
    let (mut problem, mut search) = search_growing(problem, &search_opts, &mut append);
    let mut prior_nodes = 0u64;
    let mut prior_secs = 0.0;
    let mut restarts = 0usize;
    if let Some(canonical) = pending_restart {
        prior_nodes = search.nodes;
        prior_secs = search.secs;
        let remaining = policy.effective_node_cap().saturating_sub(prior_nodes);
        // Dropping the entire old report drops every prune, level and warm
        // point. Only scalar work counters survive into the canonical plan.
        search = super::SearchReport::empty();
        problem = canonical;
        restarts = 1;
        if past(Some(deadline)) {
            search.verdict = SearchVerdict::Open(OpenReason::Deadline);
        } else if remaining == 0 {
            search.verdict = SearchVerdict::Open(OpenReason::NodeCap);
        } else {
            let mut remaining_opts = search_opts;
            remaining_opts.node_cap = remaining;
            search = super::search_with_relations(&problem, &remaining_opts, None);
        }
    }
    // The final search owns the complete proof cover. An exact append kept
    // only identical prior evaluators; a canonical restart kept NO old prunes.
    let table = table(&prepared, net, frame, deadline, opts.max_state_bytes);
    let deadline_cut =
        past(Some(deadline)) || search.verdict == SearchVerdict::Open(OpenReason::Deadline);
    if deadline_cut {
        search.verdict = SearchVerdict::Open(OpenReason::Deadline);
    }
    let final_rows = prepared.state().num_unstable();
    let order = problem.order().to_vec();
    // A timed-out compatible append still has an exact old prefix. An
    // incompatible canonical plan replaced both the problem and search report,
    // even when no remaining node/time allowance could start its search.
    let proof = capture.then(|| {
        Box::new(
            FdProofSeed::of(prepared.state(), frame, &order, table).finish(&problem, &mut search),
        )
    });
    let stats = ProgressiveStats {
        stages,
        restarts,
        columns_complete: prepared.columns_complete(),
        rows_built: final_rows,
        attempted_columns: prepared.attempted_columns(),
        nodes_charged: prior_nodes.saturating_add(search.nodes),
        search_secs: prior_secs + search.secs,
        unobserved_node_allowance: 0,
        deadline_cut,
        stop: if search.closed() {
            "closed"
        } else if deadline_cut {
            "deadline"
        } else {
            "search_stop"
        },
    };
    eprintln!("[frozen-dual-progressive] class={class} completed=1 stages={} rows={} attempted_columns={} nodes_charged={} search_secs={:.6} unobserved_allowance=0 restarts={} columns_complete={} stop={} owner=lazy_canonical authority=original_fd_eval",stats.stages,stats.rows_built,stats.attempted_columns,stats.nodes_charged,stats.search_secs,stats.restarts,u8::from(stats.columns_complete),stats.stop);
    Ok(ClassRun {
        state_gens: frame.n_gens,
        unstable: frame.n_unstable,
        depth_cap: problem.max_depth(),
        build_ms: prepared.build_ms() + extra_build_ms,
        state_bytes: prepared.state().bytes(),
        build_estimate_bytes: prepared.state().stats.estimate_bytes,
        build_columns: prepared.state().stats.columns,
        streamed_rows: true,
        root_g: prepared.state().root_bound(),
        search,
        state: FrozenDualStateMode::PerClass,
        root_loss: None,
        proof,
        progressive: Some(stats),
    })
}
