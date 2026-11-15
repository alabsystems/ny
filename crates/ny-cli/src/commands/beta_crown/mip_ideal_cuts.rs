// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! One bounded attempt to strengthen the existing complete MIP's root.
//! Cut metadata grants no verdict. An optional opaque native cover is used
//! only after whole-original-model binding and fresh publication checks.

use super::*;
use ny_mip::ir::Row;
use ny_mip::single_hidden_hull::bab::{HullBabOptions, HULL_BAB_MAX_WORKERS};
use ny_mip::single_hidden_hull::mip_cuts::{
    propose_root_mip_cuts, propose_root_mip_cuts_with_conditional_mass_portfolio_attempt,
    propose_root_mip_cuts_with_global_sdp, propose_root_mip_cuts_with_groups,
    HullAggregateRootProof, HullMipContinuation, HullMipCutReport, HullMipEncoding,
    ROOT_MIP_CUT_MAX_CELLS, ROOT_MIP_CUT_MAX_HEADS, ROOT_MIP_CUT_MAX_INPUTS,
};
use ny_mip::single_hidden_hull::HullMarginProblem;
use ny_mip::MipParts;
use std::time::{Duration, Instant};

fn stage_end(now: Instant, original: Instant) -> Option<Instant> {
    let remaining = original.checked_duration_since(now)?;
    let slice = remaining.mul_f64(0.25).min(Duration::from_secs(4));
    let end = now.checked_add(slice)?.min(original);
    (end > now).then_some(end)
}

pub(super) struct StrengtheningOutcome {
    pub(super) cuts: HullMipCutReport,
    pub(super) continuation: Option<HullMipContinuation>,
    pub(super) aggregate: Option<HullAggregateRootProof>,
    pub(super) cuts_elapsed: Duration,
    pub(super) continuation_elapsed: Duration,
}

/// Retained Source/MIP and required AY canaries own their original proof
/// authority. A legacy native cover must not mint either kind of receipt.
pub(super) fn short_proof_allowed(
    required_shared: bool,
    canary: bool,
    bridge_capture: bool,
) -> bool {
    !required_shared && !canary && !bridge_capture
}

fn continuation_end(now: Instant, initial_remaining: Duration, stage: Instant) -> Option<Instant> {
    let end = now
        .checked_add(initial_remaining.div_f64(8.0).min(Duration::from_secs(2)))?
        .min(stage);
    (end > now).then_some(end)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn strengthen(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    hidden_bounds: &[Vec<Bound>],
    input_bounds: &[Bound],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    original_deadline: Instant,
) -> Option<HullMipCutReport> {
    strengthen_inner(
        parts,
        single_hidden,
        weights,
        biases,
        dims,
        hidden_bounds,
        input_bounds,
        spec,
        unsafe_rows,
        original_deadline,
        false,
        false,
        false,
        false,
    )
    .map(|out| out.cuts)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn strengthen_with_short_proof(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    hidden_bounds: &[Vec<Bound>],
    input_bounds: &[Bound],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    original_deadline: Instant,
    allow_short_proof: bool,
) -> Option<StrengtheningOutcome> {
    strengthen_inner(
        parts,
        single_hidden,
        weights,
        biases,
        dims,
        hidden_bounds,
        input_bounds,
        spec,
        unsafe_rows,
        original_deadline,
        allow_short_proof,
        false,
        false,
        false,
    )
}

/// Grouped-cut test baseline; production uses signed refinement below.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn strengthen_with_groups(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    hidden_bounds: &[Vec<Bound>],
    input_bounds: &[Bound],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    original_deadline: Instant,
    allow_short_proof: bool,
) -> Option<StrengtheningOutcome> {
    strengthen_inner(
        parts,
        single_hidden,
        weights,
        biases,
        dims,
        hidden_bounds,
        input_bounds,
        spec,
        unsafe_rows,
        original_deadline,
        allow_short_proof,
        true,
        false,
        false,
    )
}

/// Global-SDP test baseline; production uses signed refinement below.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn strengthen_with_global_sdp(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    hidden_bounds: &[Vec<Bound>],
    input_bounds: &[Bound],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    original_deadline: Instant,
    allow_short_proof: bool,
) -> Option<StrengtheningOutcome> {
    strengthen_inner(
        parts,
        single_hidden,
        weights,
        biases,
        dims,
        hidden_bounds,
        input_bounds,
        spec,
        unsafe_rows,
        original_deadline,
        allow_short_proof,
        true,
        true,
        false,
    )
}

/// Default eligible signed refinement; existing helper seams remain unchanged.
#[allow(clippy::too_many_arguments)]
pub(super) fn strengthen_with_signed_refinement(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    hidden_bounds: &[Vec<Bound>],
    input_bounds: &[Bound],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    original_deadline: Instant,
    allow_short_proof: bool,
) -> Option<StrengtheningOutcome> {
    strengthen_inner(
        parts,
        single_hidden,
        weights,
        biases,
        dims,
        hidden_bounds,
        input_bounds,
        spec,
        unsafe_rows,
        original_deadline,
        allow_short_proof,
        true,
        true,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn strengthen_inner(
    parts: &mut MipParts,
    single_hidden: bool,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    dims: &[usize],
    hidden_bounds: &[Vec<Bound>],
    input_bounds: &[Bound],
    spec: &VnnLibSpec,
    unsafe_rows: &[Row],
    original_deadline: Instant,
    allow_short_proof: bool,
    groups: bool,
    global_sdp: bool,
    signed: bool,
) -> Option<StrengtheningOutcome> {
    let original_deadline = ny_core::instance_budget::deadline()
        .map_or(original_deadline, |global| global.min(original_deadline));
    let now = Instant::now();
    let initial_remaining = original_deadline.checked_duration_since(now)?;
    let end = stage_end(now, original_deadline)?;
    if !single_hidden
        || spec.dual_network.is_some()
        || spec.per_clause_input_bounds.iter().any(|c| !c.is_empty())
    {
        return None;
    }
    let (first, second, reverse) = certified_shared_tree_pairwise_indices(spec)?;
    let [d, h, o] = dims else { return None };
    let [unsafe_row] = unsafe_rows else {
        return None;
    };
    if *d == 0
        || *h == 0
        || *o == 0
        || *d > ROOT_MIP_CUT_MAX_INPUTS
        || *h > ROOT_MIP_CUT_MAX_HEADS
        || d.checked_mul(*h)? > ROOT_MIP_CUT_MAX_CELLS
        || h.checked_mul(*o)? > ROOT_MIP_CUT_MAX_CELLS
        || weights.len() != 2
        || biases.len() != 2
        || input_bounds.len() != *d
        || spec.num_inputs != *d
        || spec.num_outputs != *o
        || weights[0].len() != d.checked_mul(*h)?
        || weights[1].len() != h.checked_mul(*o)?
        || biases[0].len() != *h
        || biases[1].len() != *o
        || hidden_bounds.len() != 1
        || hidden_bounds[0].len() != *h
    {
        return None;
    }
    let objective = if reverse {
        [(second, 1.0), (first, -1.0)]
    } else {
        [(first, 1.0), (second, -1.0)]
    };
    let problem = HullMarginProblem::new(
        weights[0].clone(),
        biases[0].clone(),
        *d,
        &weights[1],
        &biases[1],
        *o,
        &objective,
        0.0,
        input_bounds.iter().map(|b| f64::from(b.lower())).collect(),
        input_bounds.iter().map(|b| f64::from(b.upper())).collect(),
    )
    .ok()?;
    let reserve = initial_remaining.div_f64(8.0).min(Duration::from_secs(2));
    let cuts = if signed {
        let attempt =
            propose_root_mip_cuts_with_conditional_mass_portfolio_attempt(&problem, end, reserve);
        let p = &attempt.primal;
        eprintln!(
            "[mip-lp-primal] ledger=after-root-attempt attempted={} admitted={} same_call={} columns={} positive_heads={} negative_heads={} work_units={} peak_bytes={} decoded={} bank_attempts={} banked={} cuts_returned={} stop={}",
            u8::from(p.attempted),
            u8::from(p.admitted),
            u8::from(p.same_call),
            p.columns,
            p.positive_heads,
            p.negative_heads,
            p.work_units,
            p.peak_bytes,
            u8::from(p.decoded),
            p.bank_attempts,
            u8::from(p.banked),
            u8::from(attempt.cuts.is_some()),
            p.stop_reason
        );
        attempt.cuts
    } else if global_sdp {
        propose_root_mip_cuts_with_global_sdp(&problem, end, reserve)
    } else if groups {
        propose_root_mip_cuts_with_groups(&problem, end, reserve)
    } else {
        propose_root_mip_cuts(&problem, end)
    }?;
    // Bind and bank all valid GLOBAL rows before any native closure can be
    // considered. The continuation borrows the immutable original root only.
    let report = cuts.append_to(
        parts,
        &HullMipEncoding {
            weights,
            biases,
            dimensions: dims,
            hidden_bounds,
            first_output: first,
            second_output: second,
            reverse,
            unsafe_row: *unsafe_row,
        },
        end,
    )?;
    let portfolio = cuts.mass_portfolio_report();
    eprintln!(
        "[mip-signed-mass-portfolio] attempted={} admitted={} work_units={} admitted_bytes={} ir_row_limit={} reserved_ir_rows={} collection_attempts={} completed_collections={} retained={} candidate_lower={:?} scalar_threshold={:?} export_attempts={} prepared_rows={} prepared_head_terms={} prepared_terms={} appended_rows={} appended_head_terms={} appended_terms={} stop={}",
        u8::from(portfolio.attempted),
        u8::from(portfolio.admitted),
        portfolio.work_units,
        portfolio.admitted_bytes,
        portfolio.ir_row_limit,
        portfolio.reserved_ir_rows,
        portfolio.collection_attempts,
        portfolio.completed_collections,
        u8::from(portfolio.retained),
        portfolio.candidate_lower,
        portfolio.scalar_threshold,
        portfolio.export_attempts,
        portfolio.prepared_rows,
        portfolio.prepared_head_terms,
        portfolio.prepared_terms,
        portfolio.appended_rows,
        portfolio.appended_head_terms,
        portfolio.appended_terms,
        portfolio.stop_reason
    );
    eprintln!(
        "[mip-signed-mass-conditional] attempted={} admitted={} work_units={} admitted_bytes={} collections={} prepared_rows={} prepared_terms={} appended_rows={} appended_terms={}",
        u8::from(portfolio.phase_attempted),
        u8::from(portfolio.phase_admitted),
        portfolio.phase_work_units,
        portfolio.phase_admitted_bytes,
        portfolio.phase_collections,
        portfolio.phase_prepared_rows,
        portfolio.phase_prepared_terms,
        portfolio.phase_appended_rows,
        portfolio.phase_appended_terms
    );
    let cube = &portfolio.cube;
    eprintln!(
        "[mip-signed-mass-cube] attempted={} admitted={} work_units={} admitted_bytes={} collections={} masks={} tables={} activity_components={} projected_components={} prepared_rows={} appended_rows={} appended_terms={} selector_terms={} stop={}",
        u8::from(cube.attempted),
        u8::from(cube.admitted),
        cube.work_units,
        cube.admitted_bytes,
        cube.collection_attempts,
        cube.collected_masks,
        cube.completed_tables,
        cube.activity_components,
        cube.projected_components,
        cube.prepared_rows,
        cube.appended_rows,
        cube.appended_terms,
        cube.appended_selector_terms,
        cube.stop_reason
    );
    // A separate full-q checked floor, never LP-attained scalar plus gain.
    // Original whole-model binding has already banked the global cut set.
    let aggregate = allow_short_proof
        .then(|| {
            if signed {
                cuts.best_signed_root_proof(end)
            } else if global_sdp {
                cuts.best_root_proof(end)
            } else {
                cuts.compact_root_proof(end)
            }
        })
        .flatten();
    let cuts_elapsed = now.elapsed();
    let phase_began = Instant::now();
    let continuation = if allow_short_proof && aggregate.is_none() {
        continuation_end(Instant::now(), initial_remaining, end).and_then(|phase_end| {
            let workers = std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .clamp(1, HULL_BAB_MAX_WORKERS);
            let options = HullBabOptions {
                workers,
                ..HullBabOptions::default()
            };
            if signed {
                cuts.continue_hull_bab_with_native_rows(phase_end, options)
            } else {
                cuts.continue_hull_bab(phase_end, options)
            }
        })
    } else {
        None
    };
    if signed {
        let funding = cuts.native_funding_report();
        eprintln!(
            "[mip-native-funding] calls={} setup_work={} shared_work={} per_call_work={} prepaid_work={} admitted_bytes={}",
            funding.funded_calls,
            funding.setup_work_units,
            funding.shared_work_units,
            funding.per_call_work_units,
            funding.prepaid_work_units,
            funding.admitted_bytes
        );
        let native = cuts.native_rows_report();
        eprintln!(
            "[mip-native-rows] attempted={} committed={} priced={}",
            native.attempted, native.committed, native.priced
        );
        if let Some(tree) = continuation.as_ref().map(|value| value.report()) {
            if let Some(before) = tree.root_before_native {
                let after = tree.root_certified.map(|value| value.max(before));
                eprintln!(
                    "[mip-native-root] before={before:?} after={after:?} improved={} root_lp_calls={} closed={} open={} stop={:?} root_lp_status={:?}",
                    u8::from(after.is_some_and(|value| value > before)),
                    tree.root_lp_calls,
                    tree.closed,
                    tree.open,
                    tree.stop,
                    tree.root_lp_status,
                );
            }
        }
    }
    Some(StrengtheningOutcome {
        cuts: report,
        continuation,
        aggregate,
        cuts_elapsed,
        continuation_elapsed: phase_began.elapsed(),
    })
}

#[cfg(test)]
#[path = "mip_ideal_cuts_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mip_short_proof_tests.rs"]
mod short_proof_tests;

#[cfg(test)]
#[path = "mip_compact_root_tests.rs"]
mod compact_root_tests;

#[cfg(test)]
#[path = "mip_group_root_tests.rs"]
mod group_root_tests;

#[cfg(test)]
#[path = "mip_sdp_root_tests.rs"]
mod sdp_root_tests;

#[cfg(test)]
#[path = "mip_signed_root_tests.rs"]
mod signed_root_tests;

#[cfg(test)]
#[path = "mip_signed_direction_tests.rs"]
mod signed_direction_tests;

#[cfg(test)]
#[path = "mip_signed_child_tests.rs"]
mod signed_child_tests;

#[cfg(test)]
#[path = "mip_signed_mass_tests.rs"]
mod signed_mass_tests;

#[cfg(test)]
#[path = "mip_signed_component_rows_tests.rs"]
mod signed_component_rows_tests;
