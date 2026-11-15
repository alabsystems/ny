// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original full-column Concat pullback, including embedded original constants.
//! Property rows are independent of tensor axes. Every original slot is either
//! split without changing coefficient/error bits or discharged into one shared
//! directed bias. Preparation and publication use the existing query ledger.

use super::*;

const MAX_SLOTS: usize = 32;
#[derive(Clone, Copy)]
struct Slot<'a> {
    source: Option<&'a str>,
    bounds: &'a BoundedTensor,
    shape: [usize; 8],
    width: usize,
    start: usize,
}
impl Slot<'_> {
    fn output_column(&self, column: usize, total_width: usize, inner: usize) -> usize {
        let block = self.width * inner;
        (column / block) * total_width * inner + self.start * inner + column % block
    }
}

pub(super) fn propagate(
    query: &mut AttentionRegionQuery,
    node: &GraphNode,
    incoming: &BatchedLinearBounds,
    input: &BoundedTensor,
    boxes: &HashMap<String, BoundedTensor>,
    accumulator: &mut BatchedCrownAccumulator,
) -> Result<bool> {
    let Layer::Concat(layer) = &node.layer else {
        return Ok(false);
    };
    let count = layer
        .constant_inputs
        .as_ref()
        .map_or(node.inputs.len(), Vec::len);
    if count == 0
        || count > MAX_SLOTS
        || query.global_steps >= query.global_step_limit
        || !query.has_held_slot()
    {
        return Ok(false);
    }
    let Some(budget) = query.budget.clone() else {
        return Ok(false);
    };
    // The bounded metadata/shape/slot scan is paid before traversing inputs.
    let mut header = match budget.lease(WorkRequest {
        work: 8192,
        ..WorkRequest::default()
    }) {
        Ok(lease) => lease,
        Err(_) => return Ok(false),
    };
    let deadline = match (query.deadline, header.deadline()) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    poll(deadline, &budget)?;
    ny_core::require_f64_interval_proof_environment()?;
    let Some(output) = boxes.get(node.name()) else {
        return Ok(false);
    };
    incoming.validate_internal_shapes()?;
    let rank = output.shape().len();
    if rank == 0
        || rank > 8
        || output.shape().contains(&0)
        || !full_columns(incoming, output.shape())
    {
        return Ok(false);
    }
    let mut slots = [None; MAX_SLOTS];
    let mut dynamic = 0usize;
    let mut restored_batch = false;
    for (slot, entry) in slots.iter_mut().enumerate().take(count) {
        poll(deadline, &budget)?;
        let embedded = layer.constant_input(slot);
        let (source, bounds) = if let Some(bounds) = embedded {
            (None, bounds)
        } else {
            let Some(name) = node.inputs.get(dynamic) else {
                return Ok(false);
            };
            dynamic += 1;
            let bounds = if name == NETWORK_INPUT {
                input
            } else {
                let Some(bounds) = boxes.get(name) else {
                    return Ok(false);
                };
                bounds
            };
            (Some(name.as_str()), bounds)
        };
        let raw = bounds.shape();
        if raw.is_empty() || raw.len() > 8 || raw.contains(&0) {
            return Ok(false);
        }
        let mut shape = [1; 8];
        if raw.len() == rank {
            shape[..rank].copy_from_slice(raw);
        } else if raw.len() == rank + 1 && raw[0] == 1 {
            // Removing only an actual singleton storage batch is value-exact.
            shape[..rank].copy_from_slice(&raw[1..]);
        } else if raw.len() + 1 == rank && output.shape()[0] == 1 {
            shape[1..rank].copy_from_slice(raw);
            restored_batch = true;
        } else {
            return Ok(false);
        }
        if checked_shape_product(&shape[..rank]) != Some(bounds.len())
            || !bounds.lower().is_standard_layout()
            || !bounds.upper().is_standard_layout()
        {
            return Ok(false);
        }
        *entry = Some(Slot {
            source,
            bounds,
            shape,
            width: 0,
            start: 0,
        });
    }
    if dynamic != node.inputs.len() || dynamic == 0 {
        return Ok(false);
    }
    let axis = match layer.normalize_axis_with_restored_batch(rank, restored_batch) {
        Ok(axis) => axis,
        Err(_) => return Ok(false),
    };
    let mut total_width = 0usize;
    for entry in slots.iter_mut().take(count) {
        let slot = entry.as_mut().expect("resolved original slot");
        for (dim, &extent) in slot.shape[..rank].iter().enumerate() {
            if dim != axis && extent != output.shape()[dim] {
                return Ok(false);
            }
        }
        slot.width = slot.shape[axis];
        slot.start = total_width;
        total_width = total_width.checked_add(slot.width).ok_or_else(declined)?;
    }
    if total_width != output.shape()[axis] {
        return Ok(false);
    }
    let inner = checked_shape_product(&output.shape()[axis + 1..]).ok_or_else(declined)?;
    let rows = incoming.lower_b.len();
    let all_cells = rows.checked_mul(output.len()).ok_or_else(declined)?;
    let dynamic_columns = slots
        .iter()
        .take(count)
        .flatten()
        .filter(|s| s.source.is_some())
        .try_fold(0usize, |sum, s| sum.checked_add(s.bounds.len()))
        .ok_or_else(declined)?;
    let persistent = rows
        .checked_mul(dynamic_columns)
        .and_then(|n| n.checked_mul(32))
        .and_then(|n| {
            rows.checked_mul(dynamic)
                .and_then(|m| m.checked_mul(32))
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| n.checked_add(count.checked_mul(1024)?))
        .ok_or_else(declined)?;
    let scratch = incoming
        .memory_bytes()
        .checked_mul(2)
        .and_then(|n| n.checked_add(8192))
        .ok_or_else(declined)?;
    let work = all_cells
        .checked_mul(256)
        .and_then(|n| n.checked_add(8192))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(declined)?;
    let request = WorkRequest {
        persistent_bytes: persistent,
        scratch_bytes: scratch,
        work,
        ..WorkRequest::default()
    };
    if query
        .admit_peak(accumulator.total_memory_bytes(), request, 0)
        .is_err()
    {
        return Ok(false);
    }
    let mut lease = match budget.lease(request) {
        Ok(lease) => lease,
        Err(_) => return Ok(false),
    };
    let candidate = (|| {
        let flat = consume::flat_incoming(incoming, output.len(), deadline, &budget)?;
        let (lb, ub) = constant_bias(
            &flat,
            &slots[..count],
            total_width,
            inner,
            deadline,
            &budget,
        )?;
        let mut parts = Vec::new();
        parts.try_reserve_exact(dynamic).map_err(|_| declined())?;
        if parts.capacity() != dynamic {
            return Err(declined());
        }
        let mut first = true;
        for slot in slots[..count].iter().flatten() {
            let Some(source) = slot.source else { continue };
            let bounds = split(
                &flat,
                slot,
                total_width,
                inner,
                if first { Some((&lb, &ub)) } else { None },
                deadline,
                &budget,
            )?;
            first = false;
            parts.push((
                source,
                global_source(bounds, slot.bounds.shape(), deadline, &budget)?,
            ));
        }
        consume::validate_linear(&flat, deadline, &budget)?;
        poll(deadline, &budget)?;
        Ok::<_, NyError>(parts)
    })();
    let candidate = match candidate {
        Ok(parts) => parts,
        Err(error) => {
            tracing::debug!(node=node.name(), %error, "original global Concat candidate declined");
            return Ok(false);
        }
    };
    poll(deadline, &budget)?;
    ny_core::require_f64_interval_proof_environment()?;
    lease.mark_materialized().map_err(refusal)?;
    lease.complete().map_err(refusal)?;
    header.complete().map_err(refusal)?;
    drop(header);
    query.hold_completed(lease)?;
    query.global_steps += 1;
    // All candidates are checked before the first merge. A failed merge aborts
    // the whole graph walk; it must never retry the original contribution.
    for (source, part) in candidate {
        accumulator.accumulate(source, BatchedCrownBounds::Dense(part))?;
    }
    tracing::debug!(
        node = node.name(),
        completed = query.global_steps,
        "original global Concat propagated"
    );
    Ok(true)
}

fn constant_bias(
    incoming: &LinearBounds,
    slots: &[Option<Slot<'_>>],
    total_width: usize,
    inner: usize,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let rows = incoming.num_outputs();
    let mut lb = consume::zeroes(rows, deadline, budget)?;
    let mut ub = consume::zeroes(rows, deadline, budget)?;
    for slot in slots.iter().flatten().filter(|s| s.source.is_none()) {
        for (i, (&lo, &hi)) in slot
            .bounds
            .lower()
            .iter()
            .zip(slot.bounds.upper())
            .enumerate()
        {
            if i.is_multiple_of(256) {
                poll(deadline, budget)?;
            }
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(declined());
            }
        }
    }
    for row in 0..rows {
        poll(deadline, budget)?;
        for (a, error, original, store, direction) in [
            (
                &incoming.lower_a,
                incoming.lower_a_err.as_ref(),
                incoming.lower_b[row],
                &mut lb,
                OutwardDirection::Lower,
            ),
            (
                &incoming.upper_a,
                incoming.upper_a_err.as_ref(),
                incoming.upper_b[row],
                &mut ub,
                OutwardDirection::Upper,
            ),
        ] {
            let terms = slots
                .iter()
                .flatten()
                .filter(|s| s.source.is_none())
                .flat_map(|slot| {
                    let lo = slot
                        .bounds
                        .lower()
                        .as_slice()
                        .expect("standard original constant");
                    let hi = slot
                        .bounds
                        .upper()
                        .as_slice()
                        .expect("standard original constant");
                    (0..slot.bounds.len()).map(move |j| {
                        let c = a[[row, slot.output_column(j, total_width, inner)]];
                        let x = match direction {
                            OutwardDirection::Lower => {
                                if c >= 0.0 {
                                    lo[j]
                                } else {
                                    hi[j]
                                }
                            }
                            OutwardDirection::Upper => {
                                if c >= 0.0 {
                                    hi[j]
                                } else {
                                    lo[j]
                                }
                            }
                        };
                        (c, x)
                    })
                });
            let sum = certified_affine_sum_f32_with_poll(original, terms, direction, |_| {
                poll(deadline, budget)
            })?;
            let carry = if let Some(error) = error {
                let terms = slots
                    .iter()
                    .flatten()
                    .filter(|s| s.source.is_none())
                    .flat_map(|slot| {
                        let lo = slot
                            .bounds
                            .lower()
                            .as_slice()
                            .expect("standard original constant");
                        let hi = slot
                            .bounds
                            .upper()
                            .as_slice()
                            .expect("standard original constant");
                        (0..slot.bounds.len()).map(move |j| {
                            (
                                error[[row, slot.output_column(j, total_width, inner)]],
                                lo[j].abs().max(hi[j].abs()),
                            )
                        })
                    });
                certified_affine_sum_f32_with_poll(0.0, terms, OutwardDirection::Upper, |_| {
                    poll(deadline, budget)
                })?
            } else {
                0.0
            };
            store[row] = match direction {
                OutwardDirection::Lower => f64_to_f32_down_no_subnormal(if carry == 0.0 {
                    sum
                } else {
                    next_down_f64(sum - carry)
                }),
                OutwardDirection::Upper => f64_to_f32_up_no_subnormal(if carry == 0.0 {
                    sum
                } else {
                    next_up_f64(sum + carry)
                }),
            };
        }
    }
    Ok((lb, ub))
}

#[allow(clippy::too_many_arguments)]
fn split(
    incoming: &LinearBounds,
    slot: &Slot<'_>,
    total_width: usize,
    inner: usize,
    bias: Option<(&[f32], &[f32])>,
    deadline: Option<Instant>,
    budget: &WorkBudget,
) -> Result<LinearBounds> {
    let rows = incoming.num_outputs();
    let columns = slot.bounds.len();
    let cells = rows.checked_mul(columns).ok_or_else(declined)?;
    let mut la = consume::zeroes(cells, deadline, budget)?;
    let mut ua = consume::zeroes(cells, deadline, budget)?;
    let mut le = consume::zeroes(cells, deadline, budget)?;
    let mut ue = consume::zeroes(cells, deadline, budget)?;
    for row in 0..rows {
        for column in 0..columns {
            if column.is_multiple_of(256) {
                poll(deadline, budget)?;
            }
            let output = slot.output_column(column, total_width, inner);
            let i = row * columns + column;
            la[i] = incoming.lower_a[[row, output]];
            ua[i] = incoming.upper_a[[row, output]];
            le[i] = incoming
                .lower_a_err
                .as_ref()
                .map_or(0.0, |a| a[[row, output]]);
            ue[i] = incoming
                .upper_a_err
                .as_ref()
                .map_or(0.0, |a| a[[row, output]]);
        }
    }
    let mut lb = consume::zeroes(rows, deadline, budget)?;
    let mut ub = consume::zeroes(rows, deadline, budget)?;
    if let Some((lower, upper)) = bias {
        lb.copy_from_slice(&lower[..rows]);
        ub.copy_from_slice(&upper[..rows]);
    }
    finish(rows, columns, la, ua, lb, ub, le, ue, deadline, budget)
}

#[cfg(test)]
#[path = "concat_tests.rs"]
mod tests;
