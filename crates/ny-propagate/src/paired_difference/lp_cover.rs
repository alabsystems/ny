// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Global affine cuts obtained from a source-owned COMPLETE input cover.
//!
//! A proposed direction q and root slope a produce q*v(x)-a*x >= L_C
//! on each closed cell C by native outward backward substitution. Since all
//! cells cover the original box, q*v(x)-a*x >= min_C L_C is unconditional.
//! The rows can correlate an entire layer, rather than two scalar activations.
//! No selected cell, sampled value, or conditional proof becomes a root cut.
//! A refused cell or expired construction discards the entire result.

use super::lp_certificate::LinearRow;
use super::*;

const MAX_COVER_DEPTH: usize = 6;
const DEFAULT_COVER_DEPTH: usize = 3;
const MAX_CELLS: usize = 1 << MAX_COVER_DEPTH;
const MAX_DIRECTIONS: usize = 128;

pub(super) struct CoverRows {
    pub rows: Vec<LinearRow>,
    pub output_lower: Vec<f64>,
    pub output_upper: Vec<f64>,
    pub cells: usize,
}

struct Direction {
    before: usize,
    cf: Vec<f64>,
    cg: Vec<f64>,
    columns: Vec<(usize, f64)>,
    // An arbitrary exact-f64 affine normal. Its intercept is always replayed.
    root_slope: Vec<f64>,
    box_lower: f64,
    affine_lower: f64,
    output: Option<(usize, f64)>,
}

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

/// Both closed children share an exact stored midpoint. Induction on these
/// replacements proves complete coverage; no caller can supply a partial list.
fn cells(root: &BoxDomain, depth: usize, deadline: Option<Instant>) -> Option<Vec<BoxDomain>> {
    if depth > MAX_COVER_DEPTH || root.lo.len() != root.hi.len() || root.lo.is_empty() {
        return None;
    }
    let mut cells = vec![root.clone()];
    for _ in 0..depth {
        let mut next = Vec::with_capacity(cells.len() * 2);
        for cell in cells {
            if !alive(deadline) {
                return None;
            }
            let mut best: Option<(usize, f64, f64)> = None;
            for j in 0..root.lo.len() {
                let middle = cell.lo[j] * 0.5 + cell.hi[j] * 0.5;
                if !(cell.lo[j] < middle && middle < cell.hi[j]) {
                    continue;
                }
                let width = root.hi[j] - root.lo[j];
                let fraction = (cell.hi[j] - cell.lo[j]) / width;
                if best.as_ref().is_none_or(|&(old, score, _)| {
                    fraction.total_cmp(&score).then(j.cmp(&old)).is_gt()
                }) {
                    best = Some((j, fraction, middle));
                }
            }
            if let Some((j, _, middle)) = best {
                let mut left = cell.clone();
                let mut right = cell;
                left.hi[j] = middle;
                right.lo[j] = middle;
                next.push(left);
                next.push(right);
            } else {
                // A point/adjacent-float cell remains in the cover.
                next.push(cell);
            }
        }
        if next.len() > MAX_CELLS {
            return None;
        }
        cells = next;
    }
    alive(deadline).then_some(cells)
}

pub(super) fn rows(
    chain: &PairedChain,
    domain: &BoxDomain,
    reference: &[Enclosure],
    bindings: &[(Vec<usize>, Vec<usize>)],
    deadline: Option<Instant>,
) -> Option<CoverRows> {
    chain.validate().ok()?;
    chain.input_state(domain)?;
    if bindings.len() != chain.stages.len() + 1
        || reference.len() != bindings.len()
        || bindings[0].0.len() != chain.input_dim
        || bindings[0].0 != bindings[0].1
        || bindings[0].0 != (0..chain.input_dim).collect::<Vec<_>>()
        || !alive(deadline)
    {
        return None;
    }
    let final_state = chain.stages.len();
    let hidden_state = chain
        .stages
        .iter()
        .rposition(|s| matches!(s, Stage::Relu { .. }))
        .map(|s| s + 1);
    let mut directions = Vec::new();
    // Whole-output directions couple every contributing hidden neuron. Signed
    // hidden directions independently strengthen the final activation strips.
    for before in [Some(final_state), hidden_state].into_iter().flatten() {
        if before == final_state && hidden_state == Some(final_state) && !directions.is_empty() {
            continue;
        }
        let (f, g) = &bindings[before];
        if f.len() != g.len() || f.len() > 512 {
            return None;
        }
        for j in 0..f.len() {
            if directions.len() + 2 > MAX_DIRECTIONS {
                break;
            }
            for sign in [1.0, -1.0] {
                let mut cf = vec![0.0; f.len()];
                let mut cg = cf.clone();
                cf[j] = sign;
                cg[j] = -sign;
                let (_, slope) = chain.back(
                    before,
                    cf.clone(),
                    cg.clone(),
                    0.0,
                    reference,
                    domain,
                    true,
                    deadline,
                )?;
                directions.push(Direction {
                    before,
                    cf,
                    cg,
                    columns: vec![(f[j], sign), (g[j], -sign)],
                    root_slope: slope,
                    box_lower: f64::INFINITY,
                    affine_lower: f64::INFINITY,
                    output: (before == final_state).then_some((j, sign)),
                });
            }
        }
    }
    let cover = cells(domain, DEFAULT_COVER_DEPTH, deadline)?;
    for cell in &cover {
        let states = forward::states_with_reference(chain, cell, deadline, true, Some(reference))?;
        for direction in &mut directions {
            if !alive(deadline) {
                return None;
            }
            let lower = chain
                .back(
                    direction.before,
                    direction.cf.clone(),
                    direction.cg.clone(),
                    0.0,
                    &states,
                    cell,
                    true,
                    deadline,
                )?
                .0;
            let affine = chain
                .back_with_root_cost(
                    direction.before,
                    direction.cf.clone(),
                    direction.cg.clone(),
                    0.0,
                    &states,
                    cell,
                    true,
                    deadline,
                    Some(&direction.root_slope),
                )?
                .0;
            direction.box_lower = direction.box_lower.min(lower);
            direction.affine_lower = direction.affine_lower.min(affine);
        }
    }
    let mut result = CoverRows {
        rows: Vec::with_capacity(directions.len() * 2),
        output_lower: reference[final_state].d_lo.clone(),
        output_upper: reference[final_state].d_hi.clone(),
        cells: cover.len(),
    };
    for direction in directions {
        if !direction.box_lower.is_finite() || !direction.affine_lower.is_finite() {
            return None;
        }
        if let Some((j, sign)) = direction.output {
            if sign > 0.0 {
                result.output_lower[j] = result.output_lower[j].max(direction.box_lower);
            } else {
                result.output_upper[j] = result.output_upper[j].min(-direction.box_lower);
            }
        }
        result.rows.push(LinearRow {
            terms: direction.columns.clone(),
            rhs: direction.box_lower,
        });
        let mut terms = direction.columns;
        for (&column, &a) in bindings[0].0.iter().zip(&direction.root_slope) {
            if a != 0.0 {
                terms.push((column, -a));
            }
        }
        result.rows.push(LinearRow {
            terms,
            rhs: direction.affine_lower,
        });
    }
    if result
        .output_lower
        .iter()
        .zip(&result.output_upper)
        .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return None;
    }
    alive(deadline).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internally_constructed_closed_cover_contains_every_grid_point() {
        let root = BoxDomain {
            lo: vec![-1.0, -2.0],
            hi: vec![1.0, 2.0],
        };
        let cover = cells(&root, 6, None).unwrap();
        assert_eq!(cover.len(), 64);
        for i in 0..=32 {
            for j in 0..=32 {
                let x = [-1.0 + i as f64 / 16.0, -2.0 + j as f64 / 8.0];
                assert!(cover
                    .iter()
                    .any(|b| (0..2).all(|a| b.lo[a] <= x[a] && x[a] <= b.hi[a])));
            }
        }
    }

    #[test]
    fn degenerate_cells_are_retained_and_expiry_discards_all_cuts() {
        let root = BoxDomain {
            lo: vec![1.0],
            hi: vec![1.0_f64.next_up()],
        };
        let cover = cells(&root, 6, None).unwrap();
        assert_eq!(cover.len(), 1);
        assert_eq!(cover[0].lo, root.lo);
        assert_eq!(cover[0].hi, root.hi);
        assert!(cells(&root, 6, Some(Instant::now())).is_none());
        assert!(cells(&root, 7, None).is_none());
    }

    #[test]
    fn root_linear_cost_keeps_exact_authored_bias_and_shared_input() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![Stage::Affine {
                rows: 1,
                cols: 1,
                wf: vec![2.0],
                wg: vec![1.0],
                bf: vec![0.25],
                bg: vec![0.0],
            }],
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let states = forward::states(&chain, &domain, None, true).unwrap();
        let (value, residual) = chain
            .back_with_root_cost(
                1,
                vec![1.0],
                vec![-1.0],
                0.0,
                &states,
                &domain,
                true,
                None,
                Some(&[1.0]),
            )
            .unwrap();
        assert_eq!(residual, vec![0.0]);
        assert!(value <= 0.25 && value > 0.25 - 1e-12);
        assert!(chain
            .back_with_root_cost(
                1,
                vec![1.0],
                vec![-1.0],
                0.0,
                &states,
                &domain,
                true,
                None,
                Some(&[f64::NAN])
            )
            .is_none());
    }

    #[test]
    fn common_input_roundoff_cannot_erase_a_real_unit_residual() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![Stage::Relu { width: 1 }],
        };
        let domain = BoxDomain {
            lo: vec![-2.0f64.powi(53)],
            hi: vec![-2.0f64.powi(53)],
        };
        let states = forward::states(&chain, &domain, None, true).unwrap();
        // Exact real residual: (1+2^-53)*(-2^53)-1*(-2^53) = -1.
        // Ordinary rounded coefficient combination would produce zero.
        let (lower, _) = chain
            .back_with_root_cost(
                0,
                vec![1.0],
                vec![2.0f64.powi(-53)],
                0.0,
                &states,
                &domain,
                true,
                None,
                Some(&[1.0]),
            )
            .unwrap();
        assert!(lower <= -1.0);
    }

    #[test]
    fn a_safe_cell_cannot_replace_the_minimum_over_the_complete_cover() {
        let chain = PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![0.0],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let halves = [
            BoxDomain {
                lo: vec![-1.0],
                hi: vec![0.0],
            },
            BoxDomain {
                lo: vec![0.0],
                hi: vec![1.0],
            },
        ];
        let mut values = Vec::new();
        for domain in &halves {
            let states = forward::states(&chain, domain, None, true).unwrap();
            values.push(
                chain
                    .back_with_root_cost(
                        2,
                        vec![1.0],
                        vec![-1.0],
                        0.0,
                        &states,
                        domain,
                        true,
                        None,
                        Some(&[-1.0]),
                    )
                    .unwrap()
                    .0,
            );
        }
        assert!(values[0] <= -1.0 && values[0] > -1.0 - 1e-12);
        assert!(values[1] <= 0.0 && values[1] > -1e-12);
        assert!(values.into_iter().fold(f64::INFINITY, f64::min) <= -1.0);
    }
}
