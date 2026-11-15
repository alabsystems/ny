// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Opt-in forward substitution of source-certified affine-input ReLU planes.
//!
//! A private snapshot binds the last ReLU/input state to its rigorous root
//! affine forms and boxes. The following authored affine run is composed only
//! as a proposal, with exact composition intervals and an explicit source-box
//! error charge. If `a*v-r >= k` is source-certified, sign-aware substitution
//! gives a valid root upper form for `r <= a*v-k`.
//!
//! Choosing among valid forms is numerical advice. A small score tolerance
//! permits a sparser tied plane that can cancel a later output coefficient.
//! The old scalar upper enclosure is independently retained by the caller;
//! this tolerance never repairs a coefficient, intercept, or proof threshold.

use super::forward::{combine, concretize, Form, Forms};
use super::lp_affine_hull::{certified_affine_relu_cut, chain_slopes, compose, composition_error};
use super::{add_down, BoxDomain, Enclosure, Instant, PairedChain, Stage};

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

fn portfolio(weights: &[f64], bias: f64, bounds: &[(f64, f64)]) -> Option<Vec<Vec<f64>>> {
    // Four source-only chains. Negative-first approximates the sign-flipped
    // normalized order of sparse nonnegative preceding activations.
    let mut order: Vec<_> = (0..weights.len()).collect();
    order.sort_by(|&i, &j| {
        (weights[i] >= 0.0)
            .cmp(&(weights[j] >= 0.0))
            .then(i.cmp(&j))
    });
    let mut result = Vec::new();
    let mut push = |permutation: &[usize]| -> Option<()> {
        let slopes = chain_slopes(weights, bias, bounds, permutation)?;
        if !result.iter().any(|old| old == &slopes) {
            result.push(slopes);
        }
        Some(())
    };
    push(&order)?;
    order.reverse();
    push(&order)?;
    order = (0..weights.len()).collect();
    push(&order)?;
    order.reverse();
    push(&order)?;
    Some(result)
}

fn complexity(form: &Form) -> usize {
    form.a.iter().filter(|&&a| a != 0.0).count()
}

fn prefer(candidate: &Form, current: &Form, domain: &BoxDomain) -> Option<bool> {
    let new = concretize(candidate, domain, false)?;
    let old = concretize(current, domain, false)?;
    let tolerance = 64.0 * f64::EPSILON * (1.0 + old.abs().max(new.abs()));
    Some(
        new < old - tolerance
            || (new <= old + tolerance && complexity(candidate) < complexity(current)),
    )
}

/// All arguments originate inside forward::states_impl. `start` binds the
/// snapshot to the complete preceding affine run; no proposal supplies boxes
/// or source bindings. Stable activations retain their authored identities.
pub(super) fn strengthen_tower(
    chain: &PairedChain,
    start: usize,
    stage: usize,
    predecessor: &Enclosure,
    preactivation: &Enclosure,
    previous_forms: &[Forms],
    current: &mut [Forms],
    domain: &BoxDomain,
    g: bool,
    deadline: Option<Instant>,
) -> Option<()> {
    if !alive(deadline) || start > stage {
        return None;
    }
    let Stage::Relu { width } = chain.stages.get(stage)? else {
        return None;
    };
    let (lo, hi, pre_lo, pre_hi) = if g {
        (
            &predecessor.g_lo,
            &predecessor.g_hi,
            &preactivation.g_lo,
            &preactivation.g_hi,
        )
    } else {
        (
            &predecessor.f_lo,
            &predecessor.f_hi,
            &preactivation.f_lo,
            &preactivation.f_hi,
        )
    };
    if lo.len() != hi.len()
        || lo.len() != previous_forms.len()
        || current.len() != *width
        || pre_lo.len() != *width
        || pre_hi.len() != *width
        || previous_forms
            .iter()
            .any(|f| f.lower.a.len() != domain.lo.len() || f.upper.a.len() != domain.lo.len())
    {
        return None;
    }
    if start == stage || (0..*width).all(|i| pre_lo[i] >= 0.0 || pre_hi[i] <= 0.0) {
        return Some(());
    }
    let bounds: Vec<_> = lo.iter().copied().zip(hi.iter().copied()).collect();
    let affine = compose(chain, start, stage, previous_forms.len(), g, deadline)?;
    if affine.rows != *width {
        return None;
    }
    for i in 0..*width {
        if !alive(deadline) {
            return None;
        }
        if pre_lo[i] >= 0.0 || pre_hi[i] <= 0.0 {
            continue;
        }
        let weights = &affine.weights[i * affine.cols..(i + 1) * affine.cols];
        let error = composition_error(&affine, i, &bounds, deadline)?;
        for slopes in portfolio(weights, affine.bias[i], &bounds)? {
            if !alive(deadline) {
                return None;
            }
            let k = certified_affine_relu_cut(weights, affine.bias[i], &bounds, &slopes, deadline)?;
            let rhs = add_down(k, -error);
            let terms: Vec<_> = slopes.into_iter().zip(previous_forms).collect();
            // -rhs is an upper intercept; negation is exact. Existing combine
            // selects previous lower/upper forms by sign and charges each root
            // coefficient sum against root magnitudes with outward constants.
            let proposal = combine(&terms, -rhs, domain, false)?;
            if prefer(&proposal, &current[i].upper, domain)? {
                current[i].upper = proposal;
            }
        }
    }
    alive(deadline).then_some(())
}

#[cfg(test)]
mod tests {
    use super::super::forward;
    use super::*;
    use std::time::Duration;

    fn output_cancellation_chain() -> PairedChain {
        PairedChain {
            input_dim: 2,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 2,
                    wf: vec![1.0, 1.0, 0.0, 1.0],
                    wg: vec![0.0; 4],
                    bf: vec![-1.0, 0.0],
                    bg: vec![0.0; 2],
                },
                Stage::Relu { width: 2 },
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, -1.0],
                    wg: vec![0.0; 2],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
            ],
        }
    }

    fn unit_square() -> BoxDomain {
        BoxDomain {
            lo: vec![0.0, 0.0],
            hi: vec![1.0, 1.0],
        }
    }

    fn scalar_bound_chain() -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 1,
                    wf: vec![1.0, 1.0],
                    wg: vec![1.0, 1.0],
                    bf: vec![0.0, 0.5],
                    bg: vec![0.0, 0.5],
                },
                Stage::Relu { width: 2 },
                // Keep MatMul and Add separately authored.
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, -2.0],
                    wg: vec![1.0, -2.0],
                    bf: vec![0.0],
                    bg: vec![0.0],
                },
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![0.75],
                    bg: vec![0.75],
                },
                Stage::Relu { width: 1 },
            ],
        }
    }

    #[test]
    fn a_tied_sparse_affine_hull_form_tightens_the_whole_output() {
        let chain = output_cancellation_chain();
        let root = unit_square();
        let baseline = forward::states(&chain, &root, None, true).unwrap();
        let strong = forward::states_strong(&chain, &root, None, true).unwrap();
        let old = baseline.last().unwrap();
        let new = strong.last().unwrap();
        assert!(old.f_hi[0] > 0.49);
        assert!(new.f_hi[0] < 1e-10 && new.f_hi[0] >= 0.0);
        assert!(new.d_hi[0] < 1e-10);
        // rho(x1+x2-1) <= x2 throughout the whole root domain.
        for i in 0..=16 {
            for j in 0..=16 {
                let x = [i as f64 / 16.0, j as f64 / 16.0];
                let (f, g) = chain.forward(&x).unwrap();
                assert!(new.f_lo[0] <= f[0] && f[0] <= new.f_hi[0]);
                assert!(new.d_lo[0] <= f[0] - g[0] && f[0] - g[0] <= new.d_hi[0]);
            }
        }
    }

    #[test]
    fn affine_input_upper_forms_strictly_tighten_a_later_relu_box() {
        let chain = scalar_bound_chain();
        let root = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let old = forward::states(&chain, &root, None, true).unwrap();
        let strong = forward::states_strong(&chain, &root, None, true).unwrap();
        assert!(old.last().unwrap().f_hi[0] > 1.7);
        assert!(strong.last().unwrap().f_hi[0] < 1.01);
        for i in -32..=32 {
            let (f, g) = chain.forward(&[i as f64 / 32.0]).unwrap();
            let last = strong.last().unwrap();
            assert!(last.f_lo[0] <= f[0] && f[0] <= last.f_hi[0]);
            assert!(last.g_lo[0] <= g[0] && g[0] <= last.g_hi[0]);
        }
    }

    #[test]
    fn source_mutation_rebuilds_planes_and_preserves_violating_values() {
        let mut chain = output_cancellation_chain();
        let root = unit_square();
        let before = forward::states_strong(&chain, &root, None, true).unwrap();
        let Stage::Affine { bf, .. } = &mut chain.stages[0] else {
            unreachable!()
        };
        bf[0] = 0.0;
        let after = forward::states_strong(&chain, &root, None, true).unwrap();
        let (f, _) = chain.forward(&[1.0, 1.0]).unwrap();
        assert_eq!(f[0], 1.0);
        assert!(before.last().unwrap().f_hi[0] < 1e-10);
        assert!(after.last().unwrap().f_hi[0] >= f[0]);
        let Stage::Affine { wf, .. } = &mut chain.stages[0] else {
            unreachable!()
        };
        wf[0] = f64::NAN;
        assert!(forward::states_strong(&chain, &root, None, true).is_none());
    }

    #[test]
    fn reference_meets_are_private_same_source_and_keep_subbox_traces() {
        let chain = output_cancellation_chain();
        let root = unit_square();
        let reference = forward::states_strong(&chain, &root, None, true).unwrap();
        let child = BoxDomain {
            lo: vec![0.25, 0.25],
            hi: vec![0.75, 0.75],
        };
        let strong =
            forward::states_strong_with_reference(&chain, &child, None, true, Some(&reference))
                .unwrap();
        let last = strong.last().unwrap();
        for i in 4..=12 {
            for j in 4..=12 {
                let (f, g) = chain.forward(&[i as f64 / 16.0, j as f64 / 16.0]).unwrap();
                assert!(last.f_lo[0] <= f[0] && f[0] <= last.f_hi[0]);
                assert!(last.d_lo[0] <= f[0] - g[0] && f[0] - g[0] <= last.d_hi[0]);
            }
        }
        assert!(forward::states_strong_with_reference(
            &chain,
            &child,
            None,
            true,
            Some(&reference[..2])
        )
        .is_none());
    }

    #[test]
    fn opt_in_does_not_change_the_old_forward_entrypoint() {
        let chain = output_cancellation_chain();
        let root = unit_square();
        let before = forward::states(&chain, &root, None, true).unwrap();
        let _ = forward::states_strong(&chain, &root, None, true).unwrap();
        let after = forward::states_with_reference(&chain, &root, None, true, None).unwrap();
        for (before, after) in before.iter().zip(&after) {
            for (a, b) in [
                (&before.f_lo, &after.f_lo),
                (&before.f_hi, &after.f_hi),
                (&before.g_lo, &after.g_lo),
                (&before.g_hi, &after.g_hi),
                (&before.d_lo, &after.d_lo),
                (&before.d_hi, &after.d_hi),
            ] {
                assert_eq!(
                    a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        assert!(after.last().unwrap().f_hi[0] > 0.49);
    }

    #[test]
    fn expiry_and_malformed_sources_never_return_strong_bounds() {
        let mut chain = output_cancellation_chain();
        let root = unit_square();
        let past = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        assert!(forward::states_strong(&chain, &root, Some(past), true).is_none());
        let Stage::Affine { wf, .. } = &mut chain.stages[0] else {
            unreachable!()
        };
        wf.pop();
        assert!(forward::states_strong(&chain, &root, None, true).is_none());
    }
}
