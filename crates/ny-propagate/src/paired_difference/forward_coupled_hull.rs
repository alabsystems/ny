// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Opt-in direct signed-difference propagation of source paired affine hulls.
//!
//! A source-replayed support of either Dpost or -Dpost has the form
//! `cf*rF+cg*rG >= af*Fprev+ag*Gprev+k`. The exact identity
//! `Fprev=Gprev+Dprev` gives `af*Dprev+(af+ag)*Gprev+k`.
//! TwoSum encloses the rounded coefficient addition against source G boxes;
//! existing sign-aware forward substitution repairs root coefficient sums.
//! The support authority covers all four closed activation phases upstream.
//! Numerical improvement scores choose only between already valid forms.

use super::forward::{combine, concretize, scalar_sum_error, Form, Forms};
use super::lp_coupled_affine::{self, Support};
use super::{add_down, add_up, magnitude, mul_up, BoxDomain, Enclosure, Instant, PairedChain};

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

fn prefer(candidate: &Form, current: &Form, domain: &BoxDomain, lower: bool) -> Option<bool> {
    let new = concretize(candidate, domain, lower)?;
    let old = concretize(current, domain, lower)?;
    let tolerance = 64.0 * f64::EPSILON * (1.0 + old.abs().max(new.abs()));
    let sparse = |form: &Form| {
        let count = form.a.iter().filter(|&&a| a != 0.0).count();
        // A tied constant adds no relation beyond the independently retained
        // scalar box. Keep a sparse affine relation for downstream cancellation.
        if count == 0 {
            usize::MAX
        } else {
            count
        }
    };
    // Both planes already have source authority. The tolerance only selects
    // a sparser tied form; the caller independently meets old scalar bounds.
    Some(if lower {
        new > old + tolerance || (new >= old - tolerance && sparse(candidate) < sparse(current))
    } else {
        new < old - tolerance || (new <= old + tolerance && sparse(candidate) < sparse(current))
    })
}

fn transported_lower(
    support: &Support,
    source: &Enclosure,
    previous_d: &[Forms],
    previous_g: &[Forms],
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<Form> {
    let n = support.af.len();
    if !alive(deadline)
        || support.ag.len() != n
        || previous_d.len() != n
        || previous_g.len() != n
        || source.g_lo.len() != n
        || source.g_hi.len() != n
        || previous_d
            .iter()
            .chain(previous_g)
            .any(|f| f.lower.a.len() != domain.lo.len() || f.upper.a.len() != domain.lo.len())
    {
        return None;
    }
    let mut terms = Vec::with_capacity(2 * n);
    let mut charge = 0.0;
    for j in 0..n {
        if !alive(deadline) {
            return None;
        }
        let (a, b) = (support.af[j], support.ag[j]);
        let common = a + b;
        let error = scalar_sum_error(a, b, common)?;
        charge = add_up(
            charge,
            mul_up(error, magnitude(source.g_lo[j], source.g_hi[j])),
        );
        terms.push((a, &previous_d[j]));
        terms.push((common, &previous_g[j]));
    }
    let intercept = add_down(support.k, -charge);
    let lower = combine(&terms, intercept, domain, true)?;
    alive(deadline).then_some(lower)
}

/// Source-owned support construction and direct propagation, called only by
/// the opt-in strong root forward path at every ReLU. Prefix states carry
/// no future values or imported proposal bounds. Original forms survive
/// unless a newly certified endpoint improves or a tied form is sparser.
pub(super) fn strengthen(
    chain: &PairedChain,
    domain: &BoxDomain,
    prefix_states: &[Enclosure],
    stage: usize,
    snapshot_start: usize,
    previous_d: &[Forms],
    previous_g: &[Forms],
    current_d: &mut [Forms],
    deadline: Option<Instant>,
) -> Option<()> {
    if !alive(deadline) {
        return None;
    }
    for support in
        lp_coupled_affine::supports_difference(chain, domain, prefix_states, stage, deadline)?
    {
        if !alive(deadline) {
            return None;
        }
        // Other objective normals are useful LP cuts, but do not represent a
        // scalar signed difference and cannot be silently treated as one.
        let sign = match (support.cf, support.cg) {
            (1.0, -1.0) => 1.0,
            (-1.0, 1.0) => -1.0,
            _ => continue,
        };
        if support.start != snapshot_start || support.neuron >= current_d.len() {
            return None;
        }
        let source = prefix_states.get(snapshot_start)?;
        let lower = transported_lower(&support, source, previous_d, previous_g, domain, deadline)?;
        let target = &mut current_d[support.neuron];
        if sign > 0.0 {
            if prefer(&lower, &target.lower, domain, true)? {
                target.lower = lower;
            }
        } else {
            // -D >= L(x) implies D <= -L(x). Exact stored coefficient/sign
            // negation needs no new arithmetic compensation.
            let upper = Form {
                a: lower.a.into_iter().map(|a| -a).collect(),
                b: -lower.b,
            };
            if prefer(&upper, &target.upper, domain, false)? {
                target.upper = upper;
            }
        }
    }
    alive(deadline).then_some(())
}

#[cfg(test)]
mod tests {
    use super::super::{forward, Stage};
    use super::*;
    use std::time::Duration;

    fn root_forms(n: usize) -> (Vec<Forms>, Vec<Forms>) {
        let zero = Form {
            a: vec![0.0; n],
            b: 0.0,
        };
        let d = vec![
            Forms {
                lower: zero.clone(),
                upper: zero
            };
            n
        ];
        let g = (0..n)
            .map(|j| {
                let mut a = vec![0.0; n];
                a[j] = 1.0;
                let form = Form { a, b: 0.0 };
                Forms {
                    lower: form.clone(),
                    upper: form,
                }
            })
            .collect();
        (d, g)
    }

    fn loose(n: usize, outputs: usize) -> Vec<Forms> {
        vec![
            Forms {
                lower: Form {
                    a: vec![0.0; n],
                    b: -3.0
                },
                upper: Form {
                    a: vec![0.0; n],
                    b: 3.0
                },
            };
            outputs
        ]
    }

    fn shifted(shift: f64) -> PairedChain {
        PairedChain {
            input_dim: 1,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 1,
                    wf: vec![1.0],
                    wg: vec![1.0],
                    bf: vec![shift],
                    bg: vec![0.0],
                },
                Stage::Relu { width: 1 },
            ],
        }
    }

    #[test]
    fn source_supports_tighten_identical_and_signed_shift_difference_forms() {
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        for shift in [0.0, 0.125, -0.125] {
            let chain = shifted(shift);
            let states = forward::states(&chain, &domain, None, true).unwrap();
            let (d, g) = root_forms(1);
            let mut current = loose(1, 1);
            strengthen(
                &chain,
                &domain,
                &states[..2],
                1,
                0,
                &d,
                &g,
                &mut current,
                None,
            )
            .unwrap();
            let lower = concretize(&current[0].lower, &domain, true).unwrap();
            let upper = concretize(&current[0].upper, &domain, false).unwrap();
            assert!(lower > shift.min(0.0) - 0.01 && upper < shift.max(0.0) + 0.01);
            for i in -32..=32 {
                let (f, g) = chain.forward(&[i as f64 / 32.0]).unwrap();
                assert!(lower <= f[0] - g[0] && f[0] - g[0] <= upper);
            }
        }
    }

    #[test]
    fn all_four_closed_activation_phases_remain_enclosed() {
        let chain = PairedChain {
            input_dim: 2,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, 0.0],
                    wg: vec![0.0, 1.0],
                    bf: vec![0.125],
                    bg: vec![-0.125],
                },
                Stage::Relu { width: 1 },
            ],
        };
        let domain = BoxDomain {
            lo: vec![-1.0; 2],
            hi: vec![1.0; 2],
        };
        let states = forward::states(&chain, &domain, None, true).unwrap();
        let (d, g) = root_forms(2);
        let mut current = loose(2, 1);
        strengthen(
            &chain,
            &domain,
            &states[..2],
            1,
            0,
            &d,
            &g,
            &mut current,
            None,
        )
        .unwrap();
        let mut seen = [false; 4];
        for i in -16..=16 {
            for j in -16..=16 {
                let input = [i as f64 / 16.0, j as f64 / 16.0];
                let (f, g) = chain.forward(&input).unwrap();
                let lower = current[0]
                    .lower
                    .a
                    .iter()
                    .zip(input)
                    .map(|(&a, x)| a * x)
                    .sum::<f64>()
                    + current[0].lower.b;
                let upper = current[0]
                    .upper
                    .a
                    .iter()
                    .zip(input)
                    .map(|(&a, x)| a * x)
                    .sum::<f64>()
                    + current[0].upper.b;
                assert!(lower <= f[0] - g[0] && f[0] - g[0] <= upper);
                let active_f = input[0] + 0.125 >= 0.0;
                let active_g = input[1] - 0.125 >= 0.0;
                seen[usize::from(active_f) + 2 * usize::from(active_g)] = true;
            }
        }
        assert!(seen.into_iter().all(|phase| phase));
    }

    #[test]
    fn changed_source_shift_preserves_a_real_band_violation() {
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let (d, g) = root_forms(1);
        let mut chain = shifted(0.125);
        let old = forward::states(&chain, &domain, None, true).unwrap();
        let mut current = loose(1, 1);
        strengthen(&chain, &domain, &old[..2], 1, 0, &d, &g, &mut current, None).unwrap();
        assert!(concretize(&current[0].upper, &domain, false).unwrap() < 0.14);
        let Stage::Affine { bf, .. } = &mut chain.stages[0] else {
            unreachable!()
        };
        bf[0] = 0.5;
        let changed = forward::states(&chain, &domain, None, true).unwrap();
        let mut current = loose(1, 1);
        strengthen(
            &chain,
            &domain,
            &changed[..2],
            1,
            0,
            &d,
            &g,
            &mut current,
            None,
        )
        .unwrap();
        let (f, g) = chain.forward(&[1.0]).unwrap();
        assert_eq!(f[0] - g[0], 0.5);
        assert!(concretize(&current[0].upper, &domain, false).unwrap() >= 0.5);
        assert!(concretize(&current[0].upper, &domain, false).unwrap() >= 0.25);
    }

    #[test]
    fn common_coefficient_cancellation_retains_two_sum_error() {
        // (2^53+1) rounds to2^53; the omitted unit is paid against sourceG.
        let support = Support {
            start: 0,
            neuron: 0,
            cf: 1.0,
            cg: -1.0,
            af: vec![9_007_199_254_740_992.0],
            ag: vec![1.0],
            k: 0.0,
        };
        let source = Enclosure {
            f_lo: vec![-1.0],
            f_hi: vec![1.0],
            g_lo: vec![-1.0],
            g_hi: vec![1.0],
            d_lo: vec![0.0],
            d_hi: vec![0.0],
            hulls: None,
        };
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let (d, g) = root_forms(1);
        let form = transported_lower(&support, &source, &d, &g, &domain, None).unwrap();
        assert!(form.b <= -1.0);
        assert_eq!(form.a[0], 9_007_199_254_740_992.0);
    }

    #[test]
    fn paired_forward_scope_expiry_and_snapshot_shapes_fail_closed() {
        let chain = shifted(0.125);
        let domain = BoxDomain {
            lo: vec![-1.0],
            hi: vec![1.0],
        };
        let states = forward::states(&chain, &domain, None, true).unwrap();
        let (d, g) = root_forms(1);
        let mut current = loose(1, 1);
        let past = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        assert!(strengthen(
            &chain,
            &domain,
            &states[..2],
            1,
            0,
            &d,
            &g,
            &mut current,
            Some(past)
        )
        .is_none());
        assert!(strengthen(
            &chain,
            &domain,
            &states[..2],
            1,
            0,
            &[],
            &g,
            &mut current,
            None
        )
        .is_none());
        assert!(strengthen(
            &chain,
            &domain,
            &states[..1],
            1,
            0,
            &d,
            &g,
            &mut current,
            None
        )
        .is_none());
    }

    fn earlier_difference_cancellation_chain() -> PairedChain {
        PairedChain {
            input_dim: 2,
            output_dim: 1,
            stages: vec![
                Stage::Affine {
                    rows: 2,
                    cols: 2,
                    wf: vec![1.0, 1.0, 0.0, 2.0],
                    wg: vec![0.0, 0.0, 0.0, 1.0],
                    bf: vec![-1.0, 0.0],
                    bg: vec![0.0, 0.0],
                },
                Stage::Relu { width: 2 },
                Stage::Affine {
                    rows: 1,
                    cols: 2,
                    wf: vec![1.0, -1.0],
                    wg: vec![1.0, -1.0],
                    bf: vec![2.5],
                    bg: vec![2.0],
                },
                // Both towers are strictly positive at this later ReLU.
                // No final unstable-neuron support can explain improvement.
                Stage::Relu { width: 1 },
            ],
        }
    }

    #[test]
    fn earlier_difference_correlation_closes_a_later_stable_relu_band() {
        let chain = earlier_difference_cancellation_chain();
        let root = BoxDomain {
            lo: vec![0.0; 2],
            hi: vec![1.0; 2],
        };
        let baseline = forward::states(&chain, &root, None, true).unwrap();
        let strong = forward::states_strong(&chain, &root, None, true).unwrap();
        let old = baseline.last().unwrap();
        let new = strong.last().unwrap();
        assert!(old.d_hi[0] > 0.9);
        assert!(new.d_lo[0] > -0.6 && new.d_hi[0] < 0.6);
        assert!(strong[3].f_lo[0] > 0.49 && strong[3].g_lo[0] > 0.99);
        assert!(
            lp_coupled_affine::supports_difference(&chain, &root, &strong[..4], 3, None)
                .unwrap()
                .is_empty()
        );
        // The useful earlier plane D1<=x2 ties the scalar maximum1. The
        // original interval is independently retained before later transport.
        assert!(strong[2].d_hi[0] <= baseline[2].d_hi[0]);
        for i in 0..=16 {
            for j in 0..=16 {
                let (f, g) = chain.forward(&[i as f64 / 16.0, j as f64 / 16.0]).unwrap();
                let difference = f[0] - g[0];
                // Exact dyadic identity: D=rho(x1+x2-1)-x2+1/2.
                assert!((-0.5..=0.5).contains(&difference));
                assert!(new.d_lo[0] <= difference && difference <= new.d_hi[0]);
            }
        }
    }

    #[test]
    fn all_layer_mutation_and_expiry_preserve_a_real_violating_trace() {
        let mut chain = earlier_difference_cancellation_chain();
        let root = BoxDomain {
            lo: vec![0.0; 2],
            hi: vec![1.0; 2],
        };
        let Stage::Affine { bf, .. } = &mut chain.stages[2] else {
            unreachable!()
        };
        bf[0] = 3.0;
        let strong = forward::states_strong(&chain, &root, None, true).unwrap();
        let (f, g) = chain.forward(&[0.0, 0.0]).unwrap();
        assert_eq!(f[0] - g[0], 1.0);
        assert!(strong.last().unwrap().d_hi[0] >= 1.0);
        let past = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        assert!(forward::states_strong(&chain, &root, Some(past), true).is_none());
    }

    #[test]
    fn sparse_tie_selection_keeps_proof_endpoints_out_of_the_tolerance() {
        let root = BoxDomain {
            lo: vec![0.0; 2],
            hi: vec![1.0; 2],
        };
        let old = Form {
            a: vec![0.5, 0.5],
            b: 0.0,
        };
        let sparse = Form {
            a: vec![0.0, 1.0],
            b: 8.0 * f64::EPSILON,
        };
        assert!(
            concretize(&sparse, &root, false).unwrap() > concretize(&old, &root, false).unwrap()
        );
        assert!(prefer(&sparse, &old, &root, false).unwrap());
        let materially_worse = Form {
            a: vec![0.0, 1.0],
            b: 1e-4,
        };
        assert!(!prefer(&materially_worse, &old, &root, false).unwrap());
        let scalar = Form {
            a: vec![0.0, 0.0],
            b: 1.0,
        };
        assert!(!prefer(&scalar, &old, &root, false).unwrap());
        assert!(prefer(&old, &scalar, &root, false).unwrap());
    }
}
