// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Exact authored family controls; no default-discovery or target claim.
use super::super::tests::{checked, cuts, end, graph, rat};
use super::*;
use num_rational::BigRational as Q;
use num_traits::Zero;

pub(super) fn fixture(bias: f64, dense: bool) -> HullMarginProblem {
    let d = if dense { 30 } else { 1 };
    let mut w = vec![1.; 17 * d];
    if dense {
        for j in 0..17 {
            for i in 0..d {
                w[j * d + i] = if i < 2 { 1. / 16. } else { 1. / 32. };
            }
            w[j * d + 29] += if j == 0 { 1. / 512. } else { j as f64 / 16384. };
        }
    }
    let objective: Vec<_> = (0..17).map(|j| if j == 0 { 16. } else { -1. }).collect();
    let mut p = HullMarginProblem::new(
        w,
        vec![0.; 17],
        d,
        &objective,
        &[bias],
        1,
        &[(0, 1.)],
        0.,
        vec![-1.; d],
        vec![1.; d],
    )
    .unwrap();
    // Exact authored Source problem, separate from an IEEE encoder envelope.
    p.c = objective;
    p.d = bias;
    p
}
pub(super) fn shared() -> Plan {
    Plan {
        components: (0..2)
            .map(|i| Component {
                negative: (1 + 8 * i..9 + 8 * i).collect(),
                positive: vec![0],
                positive_units: vec![1],
                g: vec![0.],
                alpha: (0u32..256).map(|m| m.count_ones() as f64 / 8.).collect(),
            })
            .collect(),
        leftover: vec![],
    }
}
pub(super) fn whole() -> Plan {
    let mut p = shared();
    p.components[0].positive_units[0] = 2;
    p.components[0].g[0] = 4.;
    p.components[0].alpha = (0u32..256)
        .map(|m| (m.count_ones() as f64 + 4.) / 16.)
        .collect();
    p.components[1].positive.clear();
    p.components[1].positive_units.clear();
    p.components[1].alpha.clear();
    p.components[1].g[0] = -4.;
    p
}
fn exact_mask(p: &HullMarginProblem, c: &Component, mask: usize) -> Q {
    let mut constant = Q::zero();
    let mut coeff: Vec<_> = c.g.iter().map(|v| -rat(*v)).collect();
    for (slot, (&j, &u)) in c.positive.iter().zip(&c.positive_units).enumerate() {
        let k = rat(p.c[j]) * rat(u as f64 / 2.) * rat(c.alpha[mask * c.positive.len() + slot]);
        constant += k.clone() * rat(p.b1[j]);
        for i in 0..p.input_dim {
            coeff[i] += k.clone() * rat(p.w_row(j)[i]);
        }
    }
    for (bit, &j) in c.negative.iter().enumerate() {
        if mask & (1 << bit) != 0 {
            constant += rat(p.c[j]) * rat(p.b1[j]);
            for i in 0..p.input_dim {
                coeff[i] += rat(p.c[j]) * rat(p.w_row(j)[i]);
            }
        }
    }
    constant
        + coeff
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v.clone() * rat(p.lower[i])).min(v * rat(p.upper[i])))
            .sum::<Q>()
}

#[test]
fn signed_mass_exact_512_masks_strictly_separate_the_whole_assignment_family() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let plan = shared();
    let mut masks = 0;
    for component in &plan.components {
        for mask in 0..256 {
            assert_eq!(exact_mask(&p, component, mask), Q::zero());
            masks += 1;
        }
    }
    assert_eq!(masks, 512);
    // At x=0, any affine minorant of the positive-owning component is <=0.
    // The other 16-k negative heads have value -n/2 after averaging +/-1.
    // Sum-G cannot raise a box minimum above the affine value at x=0.
    let universal_old_upper = (0..=8)
        .map(|k| rat(0.25) - rat((16 - k) as f64 / 2.))
        .max()
        .unwrap();
    assert_eq!(universal_old_upper, rat(-3.75));
    let old = checked(&c, &whole());
    let new = checked(&c, &plan);
    assert!(old.root_lower <= -3.75 && old.root_lower > -3.750001);
    assert!(new.root_lower > 0.249 && new.root_lower <= 0.25);
}
#[test]
fn signed_mass_dense_nonproportional_dyadics_price_all_512_masks_and_last_input() {
    let p = fixture(0.25, true);
    let c = cuts(&p);
    let mut plan = shared();
    for component in &mut plan.components {
        component.g = vec![0.; 30];
    }
    for j in 0..17 {
        for k in j + 1..17 {
            assert_ne!(
                rat(p.w_row(j)[0]) * rat(p.w_row(k)[29]),
                rat(p.w_row(k)[0]) * rat(p.w_row(j)[29])
            );
        }
    }
    let mut negative_norms: Vec<_> = (1..17)
        .map(|j| p.w_row(j).iter().map(|w| rat(*w)).sum::<Q>())
        .collect();
    negative_norms.sort();
    let outside = negative_norms[..8].iter().cloned().sum::<Q>();
    let old_upper = rat(0.25) - outside / rat(2.);
    assert!(old_upper <= rat(-479. / 128.));
    let mut sum = Q::zero();
    for component in &plan.components {
        let value = (0..256)
            .map(|mask| exact_mask(&p, component, mask))
            .min()
            .unwrap();
        assert!(value >= -rat(16. / 512.));
        sum += value;
    }
    assert!(rat(p.d) + sum >= rat(3. / 16.));
    let row = checked(&c, &plan);
    assert!(row.root_lower > 0.187 && row.root_lower <= 0.25);
    assert!(exact_mask(&p, &plan.components[0], 255) < Q::zero());
}
#[test]
fn signed_mass_safe_unsafe_and_tie_keep_original_zero_witness() {
    for bias in [0.25, -0.25, 0.] {
        let p = fixture(bias, false);
        let row = checked(&cuts(&p), &shared());
        assert_eq!(graph(&p, &[0.]), rat(bias));
        assert!(rat(row.root_lower) <= graph(&p, &[0.]));
        if bias > 0. {
            assert!(row.root_lower > 0.249);
        } else {
            assert!(row.root_lower <= bias);
        }
    }
}
#[test]
fn signed_mass_conservation_accepts_only_whole_two_distinct_halves_or_leftover() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    assert!(shared().valid(&c).is_some() && whole().valid(&c).is_some());
    let mut leftover = shared();
    for component in &mut leftover.components {
        component.positive.clear();
        component.positive_units.clear();
        component.alpha.clear();
    }
    leftover.leftover.push(0);
    assert!(leftover.valid(&c).is_some());
    for case in 0..9 {
        let mut plan = shared();
        match case {
            0 => plan.components[0].positive_units.clear(),
            1 => plan.components[0].positive_units[0] = 0,
            2 => plan.components[0].positive_units[0] = 3,
            3 => plan.components[0].positive_units[0] = 2,
            4 => plan.leftover.push(0),
            5 => {
                plan.components[1].positive.clear();
                plan.components[1].positive_units.clear();
                plan.components[1].alpha.clear();
            }
            6 => {
                plan.components[0].positive = vec![0, 0];
                plan.components[0].positive_units = vec![1, 1];
                plan.components[0].alpha = vec![0.5; 512];
            }
            7 => plan.components[0].positive[0] = 1,
            _ => plan.components[0].negative[0] = 0,
        }
        assert!(plan.valid(&c).is_none(), "case {case}");
    }
}
#[test]
fn signed_mass_third_half_is_refused_even_when_negative_partition_is_complete() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let plan = Plan {
        components: [(1, 7), (7, 13), (13, 17)]
            .into_iter()
            .map(|(l, u)| Component {
                negative: (l..u).collect(),
                positive: vec![0],
                positive_units: vec![1],
                g: vec![0.],
                alpha: vec![0.5; 1 << (u - l)],
            })
            .collect(),
        leftover: vec![],
    };
    assert_eq!(
        plan.components
            .iter()
            .map(|c| c.negative.len())
            .sum::<usize>(),
        16
    );
    assert!(plan.valid(&c).is_none());
}
#[test]
fn signed_mass_half_subnormal_uses_interval_enclosure_instead_of_a_zero_point() {
    require_hull_environment().unwrap();
    let mut p = fixture(0., false);
    p.c[0] = f64::from_bits(1);
    let plan = shared();
    let component = &plan.components[0];
    let exact = rat(p.c[0]) / rat(2.);
    assert_eq!(p.c[0] * 0.5, 0.);
    let interval = component.coefficient(&p, 0).unwrap();
    assert!(rat(interval.lo) <= exact && exact <= rat(interval.hi));
    assert!(interval.hi > 0.);
    let mut whole = component.clone();
    whole.positive_units[0] = 2;
    let interval = whole.coefficient(&p, 0).unwrap();
    assert_eq!(interval.lo.to_bits(), p.c[0].to_bits());
    assert_eq!(interval.hi.to_bits(), p.c[0].to_bits());
}
#[test]
fn signed_mass_omitted_half_scaling_would_falsely_close_an_original_unsafe_graph() {
    let mut p = fixture(0.25, false);
    p.c[0] = 8.;
    let row = checked(&cuts(&p), &shared());
    assert_eq!(graph(&p, &[1.]), rat(-7.75));
    assert!(rat(row.root_lower) <= graph(&p, &[1.]));
    assert!(row.root_lower < -7.749);
}
#[test]
fn signed_mass_nonzero_bias_and_asymmetric_box_keep_d_and_all_affine_terms_once() {
    let mut p = fixture(0.25, false);
    p.b1.fill(0.125);
    p.lower[0] = -0.5;
    p.upper[0] = 1.25;
    let c = cuts(&p);
    let plan = shared();
    let row = checked(&c, &plan);
    for component in &plan.components {
        for mask in 0..256 {
            assert_eq!(exact_mask(&p, component, mask), Q::zero());
        }
    }
    assert!(row.root_lower > 0.249 && row.root_lower <= 0.25);
    for x in [-0.5, -0.125, 0., 1.25] {
        assert_eq!(graph(&p, &[x]), rat(0.25));
    }
}
#[test]
fn signed_mass_empty_and_last_mask_changes_are_both_visible_to_the_complete_checker() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    for index in [0, 255] {
        let mut plan = shared();
        plan.components[1].alpha[index] = if index == 0 { 1. } else { 0. };
        let row = checked(&c, &plan);
        assert!(row.root_lower < -7.749);
    }
}
#[test]
fn signed_mass_generic_top_two_affinity_conserves_mass_on_a_supplied_partition() {
    for reverse in [false, true] {
        let p = fixture(0.25, false);
        let c = cuts(&p);
        let mut partition = whole();
        if reverse {
            partition.components.reverse();
        }
        let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
        let plan = plan(&c, &partition, &b).unwrap();
        assert!(plan.valid(&c).is_some());
        assert_eq!(
            plan.components
                .iter()
                .map(|c| c.positive.len())
                .sum::<usize>(),
            2
        );
        assert!(plan.components.iter().all(|c| c.positive_units == vec![1]));
        assert_eq!(plan.leftover, [] as [usize; 0]);
    }
}
#[test]
fn signed_mass_source_and_witness_unit_lengths_are_explicit_and_not_implicit_whole() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let mut plan = shared();
    let old = witness(&plan);
    plan.components[0].positive_units[0] = 2;
    assert_ne!(witness(&plan), old);
    assert!(plan.valid(&c).is_none());
    plan.components[0].positive_units.clear();
    assert_ne!(witness(&plan), old);
    assert!(plan.components[0].coefficient(&p, 0).is_none());
}
#[test]
fn signed_mass_pressure_quote_conservatively_counts_direction_and_both_future_seeds() {
    let p = fixture(0.25, false);
    let c = cuts(&p);
    let plan = whole();
    let b = Budget::new(&c, end(), Duration::ZERO).unwrap();
    let cost = plan.cost(&p).unwrap();
    b.reserve(cost).unwrap();
    assert!(!pressured(&c, &plan, cost, &b).unwrap());
    b.charge(b.remaining().unwrap() - cost).unwrap();
    assert!(pressured(&c, &plan, cost, &b).is_none_or(|v| v));
}
