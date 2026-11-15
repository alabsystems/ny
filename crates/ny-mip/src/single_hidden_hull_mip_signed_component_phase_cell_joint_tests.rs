// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original-encoder separation and exact closed-cell controls for Joint.
use super::*;
use certificate::I;

struct Axis {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Axis {
    fn new() -> Self {
        Self {
            weights: vec![
                vec![1., 0., 0., 1., 1., 0., 0., 1.],
                vec![2., 2., -1., -1., 0., 0., 0., 0.],
            ],
            biases: vec![vec![0.5, 0.5, 0., 0.], vec![0., 7. / 8.]],
            dims: vec![2, 4, 2],
            bounds: vec![vec![
                Bound::new(-0.5, 1.5),
                Bound::new(-0.5, 1.5),
                Bound::new(-1., 1.),
                Bound::new(-1., 1.),
            ]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        // The actual nonzero output bias is part of Source authentication.
        HullMarginProblem::new(
            self.weights[0].clone(),
            self.biases[0].clone(),
            2,
            &self.weights[1],
            &self.biases[1],
            2,
            &[(0, 1.), (1, -1.)],
            0.,
            vec![-1.; 2],
            vec![1.; 2],
        )
        .unwrap()
    }
    fn plan(&self) -> Plan {
        Plan {
            components: vec![Component {
                negative: vec![2, 3],
                positive: vec![0, 1],
                positive_units: vec![2, 2],
                g: vec![0.; 2],
                alpha: vec![0., 0., 0.5, 0., 0., 0.5, 0.5, 0.5],
            }],
            leftover: vec![],
        }
    }
    fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &[Bound::new(-1., 1.); 2],
            &self.bounds,
        )
        .unwrap();
        encoder.constrain_output_leq_row(0, 1).unwrap();
        encoder.into_parts()
    }
    fn encoding(&self) -> HullMipEncoding<'_> {
        HullMipEncoding {
            weights: &self.weights,
            biases: &self.biases,
            dimensions: &self.dims,
            hidden_bounds: &self.bounds,
            first_output: 0,
            second_output: 1,
            reverse: false,
            unsafe_row: Row(self.original().problem.num_rows() - 1),
        }
    }
}

fn omit_joint_after_payment(cuts: &HullRootMipCuts<'_>) {
    let b = &cuts.signed.as_ref().unwrap().budget;
    let mut counts = b.state.lock().unwrap();
    assert_eq!(counts.replay_attempts, 0);
    assert!(!counts.phase_cube.cell_attempted);
    let debt = (
        counts.work,
        counts.reserved,
        counts.component_reserved,
        counts.component_rows.admitted_bytes,
    );
    let cell = counts
        .phase_cube
        .admission
        .as_mut()
        .unwrap()
        .quote
        .cell
        .as_mut()
        .unwrap();
    assert!(cell.ray.is_some() && cell.lp.is_some());
    assert!(cell.joint.take().is_some());
    assert_eq!(
        debt,
        (
            counts.work,
            counts.reserved,
            counts.component_reserved,
            counts.component_rows.admitted_bytes,
        )
    );
}

fn check_row(row: &RowSpec, point: &[Q], label: &str) {
    let lhs: Q = row.coeffs.iter().map(|(j, a)| rat(*a) * &point[*j]).sum();
    assert!(!row.lb.is_finite() || lhs >= rat(row.lb), "{label}: lower");
    assert!(!row.ub.is_finite() || lhs <= rat(row.ub), "{label}: upper");
}

#[test]
fn joint_original_encoder_separates_full_old_family_and_keeps_closed_graph() {
    let f = Axis::new();
    let p = f.source();
    let plan = f.plan();
    let encoding = f.encoding();
    let initial = f.original();
    let heads = binding::head_columns(&encoding, &initial).unwrap();
    // Identical paid admission; only the old owner's new capability is removed.
    let old = lp_tests::advised(&p, &plan);
    let new = lp_tests::advised(&p, &plan);
    omit_joint_after_payment(&old);
    assert_eq!(controls::total(&old), controls::total(&new));
    let (mut before, mut after) = (f.original(), f.original());
    let old_report = old.append_to(&mut before, &encoding, end()).unwrap();
    let new_report = new.append_to(&mut after, &encoding, end()).unwrap();
    assert_eq!(before.problem.cols(), after.problem.cols());
    assert_eq!(after.problem.num_rows(), before.problem.num_rows() + 1);
    assert_eq!(
        before.problem.rows(),
        &after.problem.rows()[..before.problem.num_rows()]
    );
    let old_stats = old_report.signed.unwrap();
    let new_stats = new_report.signed.unwrap();
    for stats in [&old_stats, &new_stats] {
        let cell = &stats.component_rows.phase.cube.cell;
        assert!(cell.admitted && cell.ray.admitted && cell.lp.admitted);
        assert_eq!(cell.collected_masks, 4);
        assert_eq!(cell.ray.masks, 4);
        assert_eq!(cell.lp.calls, 1);
        assert_eq!(cell.lp.checked_masks, 4);
        assert_eq!(stats.source_replays, 1);
    }
    let joint = &new_stats.component_rows.phase.cube.cell.joint;
    assert!(joint.admitted);
    assert_eq!(joint.masks, 4);
    assert!(joint.improved_masks > 0);
    assert_eq!(joint.prepared_rows, 1);
    assert_eq!(joint.appended_rows, 1);
    assert_eq!(
        old_stats.component_rows.phase.cube.cell.joint.appended_rows,
        0
    );

    let mut fake = vec![Q::zero(); initial.num_cols];
    for i in 0..2 {
        fake[initial.input_vars[i].0] = rat(-3. / 32.);
        fake[2 + i] = rat(13. / 32.);
        fake[4 + i] = rat(-3. / 32.);
        fake[heads[i].0] = rat(13. / 32.);
        fake[heads[i].0 + 1] = rat(1.);
        fake[heads[2 + i].0] = rat(3. / 8.);
        fake[heads[2 + i].0 + 1] = rat(0.5);
    }
    fake[initial.output_vars[0].0] = rat(7. / 8.);
    fake[initial.output_vars[1].0] = rat(7. / 8.);
    // This includes the actual unsafe row. Integrality is intentionally relaxed.
    assert!(feasible(&before, &fake));
    assert!(!feasible(&after, &fake));
    let last = after.problem.rows().last().unwrap();
    let lhs: Q = last.coeffs.iter().map(|(j, a)| rat(*a) * &fake[*j]).sum();
    assert!(rat(last.lb) - lhs > rat(1. / 8.));

    // Both positive zero planes (-1/2) and negative zero planes (0), with
    // every closed-zero selector choice, are checked against the real encoder.
    for x in [-1., -0.5, -3. / 32., 0., 1.] {
        for y in [-1., -0.5, -3. / 32., 0., 1.] {
            for ties in 0..16 {
                let input = [rat(x), rat(y)];
                let mut point = vec![Q::zero(); initial.num_cols];
                for i in 0..2 {
                    point[initial.input_vars[i].0] = input[i].clone();
                }
                for j in 0..4 {
                    let z = rat(f.biases[0][j])
                        + (0..2)
                            .map(|i| rat(f.weights[0][2 * j + i]) * &input[i])
                            .sum::<Q>();
                    point[2 + j] = z.clone();
                    point[heads[j].0] = z.clone().max(Q::zero());
                    point[heads[j].0 + 1] = rat(if z.is_zero() {
                        ((ties >> j) & 1) as f64
                    } else {
                        usize::from(z > Q::zero()) as f64
                    });
                }
                // Original outputs use raw W2/b2, not downward-rounded p.c/d.
                for output in 0..2 {
                    point[initial.output_vars[output].0] = rat(f.biases[1][output])
                        + (0..4)
                            .map(|j| rat(f.weights[1][4 * output + j]) * &point[heads[j].0])
                            .sum::<Q>();
                }
                for (j, column) in after.problem.cols().iter().enumerate() {
                    assert!(!column.lb.is_finite() || point[j] >= rat(column.lb));
                    assert!(!column.ub.is_finite() || point[j] <= rat(column.ub));
                }
                for (index, row) in after.problem.rows().iter().enumerate() {
                    // A true graph point need not satisfy the unsafe predicate.
                    if index != encoding.unsafe_row.0 {
                        check_row(row, &point, &format!("row={index} x={x} y={y} ties={ties}"));
                    }
                }
            }
        }
    }
}

#[test]
fn joint_complete_256_mask_source_table_preserves_old_values_and_exact_graph() {
    let (d, h) = (8usize, 16usize);
    let mut w = vec![0.; d * h];
    for i in 0..d {
        w[i * d + i] = 1.;
        w[(d + i) * d + i] = 1.;
    }
    let mut bias = vec![0.5; d];
    bias.extend(vec![0.; d]);
    let mut output = vec![2.; d];
    output.extend(vec![-1.; d]);
    let p = source(w, bias, &output, d);
    let plan = Plan {
        components: vec![Component {
            negative: (d..h).collect(),
            positive: (0..d).collect(),
            positive_units: vec![2; d],
            g: vec![0.; d],
            alpha: (0..256)
                .flat_map(|mask| (0..d).map(move |i| if mask & (1 << i) == 0 { 0. } else { 0.5 }))
                .collect(),
        }],
        leftover: vec![],
    };
    let old = lp_tests::advised(&p, &plan);
    let new = lp_tests::advised(&p, &plan);
    omit_joint_after_payment(&old);
    let a = old
        .signed
        .as_ref()
        .unwrap()
        .replay(&old, end(), &mut || true)
        .unwrap();
    let b = new
        .signed
        .as_ref()
        .unwrap()
        .replay(&new, end(), &mut || true)
        .unwrap();
    assert_eq!(a.constant.to_bits(), b.constant.to_bits());
    assert_eq!(a.root_lower.to_bits(), b.root_lower.to_bits());
    assert_eq!(a.g, b.g);
    let (a, b) = (a.cube.unwrap(), b.cube.unwrap());
    assert_eq!(a.values, b.values);
    let (a, b) = (a.cell.unwrap(), b.cell.unwrap());
    assert_eq!(a.values, b.values);
    assert_eq!(a.ray, b.ray);
    assert!(a.joint.is_none());
    let values = b.joint.as_ref().unwrap();
    let ray = b.ray.as_ref().unwrap();
    assert_eq!(values.len(), 256);
    assert!(values[255] > ray[255] + 0.49);
    for mask in 0..256 {
        assert!(values[mask] >= ray[mask]);
        // Interior, exact cell minimizer in the ideal raw network, and the
        // all-zero common closed boundary. Every one of 256 cells is tested.
        for mode in 0..3 {
            let input: Vec<Q> = (0..d)
                .map(|i| {
                    rat(match (mode, mask & (1 << i) != 0) {
                        (0, false) => -0.75,
                        (0, true) => 0.25,
                        (1, false) => -0.5,
                        _ => 0.,
                    })
                })
                .collect();
            for i in 0..d {
                assert!(if mask & (1 << i) == 0 {
                    input[i] <= Q::zero()
                } else {
                    input[i] >= Q::zero()
                });
            }
            let exact: Q = (0..h)
                .map(|j| {
                    let z =
                        rat(p.b1[j]) + (0..d).map(|i| rat(p.w_row(j)[i]) * &input[i]).sum::<Q>();
                    rat(p.c[j]) * z.max(Q::zero())
                })
                .sum();
            assert!(rat(values[mask]) <= exact, "mask={mask} mode={mode}");
        }
    }
    let stats = new.signed.as_ref().unwrap().stats();
    assert_eq!(stats.source_replays, 1);
    assert_eq!(stats.component_rows.phase.cube.cell.joint.masks, 256);
    assert!(stats.component_rows.phase.cube.cell.joint.improved_masks > 0);
    let weights = support_weights(values, &plan.components[0].negative, Some(&[0.5; 8])).unwrap();
    let a = weights.map(|w| -2. * w);
    let lower = repair(values, &a, end(), &mut || true).unwrap();
    let (a, lower) = support::choose(
        values,
        a,
        lower,
        &[true; 8],
        Some(&[0.5; 8]),
        end(),
        &mut || true,
    )
    .unwrap();
    for (mask, &value) in values.iter().enumerate() {
        let exact = rat(2.) * rat(value)
            + (0..8)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| rat(a[i]))
                .sum::<Q>();
        assert!(rat(lower) <= exact, "support mask={mask}");
    }
}

#[test]
fn joint_duplicate_positive_slot_is_rejected_and_every_numeric_poll_is_sticky() {
    let f = Axis::new();
    let p = f.source();
    let plan = f.plan();
    let c = &plan.components[0];
    let mut q = vec![I::zero(); 2];
    let mut constant = I::zero();
    for i in 0..2 {
        let k = c.coefficient(&p, i).unwrap().scale(0.5).unwrap();
        numeric::add_head(&p, i, k, &mut q, &mut constant).unwrap();
        numeric::add_head(
            &p,
            2 + i,
            I::point(p.c[2 + i]).unwrap(),
            &mut q,
            &mut constant,
        )
        .unwrap();
    }
    let pairs = numeric::pairs(&p, c, end(), &mut || true).unwrap();
    let base = numeric::price(&p, &q, constant).unwrap();
    let cell = numeric::strengthen(&p, c, 3, &q, constant, base, &pairs, end(), &mut || true)
        .unwrap()
        .0;
    let old = ray::strengthen(&p, c, 3, &q, constant, cell, end(), &mut || true).unwrap();
    let mut polls = 0;
    let improved = joint::strengthen(&p, c, 3, &q, constant, old, &pairs, end(), &mut || {
        polls += 1;
        true
    })
    .unwrap();
    assert!(improved > old + 0.49);
    for cutoff in 1..=polls {
        let mut count = 0;
        assert!(
            joint::strengthen(&p, c, 3, &q, constant, old, &pairs, end(), &mut || {
                count += 1;
                count != cutoff
            })
            .is_none(),
            "cutoff={cutoff}"
        );
    }

    // Nonfinite nominal advice is rejected; it cannot weaken the retained row.
    let mut overflow = pairs;
    for candidate in overflow.iter_mut().flatten().flatten() {
        candidate.mass = f64::MAX;
        candidate.ratio = f64::MAX;
    }
    assert_eq!(
        joint::strengthen(&p, c, 3, &q, constant, old, &overflow, end(), &mut || true)
            .unwrap()
            .to_bits(),
        old.to_bits()
    );

    // Two identical negative heads choose the SAME positive slope. If that
    // endpoint is applied twice, alpha=1/2 becomes3/2 and the routine falsely
    // returns3/2, although the actual closed graph at x=0 has value1.
    let p = source(vec![1.; 3], vec![0.5, 0., 0.], &[2., -0.5, -0.5], 1);
    let c = Component {
        negative: vec![1, 2],
        positive: vec![0],
        positive_units: vec![2],
        g: vec![0.],
        alpha: vec![0.5; 4],
    };
    let mut q = vec![I::zero()];
    let mut constant = I::zero();
    numeric::add_head(
        &p,
        0,
        c.coefficient(&p, 0).unwrap().scale(0.5).unwrap(),
        &mut q,
        &mut constant,
    )
    .unwrap();
    for head in [1, 2] {
        numeric::add_head(
            &p,
            head,
            I::point(p.c[head]).unwrap(),
            &mut q,
            &mut constant,
        )
        .unwrap();
    }
    let duplicate = numeric::pairs(&p, &c, end(), &mut || true).unwrap();
    assert_eq!(duplicate[1][0].unwrap().slot, duplicate[1][1].unwrap().slot);
    let base = numeric::price(&p, &q, constant).unwrap();
    let old = numeric::strengthen(
        &p,
        &c,
        3,
        &q,
        constant,
        base,
        &duplicate,
        end(),
        &mut || true,
    )
    .unwrap()
    .0;
    assert!(old > 0.99);
    let retained = joint::strengthen(&p, &c, 3, &q, constant, old, &duplicate, end(), &mut || {
        true
    })
    .unwrap();
    assert_eq!(retained.to_bits(), old.to_bits());
    assert!(rat(retained) <= rat(p.c[0]) * rat(0.5));
}

#[test]
fn joint_sealed_payload_mutation_refuses_publication_without_reopening_work() {
    for fault in 0..5 {
        let f = Axis::new();
        let p = f.source();
        let cuts = lp_tests::advised(&p, &f.plan());
        let owner = cuts.signed.as_ref().unwrap();
        let parts = f.original();
        let encoding = f.encoding();
        let heads = binding::head_columns(&encoding, &parts).unwrap();
        let replay = owner.replay(&cuts, end(), &mut || true).unwrap();
        let minima = replay.component_minima.unwrap();
        let conditional = replay.conditional.unwrap();
        let mut table = replay.cube.unwrap();
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("original component rows");
        };
        let before = parts.problem.clone();
        let debt = owner.stats().work_units;
        binding::with_phase_binding_test(&cuts, &parts, &encoding, |binding| {
            phase::prepare(
                owner,
                &cuts,
                &parts,
                binding,
                &conditional,
                &minima,
                &old,
                end(),
                &mut || true,
            )
            .unwrap();
            let cell = table.cell.as_mut().unwrap();
            match fault {
                0 => {
                    cell.joint.as_mut().unwrap().pop();
                }
                1 => {
                    let value = cell.joint.as_ref().unwrap()[3].next_up();
                    cell.joint.as_mut().unwrap()[3] = value;
                }
                2 => {
                    cell.joint.as_mut().unwrap()[3] = f64::NAN;
                }
                3 => {
                    cell.joint = None;
                }
                _ => {
                    let value = cell.ray.as_ref().unwrap()[3].next_down();
                    cell.joint.as_mut().unwrap()[3] = value;
                }
            }
            assert_eq!(
                (cube::prepare(
                    owner,
                    &cuts,
                    &parts,
                    binding,
                    &table,
                    &minima,
                    &old,
                    end(),
                    &mut || true
                )
                .unwrap())
                .len(),
                0
            );
            assert_eq!(
                (cube::prepare(
                    owner,
                    &cuts,
                    &parts,
                    binding,
                    &table,
                    &minima,
                    &old,
                    end(),
                    &mut || true
                )
                .unwrap())
                .len(),
                0
            );
        })
        .unwrap();
        assert_eq!(parts.problem, before);
        assert_eq!(owner.stats().work_units, debt);
        assert_eq!(
            owner
                .stats()
                .component_rows
                .phase
                .cube
                .cell
                .joint
                .appended_rows,
            0
        );
        assert_eq!(owner.stats().component_rows.phase.cube.prepare_attempts, 1);
    }
}

#[test]
fn joint_exact_and_adjacent_admission_retains_every_older_family() {
    let f = Axis::new();
    let p = f.source();
    let plan = f.plan();
    let full = owner(&p, &plan, true);
    let (work, bytes) = controls::total(&full);
    assert!(
        full.signed
            .as_ref()
            .unwrap()
            .stats()
            .component_rows
            .phase
            .cube
            .cell
            .joint
            .admitted
    );
    for memory in [false, true] {
        for excess in [0, 1] {
            let cuts = owner_with(&p, &plan, true, |b| {
                let mut c = b.state.lock().unwrap();
                if memory {
                    c.shared_bytes += MAX_BYTES - bytes + excess;
                } else {
                    c.work += MAX_WORK - work + excess;
                }
            });
            let stats = cuts.signed.as_ref().unwrap().stats();
            let cube = &stats.component_rows.phase.cube;
            assert!(cube.admitted && cube.cell.admitted);
            assert!(cube.cell.ray.admitted && cube.cell.lp.admitted);
            assert_eq!(
                cube.cell.joint.admitted,
                excess == 0,
                "memory={memory} excess={excess}"
            );
            let (actual_work, actual_bytes) = controls::total(&cuts);
            assert!(actual_work <= MAX_WORK && actual_bytes <= MAX_BYTES);
            if excess == 0 {
                assert_eq!(
                    if memory { actual_bytes } else { actual_work },
                    if memory { MAX_BYTES } else { MAX_WORK }
                );
            }
        }
    }
}

#[test]
fn joint_quote_accounts_for_actual_metadata_and_the_v36_shaped_payload_margin() {
    use std::mem::size_of;
    let quote = joint::quote(30, 128, 8, 62).unwrap();
    assert_eq!(quote.work, 2_832_000);
    assert_eq!(quote.bytes, 8_144);
    assert_eq!(size_of::<usize>(), 8);
    assert_eq!(size_of::<RowSpec>(), 40);
    assert_eq!(size_of::<joint::SignedPhaseJointReport>(), 72);
    assert_eq!(size_of::<SignedPhaseCellReport>(), 464);
    assert_eq!(size_of::<Quote>(), 152);
    assert_eq!(size_of::<Option<Quote>>(), 152);
    assert_eq!(size_of::<Table>(), 176);
    assert_eq!(size_of::<Option<Table>>(), 176);
    assert_eq!(size_of::<Collector>(), 1200);
    assert_eq!(size_of::<Option<input_phase_collect::State>>(), 192);
    assert_eq!(size_of::<input_phase_resources::Report>(), 72);
    assert_eq!(size_of::<Option<input_phase_resources::Quote>>(), 32);
    // v36's fixed census was2264B. Joint added488B. Input/phase adds968B:
    // report/quote/table growth464B and collector/admission/transients504B.
    const V36_FIXED: usize = 2264;
    assert_eq!(FIXED_METADATA_BYTES, 3720);
    assert_eq!(FIXED_METADATA_BYTES - V36_FIXED, 488 + 968);
    let child_credit_fields = 2 * size_of::<usize>();
    assert_eq!(child_credit_fields, 16);
    // Synthetic accounting only: the observed census is not a benchmark
    // Source, a checked plan, a minted cache, or a performance measurement.
    let old_cube_terms = 16 * (30 + 32);
    let tighter_cube_terms = term_bound(30, 16, 54).unwrap();
    assert_eq!((old_cube_terms, tighter_cube_terms), (992, 844));
    let cube_saving = 64 * (old_cube_terms - tighter_cube_terms);
    assert_eq!(cube_saving, 9_472);
    let revised =
        8_388_080 - 8_192 + quote.bytes + (FIXED_METADATA_BYTES - V36_FIXED) + child_credit_fields
            - cube_saving;
    assert_eq!(revised, 8_380_032);
    assert_eq!(MAX_BYTES - revised, 8_576);
    let input = input_phase_resources::quote(30, 128, 8, 62, 16).unwrap();
    assert_eq!((input.work, input.bytes), (12_230_128, 7_584));
    assert_eq!(MAX_BYTES - revised - input.bytes, 992);
    assert!(joint::quote(30, 128, 1, 62).is_none());
    assert!(joint::quote(33, 128, 8, 65).is_none());
}

#[test]
fn joint_half_allocated_mass_uses_exact_source_weight_despite_bad_nominal_advice() {
    // These local component planes have half of each positive head. The exact
    // gain checks that replacing alpha uses the allocated Source coefficient.
    let p = source(
        vec![1., 0., 0., 1., 1., 0., 0., 1.],
        vec![0.5, 0.5, 0., 0.],
        &[2., 2., -0.5, -0.5],
        2,
    );
    let c = Component {
        negative: vec![2, 3],
        positive: vec![0, 1],
        positive_units: vec![1, 1],
        g: vec![0.; 2],
        alpha: vec![0.5; 8],
    };
    let mut q = vec![I::zero(); 2];
    let mut constant = I::zero();
    for slot in 0..2 {
        numeric::add_head(
            &p,
            slot,
            c.coefficient(&p, slot).unwrap().scale(0.5).unwrap(),
            &mut q,
            &mut constant,
        )
        .unwrap();
        numeric::add_head(
            &p,
            slot + 2,
            I::point(p.c[slot + 2]).unwrap(),
            &mut q,
            &mut constant,
        )
        .unwrap();
    }
    let pairs = numeric::pairs(&p, &c, end(), &mut || true).unwrap();
    let old = numeric::price(&p, &q, constant).unwrap();
    for scale in [1., 0., 16.] {
        let mut advice = pairs;
        for pair in advice.iter_mut().flatten().flatten() {
            pair.mass *= scale;
        }
        let lower =
            joint::strengthen(&p, &c, 3, &q, constant, old, &advice, end(), &mut || true).unwrap();
        assert!(lower >= old);
        if scale == 1. {
            assert!(lower > 0.99);
        }
        for x in [0., 0.25, 1.] {
            for y in [0., 0.5, 1.] {
                let exact = rat(p.c[0]) * rat(0.5) * (rat(x) + rat(0.5))
                    + rat(p.c[1]) * rat(0.5) * (rat(y) + rat(0.5))
                    + rat(p.c[2]) * rat(x)
                    + rat(p.c[3]) * rat(y);
                assert!(rat(lower) <= exact);
            }
        }
    }
}
