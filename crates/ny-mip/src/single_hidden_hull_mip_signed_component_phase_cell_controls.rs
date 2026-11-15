// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Resource, authenticated publication and cancellation boundary controls.
use super::*;

struct Fixture {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    dims: Vec<usize>,
    bounds: Vec<Vec<Bound>>,
}
impl Fixture {
    fn mixed() -> Self {
        Self {
            weights: vec![vec![1., 1. / 16., 1., -1. / 16.], vec![-1., -1., 0., 0.]],
            biases: vec![vec![0.; 2], vec![0.; 2]],
            dims: vec![2, 2, 2],
            bounds: vec![vec![Bound::new(-17. / 16., 17. / 16.); 2]],
        }
    }
    fn source(&self) -> HullMarginProblem {
        source(
            self.weights[0].clone(),
            self.biases[0].clone(),
            &self.weights[1][..self.dims[1]],
            self.dims[0],
        )
    }
    fn original(&self) -> MipParts {
        let mut encoder = crate::encode_feedforward(
            &self.weights,
            &self.biases,
            &self.dims,
            &vec![Bound::new(-1., 1.); self.dims[0]],
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
pub(super) fn total(cuts: &HullRootMipCuts<'_>) -> (usize, usize) {
    let b = &cuts.signed.as_ref().unwrap().budget;
    let c = b.state.lock().unwrap();
    (
        c.work + c.reserved + c.component_reserved,
        b.admitted_bytes
            + c.shared_bytes
            + c.direction.cache_bytes
            + c.child.cache_bytes
            + c.component_rows.admitted_bytes,
    )
}
#[test]
fn cell_exact_and_adjacent_work_and_bytes_preserve_base_cube_on_refusal() {
    let (p, plan) = mixed();
    let full = owner(&p, &plan, true);
    let base = owner(&p, &plan, false);
    let (mut full_work, mut full_bytes) = total(&full);
    // Exercise the historical Cell boundary with the new optional ray refused.
    // All metadata stays charged; subtract only its independently quoted hold.
    let quote = full
        .signed
        .as_ref()
        .unwrap()
        .budget
        .state
        .lock()
        .unwrap()
        .phase_cube
        .admission
        .as_ref()
        .unwrap()
        .quote
        .cell
        .unwrap();
    let joint = quote.joint.unwrap();
    full_work -= joint.work;
    full_bytes -= joint.bytes;
    let ray = quote.ray.unwrap();
    full_work -= ray.work;
    full_bytes -= ray.bytes;
    let lp = quote.lp.unwrap();
    full_work -= lp.work;
    full_bytes -= lp.bytes;
    let (base_work, _) = total(&base);
    for memory in [false, true] {
        for excess in [0, 1] {
            let cuts = owner_with(&p, &plan, true, |b| {
                let mut c = b.state.lock().unwrap();
                if memory {
                    c.shared_bytes += MAX_BYTES - full_bytes + excess;
                } else {
                    c.work += MAX_WORK - full_work + excess;
                }
            });
            let r = cuts
                .signed
                .as_ref()
                .unwrap()
                .stats()
                .component_rows
                .phase
                .cube;
            assert!(r.admitted);
            assert_eq!(r.cell.admitted, excess == 0);
            let (work, bytes) = total(&cuts);
            assert!(work <= MAX_WORK && bytes <= MAX_BYTES);
        }
    }
    // Even Cell's setup cannot displace an exactly affordable old Cube.
    let exact_old = owner_with(&p, &plan, true, |b| {
        b.state.lock().unwrap().work += MAX_WORK - base_work;
    });
    let r = exact_old
        .signed
        .as_ref()
        .unwrap()
        .stats()
        .component_rows
        .phase
        .cube;
    assert!(r.admitted && r.cell.attempted && !r.cell.admitted);
    assert_eq!(r.cell.work_units, 0);
    assert_eq!(total(&exact_old).0, MAX_WORK);
}

#[test]
fn cell_every_publication_poll_can_cancel_without_mutex_reentry_or_partial_ir() {
    let f = Fixture::mixed();
    let p = f.source();
    let (_, plan) = mixed();
    let mut count = 0;
    let complete = lp_tests::advised(&p, &plan);
    complete
        .append_with_poll(&mut f.original(), &f.encoding(), end(), &mut || {
            assert!(complete
                .signed
                .as_ref()
                .unwrap()
                .budget
                .state
                .try_lock()
                .is_ok());
            count += 1;
            true
        })
        .unwrap();
    assert!(count > 16);
    assert_eq!(
        complete
            .signed
            .as_ref()
            .unwrap()
            .stats()
            .component_rows
            .phase
            .cube
            .cell
            .lp
            .calls,
        1
    );
    for cutoff in 1..=count {
        let cuts = lp_tests::advised(&p, &plan);
        let owner = cuts.signed.as_ref().unwrap();
        let mut parts = f.original();
        let before = parts.problem.clone();
        let mut polls = 0;
        let result = cuts.append_with_poll(&mut parts, &f.encoding(), end(), &mut || {
            assert!(owner.budget.state.try_lock().is_ok());
            polls += 1;
            polls != cutoff // one false stays terminal even if later calls return true
        });
        assert!(result.is_none(), "cutoff={cutoff}/{count}");
        assert_eq!(parts.problem, before);
        let r = owner.stats();
        assert_eq!(r.component_rows.phase.cube.cell.appended_rows, 0);
        assert_eq!(r.component_rows.phase.cube.cell.prepared_rows, 0);
        assert_eq!(r.component_rows.phase.cube.cell.ray.prepared_rows, 0);
        assert_eq!(r.component_rows.phase.cube.cell.ray.appended_rows, 0);
        assert_eq!(r.component_rows.phase.cube.cell.lp.prepared_rows, 0);
        assert_eq!(r.component_rows.phase.cube.cell.lp.appended_rows, 0);
        assert!(r.work_units <= MAX_WORK);
        let spent = r.component_rows.phase.cube.cell.work_units;
        let attempts = owner.budget.state.lock().unwrap().phase_cube.cell_attempted;
        if attempts {
            let _ = cuts.append_to(&mut parts, &f.encoding(), end());
            let after = owner.stats();
            assert_eq!(after.component_rows.phase.cube.cell.work_units, spent);
            assert_eq!(after.component_rows.phase.cube.cell.appended_rows, 0);
            assert_eq!(after.component_rows.phase.cube.collection_attempts, 1);
            assert_eq!(
                after.component_rows.phase.cube.cell.lp.calls,
                r.component_rows.phase.cube.cell.lp.calls
            );
        }
    }
}

#[test]
fn table_lp_original_publication_keeps_prefix_and_omits_duplicate_support() {
    let f = Fixture::mixed();
    let p = f.source();
    let (_, plan) = mixed();
    let old = lp_tests::advised(&p, &plan);
    old.signed
        .as_ref()
        .unwrap()
        .budget
        .state
        .lock()
        .unwrap()
        .phase_cube
        .admission
        .as_mut()
        .unwrap()
        .quote
        .cell
        .as_mut()
        .unwrap()
        .lp = None;
    let new = lp_tests::advised(&p, &plan);
    let mut old_parts = f.original();
    let mut new_parts = f.original();
    old.append_to(&mut old_parts, &f.encoding(), end()).unwrap();
    let result = new.append_to(&mut new_parts, &f.encoding(), end()).unwrap();
    assert_eq!(new_parts.problem, old_parts.problem);
    let stats = result.signed.unwrap();
    let lp = &stats.component_rows.phase.cube.cell.lp;
    assert!(lp.admitted);
    assert_eq!(lp.calls, 1);
    assert!(lp.activity_from_root);
    assert_eq!(lp.appended_rows, 0);
    assert_eq!(lp.checked_masks, 4);
    assert!(lp.gap_upper.unwrap() < 1e-10);
    assert_eq!(stats.source_replays, 1);
}

#[test]
fn cell_collected_payload_mutation_refuses_authenticated_prepare_once() {
    for fault in 0..10 {
        let f = Fixture::mixed();
        let p = f.source();
        let (_, plan) = mixed();
        let cuts = lp_tests::advised(&p, &plan);
        let owner = cuts.signed.as_ref().unwrap();
        let parts = f.original();
        let heads = binding::head_columns(&f.encoding(), &parts).unwrap();
        let replay = owner.replay(&cuts, end(), &mut || true).unwrap();
        let minima = replay.component_minima.unwrap();
        let conditional = replay.conditional.unwrap();
        let mut table = replay.cube.unwrap();
        let component_rows::Prepared::Rows(old) =
            component_rows::prepare(owner, &cuts, &parts, &heads, &minima, end(), &mut || true)
                .unwrap()
        else {
            panic!("original rows");
        };
        let before = parts.problem.clone();
        let debt = owner.stats().work_units;
        binding::with_phase_binding_test(&cuts, &parts, &f.encoding(), |binding| {
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
                    cell.values.pop();
                }
                1 => cell.values[1] = f64::NAN,
                2 => cell.values[1] = cell.values[1].next_up(),
                3 => cell.component += 1,
                4 => cell.activity.as_mut().unwrap()[0] = 0.25,
                5 => cell.values[1] = table.values[0][1].next_down(),
                6 => {
                    cell.ray.as_mut().unwrap().pop();
                }
                7 => {
                    let v = &mut cell.ray.as_mut().unwrap()[1];
                    *v = v.next_up();
                }
                8 => cell.ray.as_mut().unwrap()[1] = f64::NAN,
                _ => cell.ray = None,
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
            let mut polls = 0;
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
                    &mut || {
                        polls += 1;
                        true
                    }
                )
                .unwrap())
                .len(),
                0
            );
            assert_eq!(polls, 1);
        })
        .unwrap();
        assert_eq!(parts.problem, before);
        let after = owner.stats();
        assert_eq!(after.work_units, debt);
        assert_eq!(after.component_rows.phase.cube.prepare_attempts, 1);
        assert_eq!(after.component_rows.phase.cube.cell.prepared_rows, 0);
        assert_eq!(after.component_rows.phase.cube.cell.lp.calls, 0);
    }
}

#[test]
fn cell_full_eight_bit_table_checks_last_mask_and_keeps_exact_closed_graph() {
    let p = source(
        (0..8).flat_map(|j| [1., (j as f64 - 3.) / 32.]).collect(),
        vec![0.; 8],
        &[-1.; 8],
        2,
    );
    let plan = Plan {
        components: vec![Component {
            negative: (0..8).collect(),
            positive: vec![],
            positive_units: vec![],
            g: vec![0.; 2],
            alpha: vec![],
        }],
        leftover: vec![],
    };
    let cuts = owner(&p, &plan, true);
    let replay = cuts
        .signed
        .as_ref()
        .unwrap()
        .replay(&cuts, end(), &mut || true)
        .unwrap();
    let cube = replay.cube.unwrap();
    let strong = cube.cell.unwrap();
    let ray = strong.ray.as_ref().unwrap();
    assert_eq!(ray.len(), 256);
    assert!(ray.iter().zip(&strong.values).all(|(a, b)| a >= b));
    assert_eq!(strong.values.len(), 256);
    assert!(strong
        .values
        .iter()
        .zip(&cube.values[0])
        .all(|(new, old)| new >= old));
    for x in [-1., 0., 1.] {
        for t in [-1., 0., 1.] {
            for mask in 0..256 {
                let z: Vec<_> = (0..8)
                    .map(|j| rat(x) + rat((j as f64 - 3.) / 32.) * rat(t))
                    .collect();
                if z.iter().enumerate().any(|(j, z)| {
                    (mask & (1 << j) == 0 && *z > Q::zero())
                        || (mask & (1 << j) != 0 && *z < Q::zero())
                }) {
                    continue;
                }
                let exact: Q = z.into_iter().map(|z| -z.max(Q::zero())).sum();
                assert!(exact >= rat(strong.values[mask]));
                assert!(exact >= rat(ray[mask]));
            }
        }
    }
    let stopped = owner(&p, &plan, true);
    let s = stopped.signed.as_ref().unwrap();
    let initial = s.stats().checked_masks;
    assert!(s
        .replay(&stopped, end(), &mut || s.stats().checked_masks
            < initial + 255)
        .is_none());
    assert_eq!(s.stats().source_replays, 0);
    assert_eq!(s.stats().component_rows.phase.cube.cell.collected_masks, 0);
}

#[test]
fn cell_numeric_overflow_keeps_base_but_false_poll_is_terminal() {
    let (p, plan) = mixed();
    let c = &plan.components[0];
    let pairs = numeric::pairs(&p, c, end(), &mut || true).unwrap();
    let coefficients = vec![certificate::I::point(f64::MAX).unwrap(); 2];
    let constant = certificate::I::point(f64::MAX).unwrap();
    assert_eq!(
        numeric::strengthen(
            &p,
            c,
            0,
            &coefficients,
            constant,
            -3.,
            &pairs,
            end(),
            &mut || true
        ),
        Some((-3., false))
    );
    let mut polls = 0;
    assert!(numeric::strengthen(
        &p,
        c,
        0,
        &coefficients,
        constant,
        -3.,
        &pairs,
        end(),
        &mut || {
            polls += 1;
            polls != 2
        }
    )
    .is_none());
    assert_eq!(polls, 2);
}

#[test]
fn cell_unary_and_projected_constant_publish_exact_selector_counts() {
    for constant in [false, true] {
        let (bias, bounds, alpha) = if constant {
            (
                vec![2.5, 2.],
                vec![Bound::new(1.5, 3.5), Bound::new(1., 3.)],
                vec![0., 0.5],
            )
        } else {
            (
                vec![0., -0.5],
                vec![Bound::new(-1., 1.), Bound::new(-1.5, 0.5)],
                vec![0., 0.5],
            )
        };
        let f = Fixture {
            weights: vec![vec![1., 1.], vec![2., -1., 0., 0.]],
            biases: vec![bias, vec![0., 0.]],
            dims: vec![1, 2, 2],
            bounds: vec![bounds],
        };
        let p = f.source();
        let plan = Plan {
            components: vec![Component {
                negative: vec![1],
                positive: vec![0],
                positive_units: vec![2],
                g: vec![0.],
                alpha,
            }],
            leftover: vec![],
        };
        let old = owner(&p, &plan, false);
        let new = owner(&p, &plan, true);
        let mut before = f.original();
        let mut after = f.original();
        old.append_to(&mut before, &f.encoding(), end()).unwrap();
        let report = new.append_to(&mut after, &f.encoding(), end()).unwrap();
        assert_eq!(after.problem.num_rows(), before.problem.num_rows() + 1);
        let r = report.signed.unwrap().component_rows.phase.cube.cell;
        assert_eq!(r.appended_rows, 1);
        assert_eq!(r.prepared_selectors, usize::from(!constant));
        let stats = new.signed.as_ref().unwrap().stats();
        assert_eq!(stats.source_replays, 1);
        let heads = binding::head_columns(&f.encoding(), &after).unwrap();
        for x in [-1., 0., 0.5, 1.] {
            for tie in [0., 1.] {
                let mut point = vec![Q::zero(); after.num_cols];
                point[0] = rat(x);
                for j in 0..2 {
                    let z = rat(x) + rat(f.biases[0][j]);
                    point[1 + j] = z.clone();
                    point[heads[j].0] = z.clone().max(Q::zero());
                    if !constant {
                        point[heads[j].0 + 1] = if z.is_zero() {
                            rat(tie)
                        } else {
                            rat(usize::from(z > Q::zero()) as f64)
                        };
                    }
                }
                point[after.output_vars[0].0] = rat(2.) * &point[heads[0].0] - &point[heads[1].0];
                for (i, row) in after.problem.rows().iter().enumerate() {
                    if i == f.encoding().unsafe_row.0 {
                        continue;
                    }
                    let lhs: Q = row.coeffs.iter().map(|(j, a)| rat(*a) * &point[*j]).sum();
                    assert!(!row.lb.is_finite() || lhs >= rat(row.lb));
                    assert!(!row.ub.is_finite() || lhs <= rat(row.ub));
                }
            }
        }
    }
}
