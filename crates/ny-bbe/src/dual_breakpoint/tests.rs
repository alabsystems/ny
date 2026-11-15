// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Independent objective and exhaustive ray-vertex oracles. These tests check
//! search quality and request handling; they make no certification claim.

use std::time::{Duration, Instant};

use ny_core::search::{DualSearchRow, SearchBackend};

use super::*;
use crate::CpuF32Backend;

fn objective(req: &DualSearchRequest<'_>, l0: &[f64], l1: &[f64]) -> f64 {
    let norm: f64 = (0..req.rc.len())
        .map(|j| {
            let rc = req.rc[j]
                + req
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        row.a[j] * row.sign * ((l0[i] - req.l0[i]) - (l1[i] - req.l1[i]))
                    })
                    .sum::<f64>();
            rc.abs()
        })
        .sum();
    -norm
        - req
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| l0[i] * row.b0 + l1[i] * row.b1)
            .sum::<f64>()
}

fn request<'a>(
    rc: &'a [f64],
    rows: &'a [DualSearchRow<'a>],
    l0: &'a [f64],
    l1: &'a [f64],
) -> DualSearchRequest<'a> {
    DualSearchRequest {
        rc,
        rows,
        l0,
        l1,
        steps: 24,
        deadline: None,
        mem_cap_bytes: 1 << 20,
    }
}

struct Lcg(u64);

impl Lcg {
    fn sample(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
        2.0 * unit - 1.0
    }
}

#[test]
fn ray_maximum_matches_exhaustive_vertices_and_a_dense_grid() {
    let mut rng = Lcg(0x627265616b);
    for _ in 0..160 {
        let rc: Vec<f64> = (0..7).map(|_| rng.sample() * 2.0).collect();
        let coefficients: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..7).map(|_| rng.sample()).collect())
            .collect();
        let rows: Vec<_> = coefficients
            .iter()
            .map(|a| DualSearchRow {
                a,
                sign: if rng.sample() > 0.0 { 1.0 } else { -1.0 },
                b0: rng.sample(),
                b1: rng.sample(),
                vacuous: false,
            })
            .collect();
        let initial0 = [2.0; 3];
        let initial1 = [2.0; 3];
        let req = request(&rc, &rows, &initial0, &initial1);
        let baseline = objective(&req, &initial0, &initial1);
        let mut work = Work::new(&req, baseline, working_bytes(7, 3)).unwrap();
        // Negative components bound every test ray by a small finite t.
        work.d0 = (0..3).map(|_| rng.sample()).collect();
        work.d1 = (0..3).map(|_| rng.sample()).collect();
        work.d0[0] = -1.0;
        let d0 = work.d0.clone();
        let d1 = work.d1.clone();
        let mut limit = f64::INFINITY;
        for &direction in d0.iter().chain(&d1) {
            if direction < 0.0 {
                limit = limit.min(-2.0 / direction);
            }
        }
        let score = |t: f64| {
            let l0: Vec<f64> = d0.iter().map(|&d| 2.0 + t * d).collect();
            let l1: Vec<f64> = d1.iter().map(|&d| 2.0 + t * d).collect();
            objective(&req, &l0, &l1)
        };
        let mut oracle = baseline.max(score(limit));
        for j in 0..rc.len() {
            let direction: f64 = rows
                .iter()
                .enumerate()
                .map(|(i, row)| row.sign * (d0[i] - d1[i]) * row.a[j])
                .sum();
            let t = -rc[j] / direction;
            if t.is_finite() && t > 0.0 && t < limit {
                oracle = oracle.max(score(t));
            }
        }
        for k in 0..=1000 {
            oracle = oracle.max(score(limit * k as f64 / 1000.0));
        }
        work.ray(&req)
            .unwrap_or_else(|_| panic!("finite ray refused"));
        assert!(
            (work.objective - oracle).abs() < 1e-9 * (1.0 + oracle.abs()),
            "got {}, exhaustive ray maximum {oracle}",
            work.objective
        );
        assert!(work.l0.iter().chain(&work.l1).all(|&l| l >= 0.0));
    }
}

#[test]
fn coupled_rows_improve_and_match_an_independent_grid_maximum() {
    let rows = [
        DualSearchRow {
            a: &[1.0, 1.0],
            sign: 1.0,
            b0: 0.1,
            b1: 4.0,
            vacuous: true,
        },
        DualSearchRow {
            a: &[1.0, -1.0],
            sign: 1.0,
            b0: 0.1,
            b1: 4.0,
            vacuous: true,
        },
    ];
    let req = request(&[-2.0, 1.0], &rows, &[0.0, 0.0], &[0.0, 0.0]);
    let out = CpuF32Backend::new().refine_dual(&req).unwrap();
    let steps = out.steps;
    let gain = out.gain.into_hint().into_ingest();
    let hint = out.multipliers.into_hint().into_ingest();
    let actual = objective(&req, &hint.l0, &hint.l1);
    let mut grid = f64::NEG_INFINITY;
    for i in 0..=60 {
        for j in 0..=60 {
            grid = grid.max(objective(
                &req,
                &[i as f64 / 20.0, j as f64 / 20.0],
                &[0.0, 0.0],
            ));
        }
    }
    assert!(steps > 0);
    assert!(gain > 2.7);
    assert!(actual >= grid - 1e-10, "actual={actual} grid={grid}");
    assert!((gain - (actual - objective(&req, req.l0, req.l1))).abs() < 1e-10);
}

#[test]
fn coordinate_rescue_escapes_a_joint_direction_blocked_by_a_zero_cost() {
    let rows = [
        DualSearchRow {
            a: &[2.0, -1.0],
            sign: 1.0,
            b0: 0.0,
            b1: 0.0,
            vacuous: true,
        },
        DualSearchRow {
            a: &[0.0, -1.0],
            sign: 1.0,
            b0: 0.9,
            b1: 0.0,
            vacuous: true,
        },
    ];
    let req = request(&[0.0, 1.0], &rows, &[0.0, 0.0], &[0.0, 0.0]);
    let mut work = Work::new(&req, -1.0, working_bytes(2, 2)).unwrap();
    work.projected_direction(&req)
        .unwrap_or_else(|_| panic!("finite direction"));
    assert!(!work.ray(&req).unwrap_or(false), "joint ray must stall");
    assert!(
        work.rescue(&req).unwrap_or(false),
        "coordinate ray must improve"
    );
    assert_eq!(work.l0, vec![0.0, 1.0]);
    assert!((work.objective + 0.9).abs() < 1e-14);
}

fn pair_trap_rows() -> [DualSearchRow<'static>; 2] {
    [
        DualSearchRow {
            a: &[2.0, -1.0, 0.0],
            sign: 1.0,
            b0: 0.8,
            b1: 0.0,
            vacuous: true,
        },
        DualSearchRow {
            a: &[-1.0, 0.0, -1.0],
            sign: 1.0,
            b0: 0.8,
            b1: 0.0,
            vacuous: true,
        },
    ]
}

#[test]
fn pair_rescue_escapes_an_exact_joint_and_coordinate_trap() {
    // F(u,v)=-|2u-v|-|1-u|-|1-v|-.8(u+v). At zero the joint
    // derivative is -.6 and coordinate derivatives are -1.8 and -.8.
    // The balanced (1,2) ray instead improves by .3 at (.5,1).
    let rows = pair_trap_rows();
    let req = request(&[0.0, 1.0, 1.0], &rows, &[0.0; 2], &[0.0; 2]);
    let baseline = objective(&req, req.l0, req.l1);
    let mut work = Work::new(&req, baseline, working_bytes(3, 2)).unwrap();
    work.projected_direction(&req)
        .unwrap_or_else(|_| panic!("finite gradient"));
    assert!(!work.ray(&req).unwrap_or(false));
    assert!(!work.rescue(&req).unwrap_or(false));
    assert!(work.pair_rescue(&req).unwrap_or(false));
    assert_eq!(work.l0, [0.5, 1.0]);
    assert_eq!(work.l1, [0.0, 0.0]);
    assert_eq!(work.pair_rays, 1);
    assert!((work.objective + 1.7).abs() < 1e-14);
    assert_eq!(work.objective, objective(&req, &work.l0, &work.l1));

    // Independent exact rational oracle for the actual f64 price, including
    // its binary rounding. Signs ((1-b)/2,1,(3b-1)/2) in [-1,1] give
    // A*signs+b=0. The point (1/2,1) attains the resulting global bound.
    let price_num = 3_602_879_701_896_397_i128;
    let price_den = 4_503_599_627_370_496_i128;
    assert_eq!(price_num as f64 / price_den as f64, req.rows[0].b0);
    let sign_den = 2 * price_den;
    let signs = [price_den - price_num, sign_den, 3 * price_num - price_den];
    for a in [[2_i128, -1, 0], [-1_i128, 0, -1]] {
        assert_eq!(
            a.iter().zip(signs).map(|(a, z)| a * z).sum::<i128>() + 2 * price_num,
            0
        );
    }
    assert!(signs.iter().all(|s| s.abs() <= sign_den));
    let optimal_num = price_den + 3 * price_num;
    assert_eq!(signs[1] + signs[2], optimal_num);
    // Integer enumeration at twentieths is separate from the f64 scorer.
    let mut oracle_cost = i128::MAX;
    for u in 0_i128..=40 {
        for v in 0_i128..=40 {
            oracle_cost = oracle_cost.min(
                price_den * ((2 * u - v).abs() + (20 - u).abs() + (20 - v).abs())
                    + price_num * (u + v),
            );
        }
    }
    assert_eq!(oracle_cost, 10 * optimal_num);
    let oracle = -(oracle_cost as f64) / (20 * price_den) as f64;
    assert!((work.objective - oracle).abs() < 1e-14);
    let req = DualSearchRequest { steps: 4, ..req };
    let out = CpuF32Backend::new().refine_dual(&req).unwrap();
    assert_eq!(out.steps, 1);
    let hint = out.multipliers.into_hint().into_ingest();
    assert_eq!(hint.l0, [0.5, 1.0]);
    assert!((objective(&req, &hint.l0, &hint.l1) + 1.7).abs() < 1e-14);
}

#[test]
fn pair_rescue_handles_far_coordinates_orientation_and_finite_scaling() {
    let cases = [
        ([2.0, -1.0, 0.0], 1.0, 1.0),
        ([-2.0, 1.0, 0.0], -1.0, 1.0),
        ([2.0e200, -1.0, 0.0], 1.0, 1.0e200),
    ];
    for (a0, sign, scale) in cases {
        let a1 = [scale, 0.0, 1.0];
        let rows = [
            DualSearchRow {
                a: &a0,
                sign,
                b0: 0.8,
                b1: 0.0,
                vacuous: true,
            },
            DualSearchRow {
                a: &a1,
                sign: 1.0,
                b0: 4.0,
                b1: 0.8,
                vacuous: false,
            },
        ];
        let req = request(&[0.0, 1.0, 1.0], &rows, &[0.0; 2], &[0.0; 2]);
        let mut work = Work::new(&req, -2.0, working_bytes(3, 2)).unwrap();
        assert!(work.pair_rescue(&req).unwrap_or(false));
        assert_eq!(work.l0, [0.5, 0.0]);
        assert_eq!(work.l1, [0.0, 1.0]);
        assert!(work
            .l0
            .iter()
            .chain(&work.l1)
            .all(|v| v.is_finite() && *v >= 0.0));
        assert!((work.objective + 1.7).abs() < 1e-14);
    }
}

#[test]
fn pair_rescue_scores_the_actual_near_zero_residual() {
    let rows = pair_trap_rows();
    let req = request(&[f64::EPSILON, 1.0, 1.0], &rows, &[0.0; 2], &[0.0; 2]);
    let baseline = objective(&req, req.l0, req.l1);
    let mut work = Work::new(&req, baseline, working_bytes(3, 2)).unwrap();
    assert!(work.pair_rescue(&req).unwrap_or(false));
    assert!(work.objective > baseline);
    assert_eq!(work.objective, objective(&req, &work.l0, &work.l1));
    assert!(
        work.rc[0] != 0.0,
        "the heuristic must not erase the residual"
    );
}

#[test]
fn pair_rescue_deadline_and_overflow_preserve_the_last_point() {
    let rows = pair_trap_rows();
    let mut req = request(&[0.0, 1.0, 1.0], &rows, &[0.0; 2], &[0.0; 2]);
    let mut work = Work::new(&req, -2.0, working_bytes(3, 2)).unwrap();
    req.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    assert!(matches!(work.pair_rescue(&req), Err(Halt::Deadline)));
    assert_eq!(work.pair_scans, 0);
    assert_eq!(work.pair_rays, 0);
    assert_eq!(work.l0, [0.0; 2]);
    assert_eq!(work.objective, -2.0);

    let a0 = [f64::MAX, f64::MAX];
    let a1 = [-f64::MAX, f64::MAX];
    let rows = [
        DualSearchRow {
            a: &a0,
            sign: 1.0,
            b0: -0.1,
            b1: 0.0,
            vacuous: true,
        },
        DualSearchRow {
            a: &a1,
            sign: 1.0,
            b0: -0.1,
            b1: 0.0,
            vacuous: true,
        },
    ];
    let req = request(&[0.0; 2], &rows, &[0.0; 2], &[0.0; 2]);
    assert_eq!(validate(&req).unwrap(), 0.0);
    let mut work = Work::new(&req, 0.0, working_bytes(2, 2)).unwrap();
    assert!(matches!(work.pair_rescue(&req), Err(Halt::Nonfinite)));
    assert_eq!(work.l0, [0.0; 2]);
    assert_eq!(work.rc, [0.0; 2]);
    assert_eq!(work.objective, 0.0);
}

#[test]
fn pair_rescue_geometry_admission_preserves_a_valid_but_large_trap() {
    for (n, depth) in [
        (MAX_PAIR_GENERATORS + 1, 2),
        (1024, MAX_PAIR_ROW_TERMS / 1024 + 1),
    ] {
        let mut a0 = vec![0.0; n];
        let mut a1 = vec![0.0; n];
        let mut rc = vec![0.0; n];
        a0[..3].copy_from_slice(&[2.0, -1.0, 0.0]);
        a1[..3].copy_from_slice(&[-1.0, 0.0, -1.0]);
        rc[..3].copy_from_slice(&[0.0, 1.0, 1.0]);
        let zeros = vec![0.0; n];
        let mut rows = vec![
            DualSearchRow {
                a: &zeros,
                sign: 1.0,
                b0: 4.0,
                b1: 0.0,
                vacuous: true
            };
            depth
        ];
        rows[0].a = &a0;
        rows[1].a = &a1;
        rows[0].b0 = 0.8;
        rows[1].b0 = 0.8;
        let l = vec![0.0; depth];
        let req = request(&rc, &rows, &l, &l);
        let mut work = Work::new(&req, -2.0, working_bytes(n, depth)).unwrap();
        assert!(!work.pair_rescue(&req).unwrap_or(false));
        assert_eq!(work.pair_scans, 0);
        assert_eq!(work.pair_rays, 0);
        // Admission skipped useful work, not a stationary or malformed input.
        work.d0[0] = 0.5;
        work.d0[1] = 1.0;
        assert!(work.ray(&req).unwrap_or(false));
        assert!((work.objective + 1.7).abs() < 1e-14);
    }
    assert!(!pair_admitted(usize::MAX, usize::MAX));
}

#[test]
fn pair_rescue_enforces_the_inspection_and_ray_limits() {
    // All rows point the same way, so every cancellation requires an
    // infeasible negative multiplier. Stop scanning without trying a ray.
    let a = vec![1.0; 8192];
    let rows = vec![
        DualSearchRow {
            a: &a,
            sign: 1.0,
            b0: -0.25,
            b1: 0.0,
            vacuous: true
        };
        PAIR_COORDINATES
    ];
    let rc = vec![0.0; a.len()];
    let l = vec![0.0; rows.len()];
    let req = request(&rc, &rows, &l, &l);
    let mut work = Work::new(&req, 0.0, working_bytes(a.len(), rows.len())).unwrap();
    assert!(!work.pair_rescue(&req).unwrap_or(false));
    assert_eq!(work.pair_scans, MAX_PAIR_GENERATOR_SCANS);
    assert_eq!(work.pair_rays, 0);
    assert_eq!(work.objective, 0.0);

    // Different affine column patterns allow many feasible cancellations of
    // one zero cost. Each worsens other zero costs, exhausting the ray cap.
    let coefficients: Vec<Vec<f64>> = (0..PAIR_COORDINATES)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            (0..16)
                .map(|j| sign * (1 + (i + 1) * (j + 1)) as f64)
                .collect()
        })
        .collect();
    let rows: Vec<_> = coefficients
        .iter()
        .map(|a| DualSearchRow {
            a,
            sign: 1.0,
            b0: -0.01,
            b1: 0.0,
            vacuous: true,
        })
        .collect();
    let req = request(&[0.0; 16], &rows, &l, &l);
    let mut work = Work::new(&req, 0.0, working_bytes(16, rows.len())).unwrap();
    assert!(!work.pair_rescue(&req).unwrap_or(false));
    assert_eq!(work.pair_rays, MAX_PAIR_RAYS);
    assert!(work.pair_scans <= MAX_PAIR_GENERATOR_SCANS);
    assert_eq!(work.objective, 0.0);
}

#[test]
fn pair_rescue_keeps_every_legacy_four_step_candidate() {
    let mut rng = Lcg(0x7061697273);
    for _ in 0..64 {
        let mut rc: Vec<f64> = (0..7).map(|_| rng.sample()).collect();
        rc[0] = 0.0;
        let coefficients: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..7).map(|_| rng.sample()).collect())
            .collect();
        let rows: Vec<_> = coefficients
            .iter()
            .map(|a| DualSearchRow {
                a,
                sign: 1.0,
                b0: rng.sample(),
                b1: 3.0,
                vacuous: true,
            })
            .collect();
        let req = DualSearchRequest {
            steps: 4,
            ..request(&rc, &rows, &[0.0; 3], &[0.0; 3])
        };
        let baseline = objective(&req, req.l0, req.l1);
        let mut legacy = Work::new(&req, baseline, working_bytes(7, 3)).unwrap();
        for _ in 0..req.steps {
            let moved = legacy
                .projected_direction(&req)
                .and_then(|()| legacy.ray(&req))
                .and_then(|moved| if moved { Ok(true) } else { legacy.rescue(&req) });
            if !matches!(moved, Ok(true)) {
                break;
            }
        }
        let out = search(&req).unwrap();
        let hint = out.multipliers.into_hint().into_ingest();
        let improved = objective(&req, &hint.l0, &hint.l1);
        let previous = objective(&req, &legacy.l0, &legacy.l1);
        let tol = 1e-12 * (1.0 + previous.abs());
        assert!(
            improved + tol >= previous,
            "new={improved}, legacy={previous}"
        );
        assert!(out.steps <= 4);
    }
}

#[test]
fn nonzero_initial_multipliers_use_deltas_and_vacuous_far_stays_fixed() {
    let rows = [DualSearchRow {
        a: &[2.0],
        sign: -1.0,
        b0: 0.5,
        b1: 0.25,
        vacuous: true,
    }];
    let req = request(&[-2.0], &rows, &[3.0], &[0.4]);
    let out = search(&req).unwrap();
    let hint = out.multipliers.into_hint().into_ingest();
    assert_eq!(hint.l1, vec![0.4]);
    assert!((hint.l0[0] - 2.0).abs() < 1e-14);
    assert!(objective(&req, &hint.l0, &hint.l1) > objective(&req, req.l0, req.l1));
}

#[test]
fn no_steps_is_an_unchanged_zero_gain_hint() {
    let rows = [DualSearchRow {
        a: &[1.0],
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let mut req = request(&[-1.0], &rows, &[0.25], &[0.5]);
    req.steps = 0;
    let out = search(&req).unwrap();
    assert_eq!(out.steps, 0);
    assert_eq!(out.stop, "steps");
    assert_eq!(out.gain.into_hint().into_ingest(), 0.0);
    let hint = out.multipliers.into_hint().into_ingest();
    assert_eq!(hint.l0, req.l0);
    assert_eq!(hint.l1, req.l1);
}

#[test]
fn an_unbounded_search_objective_still_returns_finite_multipliers() {
    let rows = [DualSearchRow {
        a: &[0.0],
        sign: 1.0,
        b0: -1.0,
        b1: 1.0,
        vacuous: true,
    }];
    let req = request(&[0.0], &rows, &[0.0], &[0.0]);
    let out = search(&req).unwrap();
    assert!(out.gain.into_hint().into_ingest().is_finite());
    let hint = out.multipliers.into_hint().into_ingest();
    assert_eq!(hint.l0, vec![MULTIPLIER_LIMIT]);
    assert_eq!(hint.l1, vec![0.0]);
}

#[test]
fn malformed_deadline_and_memory_requests_refuse() {
    let rows = [DualSearchRow {
        a: &[1.0],
        sign: 1.0,
        b0: 0.0,
        b1: 1.0,
        vacuous: false,
    }];
    let valid = request(&[-1.0], &rows, &[0.0], &[0.0]);
    let mut bad = valid;
    bad.l0 = &[];
    assert!(matches!(search(&bad), Err(Refusal::Invalid(_))));
    bad = valid;
    bad.l1 = &[-1.0];
    assert!(matches!(search(&bad), Err(Refusal::Invalid(_))));
    bad = valid;
    bad.rc = &[f64::NAN];
    assert!(matches!(search(&bad), Err(Refusal::Invalid(_))));
    let bad_rows = [DualSearchRow {
        sign: 0.0,
        ..rows[0]
    }];
    bad = valid;
    bad.rows = &bad_rows;
    assert!(matches!(search(&bad), Err(Refusal::Invalid(_))));
    let bad_rows = [DualSearchRow {
        a: &[f64::INFINITY],
        ..rows[0]
    }];
    bad.rows = &bad_rows;
    assert!(matches!(search(&bad), Err(Refusal::Invalid(_))));
    let bad_rows = [DualSearchRow { a: &[], ..rows[0] }];
    bad.rows = &bad_rows;
    assert!(matches!(search(&bad), Err(Refusal::Invalid(_))));
    bad = valid;
    bad.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    assert_eq!(search(&bad).unwrap_err(), Refusal::Deadline);
    bad = valid;
    bad.mem_cap_bytes = working_bytes(1, 1) - 1;
    assert!(matches!(search(&bad), Err(Refusal::Memory { .. })));
    bad = valid;
    bad.rows = &[];
    bad.l0 = &[];
    bad.l1 = &[];
    assert!(matches!(search(&bad), Err(Refusal::Nothing(_))));
}

#[test]
fn numerical_overflow_returns_the_last_complete_hint() {
    let rows = [DualSearchRow {
        a: &[f64::MAX],
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let req = request(&[-1.0], &rows, &[0.0], &[0.0]);
    let out = search(&req).unwrap();
    assert_eq!(out.stop, "nonfinite");
    assert_eq!(out.steps, 0);
    let hint = out.multipliers.into_hint().into_ingest();
    assert_eq!(hint.l0, req.l0);
    assert_eq!(hint.l1, req.l1);
}

fn compiled_work(req: &DualSearchRequest<'_>) -> Work {
    let mut receipt = SupportSearchStats::default();
    let view = Support::compile(req, &mut receipt);
    assert_eq!(
        receipt.compiled, 1,
        "fixture must reach actual support compilation"
    );
    Work::with_support(
        req,
        objective(req, req.l0, req.l1),
        working_bytes(req.rc.len(), req.rows.len()),
        view,
        receipt,
    )
    .unwrap()
}

#[test]
fn support_quotient_production_groups_equal_and_opposite_events_without_matrix_copy() {
    let n = 4096;
    let a: Vec<_> = (0..n)
        .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let rc: Vec<_> = a.iter().map(|&x| -x).collect();
    let rows = [DualSearchRow {
        a: &a,
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let req = request(&rc, &rows, &[0.0], &[0.0]);
    let out = CpuF32Backend::new().refine_dual(&req).unwrap();
    assert_eq!(out.support.compiled, 1);
    assert_eq!(out.support.active_columns, 1);
    assert_eq!(out.support.grouped_columns, (n - 1) as u64);
    assert!(out.support.events > 0);
    assert_eq!(out.support.ungrouped_events, out.support.events * n as u64);
    assert!(out.support.search_reads < out.support.dense_search_reads);
    assert!(
        out.support.build_reads + out.support.search_reads < out.support.dense_search_reads,
        "the measured operation savings must repay compilation"
    );
    assert!(out.support.peak_bytes < out.support.dense_bytes);
    assert!(out.support.peak_bytes < working_bytes(n, 1));
    let hint = out.multipliers.into_hint().into_ingest();
    assert_eq!(hint.l0, [1.0]);
    assert_eq!(hint.l1, [0.0]);
    assert_eq!(objective(&req, &hint.l0, &hint.l1), 0.0);
    // The original ray oracle is still available and chooses the same kink.
    let mut dense = Work::new(&req, -(n as f64), working_bytes(n, 1)).unwrap();
    dense.d0[0] = 1.0;
    assert!(dense.ray(&req).unwrap_or(false));
    assert_eq!(hint.l0, dense.l0);
    assert_eq!(hint.l1, dense.l1);
}

#[test]
fn untouched_fold_matches_original_support_at_signed_nonzero_multiplier_points() {
    let n = 4096;
    let mut a0 = vec![0.0; n];
    let mut a1 = vec![0.0; n];
    a0[..4].copy_from_slice(&[1.0, -2.0, 0.5, 0.0]);
    a1[..4].copy_from_slice(&[0.0, 1.0, -0.5, 2.0]);
    let mut rc = vec![0.25; n];
    rc[..4].copy_from_slice(&[-1.0, 2.0, -0.5, 1.0]);
    let rows = [
        DualSearchRow {
            a: &a0,
            sign: -1.0,
            b0: 0.5,
            b1: -0.25,
            vacuous: false,
        },
        DualSearchRow {
            a: &a1,
            sign: 1.0,
            b0: -0.25,
            b1: 0.5,
            vacuous: false,
        },
    ];
    let req = request(&rc, &rows, &[0.5, 0.25], &[0.25, 0.5]);
    let mut work = compiled_work(&req);
    assert_eq!(work.support_stats.untouched_columns, (n - 4) as u64);
    for a in [0.0, 0.25, 1.0, 2.0] {
        for b in [0.0, 0.5, 1.0] {
            work.l0.copy_from_slice(&[a, b]);
            work.l1.copy_from_slice(&[b, a]);
            work.d0.fill(0.0);
            work.d1.fill(0.0);
            let score = work.evaluate_candidate(&req, 0.0).unwrap();
            assert_eq!(score, objective(&req, &work.l0, &work.l1));
        }
    }
}

#[test]
fn support_sign_fold_is_guarded_over_entire_ray_box_and_preserves_old_oracle() {
    let n = 128;
    let mut a = vec![0.0; n];
    let mut rc = vec![0.0; n];
    for j in 0..64 {
        a[j] = 2.0f64.powi(-45);
        rc[j] = if j < 48 { 2.0 } else { -2.0 };
    }
    a[127] = 1.0;
    rc[127] = -1.0;
    let rows = [DualSearchRow {
        a: &a,
        sign: -1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let req = request(&rc, &rows, &[0.25], &[0.5]);
    let mut work = compiled_work(&req);
    assert_eq!(work.support_stats.sign_folded_columns, 64);
    assert_eq!(work.support_stats.untouched_columns, 63);
    assert_eq!(work.rc.len(), 1);
    for (&near, &far) in [0.0, 1.0, MULTIPLIER_LIMIT]
        .iter()
        .zip([MULTIPLIER_LIMIT, 0.0, MULTIPLIER_LIMIT].iter())
    {
        work.l0[0] = near;
        work.l1[0] = far;
        work.d0[0] = 0.0;
        work.d1[0] = 0.0;
        let compressed = work.evaluate_candidate(&req, 0.0).unwrap();
        let original = objective(&req, &[near], &[far]);
        assert!((compressed - original).abs() <= 1e-12 * (1.0 + original.abs()));
    }
}

#[test]
fn support_sign_guard_includes_inherited_values_above_search_ceiling() {
    let n = 128;
    let a = vec![2.0f64.powi(-45); n];
    let rc = vec![0.07; n];
    let rows = [DualSearchRow {
        a: &a,
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: true,
    }];
    let req = request(&rc, &rows, &[4.0e12], &[0.0]);
    let mut work = compiled_work(&req);
    assert_eq!(
        work.support_stats.sign_folded_columns, 0,
        "the inherited-to-zero ray crosses a real sign breakpoint"
    );
    assert_eq!(work.support_stats.grouped_columns, 127);
    work.d0[0] = -1.0;
    assert!(work.ray(&req).unwrap_or(false));
    assert!(work.l0[0] < req.l0[0]);
    assert!((work.objective - objective(&req, &work.l0, &work.l1)).abs() < 1e-10);
}

#[test]
fn support_signatures_never_group_nearly_equal_coefficients_or_ignore_zero_duals() {
    let n = 128;
    let mut a = vec![1.0; n];
    a[1] = f64::from_bits(1.0f64.to_bits() + 1);
    let rc = vec![-1.0; n];
    let rows = [DualSearchRow {
        a: &a,
        sign: -1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let req = request(&rc, &rows, &[0.0], &[0.0]);
    let work = compiled_work(&req);
    assert_eq!(work.rc.len(), 2);
    assert_eq!(work.support_stats.untouched_columns, 0);
    assert_eq!(work.support_stats.sign_folded_columns, 0);
    assert_eq!(work.support_stats.grouped_columns, 126);
}

#[test]
fn support_request_local_substitution_cancellation_cannot_reuse_stale_grouping() {
    let n = 128;
    let a = vec![1.0; n];
    let before = vec![0.0; n]; // d=1 and fixed signed row offset=-1.
    let mut after = before.clone();
    after[0] = -1.0; // The deeper enc substitution removes that d=1.
    let rows = [DualSearchRow {
        a: &a,
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let old_req = request(&before, &rows, &[0.0], &[0.0]);
    let new_req = request(&after, &rows, &[0.0], &[0.0]);
    let mut old = compiled_work(&old_req);
    let mut new = compiled_work(&new_req);
    assert_eq!(old.rc.len(), 1);
    assert_eq!(new.rc.len(), 2);
    old.d0[0] = 0.0;
    new.d0[0] = 0.0;
    assert_eq!(old.evaluate_candidate(&old_req, 0.0).unwrap(), 0.0);
    assert_eq!(new.evaluate_candidate(&new_req, 0.0).unwrap(), -1.0);
    assert_eq!(
        new.evaluate_candidate(&new_req, 0.0).unwrap(),
        objective(&new_req, new_req.l0, new_req.l1)
    );
}

#[test]
fn support_pilot_declines_dense_geometry_and_deadlines_without_changing_initial_hint() {
    let n = 128;
    let a: Vec<_> = (0..n).map(|j| 1.0 + j as f64).collect();
    let rc = vec![-1.0; n];
    let rows = [DualSearchRow {
        a: &a,
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let mut req = request(&rc, &rows, &[0.0], &[0.0]);
    let mut receipt = SupportSearchStats::default();
    let view = Support::compile(&req, &mut receipt);
    assert_eq!(receipt.compiled, 0);
    assert_eq!(receipt.fallback, 1);
    assert_eq!(view.len(n), n);
    assert!(receipt.build_reads > 0);
    assert!(receipt.peak_bytes <= req.mem_cap_bytes);
    req.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    let mut receipt = SupportSearchStats::default();
    let view = Support::compile(&req, &mut receipt);
    assert_eq!(receipt.compiled, 0);
    assert_eq!(receipt.build_reads, 0);
    assert_eq!(view.len(n), n);
}

#[test]
fn support_dense_near_original_memory_cap_remains_admitted() {
    let n = 4096;
    let a = vec![1.0; n];
    let rc = vec![-1.0; n];
    let rows = [DualSearchRow {
        a: &a,
        sign: 1.0,
        b0: 0.0,
        b1: 0.0,
        vacuous: false,
    }];
    let mut req = request(&rc, &rows, &[0.0], &[0.0]);
    let compact = search(&req).unwrap();
    assert_eq!(compact.support.compiled, 1);
    req.mem_cap_bytes = 40 * n as u64 + 48 + 1024;
    assert_eq!(req.mem_cap_bytes, working_bytes(n, 1));
    let dense = CpuF32Backend::new().refine_dual(&req).unwrap();
    assert_eq!(dense.support.compiled, 0);
    assert_eq!(dense.support.fallback, 1);
    assert_eq!(
        dense.support.build_reads, 0,
        "decline before pilot allocation"
    );
    assert!(dense.support.peak_bytes <= req.mem_cap_bytes);
    assert_eq!(dense.steps, compact.steps);
    assert_eq!(
        dense.gain.into_hint().into_ingest(),
        compact.gain.into_hint().into_ingest()
    );
    let dense_hint = dense.multipliers.into_hint().into_ingest();
    let compact_hint = compact.multipliers.into_hint().into_ingest();
    assert_eq!(dense_hint.l0, compact_hint.l0);
    assert_eq!(dense_hint.l1, compact_hint.l1);
    req.mem_cap_bytes -= 1;
    assert!(matches!(search(&req), Err(Refusal::Memory { .. })));
}

#[test]
fn support_resident_summary_over_cap_falls_back_after_bounded_compilation() {
    let n = 64;
    let depth = 1024;
    let a = vec![1.0; n];
    let rc = vec![-1.0; n];
    let rows = vec![
        DualSearchRow {
            a: &a,
            sign: 1.0,
            b0: n as f64,
            b1: 0.0,
            vacuous: true,
        };
        depth
    ];
    let initial = vec![0.0; depth];
    let mut req = request(&rc, &rows, &initial, &initial);
    req.steps = 1;
    req.mem_cap_bytes = working_bytes(n, depth);
    let out = search(&req).unwrap();
    assert_eq!(out.support.compiled, 0);
    assert_eq!(out.support.fallback, 1);
    assert!(
        out.support.build_reads > 0,
        "compiler fits, resident summary does not"
    );
    assert!(out.support.peak_bytes <= req.mem_cap_bytes);
    assert_eq!(out.multipliers.into_hint().into_ingest().l0, initial);
}
