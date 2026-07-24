// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness gate for the 2-ReLU multi-neuron relaxation (design §4).
//!
//! (a) enclosure: many true reachable (x,y) satisfy every emitted facet;
//! (b) adversarial: a search for a reachable facet-violator finds none;
//! (c) differential: V-set + excluded-corner residual match the numpy/scipy
//!     oracle `validate_hull.py` on shared inputs (via `multineuron_fixture.json`);
//! (d) certified-outward RHS: `next_up_f32` is the sound direction — the wrong
//!     way excludes a reachable lifted vertex.
//!
//! Plus: the β_c Adam multiplier (R6), the A-matrix injection error-fold (R4),
//! and an end-to-end toy 2-ReLU group showing a MEASURABLE sound tightening
//! through NY's real `LinearBounds` + `concretize_sound`.

use super::*;
use crate::bounds::LinearBounds;
use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;

const FIXTURE: &str = include_str!("multineuron_fixture.json");

/// Enclosure tolerance: covers only the outward-rounding margin (matches the
/// oracle's `EPS = 1e-6`, widened slightly for the f32-normal residual).
const ENCLOSE_TOL: f64 = 1e-5;
const ADV_TOL: f64 = 1e-4;

// --- tiny deterministic RNG (hermetic; no rand version coupling) ------------
struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform f64 in [lo, hi].
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let r = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + r * (hi - lo)
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

// --- fixture model ----------------------------------------------------------
struct Case {
    name: String,
    w: Vec<Vec<f64>>, // 2 x n
    c: Vec<f64>,      // 2
    u_lo: Vec<f64>,
    u_hi: Vec<f64>,
    both_unstable: bool,
    verts: Vec<[f64; 4]>,
    excluded_corner_residual: Option<f64>,
    u1: f64,
    u2: f64,
}

fn load_cases() -> Vec<Case> {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).expect("parse fixture json");
    let arr = v["cases"].as_array().expect("cases array");
    arr.iter()
        .map(|c| {
            let f = |k: &str| c[k].as_f64().unwrap();
            let vecf = |k: &str| {
                c[k].as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect::<Vec<_>>()
            };
            let w = c["W"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let verts = c["verts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    let row = r.as_array().unwrap();
                    [
                        row[0].as_f64().unwrap(),
                        row[1].as_f64().unwrap(),
                        row[2].as_f64().unwrap(),
                        row[3].as_f64().unwrap(),
                    ]
                })
                .collect::<Vec<_>>();
            Case {
                name: c["name"].as_str().unwrap().to_string(),
                w,
                c: vecf("c"),
                u_lo: vecf("u_lo"),
                u_hi: vecf("u_hi"),
                both_unstable: c["both_unstable"].as_bool().unwrap(),
                verts,
                excluded_corner_residual: c["excluded_corner_residual"].as_f64(),
                u1: f("u1"),
                u2: f("u2"),
            }
        })
        .collect()
}

impl Case {
    fn octa(&self) -> Octahedron2 {
        Octahedron2::from_affine(
            &self.w[0], &self.w[1], self.c[0], self.c[1], &self.u_lo, &self.u_hi,
        )
    }
    /// x = W u + c  (pre-activations) for input u.
    fn preact(&self, u: &[f64]) -> [f64; 2] {
        let mut x = [self.c[0], self.c[1]];
        for i in 0..2 {
            for j in 0..u.len() {
                x[i] += self.w[i][j] * u[j];
            }
        }
        x
    }
    fn lift(&self, u: &[f64]) -> [f64; 4] {
        let x = self.preact(u);
        [x[0], x[1], relu(x[0]), relu(x[1])]
    }
}

/// A dense set of reachable lifted points: box corners, edge/boundary-biased
/// samples, and interior randoms (the design's boundary bias — tight reachable
/// points sit at box corners).
fn reachable_points(case: &Case, rng: &mut Xorshift, n_random: usize) -> Vec<[f64; 4]> {
    let d = case.u_lo.len();
    let mut pts = Vec::new();
    // all box corners
    for mask in 0..(1u32 << d) {
        let u: Vec<f64> = (0..d)
            .map(|j| {
                if mask & (1 << j) != 0 {
                    case.u_hi[j]
                } else {
                    case.u_lo[j]
                }
            })
            .collect();
        pts.push(case.lift(&u));
    }
    // random interior + boundary-biased (snap a random coord to a face)
    for _ in 0..n_random {
        let mut u: Vec<f64> = (0..d)
            .map(|j| rng.uniform(case.u_lo[j], case.u_hi[j]))
            .collect();
        if rng.next_u64() & 1 == 0 {
            let j = (rng.next_u64() as usize) % d;
            u[j] = if rng.next_u64() & 1 == 0 {
                case.u_lo[j]
            } else {
                case.u_hi[j]
            };
        }
        pts.push(case.lift(&u));
    }
    pts
}

// ===========================================================================
// (a) ENCLOSURE — every emitted facet contains every reachable point
// ===========================================================================
#[test]
fn gate_a_enclosure_all_facets() {
    let mut rng = Xorshift::new(0x1234_1498);
    for case in load_cases() {
        if !case.both_unstable {
            continue;
        }
        let p = case.octa();
        let verts = arrangement_lifted_vertices(&p);
        let facets = hull_facets(&verts);
        assert!(
            !facets.is_empty(),
            "case {}: expected some hull facets",
            case.name
        );
        let pts = reachable_points(&case, &mut rng, 40_000);
        let mut worst = f64::NEG_INFINITY;
        for w in &pts {
            for f in &facets {
                worst = worst.max(f.residual(w));
            }
        }
        assert!(
            worst <= ENCLOSE_TOL,
            "case {}: reachable point EXCLUDED by a facet (worst residual {:+.3e} > {:.0e}) — Invariant MN broken",
            case.name,
            worst,
            ENCLOSE_TOL
        );
    }
}

// ===========================================================================
// (b) ADVERSARIAL — actively hunt a reachable facet-violator
// ===========================================================================
#[test]
fn gate_b_adversarial_no_violator() {
    let mut rng = Xorshift::new(0xa5a5_0792);
    for case in load_cases() {
        if !case.both_unstable {
            continue;
        }
        let p = case.octa();
        let verts = arrangement_lifted_vertices(&p);
        let facets = hull_facets(&verts);
        let d = case.u_lo.len();

        let eval = |u: &[f64]| {
            let w = case.lift(u);
            facets
                .iter()
                .map(|f| f.residual(&w))
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let mut best = f64::NEG_INFINITY;
        for _ in 0..300 {
            let mut u: Vec<f64> = (0..d)
                .map(|j| rng.uniform(case.u_lo[j], case.u_hi[j]))
                .collect();
            let mut step: Vec<f64> = (0..d)
                .map(|j| 0.25 * (case.u_hi[j] - case.u_lo[j]))
                .collect();
            let mut cur = eval(&u);
            best = best.max(cur);
            for _ in 0..200 {
                let cand: Vec<f64> = (0..d)
                    .map(|j| {
                        (u[j] + rng.uniform(-1.0, 1.0) * step[j]).clamp(case.u_lo[j], case.u_hi[j])
                    })
                    .collect();
                let cv = eval(&cand);
                if cv > cur {
                    u = cand;
                    cur = cv;
                    best = best.max(cur);
                }
                for s in &mut step {
                    *s *= 0.99;
                }
            }
        }
        assert!(
            best <= ADV_TOL,
            "case {}: ADVERSARIAL search found a reachable facet-violator (max viol {:+.3e} > {:.0e})",
            case.name,
            best,
            ADV_TOL
        );
    }
}

// ===========================================================================
// (c) DIFFERENTIAL — Rust pipeline matches the numpy/scipy oracle
// ===========================================================================
#[test]
fn gate_c_differential_vs_oracle() {
    for case in load_cases() {
        if !case.both_unstable {
            // Producer still reproduces the box bounds; just no facets.
            continue;
        }
        let p = case.octa();
        let verts = arrangement_lifted_vertices(&p);
        let expected = &case.verts;

        // (c1) conv(V) equality with the oracle via the SUPPORT FUNCTION
        // h_V(d) = max_{v∈V} d·v. This is the geometrically meaningful producer +
        // arrangement output and is robust to redundant / near-duplicate vertices
        // (which differ between scipy's solve and our Cramer solve on the
        // near-degenerate corner-reachable cases) — those don't change the hull.
        let support = |vs: &[[f64; 4]], d: &[f64; 4]| -> f64 {
            vs.iter()
                .map(|v| (0..4).map(|k| d[k] * v[k]).sum::<f64>())
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let mut rng = Xorshift::new(0xc0ffee ^ case.name.len() as u64);
        for _ in 0..300 {
            let d = [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ];
            let hr = support(&verts, &d);
            let ho = support(expected, &d);
            assert!(
                (hr - ho).abs() <= 1e-4,
                "case {}: conv(V) support mismatch in dir {:?}: rust={:.6} oracle={:.6}",
                case.name,
                d,
                hr,
                ho
            );
        }

        // (c3) excluded box-corner residual matches the oracle.
        let facets = hull_facets(&verts);
        let corner = [case.u1, case.u2, relu(case.u1), relu(case.u2)];
        let rust_resid = facets
            .iter()
            .map(|f| f.residual(&corner))
            .fold(f64::NEG_INFINITY, f64::max);
        let oracle_resid = case.excluded_corner_residual.unwrap();
        // sign agreement: both exclude (>EPS) or both admit (<=EPS)
        let rust_excl = rust_resid > ENCLOSE_TOL;
        let oracle_excl = oracle_resid > 1e-6;
        assert_eq!(
            rust_excl, oracle_excl,
            "case {}: corner-exclusion disagreement rust_resid={:+.4e} oracle_resid={:+.4e}",
            case.name, rust_resid, oracle_resid
        );
        // magnitude agreement (both unit-normalized facets): tol 5e-3.
        assert!(
            (rust_resid - oracle_resid).abs() <= 5e-3,
            "case {}: excluded-corner residual mismatch rust={:+.5} oracle={:+.5}",
            case.name,
            rust_resid,
            oracle_resid
        );
    }
}

// (c2) The design's closed-form diamond coupling facet y1+y2 <= 0.5(x1+x2)+1.
#[test]
fn gate_c2_diamond_closed_form_coupling_facet() {
    // x1 = u1+u2, x2 = u1-u2, u in [-1,1]^2  => P = diamond |x1|+|x2|<=2.
    let p = Octahedron2::from_affine(
        &[1.0, 1.0],
        &[1.0, -1.0],
        0.0,
        0.0,
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    let facets = coupling_facets(&p);
    assert!(
        !facets.is_empty(),
        "expected a coupling facet for the diamond"
    );
    // find the y1+y2 coupling facet: a_y1,a_y2 > 0 and roughly equal.
    let mut found = false;
    for f in &facets {
        let (ax1, ax2, ay1, ay2) = (f.a[0] as f64, f.a[1] as f64, f.a[2] as f64, f.a[3] as f64);
        if ay1 > 0.1 && ay2 > 0.1 && (ay1 - ay2).abs() < 1e-3 {
            // un-normalize to the y1+y2 <= (-ax1/ay1) x1 + ... + b/ay1 form
            let sx1 = ax1 / ay1;
            let sx2 = ax2 / ay1;
            let intercept = f.b as f64 / ay1;
            // expect y1+y2 <= 0.5(x1+x2)+1  => stored a = (-0.5,-0.5,1,1), so
            // sx1 = ax1/ay1 = -0.5, intercept = +1.0.
            if (sx1 + 0.5).abs() < 5e-3
                && (sx2 + 0.5).abs() < 5e-3
                && (intercept - 1.0).abs() < 5e-3
            {
                found = true;
            }
        }
    }
    assert!(
        found,
        "diamond coupling facet y1+y2 <= 0.5(x1+x2)+1 not reproduced; facets={:?}",
        facets
    );
}

// ===========================================================================
// (d) CERTIFIED-OUTWARD RHS — next_up_f32 is the sound direction
// ===========================================================================
#[test]
fn gate_d_certified_outward_direction() {
    for case in load_cases() {
        if !case.both_unstable {
            continue;
        }
        let p = case.octa();
        let verts = arrangement_lifted_vertices(&p);
        let facets = hull_facets(&verts);

        for f in &facets {
            // (O) every lifted vertex satisfies the outward-rounded facet.
            let mut worst = f64::NEG_INFINITY;
            let mut argmax_v = verts[0];
            let mut max_dot = f64::NEG_INFINITY;
            for v in &verts {
                let r = f.residual(v);
                worst = worst.max(r);
                let dot: f64 = (0..4).map(|k| f.a[k] as f64 * v[k]).sum();
                if dot > max_dot {
                    max_dot = dot;
                    argmax_v = *v;
                }
            }
            assert!(
                worst <= ENCLOSE_TOL,
                "case {}: outward facet excludes a lifted vertex (residual {:+.3e})",
                case.name,
                worst
            );

            // (P2, the primary assertion) the stored RHS is >= the true f64
            // support `max_dot = max_{v∈V} a·v`. `max_dot` is recomputed here with
            // the byte-identical summation order used in `hull_facets`, and
            // `b = next_up_f32(max_dot as f32) >= max_dot` because round-to-nearest
            // lands within ½ ULP and next_up adds a full ULP. This is exactly the
            // half-space `a·w ≤ b` being a proven superset of conv(V).
            assert!(
                f.b as f64 >= max_dot,
                "case {}: stored RHS {:.9} is below the true support {:.9} — UNSOUND (P2 broken)",
                case.name,
                f.b,
                max_dot
            );
            // Witness that the outward rounding is load-bearing: a RHS a clearly
            // visible amount BELOW the certified support (the R2 "wrong way")
            // EXCLUDES the argmax reachable vertex, whereas the stored next_up RHS
            // does not. Uses an absolute margin (not ULP arithmetic) so the
            // witness is robust across summation orders and at max_dot = 0 facets.
            let delta = 1e-3 * (1.0 + max_dot.abs());
            let b_wrong = (max_dot - delta) as f32;
            let wrong = Facet { a: f.a, b: b_wrong };
            assert!(
                wrong.residual(&argmax_v) > 0.0,
                "case {}: a sub-support RHS failed to exclude its argmax vertex",
                case.name
            );
            // And the correct (next_up) RHS does NOT exclude it.
            assert!(
                f.residual(&argmax_v) <= ENCLOSE_TOL,
                "case {}: correct outward RHS unexpectedly excludes its argmax vertex",
                case.name
            );
        }
    }
}

// ===========================================================================
// Exact support checker — geometry proposes; exact four-orthant LP certifies.
// ===========================================================================

#[test]
fn exact_support_certifies_closed_form_diamond() {
    // P = {|x1+x2|<=2, |x1-x2|<=2} = {|x1|+|x2|<=2}.
    let p = Octahedron2::from_bounds(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
    let checker = ExactRelu2Support::new(&p).expect("nonempty finite diamond");
    assert!(
        checker.orthant_vertex_counts().iter().all(|&n| n > 0),
        "the exact LP must inspect every ReLU orthant"
    );

    // y1+y2 <= 0.5(x1+x2)+1, whose support is EXACTLY one.
    let normal = [-0.5f32, -0.5, 1.0, 1.0];
    let certified = checker
        .certify_normal(normal)
        .expect("finite exact support must certify");
    assert_eq!(
        certified.a, normal,
        "the stored f32 normal is authoritative"
    );
    assert_eq!(certified.b, 1.0, "closed-form exact support");

    // The end-to-end repair seam must discard the legacy RHS and reproduce each
    // proposal through the independent checker.
    let facets = certified_coupling_facets_exact(&p);
    assert!(
        !facets.is_empty(),
        "diamond must yield exact-certified proposals"
    );
    for facet in facets {
        assert_eq!(
            checker.certify_normal(facet.a),
            Some(facet),
            "every stored RHS must be the checker result for its stored normal"
        );
    }
}

#[test]
fn exact_support_preserves_sub_ulp_cancellation_band() {
    use num_rational::BigRational;

    // Around x1=x2=1, ordinary f64 vertex arithmetic loses `delta` in
    // `1-delta` entirely.  The octahedral difference row nevertheless stores
    // delta as an exact dyadic, and the exact LP must recover
    // max(x1-x2)=delta rather than zero.
    let delta = 2.0f64.powi(-140);
    assert_eq!(1.0 - delta, 1.0, "the witness is below one f64 ULP at 1");
    let p = Octahedron2::from_bounds(-1.0, 1.0, -1.0, 1.0, -2.0, 2.0, -delta, delta);
    let checker = ExactRelu2Support::new(&p).expect("nonempty cancellation band");
    let certified = checker
        .certify_normal([1.0, -1.0, 0.0, 0.0])
        .expect("dyadic support is f32-representable");

    let expected = delta as f32;
    assert_eq!(certified.b, expected, "exact max(x1-x2) must be delta");
    let rhs = BigRational::from_float(f64::from(certified.b)).unwrap();
    let exact = BigRational::from_float(delta).unwrap();
    let predecessor = BigRational::from_float(f64::from(next_down_f32(certified.b))).unwrap();
    assert!(rhs >= exact, "stored RHS is outward in exact arithmetic");
    assert!(predecessor < exact, "the directed f32 RHS is tight");
}

#[test]
fn exact_support_rounds_rhs_up_when_nearest_f32_is_inward() {
    use num_rational::BigRational;

    // One quarter of an f32 ULP above 1 rounds to 1 under round-to-nearest.  A
    // support certificate must instead take the successor.  A singleton P keeps
    // the exact optimum transparent while also exercising lower-dimensional
    // exact-cell vertex enumeration.
    let support = 1.0f64 + 2.0f64.powi(-25);
    assert_eq!(support as f32, 1.0, "nearest f32 is inward");
    let p = Octahedron2::from_bounds(
        support, support, 0.0, 0.0, support, support, support, support,
    );
    let checker = ExactRelu2Support::new(&p).expect("singleton P has an exact vertex");
    let certified = checker
        .certify_normal([1.0, 0.0, 0.0, 0.0])
        .expect("finite directed RHS");
    assert_eq!(certified.b, next_up_f32(1.0));
    let encoded = BigRational::from_float(f64::from(certified.b)).unwrap();
    let optimum = BigRational::from_float(support).unwrap();
    assert!(
        encoded >= optimum,
        "exact round-trip authorizes the outward RHS"
    );
}

#[test]
fn exact_support_is_independent_of_tolerance_dedup_at_tiny_scale() {
    // Every genuine vertex lies inside the legacy 1e-7 dedup radius.  The
    // proposal geometry therefore collapses the lifted set, but a supplied f32
    // normal can still be certified directly from P with no scale floor.
    let eps = 2.0f64.powi(-40);
    let p = Octahedron2::from_bounds(
        -eps,
        eps,
        -eps,
        eps,
        -2.0 * eps,
        2.0 * eps,
        -2.0 * eps,
        2.0 * eps,
    );
    let legacy_vertices = arrangement_lifted_vertices(&p);
    assert!(
        legacy_vertices.len() < 5,
        "adversarial tiny geometry should expose tolerance dedup; got {} vertices",
        legacy_vertices.len()
    );
    assert!(
        proposed_hull_normals(&legacy_vertices).is_empty(),
        "dedup may lose proposals, never authorize an unchecked facet"
    );

    let checker = ExactRelu2Support::new(&p).expect("tiny P is exactly nonempty");
    assert!(checker.orthant_vertex_counts().iter().all(|&n| n > 0));
    let certified = checker
        .certify_normal([1.0, 1.0, 1.0, 1.0])
        .expect("tiny exact support is representable");
    assert_eq!(
        certified.b,
        (4.0 * eps) as f32,
        "max x1+x2+ReLU(x1)+ReLU(x2) = 4*eps"
    );
}

#[test]
fn exact_support_fails_closed_on_malformed_or_unrepresentable_inputs() {
    let malformed = Octahedron2::from_bounds(f64::NAN, 1.0, -1.0, 1.0, -2.0, 2.0, -2.0, 2.0);
    assert!(ExactRelu2Support::new(&malformed).is_none());

    let inconsistent = Octahedron2::from_bounds(1.0, -1.0, -1.0, 1.0, -2.0, 2.0, -2.0, 2.0);
    assert!(ExactRelu2Support::new(&inconsistent).is_none());

    let p = Octahedron2::from_bounds(-1.0, 2.0, -1.0, 1.0, -2.0, 3.0, -2.0, 3.0);
    let checker = ExactRelu2Support::new(&p).expect("valid bounded P");
    assert!(checker.certify_normal([f32::NAN, 0.0, 0.0, 0.0]).is_none());
    assert!(
        checker.certify_normal([f32::MAX, 0.0, 0.0, 0.0]).is_none(),
        "support above f32::MAX must drop the facet, not store +inf"
    );
}

// ===========================================================================
// β_c Adam multiplier (R6) — mirrors update_lambda_adam
// ===========================================================================
#[test]
fn beta_adam_projects_nonnegative_and_clamps() {
    let terms = vec![MnTerm {
        node_name: "n".into(),
        neuron_idx: 0,
        var: MnVar::PreActivation,
        coefficient: 1.0,
    }];
    // constructor rejects negative / NaN beta
    assert!(MultiNeuronConstraint::new(terms.clone(), 1.0, -0.1).is_err());
    assert!(MultiNeuronConstraint::new(terms.clone(), f32::NAN, 0.0).is_err());

    let mut g = MultiNeuronConstraint::new(terms, 1.0, 0.0).unwrap();
    // positive gradient (maximize) drives beta up, clamped at MAX_BETA=10.
    for t in 1..=500 {
        g.set_beta_grad(1.0);
        g.update_beta_adam(0.5, 0.9, 0.999, 1e-8, t);
        assert!(g.beta() >= 0.0, "beta must stay >= 0 (R6)");
        assert!(g.beta() <= 10.0 + 1e-6, "beta must clamp at MAX_BETA");
    }
    assert!(
        g.beta() > 1.0,
        "beta should have grown under positive gradient"
    );

    // strongly negative gradient projects to the 0 floor.
    for t in 1..=500 {
        g.set_beta_grad(-5.0);
        g.update_beta_adam(0.5, 0.9, 0.999, 1e-8, t);
        assert!(
            g.beta() >= 0.0,
            "beta must stay >= 0 under negative gradient (R6)"
        );
    }
    assert!(
        g.beta() <= 1e-4,
        "beta should be projected back to the 0 floor"
    );
}

// ===========================================================================
// (R4) A-matrix injection folds the f32 rounding into the certified error
// ===========================================================================
#[test]
fn injection_folds_coeff_error() {
    // 1 row, 3 cols; start with an exact err (zeros) so folding is observable.
    let la = Array2::<f32>::from_shape_vec((1, 3), vec![0.3, -0.7, 1.1]).unwrap();
    let ua = la.clone();
    let lb = Array1::<f32>::zeros(1);
    let ub = Array1::<f32>::zeros(1);
    let mut bounds = LinearBounds::new(la, lb, ua, ub).unwrap();
    bounds.set_coeff_err(Array2::zeros((1, 3)), Array2::zeros((1, 3)));

    let col = 1usize;
    let coeff = 0.123_456_79_f32; // not exactly representable after the add
    let before = bounds.lower_a()[[0, col]];
    let err_before = bounds.lower_a_err().unwrap()[[0, col]];
    bounds.add_to_lower_column(col, coeff);
    let after = bounds.lower_a()[[0, col]];
    let err_after = bounds.lower_a_err().unwrap()[[0, col]];

    // The stored coefficient moved by ~coeff; the certified error grew by the
    // exact f32 rounding gap of the mutation (folded via next_up).
    let gap = ((after as f64) - (before as f64 + coeff as f64)).abs() as f32;
    let expected = next_up_f32(err_before + gap);
    assert_eq!(
        err_after, expected,
        "error fold must equal next_up(old_err + gap)"
    );
    assert!(
        err_after >= err_before,
        "certified error may only grow under a lossy mutation (R4)"
    );
    // Untouched columns keep their (zero) error.
    assert_eq!(bounds.lower_a_err().unwrap()[[0, 0]], 0.0);
}

// ===========================================================================
// END-TO-END: measurable SOUND tightening on a real 2-ReLU group through
// NY's LinearBounds injection + concretize_sound + the β_c optimizer.
// ===========================================================================

/// Fold a group-layer lower functional `q·(x1,x2,y1,y2)+qb` down to a lower
/// bound on the input box, using the ReLU relaxation for the y-columns and the
/// affine map x=Wu+t for the x-columns, then `concretize_sound`. Returns LB.
///
/// This is the faithful 2-stage backward of §2.2: the facet is injected on the
/// (x,y) columns; the y-terms ride the ReLU relaxation; everything concretizes
/// JOINTLY over U (the coupling Lemma 1 says is the whole point).
#[allow(clippy::too_many_arguments)]
fn toy_lb(
    q: [f64; 4],
    qb: f64,
    w1: &[f64],
    w2: &[f64],
    t1: f64,
    t2: f64,
    l: [f64; 2],
    u: [f64; 2],
    u_lo: &[f64],
    u_hi: &[f64],
) -> f64 {
    // ReLU fold of y-cols (2,3) into x-cols (0,1) + bias, in f64, outward-safe.
    let mut qx = [q[0], q[1]];
    let mut bias = qb;
    for i in 0..2 {
        let qy = q[2 + i];
        let (li, ui) = (l[i], u[i]);
        if qy >= 0.0 {
            // lower facet y_i >= alpha_i x_i ; adaptive alpha in {0,1}
            let alpha = if ui > -li { 1.0 } else { 0.0 };
            qx[i] += qy * alpha;
        } else {
            // upper chord y_i <= c_i (x_i - l_i), c_i = u_i/(u_i-l_i)
            let c = ui / (ui - li);
            qx[i] += qy * c;
            bias += -qy * c * li; // qy<0, c>0, li<0 => lowers the bias (sound)
        }
    }
    // Affine fold x = W u + t into u-columns.
    let n = w1.len();
    let mut au = vec![0.0f64; n];
    for j in 0..n {
        au[j] = qx[0] * w1[j] + qx[1] * w2[j];
    }
    bias += qx[0] * t1 + qx[1] * t2;

    // u-level LinearBounds (1 row), concretize_sound over the input box.
    let a_row: Vec<f32> = au.iter().map(|&x| x as f32).collect();
    let la = Array2::from_shape_vec((1, n), a_row).unwrap();
    let lbias = Array1::from_vec(vec![next_down_f32(bias as f32)]);
    let bounds =
        LinearBounds::new(la.clone(), lbias, la, Array1::from_vec(vec![bias as f32])).unwrap();
    let lo = Array1::from_vec(u_lo.iter().map(|&x| x as f32).collect::<Vec<_>>()).into_dyn();
    let hi = Array1::from_vec(u_hi.iter().map(|&x| x as f32).collect::<Vec<_>>()).into_dyn();
    let ubox = BoundedTensor::new(lo, hi).unwrap();
    let out = bounds.concretize_sound(&ubox);
    out.lower().iter().next().copied().unwrap() as f64
}

#[test]
fn end_to_end_measurable_sound_tightening() {
    // Diamond group: x1=u1+u2, x2=u1-u2, u in [-1,1]^2.
    let w1 = [1.0, 1.0];
    let w2 = [1.0, -1.0];
    let (t1, t2) = (0.0, 0.0);
    let u_lo = [-1.0, -1.0];
    let u_hi = [1.0, 1.0];
    let p = Octahedron2::from_affine(&w1, &w2, t1, t2, &u_lo, &u_hi);
    assert!(p.both_unstable());
    let l = [p.l1, p.l2];
    let u = [p.u1, p.u2];

    // Margin m = -(y1+y2): a spec whose LB the coupling facet (an UPPER bound on
    // y1+y2) tightens. Group-layer functional: lower_a = [0,0,-1,-1], lb = 0.
    let base_q = [0.0, 0.0, -1.0, -1.0];
    let base_qb = 0.0;

    // Baseline (β=0): no facet.
    let lb0 = toy_lb(base_q, base_qb, &w1, &w2, t1, t2, l, u, &u_lo, &u_hi);

    // The coupling facet (certified-outward), converted to a group constraint.
    let facets = coupling_facets(&p);
    // pick the y1+y2 coupling facet (a_y1,a_y2 > 0 ~ equal).
    let facet = facets
        .iter()
        .find(|f| f.a[2] > 0.1 && f.a[3] > 0.1 && (f.a[2] - f.a[3]).abs() < 1e-3)
        .copied()
        .expect("y1+y2 coupling facet");
    let mut group =
        MultiNeuronConstraint::from_facet_for_group(&facet, "relu", 0, "relu", 1).unwrap();

    // Evaluate LB under a given β via the injection primitive on the (x,y) layer.
    let lb_at = |beta: f32| -> f64 {
        // group-layer LinearBounds over (x1,x2,y1,y2): m = -(y1+y2).
        let la = Array2::from_shape_vec((1, 4), vec![0.0f32, 0.0, -1.0, -1.0]).unwrap();
        let mut lbnd = LinearBounds::new(
            la.clone(),
            Array1::from_vec(vec![0.0f32]),
            la,
            Array1::from_vec(vec![0.0f32]),
        )
        .unwrap();
        // Lagrangian: + beta * (a·(x,y) - b) on the lower bound.
        for k in 0..4 {
            lbnd.add_to_lower_column(k, beta * facet.a[k]);
        }
        lbnd.add_lower_bias_outward(-beta * facet.b);
        let q = [
            lbnd.lower_a()[[0, 0]] as f64,
            lbnd.lower_a()[[0, 1]] as f64,
            lbnd.lower_a()[[0, 2]] as f64,
            lbnd.lower_a()[[0, 3]] as f64,
        ];
        let qb = lbnd.lower_b[0] as f64;
        toy_lb(q, qb, &w1, &w2, t1, t2, l, u, &u_lo, &u_hi)
    };

    // β_c optimizer loop (Adam) with a finite-difference gradient of LB(β).
    let h = 1e-2f32;
    let mut best = lb0;
    for t in 1..=200 {
        let b = group.beta();
        let grad = ((lb_at(b + h) - lb_at((b - h).max(0.0))) / (2.0 * h as f64)) as f32;
        group.set_beta_grad(grad);
        group.update_beta_adam(0.2, 0.9, 0.999, 1e-8, t);
        best = best.max(lb_at(group.beta()));
    }
    let lb_star = best;

    // True min of m over U by dense sampling (an OVER-estimate of the true min,
    // so LB <= true_min <= true_min_sampled is a valid soundness necessary cond).
    let mut rng = Xorshift::new(0xf00d_1498);
    let mut true_min = f64::INFINITY;
    let grid = 41;
    for gi in 0..grid {
        for gj in 0..grid {
            let uu = [
                u_lo[0] + (u_hi[0] - u_lo[0]) * gi as f64 / (grid - 1) as f64,
                u_lo[1] + (u_hi[1] - u_lo[1]) * gj as f64 / (grid - 1) as f64,
            ];
            let x = [w1[0] * uu[0] + w1[1] * uu[1], w2[0] * uu[0] + w2[1] * uu[1]];
            let m = -(relu(x[0]) + relu(x[1]));
            true_min = true_min.min(m);
        }
    }
    for _ in 0..200_000 {
        let uu = [rng.uniform(u_lo[0], u_hi[0]), rng.uniform(u_lo[1], u_hi[1])];
        let x = [w1[0] * uu[0] + w1[1] * uu[1], w2[0] * uu[0] + w2[1] * uu[1]];
        let m = -(relu(x[0]) + relu(x[1]));
        true_min = true_min.min(m);
    }

    // SOUNDNESS: the multi-neuron LB never exceeds the (sampled) true min.
    assert!(
        lb_star <= true_min + 1e-4,
        "UNSOUND: multi-neuron LB {:.5} exceeds sampled true min {:.5}",
        lb_star,
        true_min
    );
    assert!(
        lb0 <= true_min + 1e-4,
        "baseline LB {:.5} exceeds sampled true min {:.5}",
        lb0,
        true_min
    );
    // TIGHTENING: measurable, and monotone (LB only improves).
    assert!(
        lb_star > lb0 + 1e-3,
        "expected measurable tightening: lb0={:.5} lb*={:.5}",
        lb0,
        lb_star
    );
    eprintln!(
        "[multineuron toy] baseline LB={:.5}  multineuron LB={:.5}  (gain {:+.5})  true_min≈{:.5}",
        lb0,
        lb_star,
        lb_star - lb0,
        true_min
    );
}

// ===========================================================================
// Producer soundness (Invariant P1) — octahedral bounds enclose sampled x(u).
// ===========================================================================
#[test]
fn producer_p1_encloses_sampled_preactivations() {
    let mut rng = Xorshift::new(0x0c7a_beef);
    for case in load_cases() {
        let p = case.octa();
        let d = case.u_lo.len();
        for _ in 0..20_000 {
            let uu: Vec<f64> = (0..d)
                .map(|j| rng.uniform(case.u_lo[j], case.u_hi[j]))
                .collect();
            let x = case.preact(&uu);
            let s = x[0] + x[1];
            let dd = x[0] - x[1];
            let tol = 1e-5;
            assert!(x[0] <= p.u1 + tol && x[0] >= p.l1 - tol, "P1 x1 out of box");
            assert!(x[1] <= p.u2 + tol && x[1] >= p.l2 - tol, "P1 x2 out of box");
            assert!(s <= p.s_hi + tol && s >= p.s_lo - tol, "P1 x1+x2 out of P");
            assert!(
                dd <= p.d_hi + tol && dd >= p.d_lo - tol,
                "P1 x1-x2 out of P"
            );
        }
    }
}

// Degeneracy (R7): a rank-deficient reachable set (x2 = -x1) yields < 5 lifted
// vertices, so hull_facets emits NO facet — sound (fewer half-spaces).
#[test]
fn degenerate_pair_emits_no_facet_soundly() {
    // x1 = u1+u2, x2 = -(u1+u2) => reachable set is the line x2 = -x1.
    let p = Octahedron2::from_affine(
        &[1.0, 1.0],
        &[-1.0, -1.0],
        0.0,
        0.0,
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    let verts = arrangement_lifted_vertices(&p);
    let facets = hull_facets(&verts);
    assert!(
        facets.is_empty(),
        "degenerate (rank-deficient) pair must emit no facet; got {}",
        facets.len()
    );
}

// ===========================================================================
// §3 group-selection score — larger when P clips a box corner (Theorem 2(iv)).
// ===========================================================================
#[test]
fn excluded_corner_score_ranks_coupling_strength() {
    // Diamond P (|x1|+|x2|<=2): sum bound s_hi=2 << u1+u2=4 → strong coupling.
    let diamond = Octahedron2::from_affine(
        &[1.0, 1.0],
        &[1.0, -1.0],
        0.0,
        0.0,
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    assert!(diamond.both_unstable());
    let diamond_score = diamond.excluded_corner_score();
    assert!(
        diamond_score > 1.0,
        "diamond should score high (clips corners hard), got {diamond_score}"
    );

    // Box P (x1=u1, x2=u2 independent): s_hi=u1+u2 exactly → no coupling gain.
    let boxlike = Octahedron2::from_affine(
        &[1.0, 0.0],
        &[0.0, 1.0],
        0.0,
        0.0,
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    let box_score = boxlike.excluded_corner_score();
    assert!(
        box_score <= 1e-6,
        "axis-aligned (box) pair scores ~0 (Lemma 1, no free lunch), got {box_score}"
    );
    assert!(
        diamond_score > box_score + 1.0,
        "score must rank the coupled pair strictly above the box pair"
    );
}

// ===========================================================================
// (INCREMENT 2, item 1 + soundness (a)) LIVE joint-bound producer on a REAL
// NY α-CROWN backward: build a 2-layer ReLU net, run `combined_row_octahedron`
// (a genuine spec-guided CROWN backward through the intervening ReLU), then
// verify P ⊇ Z by sampling REAL network inputs → true pre-activations.
// ===========================================================================
#[test]
fn producer_combined_row_encloses_on_real_backward() {
    use crate::bounds::GraphAlphaState;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2};

    // Net: input u∈[-1,1]^2 → linear1(2→3) → relu1 → linear2(3→2) = "pre".
    // linear2 couples the two pre-activations through shared relu1 outputs.
    let w1 = arr2(&[[1.2_f32, 0.9], [0.8, -1.1], [-0.7, 0.6]]);
    let b1 = arr1(&[0.1_f32, -0.05, 0.2]);
    let w2 = arr2(&[[1.0_f32, 0.9, 0.7], [1.0, -0.9, -0.7]]);
    let b2 = arr1(&[-0.6_f32, -0.2]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "pre",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["r1".to_string()],
    ));
    graph.set_output("pre");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let node_bounds = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha
        .add_relu_node("r1", node_bounds.get("l1").unwrap(), false)
        .unwrap();

    let p = combined_row_octahedron(
        &graph,
        &input,
        &alpha,
        Some(&node_bounds),
        "pre",
        0,
        1,
        None,
    )
    .expect("producer runs on the real backward");

    // Manual f32 forward to the true pre-activations x(u) = W2·ReLU(W1 u + b1) + b2.
    let forward = |u: &[f64; 2]| -> [f64; 2] {
        let u0 = u[0] as f32;
        let u1 = u[1] as f32;
        let mut z = [0.0f32; 3];
        for k in 0..3 {
            z[k] = (w1[[k, 0]] * u0 + w1[[k, 1]] * u1 + b1[[k]]).max(0.0);
        }
        let mut x = [0.0f32; 2];
        for r in 0..2 {
            let mut acc = b2[[r]];
            for k in 0..3 {
                acc += w2[[r, k]] * z[k];
            }
            x[r] = acc;
        }
        [x[0] as f64, x[1] as f64]
    };

    // (P1) every sampled true pre-activation lies inside P (sum/diff included).
    let mut rng = Xorshift::new(0x0b0b_1498);
    let tol = 1e-4;
    let mut saw_x0_neg = false;
    let mut saw_x0_pos = false;
    let mut saw_x1_neg = false;
    let mut saw_x1_pos = false;
    let facets = coupling_facets(&p);
    for _ in 0..80_000 {
        let u = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
        let x = forward(&u);
        saw_x0_neg |= x[0] < 0.0;
        saw_x0_pos |= x[0] > 0.0;
        saw_x1_neg |= x[1] < 0.0;
        saw_x1_pos |= x[1] > 0.0;
        assert!(
            x[0] <= p.u1 + tol && x[0] >= p.l1 - tol,
            "P1 violated: x0={} not in [{}, {}]",
            x[0],
            p.l1,
            p.u1
        );
        assert!(
            x[1] <= p.u2 + tol && x[1] >= p.l2 - tol,
            "P1 violated: x1={} not in [{}, {}]",
            x[1],
            p.l2,
            p.u2
        );
        let s = x[0] + x[1];
        let d = x[0] - x[1];
        assert!(
            s <= p.s_hi + tol && s >= p.s_lo - tol,
            "P1 violated: x0+x1={s} not in [{}, {}]",
            p.s_lo,
            p.s_hi
        );
        assert!(
            d <= p.d_hi + tol && d >= p.d_lo - tol,
            "P1 violated: x0-x1={d} not in [{}, {}]",
            p.d_lo,
            p.d_hi
        );

        // (MN) every emitted coupling facet holds at the true (x, ReLU(x)).
        let y = [relu(x[0]), relu(x[1])];
        let w = [x[0], x[1], y[0], y[1]];
        for f in &facets {
            assert!(
                f.residual(&w) <= ENCLOSE_TOL,
                "Invariant MN violated on live backward: facet residual {} > tol at u={:?}",
                f.residual(&w),
                u
            );
        }
    }

    // The net was designed so both pre-activations are unstable (cross zero over
    // the reachable set) — otherwise the coupling facet family is vacuous.
    assert!(
        saw_x0_neg && saw_x0_pos && saw_x1_neg && saw_x1_pos,
        "test net must make both pre-activations unstable (x0 neg/pos: {}/{}, x1 neg/pos: {}/{})",
        saw_x0_neg,
        saw_x0_pos,
        saw_x1_neg,
        saw_x1_pos
    );
    assert!(
        p.both_unstable(),
        "producer P must be both-unstable for this net: {:?}",
        p
    );
    eprintln!(
        "[multineuron live-producer] P={:?} score={:.4} facets={}",
        p,
        p.excluded_corner_score(),
        facets.len()
    );
}

// ===========================================================================
// (INCREMENT 2, item 2 + soundness (c)) ROUTING correctness through the REAL
// ReLU relaxation (`propagate_linear_with_alpha`): the §2.2 pre/post split is
// SOUND (LB never exceeds the true min), recovers the exact coupling gain, and
// MIS-routing (swapping pre/post) is detectably different — the R5 catch.
// ===========================================================================
#[test]
fn routing_pre_post_is_sound_and_misrouting_is_detectable() {
    use crate::{Layer, ReLULayer};
    use ndarray::arr1;

    // Diamond group: x1=u1+u2, x2=u1-u2, u∈[-1,1]^2. Closed form l=-2,u=2,c=0.5;
    // the coupling facet is  -0.5 x1 -0.5 x2 + y1 + y2 <= 1  (a=[-.5,-.5], g=[1,1]).
    let w1 = [1.0f64, 1.0];
    let w2 = [1.0f64, -1.0];
    let (t1, t2) = (0.0f64, 0.0);
    let u_lo = [-1.0f64, -1.0];
    let u_hi = [1.0f64, 1.0];
    let p = Octahedron2::from_affine(&w1, &w2, t1, t2, &u_lo, &u_hi);
    let l = [p.l1, p.l2];
    let u = [p.u1, p.u2];
    let facet = coupling_facets(&p)
        .into_iter()
        .find(|f| f.a[2] > 0.1 && f.a[3] > 0.1 && (f.a[2] - f.a[3]).abs() < 1e-3)
        .expect("y1+y2 coupling facet");
    let group = MultiNeuronConstraint::from_facet_for_group(&facet, "r", 0, "r", 1).unwrap();

    // Pre-activation box for the ReLU relaxation.
    let pre = BoundedTensor::new(
        arr1(&[l[0] as f32, l[1] as f32]).into_dyn(),
        arr1(&[u[0] as f32, u[1] as f32]).into_dyn(),
    )
    .unwrap();
    // Adaptive lower slope (α=1 if u>-l else 0). Symmetric diamond → α=0.
    let alpha = arr1(&[
        if u[0] > -l[0] { 1.0f32 } else { 0.0 },
        if u[1] > -l[1] { 1.0f32 } else { 0.0 },
    ]);
    let relu_layer = ReLULayer;
    let _ = Layer::ReLU(ReLULayer); // dispatch-coverage visibility

    // LB of m = -(y1+y2) under a facet multiplier β, routed correctly or mis-routed,
    // through the REAL ReLU relaxation then the affine fold x=Wu+t and box concretize.
    let lb_at = |beta: f32, mis: bool| -> f64 {
        // Incoming carrier at the ReLU node: coeff on y = (y1,y2). m=-(y1+y2).
        let la = Array2::from_shape_vec((1, 2), vec![-1.0f32, -1.0]).unwrap();
        let mut node_lb = LinearBounds::new(
            la.clone(),
            Array1::from_vec(vec![0.0f32]),
            la,
            Array1::from_vec(vec![0.0f32]),
        )
        .unwrap();
        if !mis {
            group.inject_post_terms_before_relu(&mut node_lb, "r", beta);
        } else {
            // WRONG: put the pre-activation (x) terms on the ReLU OUTPUT before relax.
            group.inject_pre_terms_after_relu(&mut node_lb, "r", beta);
        }
        let (mut new_lb, _g, _gu) = relu_layer
            .propagate_linear_with_alpha(&node_lb, &pre, &alpha, None)
            .unwrap();
        if !mis {
            group.inject_pre_terms_after_relu(&mut new_lb, "r", beta);
        } else {
            // WRONG: put the post-activation (y) terms directly on x, skipping relax.
            group.inject_post_terms_before_relu(&mut new_lb, "r", beta);
        }
        // Fold x = W u + t into u-columns, then box-concretize (exact f64 min).
        let qx = [
            new_lb.lower_a()[[0, 0]] as f64,
            new_lb.lower_a()[[0, 1]] as f64,
        ];
        let mut bias = new_lb.lower_b[0] as f64;
        bias += qx[0] * t1 + qx[1] * t2;
        let n = w1.len();
        let mut lb = bias;
        for k in 0..n {
            let a = qx[0] * w1[k] + qx[1] * w2[k];
            lb += (a * u_lo[k]).min(a * u_hi[k]);
        }
        lb
    };

    let lb0 = lb_at(0.0, false);

    // β_c Adam ascent (finite-difference gradient of the correctly-routed LB).
    let mut group_opt = group.clone();
    let h = 1e-2f32;
    let mut best = lb0;
    let mut best_beta = 0.0f32;
    for t in 1..=200 {
        let b = group_opt.beta();
        let grad =
            ((lb_at(b + h, false) - lb_at((b - h).max(0.0), false)) / (2.0 * h as f64)) as f32;
        group_opt.set_beta_grad(grad);
        group_opt.update_beta_adam(0.2, 0.9, 0.999, 1e-8, t);
        let v = lb_at(group_opt.beta(), false);
        if v > best {
            best = v;
            best_beta = group_opt.beta();
        }
    }
    let lb_star = best;

    // True min of m over the reachable set (dense grid + random), an OVER-estimate.
    let mut rng = Xorshift::new(0xf00d_2222);
    let mut true_min = f64::INFINITY;
    let grid = 61;
    for gi in 0..grid {
        for gj in 0..grid {
            let uu = [
                u_lo[0] + (u_hi[0] - u_lo[0]) * gi as f64 / (grid - 1) as f64,
                u_lo[1] + (u_hi[1] - u_lo[1]) * gj as f64 / (grid - 1) as f64,
            ];
            let x = [w1[0] * uu[0] + w1[1] * uu[1], w2[0] * uu[0] + w2[1] * uu[1]];
            true_min = true_min.min(-(relu(x[0]) + relu(x[1])));
        }
    }
    for _ in 0..300_000 {
        let uu = [rng.uniform(u_lo[0], u_hi[0]), rng.uniform(u_lo[1], u_hi[1])];
        let x = [w1[0] * uu[0] + w1[1] * uu[1], w2[0] * uu[0] + w2[1] * uu[1]];
        true_min = true_min.min(-(relu(x[0]) + relu(x[1])));
    }

    // (SOUND) correctly-routed LB never exceeds the true min, at EVERY β tried.
    for step in 0..=40 {
        let beta = step as f32 * 0.25;
        let lb = lb_at(beta, false);
        assert!(
            lb <= true_min + 1e-4,
            "UNSOUND routing: correct-routed LB {lb:.5} exceeds true_min {true_min:.5} at β={beta}"
        );
    }
    // (TIGHTEN) the routing recovers a measurable coupling gain toward true_min.
    assert!(
        lb_star > lb0 + 1e-2,
        "correct routing should tighten measurably: lb0={lb0:.5} lb*={lb_star:.5}"
    );
    // (EXACT-ish) the coupling facet drives the LB essentially onto the true min
    // (the diamond's excluded box corner is exactly the triangle slack).
    assert!(
        (lb_star - true_min).abs() < 5e-2,
        "coupling facet should recover ~exact bound: lb*={lb_star:.5} true_min={true_min:.5}"
    );

    // (ROUTING MATTERS / R5) mis-routing yields a DIFFERENT bound at the winning β.
    let lb_mis = lb_at(best_beta.max(1.0), true);
    let lb_cor = lb_at(best_beta.max(1.0), false);
    assert!(
        (lb_mis - lb_cor).abs() > 1e-2,
        "mis-routing must be detectably different from correct routing: mis={lb_mis:.5} cor={lb_cor:.5}"
    );

    eprintln!(
        "[multineuron routing] lb0={lb0:.5} lb*={lb_star:.5} (β*={best_beta:.3}) true_min={true_min:.5} \
         mis={lb_mis:.5} cor={lb_cor:.5}"
    );
}

// ===========================================================================
// (INCREMENT 3, item 2 — SOUNDNESS-CRITICAL) [C,H,W] flatten indexing for conv
// groups. A conv group's producer spec rows index the FLATTENED pre-activation
// column; a wrong (c,h,w)->flat map would bound the WRONG neuron pair (unsound).
// This validates `chw_to_flat` against NY's real Conv2d forward+flatten AND
// against the live producer's per-neuron selection.
// ===========================================================================
#[test]
fn chw_indexing_selects_intended_conv_neurons() {
    use crate::bounds::GraphAlphaState;
    use crate::layers::convolution::conv2d::Conv2dLayer;
    use crate::{GraphNetwork, GraphNode, Layer};
    use ndarray::{arr1, Array, IxDyn};

    // input [C_in=2, H=4, W=4] -> Conv2d(out=3, k=3, stride1, pad1) -> "conv" [3,4,4].
    let (cin, hh, ww) = (2usize, 4usize, 4usize);
    let cout = 3usize;
    // Deterministic kernel/bias so the forward is reproducible.
    let mut kdata = Vec::new();
    for oc in 0..cout {
        for ic in 0..cin {
            for kh in 0..3 {
                for kw in 0..3 {
                    let v = ((oc * 37 + ic * 13 + kh * 3 + kw) % 11) as f32 * 0.05 - 0.25;
                    kdata.push(v);
                }
            }
        }
    }
    let kernel = Array::from_shape_vec(IxDyn(&[cout, cin, 3, 3]), kdata).unwrap();
    let bias = arr1(&[-0.1f32, 0.2, -0.3]);
    let conv = Conv2dLayer::new(kernel, Some(bias), (1, 1), (1, 1)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");

    // Point input (lo == hi) so every bound is exact: the conv output value at
    // (c,h,w) is deterministic and the producer's [l_i,u_i] collapse to it.
    let mut pt = Vec::new();
    for c in 0..cin {
        for h in 0..hh {
            for w in 0..ww {
                pt.push(((c * 5 + h * 3 + w) % 7) as f32 * 0.1 - 0.3);
            }
        }
    }
    let pt_arr = Array::from_shape_vec(IxDyn(&[cin, hh, ww]), pt).unwrap();
    let input = BoundedTensor::new(pt_arr.clone(), pt_arr).unwrap();

    let node_bounds = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let conv_bt = node_bounds.get("conv").expect("conv node bounds");
    let shape = conv_bt.lower().shape().to_vec();
    assert_eq!(shape, vec![cout, hh, ww], "conv output shape [C,H,W]");
    let ho = shape[1];
    let wo = shape[2];

    // (a) INDEXING CONSISTENCY: for every (c,h,w), the [C,H,W]-indexed value
    // equals the flattened value at chw_to_flat(c,h,w) — i.e. `chw_to_flat`
    // matches NY's row-major BoundedTensor::flatten order EXACTLY.
    let conv_lo = conv_bt.lower();
    let flat_lo = conv_bt.flatten();
    let flat_lo = flat_lo.lower();
    for c in 0..cout {
        for h in 0..ho {
            for w in 0..wo {
                let direct = conv_lo[IxDyn(&[c, h, w])];
                let flat = chw_to_flat(c, h, w, ho, wo);
                let via_flat = flat_lo[flat];
                assert_eq!(
                    direct, via_flat,
                    "chw_to_flat mismatch at (c={c},h={h},w={w}): [C,H,W]={direct} vs flat[{flat}]={via_flat}"
                );
            }
        }
    }

    // (b) PRODUCER SELECTION: pick two distinct (c,h,w) neurons, map to flat via
    // chw_to_flat, run the live producer, and confirm P's per-neuron bounds match
    // the intended (c,h,w) values (point input ⇒ l==u==forward value). If the map
    // were wrong the producer would bound a different neuron and this would fail.
    let (c1, h1, w1) = (0usize, 1usize, 2usize);
    let (c2, h2, w2) = (2usize, 3usize, 0usize);
    let f1 = chw_to_flat(c1, h1, w1, ho, wo);
    let f2 = chw_to_flat(c2, h2, w2, ho, wo);
    let v1 = conv_lo[IxDyn(&[c1, h1, w1])] as f64;
    let v2 = conv_lo[IxDyn(&[c2, h2, w2])] as f64;

    let alpha = GraphAlphaState::new(); // no intervening ReLU ⇒ unused
    let p = combined_row_octahedron(
        &graph,
        &input,
        &alpha,
        Some(&node_bounds),
        "conv",
        f1,
        f2,
        None,
    )
    .expect("producer runs on conv pre-activation");

    let tol = 1e-4;
    assert!(
        (p.l1 - v1).abs() < tol && (p.u1 - v1).abs() < tol,
        "producer neuron-1 [{},{}] must bracket the intended (c1,h1,w1) value {v1}",
        p.l1,
        p.u1
    );
    assert!(
        (p.l2 - v2).abs() < tol && (p.u2 - v2).abs() < tol,
        "producer neuron-2 [{},{}] must bracket the intended (c2,h2,w2) value {v2}",
        p.l2,
        p.u2
    );
    // Sum/diff rows also select the intended pair.
    assert!((p.s_hi - (v1 + v2)).abs() < tol && (p.s_lo - (v1 + v2)).abs() < tol);
    assert!((p.d_hi - (v1 - v2)).abs() < tol && (p.d_lo - (v1 - v2)).abs() < tol);

    // (c) BATCHED producer selects the same neurons for many pairs in ONE pass.
    let pairs = [(f1, f2), (f2, f1)];
    let octs = combined_rows_octahedra(
        &graph,
        &input,
        &alpha,
        Some(&node_bounds),
        "conv",
        &pairs,
        None,
    )
    .expect("batched producer runs");
    assert_eq!(octs.len(), 2);
    assert!((octs[0].l1 - v1).abs() < tol && (octs[0].l2 - v2).abs() < tol);
    assert!((octs[1].l1 - v2).abs() < tol && (octs[1].l2 - v1).abs() < tol);

    eprintln!("[multineuron chw] f1={f1}(c{c1},h{h1},w{w1})={v1:.4} f2={f2}(c{c2},h{h2},w{w2})={v2:.4} — indexing sound");
}

// ===========================================================================
// STEM-RESIDENT lever (`NY_MULTINEURON_STEM`) — INV-A / INV-B soundness gates.
// ===========================================================================

#[test]
fn multineuron_environment_authority_is_quarantined() {
    ny_test_utils::env::with_env_edits(|env| {
        env.set("NY_MULTINEURON", "1");
        env.set("NY_MULTINEURON_STEM", "1");
        env.set("NY_MN_HEAD_RESIDENT", "1");
        env.set("NY_MN_HEAD_FACET", "1");
        assert!(!root_inject::enabled());
        assert!(!root_inject::stem_enabled());
        assert!(!root_inject::head_resident_enabled());
        assert!(!root_inject::head_facet_enabled());
    });
}

/// Build a diamond-coupling pool with one facet at ReLU node `relu` (neurons
/// 0,1), reused by both stem tests.
fn diamond_stem_pool() -> MultiNeuronPool {
    let w1 = [1.0, 1.0];
    let w2 = [1.0, -1.0];
    let p = Octahedron2::from_affine(&w1, &w2, 0.0, 0.0, &[-1.0, -1.0], &[1.0, 1.0]);
    assert!(p.both_unstable());
    let facets = coupling_facets(&p);
    let facet = facets
        .iter()
        .find(|f| f.a[2] > 0.1 && f.a[3] > 0.1 && (f.a[2] - f.a[3]).abs() < 1e-3)
        .copied()
        .expect("y1+y2 coupling facet");
    let group = MultiNeuronConstraint::from_facet_for_group(&facet, "relu", 0, "relu", 1).unwrap();
    let mut pool = MultiNeuronPool::new(0);
    assert!(pool.push(group));
    pool
}

/// INV-A (conversion level): a `β = 0` pool→resident-fold is a structural NO-OP
/// — empty coeff channels and a zero bias shift — so the resident re-run at
/// β = 0 reproduces the baseline backward byte-identical (and the driver never
/// even re-runs β = 0). INV-B mechanism: coefficients scale linearly with β.
#[test]
fn stem_fold_conversion_beta_zero_is_noop_and_scales() {
    let pool = diamond_stem_pool();

    // β = 0 ⇒ inert fold (INV-A).
    let f0 = root_inject::pool_to_resident_fold(&pool, 0.0);
    assert!(f0.coeffs.is_empty(), "β=0 must yield no post-coeffs");
    assert!(f0.pre_coeffs.is_empty(), "β=0 must yield no pre-coeffs");
    assert_eq!(f0.bias_shift, 0.0, "β=0 must yield zero bias shift");
    assert!(f0.sound_round, "stem fold must use outward rounding");

    // β = 1 ⇒ some channel is populated (the facet has post AND pre terms).
    let f1 = root_inject::pool_to_resident_fold(&pool, 1.0);
    assert!(
        !f1.coeffs.is_empty() || !f1.pre_coeffs.is_empty(),
        "β>0 must populate at least one fold channel"
    );

    // β = 2 ⇒ every coefficient DOUBLES vs β = 1 (linearity of the Lagrangian
    // multiplier; the sound-max mechanism relies on this monotone family).
    let f2 = root_inject::pool_to_resident_fold(&pool, 2.0);
    let cmp = |a: &[(u32, f32)], b: &[(u32, f32)]| {
        assert_eq!(a.len(), b.len(), "channel index set must be β-invariant");
        for (&(ia, ca), &(ib, cb)) in a.iter().zip(b.iter()) {
            assert_eq!(ia, ib, "same neuron indices across β");
            assert!(
                (cb - 2.0 * ca).abs() <= 1e-5 + 1e-5 * ca.abs(),
                "coeff must scale linearly: c(β=1)={ca} c(β=2)={cb}"
            );
        }
    };
    cmp(&f1.coeffs, &f2.coeffs);
    cmp(&f1.pre_coeffs, &f2.pre_coeffs);
    assert!(
        (f2.bias_shift - 2.0 * f1.bias_shift).abs() <= 1e-5 + 1e-5 * f1.bias_shift.abs(),
        "bias shift must scale linearly with β"
    );
    eprintln!(
        "[multineuron-stem conv] β=1 post={} pre={} bias={:.5}; β=2 doubles",
        f1.coeffs.len(),
        f1.pre_coeffs.len(),
        f1.bias_shift
    );
}

/// Production-authority quarantine at the DRIVER level. Both an absent gate and
/// `NY_MULTINEURON_STEM=1` must return the baseline byte-identically; the latter
/// cannot authorize the research fold until its facet/reduction/target proof
/// obligations are discharged.
#[test]
fn stem_resident_environment_is_authority_quarantined() {
    use crate::bounds::GraphAlphaState;
    use crate::layers::convolution::conv2d::Conv2dLayer;
    use crate::{GraphNetwork, GraphNode, Layer, ReLULayer};
    use ndarray::{arr1, Array, IxDyn};

    // input [2,4,4] → Conv2d(out=3,k3,s1,p1) "conv" [3,4,4] → ReLU "relu".
    let (cin, hh, ww) = (2usize, 4usize, 4usize);
    let cout = 3usize;
    let mut kdata = Vec::new();
    for oc in 0..cout {
        for ic in 0..cin {
            for kh in 0..3 {
                for kw in 0..3 {
                    let v = ((oc * 37 + ic * 13 + kh * 3 + kw) % 11) as f32 * 0.05 - 0.25;
                    kdata.push(v);
                }
            }
        }
    }
    let kernel = Array::from_shape_vec(IxDyn(&[cout, cin, 3, 3]), kdata).unwrap();
    let bias = arr1(&[-0.1f32, 0.2, -0.3]);
    let conv = Conv2dLayer::new(kernel, Some(bias), (1, 1), (1, 1)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.set_output("relu");
    assert!(graph.has_conv_layers());

    // Interval input so the conv pre-activations have both-unstable neurons
    // (exercises the pool build + fold registration + re-run path).
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for c in 0..cin {
        for h in 0..hh {
            for w in 0..ww {
                let mid = ((c * 5 + h * 3 + w) % 7) as f32 * 0.1 - 0.3;
                lo.push(mid - 0.5);
                hi.push(mid + 0.5);
            }
        }
    }
    let lo = Array::from_shape_vec(IxDyn(&[cin, hh, ww]), lo).unwrap();
    let hi = Array::from_shape_vec(IxDyn(&[cin, hh, ww]), hi).unwrap();
    let input = BoundedTensor::new(lo, hi).unwrap();

    let node_bounds = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha
        .add_relu_node("relu", node_bounds.get("conv").unwrap(), false)
        .unwrap();

    // Two objectives over the [3,4,4]=48 flat ReLU output.
    let out_dim = cout * hh * ww;
    let mut o1 = vec![0.0f32; out_dim];
    o1[0] = 1.0;
    o1[5] = -1.0;
    let mut o2 = vec![0.0f32; out_dim];
    o2[10] = 1.0;
    o2[20] = -1.0;
    let objectives = vec![o1, o2];

    // A deliberately pessimistic baseline: the sound max can only raise it.
    let baseline = vec![(-10.0f32, 5.0f32); objectives.len()];

    // Serialized env scope (clippy env wall): point the stem resolver at our
    // ReLU node and walk the gate through OFF → ON; restore on exit.
    let on = ny_test_utils::env::with_env_edits(|env| {
        env.set("NY_MULTINEURON_STEM_NODES", "relu");

        // INV-A (default byte-identical): gate OFF ⇒ baseline verbatim.
        env.remove("NY_MULTINEURON_STEM");
        let off = root_inject::tighten_root_objective_bounds_stem_resident(
            &graph,
            &input,
            &objectives,
            None,
            &node_bounds,
            Some(&alpha),
            &baseline,
            None,
        );
        assert_eq!(
            off, baseline,
            "INV-A: gate-off must return the baseline verbatim"
        );

        // The legacy environment request cannot grant verdict authority.
        env.set("NY_MULTINEURON_STEM", "1");
        root_inject::tighten_root_objective_bounds_stem_resident(
            &graph,
            &input,
            &objectives,
            None,
            &node_bounds,
            Some(&alpha),
            &baseline,
            None,
        )
    });

    assert_eq!(
        on, baseline,
        "NY_MULTINEURON_STEM must be a byte-identical no-op while authority is quarantined"
    );
}

/// GUARD 2 (mandatory true-SAT reject-guard) — the false-UNSAT catch. A
/// deliberately-CORRUPTED injected LB vector (one entry ABOVE the known true
/// margin, modelling a RISK-2 conv-column permutation / node-index error) MUST be
/// rejected per objective: the corrupted entry keeps its baseline, while a
/// genuinely-tight entry (≤ the true margin) is still accepted. This is the
/// invariant that guarantees a layout/index error can only FAIL to tighten, never
/// lift an LB above the true margin.
#[test]
fn stem_guard2_rejects_too_high_injected_and_keeps_baseline() {
    // 3 objectives. Sampled true margins (the per-objective ceiling from the true
    // network): obj0=1.0, obj1=2.0, obj2=0.5.
    let min_true = [1.0f32, 2.0, 0.5];

    // acc starts AS the baseline lower vector (INV-A).
    let baseline = [-3.0f32, -3.0, -3.0];
    let mut acc = baseline;

    // A corrupted injected vector:
    //  * obj0 = 0.7   ≤ true 1.0  ⇒ ACCEPT (legit tightening, -3.0 → 0.7);
    //  * obj1 = 9.9   >  true 2.0 ⇒ REJECT (false UNSAT!) keep baseline -3.0;
    //  * obj2 = 0.4   ≤ true 0.5  ⇒ ACCEPT (legit tightening, -3.0 → 0.4).
    let injected = [0.7f32, 9.9, 0.4];

    // Outward tol = 0.0 (default): any exceedance of the sampled true margin rejects.
    let rejected = root_inject::max_into_true_sat_masked(&mut acc, &injected, &min_true, 0.0);

    assert_eq!(rejected, 1, "exactly the too-high obj1 must be rejected");
    assert_eq!(acc[0], 0.7, "obj0: legit-tight injected accepted");
    assert_eq!(
        acc[1], baseline[1],
        "obj1: too-high injected REJECTED — baseline -3.0 kept (no false UNSAT)"
    );
    assert_eq!(acc[2], 0.4, "obj2: legit-tight injected accepted");

    // And the rejected value is NEVER >= its true margin in the accumulator.
    assert!(
        acc[1] <= min_true[1],
        "GUARD 2: accepted acc[1] must never exceed the true margin"
    );

    // A second fold pass with an ALL-too-high vector must change nothing (every
    // entry already ≥ these, or rejected as too-high): acc is a monotone,
    // true-margin-capped accumulator.
    let all_too_high = [5.0f32, 5.0, 5.0];
    let before = acc;
    let rejected2 = root_inject::max_into_true_sat_masked(&mut acc, &all_too_high, &min_true, 0.0);
    assert_eq!(
        rejected2, 3,
        "all three too-high (obj0 5.0>1.0, obj1 5.0>2.0, obj2 5.0>0.5) rejected"
    );
    assert_eq!(
        acc, before,
        "no accumulator entry moved on the all-too-high pass"
    );

    eprintln!(
        "[multineuron-stem GUARD2] corrupted injected {injected:?} -> acc {acc:?} (obj1 false-UNSAT rejected, baseline kept)"
    );
}

/// GUARD 3 (layout precondition — RISK 2) — the CPU pool index ↔ GPU-resident
/// frontier column layout is PROVEN-EQUAL only at an NCHW `(C,H,W)` conv/BN
/// producer. A `Conv2d`/`BatchNorm`-fed stem (rank-3) is ACCEPTED; a `Linear`
/// (Gemm)-fed ReLU (where the resident conv-reshape decode does not apply) is
/// REFUSED — so a non-NCHW stem can never fold a permuted cut onto wrong neurons.
#[test]
fn stem_guard3_accepts_nchw_conv_producer_and_refuses_linear() {
    use crate::layers::convolution::conv2d::Conv2dLayer;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, Array, IxDyn};

    // (1) Conv2d "conv" [3,4,4] → ReLU: rank-3 NCHW conv producer ⇒ ACCEPT.
    let (cin, hh, ww, cout) = (2usize, 4usize, 4usize, 3usize);
    let kdata: Vec<f32> = (0..cout * cin * 9)
        .map(|k| (k % 11) as f32 * 0.05 - 0.25)
        .collect();
    let kernel = Array::from_shape_vec(IxDyn(&[cout, cin, 3, 3]), kdata).unwrap();
    let conv = Conv2dLayer::new(kernel, Some(arr1(&[-0.1f32, 0.2, -0.3])), (1, 1), (1, 1)).unwrap();
    let mut cgraph = GraphNetwork::new();
    cgraph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    cgraph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    cgraph.set_output("relu");
    let clo = Array::from_elem(IxDyn(&[cin, hh, ww]), -0.5f32);
    let chi = Array::from_elem(IxDyn(&[cin, hh, ww]), 0.5f32);
    let cinput = BoundedTensor::new(clo, chi).unwrap();
    let cbounds = cgraph
        .collect_node_bounds_with_engine(&cinput, None)
        .unwrap();
    assert_eq!(
        cbounds.get("conv").unwrap().shape().len(),
        3,
        "conv bounds are rank-3"
    );
    assert!(
        root_inject::stem_layout_precondition_ok(&cgraph, &cbounds, "conv"),
        "GUARD3: an NCHW (C,H,W) Conv2d producer must be ACCEPTED"
    );

    // (2) Linear "l1" [3] → ReLU: Gemm-fed, rank-1 ⇒ REFUSE (no proven layout).
    let w = ndarray::arr2(&[
        [0.5f32, -0.3, 0.2, 0.1],
        [0.1, 0.4, -0.2, 0.3],
        [-0.2, 0.1, 0.3, -0.1],
    ]);
    let lin = LinearLayer::new(w, Some(arr1(&[0.0f32, 0.0, 0.0]))).unwrap();
    let mut lgraph = GraphNetwork::new();
    lgraph.add_node(GraphNode::from_input("l1", Layer::Linear(lin)));
    lgraph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    lgraph.set_output("relu");
    let llo = Array::from_elem(IxDyn(&[4]), -1.0f32);
    let lhi = Array::from_elem(IxDyn(&[4]), 1.0f32);
    let linput = BoundedTensor::new(llo, lhi).unwrap();
    let lbounds = lgraph
        .collect_node_bounds_with_engine(&linput, None)
        .unwrap();
    assert!(
        !root_inject::stem_layout_precondition_ok(&lgraph, &lbounds, "l1"),
        "GUARD3: a Linear/Gemm producer must be REFUSED (layout not proven-equal)"
    );

    // (3) A missing pre-node is refused (no bounds ⇒ cannot certify layout).
    assert!(
        !root_inject::stem_layout_precondition_ok(&cgraph, &cbounds, "nonexistent"),
        "GUARD3: an unknown pre-node must be REFUSED"
    );
    eprintln!("[multineuron-stem GUARD3] conv(rank-3 NCHW)=ACCEPT, linear=REFUSE, missing=REFUSE");
}

/// #mn-head-resident ORACLE (c) — GUARD1 (head node identity) is the false-UNSAT
/// guard for the HEAD retarget. The resident lane folds at fold-order index 0 =
/// the network's LAST ReLU in exec order. If `resolve_head_dense_relu` picks a
/// dense ReLU that is NOT that last ReLU, folding its cut coefficients onto the
/// last ReLU's neurons would be an INVALID Lagrangian ⇒ a potential false UNSAT.
/// GUARD1 must REFUSE (register no fold ⇒ baseline) on that mismatch, and ACCEPT
/// only when the resolved head IS the last ReLU. Also asserts INV-A (gate-off
/// byte-identical). No GPU/propagation needed: the REFUSE fires before any fold.
#[test]
fn head_resident_guard1_refuses_node_mismatch_and_accepts_head() {
    use crate::bounds::GraphAlphaState;
    use crate::layers::convolution::conv2d::Conv2dLayer;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, Array, IxDyn};

    // A structurally-valid conv net with TWO dense-fed ReLUs: the FIRST dense ReLU
    // (`rd1`, which `resolve_head_dense_relu` picks by default) is NOT the last
    // ReLU in exec order (`rd2` is). has_conv_layers() = true (via `conv`).
    let mk_conv = || {
        let kdata: Vec<f32> = (0..3 * 2 * 9)
            .map(|k| (k % 7) as f32 * 0.05 - 0.15)
            .collect();
        let kernel = Array::from_shape_vec(IxDyn(&[3, 2, 3, 3]), kdata).unwrap();
        Conv2dLayer::new(kernel, Some(arr1(&[0.0f32, 0.0, 0.0])), (1, 1), (1, 1)).unwrap()
    };
    let mk_lin = |o: usize, i: usize| {
        let w = Array::from_shape_vec(
            (o, i),
            (0..o * i)
                .map(|k| (k % 5) as f32 * 0.1 - 0.2)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        LinearLayer::new(w, Some(Array::from_elem(o, 0.0f32))).unwrap()
    };

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(mk_conv())));
    graph.add_node(GraphNode::new(
        "rc",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    graph.add_node(GraphNode::new(
        "l1",
        Layer::Linear(mk_lin(4, 48)),
        vec!["rc".into()],
    ));
    graph.add_node(GraphNode::new(
        "rd1",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(mk_lin(4, 4)),
        vec!["rd1".into()],
    ));
    graph.add_node(GraphNode::new(
        "rd2",
        Layer::ReLU(ReLULayer),
        vec!["l2".into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(mk_lin(2, 4)),
        vec!["rd2".into()],
    ));
    graph.set_output("out");
    assert!(graph.has_conv_layers());

    let alpha = GraphAlphaState::new();
    let empty_bounds: std::collections::HashMap<String, BoundedTensor> =
        std::collections::HashMap::new();
    let dummy_in = BoundedTensor::new(
        Array::from_elem(IxDyn(&[2, 4, 4]), -0.5f32),
        Array::from_elem(IxDyn(&[2, 4, 4]), 0.5f32),
    )
    .unwrap();
    let objectives = vec![vec![1.0f32, -1.0], vec![-1.0f32, 1.0]];
    let baseline = vec![(-7.0f32, 3.0f32); objectives.len()];

    ny_test_utils::env::with_env_edits(|env| {
        // INV-A: gate OFF ⇒ baseline verbatim (early return, byte-identical).
        env.remove("NY_MN_HEAD_RESIDENT");
        let off = root_inject::tighten_root_objective_bounds_head_resident(
            &graph,
            &dummy_in,
            &objectives,
            None,
            &empty_bounds,
            Some(&alpha),
            &baseline,
            None,
        );
        assert_eq!(
            off, baseline,
            "INV-A: gate-off must return the baseline verbatim"
        );

        // GUARD1 REFUSE: gate ON, default resolver picks `rd1` (first dense ReLU) ≠
        // `rd2` (last ReLU = fold index 0) ⇒ refuse ⇒ baseline (no fold registered).
        env.set("NY_MN_HEAD_RESIDENT", "1");
        env.remove("NY_MN_HEAD_FACET_NODE");
        let refused = root_inject::tighten_root_objective_bounds_head_resident(
            &graph,
            &dummy_in,
            &objectives,
            None,
            &empty_bounds,
            Some(&alpha),
            &baseline,
            None,
        );
        assert_eq!(
            refused, baseline,
            "GUARD1 must REFUSE a head/target mismatch (rd1 != last ReLU rd2) and keep baseline"
        );
    });
    eprintln!("[mn-head-resident GUARD1] gate-off==baseline (INV-A); node mismatch REFUSED (false-UNSAT hazard guarded)");
}

/// #mn-head-resident GUARD1 POSITIVE — on a net whose DENSE head IS the last ReLU
/// (the real cifar shape: conv stem → dense `Linear→ReLU→Linear` head), the two
/// resolvers AGREE, so GUARD1 ACCEPTS (the retarget's fold-index-0 act == the
/// pool node). Verified at the resolver level (no GPU) — the fold-application
/// soundness itself is proven by the GPU oracle (b).
#[test]
fn head_resident_guard1_accepts_when_dense_head_is_last_relu() {
    use crate::layers::convolution::conv2d::Conv2dLayer;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, Array, IxDyn};

    let kdata: Vec<f32> = (0..3 * 2 * 9)
        .map(|k| (k % 7) as f32 * 0.05 - 0.15)
        .collect();
    let kernel = Array::from_shape_vec(IxDyn(&[3, 2, 3, 3]), kdata).unwrap();
    let conv = Conv2dLayer::new(kernel, Some(arr1(&[0.0f32, 0.0, 0.0])), (1, 1), (1, 1)).unwrap();
    let mk_lin = |o: usize, i: usize| {
        let w = Array::from_shape_vec(
            (o, i),
            (0..o * i)
                .map(|k| (k % 5) as f32 * 0.1 - 0.2)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        LinearLayer::new(w, Some(Array::from_elem(o, 0.0f32))).unwrap()
    };

    // conv → rc → l1(dense) → rhead(ReLU) → out(Linear). The ONLY dense-fed ReLU is
    // `rhead`, and it IS the last ReLU in exec order (output Gemm carries no ReLU).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "rc",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    graph.add_node(GraphNode::new(
        "l1",
        Layer::Linear(mk_lin(4, 48)),
        vec!["rc".into()],
    ));
    graph.add_node(GraphNode::new(
        "rhead",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(mk_lin(2, 4)),
        vec!["rhead".into()],
    ));
    graph.set_output("out");

    let resolved = root_inject::resolve_head_dense_relu_for_test(&graph);
    let last = root_inject::head_relu_in_fold_order_for_test(&graph);
    assert_eq!(
        resolved.as_deref(),
        Some("rhead"),
        "resolve_head_dense_relu must pick the dense head ReLU"
    );
    assert_eq!(
        last.as_deref(),
        Some("rhead"),
        "the last ReLU in fold order must be the dense head"
    );
    assert_eq!(
        resolved, last,
        "GUARD1 ACCEPT: the resolved head IS the retarget fold-index-0 act"
    );
    eprintln!("[mn-head-resident GUARD1] dense head == last ReLU ⇒ GUARD1 ACCEPTS (rhead)");
}

// ===========================================================================
// #mn-head-facet increment 1 — BUILD PIPELINE. `build_head_f64_fold` reduces the
// REAL head pool (`build_pool_for_node` → `coupling_facets`) to an f64
// `HeadF64Fold` β-grid on a genuine dense-fed `Linear→ReLU→Linear` head, keyed by
// the head ReLU's flat neuron index, with the target act resolved and a certified
// (tiny, non-negative) build error per coefficient. Complements the recovery-side
// oracles in `wide_alpha_true` (which drive `sound_f64_lower_bound` directly) by
// exercising the ROOT reduction the CLI uses. Facet VALIDITY (oracle (d)) is
// covered by `facet_enclosure_holds_on_reachable_points` /
// `adversarial_search_finds_no_facet_violator` above.
// ===========================================================================
#[test]
fn build_head_f64_fold_reduces_real_head_pool() {
    use crate::bounds::GraphAlphaState;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{arr1, arr2};

    // Dense head: input u∈[-1,1]^2 → l1(2→3 Linear) → r1(ReLU) → out(3→2 Linear).
    // r1's pre-activations (l1 outputs) all cross zero over the box ⇒ coupling
    // facets exist; r1 is Linear-fed ⇒ the dense head resolver picks it.
    let w1 = arr2(&[[1.2_f32, 0.9], [0.8, -1.1], [-0.7, 0.6]]);
    let b1 = arr1(&[0.1_f32, -0.05, 0.2]);
    let w2 = arr2(&[[1.0_f32, 0.9, 0.7], [1.0, -0.9, -0.7]]);
    let b2 = arr1(&[-0.6_f32, -0.2]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer),
        vec!["l1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["r1".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let node_bounds = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha
        .add_relu_node("r1", node_bounds.get("l1").unwrap(), false)
        .unwrap();

    let folds =
        root_inject::build_head_f64_fold(&graph, &input, &node_bounds, &alpha, None, None, false)
            .expect("head fold must build on a dense-fed crossing head");
    assert!(!folds.is_empty(), "at least one β must yield a fold");
    assert!(folds.len() <= 3, "default β-grid has 3 entries");

    for f in &folds {
        assert_eq!(f.relu_name, "r1", "head ReLU identity");
        assert_eq!(f.head_width, 3, "r1 width = 3 neurons");
        // r1 is the only ReLU ⇒ fold index 0 in the output→input fold order.
        assert_eq!(f.target_act, 0, "resolved global target_act");
        assert!(!f.is_empty(), "a facet-carrying head fold is non-empty");
        // Every fold key is a valid head neuron index; the certified build error is
        // non-negative and tiny (f32×f32 products are EXACT in f64 — only the
        // accumulation rounds, ~1e-16 scale).
        for (idx, coeff, err) in f.post_terms().into_iter().chain(f.pre_terms()) {
            assert!((idx as usize) < f.head_width, "in-range neuron index");
            assert!(
                coeff.is_finite() && coeff != 0.0,
                "retained coeff is nonzero"
            );
            assert!(
                (0.0..1e-9).contains(&err),
                "build error tiny & outward: idx={idx} coeff={coeff} err={err}"
            );
        }
        assert!(f.bias_shift.is_finite() && f.bias_err >= 0.0 && f.bias_err < 1e-9);
    }

    // The β-grid scales the fold monotonically: doubling β doubles the coeffs (the
    // products are exact), which the recovery `max`es over — larger β is a strictly
    // different, still-sound candidate.
    if folds.len() >= 2 {
        eprintln!(
            "[mn-head-facet build] {} folds at r1 (width 3): β0 post={:?} bias={:.4}",
            folds.len(),
            folds[0].post_terms(),
            folds[0].bias_shift
        );
    }
}

// ===========================================================================
// #mn-binding-select (NY_MN_BINDING_SELECT) — objective-gradient-informed head
// facet pair SELECTION. Three oracles:
//  (c) the scorer ranks an objective-relevant pair above an objective-irrelevant
//      one, and picks the SUM vs DIFFERENCE corner gap by the objective sign
//      relationship (correlated ⇒ sum, opposite ⇒ difference);
//  (a) gate-off (`binding_coeffs = None`) is the UNCHANGED excluded-corner
//      heuristic ranking, byte-for-byte (the kept score IS excluded_corner_score,
//      sorted descending);
//  (b) gate-on selection still yields VALID coupling facets — every facet built
//      over a binding-selected pair encloses all reachable lifted points (the
//      same enclosure oracle as gate_a / the head-resident MC soundness oracle).
// ===========================================================================

/// A dense head `input(4) → l1(4→5 Linear) → r1(ReLU) → out(5→3 Linear)` whose 5
/// hidden pre-activations all cross zero over the box (⇒ every pair is a coupling
/// candidate). Returns the graph, the input box, and the α/node bounds.
#[cfg(test)]
fn binding_select_head_fixture() -> (
    crate::GraphNetwork,
    BoundedTensor,
    crate::bounds::GraphAlphaState,
    std::collections::HashMap<String, BoundedTensor>,
) {
    use crate::bounds::GraphAlphaState;
    use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::{Array, Array1};

    // l1: 5×4 weights chosen so every hidden pre-activation straddles 0 over the
    // box, with DIFFERENT sum/difference coupling strengths per pair.
    let w1 = Array::from_shape_vec(
        (5, 4),
        vec![
            0.9_f32, -0.7, 0.5, -0.3, //
            -0.6, 0.8, -0.4, 0.7, //
            0.4, 0.5, -0.9, 0.2, //
            -0.8, -0.3, 0.6, 0.5, //
            0.7, 0.6, 0.5, -0.8, //
        ],
    )
    .unwrap();
    let b1 = Array1::from(vec![0.05_f32, -0.1, 0.15, -0.05, 0.0]);
    // out: 3×5 classifier (the "output Linear" the objective backward reads).
    let w2 = Array::from_shape_vec(
        (3, 5),
        vec![
            1.0_f32, -0.5, 0.8, 0.2, -0.9, //
            -0.7, 0.6, -0.4, 1.1, 0.3, //
            0.5, 0.9, -1.0, -0.2, 0.7, //
        ],
    )
    .unwrap();
    let b2 = Array1::from(vec![0.0_f32, 0.0, 0.0]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "l1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "r1",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["r1".into()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        Array::from_elem(ndarray::IxDyn(&[4]), -1.0_f32),
        Array::from_elem(ndarray::IxDyn(&[4]), 1.0_f32),
    )
    .unwrap();
    let node_bounds = graph.collect_node_bounds_with_engine(&input, None).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha
        .add_relu_node("r1", node_bounds.get("l1").unwrap(), false)
        .unwrap();
    (graph, input, alpha, node_bounds)
}

/// (c) The scorer ranks an objective-RELEVANT pair strictly above an
/// objective-IRRELEVANT one, scores an irrelevant pair 0, and selects the corner
/// gap (SUM vs DIFFERENCE) by the objective's sign relationship — a constructed,
/// graph-free fixture.
#[test]
fn binding_select_scorer_ranks_objective_relevant_above_irrelevant() {
    use root_inject::binding_rank_score_for_test as score;

    // Diamond P: all four corner gaps == 2 (fully symmetric strong coupling).
    let diamond = Octahedron2::from_affine(
        &[1.0, 1.0],
        &[1.0, -1.0],
        0.0,
        0.0,
        &[-1.0, -1.0],
        &[1.0, 1.0],
    );
    assert!(diamond.both_unstable());

    // RELEVANT: both neurons carry a large objective coefficient (both matter).
    let relevant = score(&diamond, 0, 1, &[-1.5, -2.0]);
    // IRRELEVANT: neuron 1's objective coefficient is ~0 ⇒ the coupling on it
    // cannot move the binding margin ⇒ score 0.
    let irrelevant = score(&diamond, 0, 1, &[-1.5, 0.0]);
    assert!(
        relevant > irrelevant,
        "objective-relevant pair must outrank the irrelevant one: relevant={relevant} irrelevant={irrelevant}"
    );
    assert_eq!(
        irrelevant, 0.0,
        "a pair with a zero objective coeff scores 0"
    );
    assert_eq!(
        score(&diamond, 0, 1, &[0.0, 0.0]),
        0.0,
        "a fully objective-irrelevant pair scores 0"
    );

    // Directional selection. P_sum clips ONLY the sum corners; P_diff ONLY the
    // difference corners (from explicit sound-shaped bounds).
    let p_sum = Octahedron2::from_bounds(-1.0, 1.0, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0);
    let p_diff = Octahedron2::from_bounds(-1.0, 1.0, -1.0, 1.0, -2.0, 2.0, -1.5, 1.5);
    assert!(p_sum.both_unstable() && p_diff.both_unstable());

    // Correlated (same-sign) objective ⇒ both neurons pushed the SAME way ⇒ the
    // SUM bound binds. Anti-correlated ⇒ the DIFFERENCE bound binds.
    let correlated = [-1.0, -1.0]; // both want y high ⇒ sum_hi corner
    let anti = [-1.0, 1.0]; //          i high, j low ⇒ dif_hi corner
    assert!(
        score(&p_sum, 0, 1, &correlated) > score(&p_sum, 0, 1, &anti),
        "for a SUM-clipped octahedron a correlated objective must outrank an anti-correlated one"
    );
    assert!(
        score(&p_diff, 0, 1, &anti) > score(&p_diff, 0, 1, &correlated),
        "for a DIFFERENCE-clipped octahedron an anti-correlated objective must outrank a correlated one"
    );
}

/// (a) GATE-OFF byte-identical: with `binding_coeffs = None` the kept pairs are
/// ranked EXACTLY by `excluded_corner_score` (the unchanged heuristic) — the
/// stored ranking score equals the recomputed geometric score and the sequence is
/// sorted descending. No objective ever enters the default path.
#[test]
fn binding_select_gate_off_is_unchanged_heuristic_ranking() {
    let (graph, input, alpha, nb) = binding_select_head_fixture();

    let heuristic = root_inject::build_pool_pairs_for_test(
        &graph, &input, &alpha, &nb, "r1", "l1", None, false,
    );
    assert!(
        heuristic.len() >= 3,
        "fixture should yield several coupling pairs, got {}",
        heuristic.len()
    );

    // The kept score IS the excluded-corner heuristic: rebuild each returned pair's
    // octahedron and compare (byte-identical selection key).
    let pairs: Vec<(usize, usize)> = heuristic.iter().map(|&(i, j, _)| (i, j)).collect();
    let octs = combined_rows_octahedra(&graph, &input, &alpha, Some(&nb), "l1", &pairs, None)
        .expect("producer backward");
    for (&(i, j, score), p) in heuristic.iter().zip(octs.iter()) {
        let geo = p.excluded_corner_score();
        assert!(
            (score - geo).abs() <= 1e-9,
            "gate-off ranking key must be excluded_corner_score (pair ({i},{j}): stored={score} geo={geo})"
        );
    }
    // Sorted strictly by the heuristic score, descending (unchanged sort).
    for w in heuristic.windows(2) {
        assert!(
            w[0].2 >= w[1].2,
            "gate-off order must be excluded-corner descending: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
}

/// (b) GATE-ON validity: the objective-gradient-informed selection (1) actually
/// REORDERS the pairs vs the heuristic (it is live), and (2) every facet built
/// over a binding-SELECTED pair still ENCLOSES all reachable lifted points — the
/// same soundness oracle as gate_a / the head-resident Monte-Carlo oracle. A valid
/// facet stays valid regardless of which pair selection chose it.
#[test]
fn binding_select_gate_on_produces_valid_facets() {
    let (graph, input, alpha, nb) = binding_select_head_fixture();
    let head_width = 5usize;

    // Real objective coeffs for the binding worst-child row (argmin baseline
    // lower). Two objectives; row 0 is made the binding one (smaller lower bound).
    let objectives = vec![vec![1.0_f32, -0.5, 0.3], vec![-0.2_f32, 0.7, 0.4]];
    let baseline = vec![(-3.0_f32, 2.0_f32), (-1.0_f32, 2.0_f32)];
    let coeffs = root_inject::head_objective_coeffs_for_test(
        &graph,
        "r1",
        head_width,
        &objectives,
        &baseline,
    )
    .expect("output Linear consumer must resolve for the dense head");
    assert_eq!(coeffs.len(), head_width, "one coeff per head neuron");
    // Sanity: c_k = Σ_o spec0[o]·W_out[o,k] for the binding row (row 0).
    // W_out row-major [3×5]; spec0 = [1.0, -0.5, 0.3].
    let expect_c0 = 1.0 * 1.0 + (-0.5) * (-0.7) + 0.3 * 0.5; // neuron 0
    assert!(
        (coeffs[0] - expect_c0).abs() < 1e-5,
        "objective coeff must be spec·W_out column: got {} want {expect_c0}",
        coeffs[0]
    );

    let binding = root_inject::build_pool_pairs_for_test(
        &graph,
        &input,
        &alpha,
        &nb,
        "r1",
        "l1",
        Some(&coeffs),
        false,
    );
    assert!(!binding.is_empty(), "binding selection must keep pairs");

    // VALID: every facet over every binding-SELECTED pair encloses all reachable
    // lifted points (x_i, x_j, relu(x_i), relu(x_j)) with x = l1(u), u in the box.
    let w1 = [
        [0.9_f64, -0.7, 0.5, -0.3],
        [-0.6, 0.8, -0.4, 0.7],
        [0.4, 0.5, -0.9, 0.2],
        [-0.8, -0.3, 0.6, 0.5],
        [0.7, 0.6, 0.5, -0.8],
    ];
    let b1 = [0.05_f64, -0.1, 0.15, -0.05, 0.0];
    let pre_of = |u: &[f64; 4], n: usize| -> f64 {
        let mut s = b1[n];
        for k in 0..4 {
            s += w1[n][k] * u[k];
        }
        s
    };

    let sel_pairs: Vec<(usize, usize)> = binding.iter().map(|&(i, j, _)| (i, j)).collect();
    let octs = combined_rows_octahedra(&graph, &input, &alpha, Some(&nb), "l1", &sel_pairs, None)
        .expect("producer backward");

    // LIVE + correct: the stored ranking key IS the objective-gradient-informed
    // score (not the geometric one), and the kept order is sorted by it descending.
    let mut prev = f64::INFINITY;
    for (&(i, j, stored), p) in binding.iter().zip(octs.iter()) {
        let recomputed = root_inject::binding_rank_score_for_test(p, i, j, &coeffs);
        assert!(
            (stored - recomputed).abs() <= 1e-9,
            "gate-on ranking key must be the objective-informed score (pair ({i},{j}): stored={stored} recomputed={recomputed})"
        );
        assert!(
            stored <= prev + 1e-12,
            "binding order must be objective-score descending: {stored} after {prev}"
        );
        prev = stored;
    }
    // The top binding pair actually carries objective weight on BOTH neurons.
    let (ti, tj, _) = binding[0];
    assert!(
        coeffs[ti].abs() > 1e-6 && coeffs[tj].abs() > 1e-6,
        "top binding pair ({ti},{tj}) must have nonzero objective coeffs on both neurons"
    );

    let mut rng = Xorshift::new(0x0511_D1C7);
    let mut worst = f64::NEG_INFINITY;
    let mut total_facets = 0usize;
    for (&(i, j), p) in sel_pairs.iter().zip(octs.iter()) {
        let facets = coupling_facets(p);
        total_facets += facets.len();
        for _ in 0..20_000 {
            let u = [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ];
            let xi = pre_of(&u, i);
            let xj = pre_of(&u, j);
            let wv = [xi, xj, relu(xi), relu(xj)];
            for f in &facets {
                worst = worst.max(f.residual(&wv));
            }
        }
    }
    assert!(total_facets > 0, "binding-selected pairs must carry facets");
    assert!(
        worst <= ENCLOSE_TOL,
        "binding-SELECTED facet EXCLUDED a reachable point (worst residual {worst:+.3e} > {ENCLOSE_TOL:.0e}) — selection cannot break facet validity"
    );
    eprintln!(
        "[mn-binding-select oracle-b] {} facets over {} binding-selected pairs; worst enclosure residual={worst:+.3e} (<= {ENCLOSE_TOL:.0e})",
        total_facets,
        sel_pairs.len()
    );
}

// ===========================================================================
// #mn-head-f64-certified-measure (NY_MN_HEAD_F64_CERTIFIED_MEASURE) — the dark,
// default-OFF RESEARCH-MEASUREMENT lane: apply EXACT-certified k=2 head coupling
// facets (`certified_coupling_facets_exact`) through the sound CPU-f64 masked head
// fold, to MEASURE (soundly) whether a well-selected facet lifts the cifar margin.
// It is a SEPARATE gate from the quarantined production levers; f64-only; monotone
// max ⇒ can only RAISE the bound. Two oracles:
//  (1) gate-OFF (`use_certified = false`) is byte-identical to the pre-measurement
//      path — same kept pairs, heuristic ranking, `coupling_facets` source — and
//      the research gate is false when its var is absent AND when the OLD production
//      vars (NY_MN_HEAD_FACET=1, …) are set (it is not their re-authorization);
//  (2) gate-ON (`use_certified = true`) draws ONLY certifier-approved facets into
//      the pool (each is the ExactRelu2Support certificate for its own normal) and
//      those facets ENCLOSE all reachable lifted ReLU2 points (20k-sample oracle,
//      worst residual ≤ ENCLOSE_TOL) — the soundness proof for the measurement.
// ===========================================================================

/// (1) GATE-OFF byte-identical: the research gate defaults false (and stays false
/// even with the OLD production vars armed — it is a distinct measurement gate, not
/// their re-authorization), and `build_pool_for_node(use_certified = false)`
/// reproduces the pre-measurement path (heuristic `excluded_corner_score` ranking,
/// `coupling_facets` source) exactly.
#[test]
fn head_f64_certified_measure_gate_off_is_byte_identical() {
    // The research gate is false when its var is ABSENT, even with every OLD
    // production/multi-neuron authority var set (it does not ride them). And it
    // arms ONLY under its own var — a genuinely separate measurement gate.
    ny_test_utils::env::with_env_edits(|env| {
        env.set("NY_MULTINEURON", "1");
        env.set("NY_MULTINEURON_STEM", "1");
        env.set("NY_MN_HEAD_RESIDENT", "1");
        env.set("NY_MN_HEAD_FACET", "1");
        env.remove("NY_MN_HEAD_F64_CERTIFIED_MEASURE");
        assert!(
            !root_inject::head_f64_certified_measure_enabled(),
            "research measure gate must stay OFF when its var is absent (old vars do not grant it)"
        );
        env.set("NY_MN_HEAD_F64_CERTIFIED_MEASURE", "1");
        assert!(
            root_inject::head_f64_certified_measure_enabled(),
            "research measure gate must arm under its OWN var"
        );
    });

    // The gate-off pool build (`use_certified = false`) is byte-identical to the
    // pre-measurement path: same kept pairs, ranked by the heuristic
    // excluded-corner score, sourced from `coupling_facets`. (The build path never
    // reads the env var — `use_certified` is threaded explicitly — so this holds
    // regardless of process env.)
    let (graph, input, alpha, nb) = binding_select_head_fixture();
    let off = root_inject::build_pool_octahedra_for_test(
        &graph, &input, &alpha, &nb, "r1", "l1", None, false,
    );
    let heuristic = root_inject::build_pool_pairs_for_test(
        &graph, &input, &alpha, &nb, "r1", "l1", None, false,
    );
    assert!(!off.is_empty(), "gate-off pool must keep coupling pairs");
    assert_eq!(
        off.len(),
        heuristic.len(),
        "gate-off kept-pair count must match the heuristic path"
    );
    for ((i, j, p), &(hi, hj, hs)) in off.iter().zip(heuristic.iter()) {
        assert_eq!(
            (*i, *j),
            (hi, hj),
            "gate-off must keep the SAME pairs in order"
        );
        // Heuristic ranking key: the objective-blind excluded-corner score.
        let geo = p.excluded_corner_score();
        assert!(
            (hs - geo).abs() <= 1e-9,
            "gate-off ranking key must be excluded_corner_score (pair ({i},{j}): stored={hs} geo={geo})"
        );
        // Facet SOURCE is the legacy `coupling_facets` producer (non-empty for a
        // kept coupling pair) — NOT the exact certifier.
        assert!(
            !coupling_facets(p).is_empty(),
            "gate-off facet source must be coupling_facets (non-empty for kept pair ({i},{j}))"
        );
    }
    eprintln!(
        "[mn-head-f64-certified-measure oracle-1] gate-off byte-identical: {} kept pairs via heuristic + coupling_facets",
        off.len()
    );
}

/// (2) GATE-ON soundness: `build_pool_for_node(use_certified = true)` populates the
/// pool from `certified_coupling_facets_exact` ONLY. Every such facet is the
/// ExactRelu2Support certificate for its own stored normal (certifier-approved), and
/// the certified facets ENCLOSE all reachable lifted ReLU2 points (x_i, x_j,
/// relu(x_i), relu(x_j)) with x = l1(u), u in the box — a 20k-sample enclosure
/// oracle, worst residual ≤ ENCLOSE_TOL. This is the soundness proof: the masked
/// f64 fold only ever `max`-intersects proven-superset half-spaces, so the measured
/// tightening can only RAISE the certified lower bound, never over-claim.
#[test]
fn head_f64_certified_measure_produces_only_certified_facets() {
    let (graph, input, alpha, nb) = binding_select_head_fixture();

    // The pairs the CERTIFIED pool actually kept (a pair is kept only if the exact
    // certifier yielded ≥1 pushable facet), with their producer octahedra.
    let kept = root_inject::build_pool_octahedra_for_test(
        &graph, &input, &alpha, &nb, "r1", "l1", None, true,
    );
    assert!(
        !kept.is_empty(),
        "certified measure pool must keep at least one exact-certified coupling pair"
    );

    // l1 weights/bias (matches `binding_select_head_fixture`) for the true forward.
    let w1 = [
        [0.9_f64, -0.7, 0.5, -0.3],
        [-0.6, 0.8, -0.4, 0.7],
        [0.4, 0.5, -0.9, 0.2],
        [-0.8, -0.3, 0.6, 0.5],
        [0.7, 0.6, 0.5, -0.8],
    ];
    let b1 = [0.05_f64, -0.1, 0.15, -0.05, 0.0];
    let pre_of = |u: &[f64; 4], n: usize| -> f64 {
        let mut s = b1[n];
        for k in 0..4 {
            s += w1[n][k] * u[k];
        }
        s
    };

    let mut rng = Xorshift::new(0x000C_E271_F1ED);
    let mut worst = f64::NEG_INFINITY;
    let mut total_facets = 0usize;
    for (i, j, p) in &kept {
        // The EXACT source `build_pool_for_node(use_certified=true)` drew from.
        let certified = certified_coupling_facets_exact(p);
        assert!(
            !certified.is_empty(),
            "a kept certified pair ({i},{j}) must recompute a non-empty certified facet set"
        );
        total_facets += certified.len();

        // CERTIFIER-APPROVED: every stored facet is exactly the ExactRelu2Support
        // certificate for its own normal (the exact-support repair seam).
        let checker = ExactRelu2Support::new(p)
            .expect("both-unstable producer octahedron has an exact support checker");
        for f in &certified {
            assert_eq!(
                checker.certify_normal(f.a),
                Some(*f),
                "every pool facet must be the exact-support certificate for its stored normal (pair ({i},{j}))"
            );
        }

        // ENCLOSURE: every certified facet holds at every reachable lifted point.
        for _ in 0..20_000 {
            let u = [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ];
            let xi = pre_of(&u, *i);
            let xj = pre_of(&u, *j);
            let wv = [xi, xj, relu(xi), relu(xj)];
            for f in &certified {
                worst = worst.max(f.residual(&wv));
            }
        }
    }
    assert!(total_facets > 0, "certified measure pool must carry facets");
    assert!(
        worst <= ENCLOSE_TOL,
        "certified-measure facet EXCLUDED a reachable lifted point (worst residual {worst:+.3e} > {ENCLOSE_TOL:.0e}) — the measurement lane would be UNSOUND"
    );
    eprintln!(
        "[mn-head-f64-certified-measure oracle-2] {} certifier-approved facets over {} kept pairs; worst enclosure residual={worst:+.3e} (<= {ENCLOSE_TOL:.0e}) — sound measurement",
        total_facets,
        kept.len()
    );
}
