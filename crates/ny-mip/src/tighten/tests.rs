// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_tensor::next_down_f32;

/// Helper: build a 2-input -> 3-hidden (ReLU) -> 2-output network.
///
/// Layer 0: W = [[1, 0], [0, 1], [1, 1]], b = [0, 0, -1]
/// Layer 1: W = [[1, -1, 0], [0, 1, 1]], b = [0, 0]
fn small_network() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
    let weights = vec![
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
    ];
    let biases = vec![vec![0.0, 0.0, -1.0], vec![0.0, 0.0]];
    let layer_dims = vec![2, 3, 2];
    (weights, biases, layer_dims)
}

fn make_tightener(
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    layer_dims: Vec<usize>,
    input_bounds: Vec<Bound>,
    intermediate_bounds: Vec<Vec<Bound>>,
) -> LpTightener {
    LpTightener::new(
        weights,
        biases,
        layer_dims,
        input_bounds,
        intermediate_bounds,
        MipConfig {
            timeout_secs: 10.0,
            ..Default::default()
        },
    )
}

#[test]
fn hard_deadline_caps_lp_slices_and_expired_phase_is_noop() {
    use std::time::{Duration, Instant};

    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let current = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ];
    let make = |deadline| {
        LpTightener::new(
            weights.clone(),
            biases.clone(),
            layer_dims.clone(),
            input_bounds.clone(),
            vec![current.clone()],
            MipConfig {
                timeout_secs: 100.0,
                ay_hard_deadline: Some(deadline),
                ..MipConfig::default()
            },
        )
    };

    let live = make(Instant::now() + Duration::from_secs(2));
    assert!(
        live.live_per_neuron_timeout_secs()
            .is_some_and(|secs| secs <= 2.0),
        "the per-neuron slice must stay inside the absolute phase deadline"
    );

    let expired = make(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("one millisecond is representable"),
    );
    assert_eq!(expired.live_per_neuron_timeout_secs(), None);
    let (unchanged, newly_stable) = expired
        .tighten_layer(0, &current)
        .expect("an expired tightening phase should decline cleanly");
    assert_eq!(unchanged, current);
    assert_eq!(newly_stable, 0);
}

#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_layer_basic() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    let tightener = make_tightener(
        weights,
        biases,
        layer_dims,
        input_bounds,
        intermediate_bounds.clone(),
    );

    let (tightened, _newly_stable) = tightener
        .tighten_layer(0, &intermediate_bounds[0])
        .expect("tightening should succeed");

    assert_eq!(tightened.len(), 3);
    // Stable neurons unchanged
    assert_eq!(tightened[0].lower(), 0.0);
    assert_eq!(tightened[0].upper(), 1.0);
    assert_eq!(tightened[1].lower(), 0.0);
    assert_eq!(tightened[1].upper(), 1.0);
    // Unstable neuron: LP finds exact bounds [-1, 1] (single linear layer)
    assert!(
        (tightened[2].lower() - (-1.0)).abs() < 1e-5,
        "lb: {}",
        tightened[2].lower()
    );
    assert!(
        (tightened[2].upper() - 1.0).abs() < 1e-5,
        "ub: {}",
        tightened[2].upper()
    );
}

#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_proves_stability() {
    // y = x + 2, input [0,1]. True bounds [2,3]. Give loose [-1,3].
    let tightener = make_tightener(
        vec![vec![1.0], vec![1.0]],
        vec![vec![2.0], vec![0.0]],
        vec![1, 1, 1],
        vec![Bound::new(0.0, 1.0)],
        vec![vec![Bound::new(2.0, 3.0)]],
    );

    let (tightened, newly_stable) = tightener
        .tighten_layer(0, &[Bound::new(-1.0, 3.0)])
        .expect("tightening should succeed");

    assert!(tightened[0].lower() >= 1.99, "got {}", tightened[0].lower());
    assert!(tightened[0].upper() <= 3.01, "got {}", tightened[0].upper());
    assert_eq!(newly_stable, 1, "neuron should become stable");
}

#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_multi_layer() {
    // 2 -> 2 (ReLU) -> 2 (ReLU) -> 1. Tighten layer 1.
    let layer0_bounds = vec![Bound::new(-0.5, 0.5), Bound::new(0.0, 1.0)];
    let layer1_bounds = vec![Bound::new(-1.0, 2.0), Bound::new(-2.0, 1.0)];

    let tightener = make_tightener(
        vec![
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, -1.0],
            vec![1.0, 1.0],
        ],
        vec![vec![-0.5, 0.0], vec![0.0, -0.5], vec![0.0]],
        vec![2, 2, 2, 1],
        vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)],
        vec![layer0_bounds, layer1_bounds.clone()],
    );

    let (tightened, _) = tightener
        .tighten_layer(1, &layer1_bounds)
        .expect("tightening should succeed");

    assert_eq!(tightened.len(), 2);
    for (i, (t, orig)) in tightened.iter().zip(layer1_bounds.iter()).enumerate() {
        assert!(
            t.lower() >= orig.lower() - 1e-6,
            "neuron {i}: tightened lower {} < original {}",
            t.lower(),
            orig.lower()
        );
        assert!(
            t.upper() <= orig.upper() + 1e-6,
            "neuron {i}: tightened upper {} > original {}",
            t.upper(),
            orig.upper()
        );
    }
}

#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_skips_stable_neurons() {
    let tightener = make_tightener(
        vec![vec![1.0], vec![1.0]],
        vec![vec![5.0], vec![0.0]],
        vec![1, 1, 1],
        vec![Bound::new(0.0, 1.0)],
        vec![vec![Bound::new(5.0, 6.0)]],
    );

    let (tightened, newly_stable) = tightener
        .tighten_layer(0, &[Bound::new(5.0, 6.0)])
        .expect("tightening should succeed");

    assert_eq!(tightened[0].lower(), 5.0);
    assert_eq!(tightened[0].upper(), 6.0);
    assert_eq!(newly_stable, 0, "already-stable neurons don't count");
}

#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_respects_max_per_layer() {
    let (weights, biases, layer_dims) = small_network();
    let input_bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)];
    let intermediate_bounds = vec![vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
    ]];

    let tightener = LpTightener::new(
        weights,
        biases,
        layer_dims,
        input_bounds,
        intermediate_bounds.clone(),
        MipConfig {
            timeout_secs: 10.0,
            max_tighten_per_layer: 1,
            ..Default::default()
        },
    );

    let (tightened, _) = tightener
        .tighten_layer(0, &intermediate_bounds[0])
        .expect("tightening should succeed");

    let first_tightened = tightened[0].lower() != -1.0 || tightened[0].upper() != 1.0;
    assert!(first_tightened, "first unstable neuron should be tightened");
    // Remaining untouched
    assert_eq!(tightened[1].lower(), -1.0);
    assert_eq!(tightened[1].upper(), 1.0);
    assert_eq!(tightened[2].lower(), -1.0);
    assert_eq!(tightened[2].upper(), 1.0);
}

#[test]
#[ntest::timeout(30_000)]
fn test_bound_tightener_trait() {
    let tightener = make_tightener(
        vec![vec![1.0], vec![1.0]],
        vec![vec![2.0], vec![0.0]],
        vec![1, 1, 1],
        vec![Bound::new(0.0, 1.0)],
        vec![vec![Bound::new(2.0, 3.0)]],
    );

    let tightener: &dyn BoundTightener<Error = MipError> = &tightener;
    let result = tightener
        .tighten(0, &[Bound::new(-1.0, 4.0)])
        .expect("trait call should succeed");
    assert_eq!(result.len(), 1);
    assert!(result[0].lower() >= 1.99, "got {}", result[0].lower());
}

/// Regression test for #3340: f64→f32 directed rounding in LP tighten.
///
/// Uses a NEGATIVE midpoint so the lower bound stays < 0, avoiding the
/// early-exit at tighten_neuron:177 (`if new_lb >= 0.0`). This ensures
/// BOTH the lower-bound (minimize → next_down_f32) and upper-bound
/// (maximize → next_up_f32) paths are exercised.
#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_directed_rounding_midpoint() {
    // Use negative adjacent f32 values so the LP optimum is negative.
    // This avoids the early exit (new_lb >= 0 → return) and exercises both paths.
    let neg_upper: f32 = -1.0000001;
    let neg_lower = next_down_f32(neg_upper); // more negative neighbor
    let midpoint_f64 = f64::midpoint(neg_lower as f64, neg_upper as f64);

    // Verify midpoint is strictly between the two adjacent f32 values
    assert!(
        midpoint_f64 > neg_lower as f64 && midpoint_f64 < neg_upper as f64,
        "midpoint {midpoint_f64} not between {neg_lower} and {neg_upper}"
    );

    // Network: y = 1*x + bias, input fixed at [0, 0].
    // LP min = LP max = midpoint_f64 (constant function).
    let tightener = make_tightener(
        vec![vec![1.0], vec![1.0]],
        vec![vec![midpoint_f64], vec![0.0]],
        vec![1, 1, 1],
        vec![Bound::new(0.0, 0.0)],
        vec![vec![Bound::new(neg_lower, neg_upper)]],
    );

    // Wide initial bounds ensure neuron is unstable (lb < 0 < ub)
    let wide_bounds = vec![Bound::new(-10.0, 10.0)];
    let (tightened, _) = tightener
        .tighten_layer(0, &wide_bounds)
        .expect("tightening should succeed");

    let tightened_lb = tightened[0].lower();
    let tightened_ub = tightened[0].upper();

    // Lower bound must round DOWN (not exclude reachable states)
    assert!(
        (tightened_lb as f64) <= midpoint_f64,
        "SOUNDNESS: lb {tightened_lb} > true opt {midpoint_f64} (should use next_down_f32)"
    );
    // Upper bound must round UP (not exclude reachable states)
    // This assertion was trivially true before (returned original 10.0 via early exit).
    assert!(
        (tightened_ub as f64) >= midpoint_f64,
        "SOUNDNESS: ub {tightened_ub} < true opt {midpoint_f64} (should use next_up_f32)"
    );
    // Verify bounds are actually tightened (not the original wide bounds)
    assert!(tightened_lb > -10.0, "lower bound not tightened from -10.0");
    assert!(tightened_ub < 10.0, "upper bound not tightened from 10.0");
}

/// P3: the OBBT tighten path reaches the same SOUND result as the per-neuron
/// pass — it proves the `y = x + 2` neuron stable while never widening the
/// original box. Exercises `obbt_rounds > 0`.
#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_obbt_path_is_sound() {
    let obbt_tightener = |rounds: usize| {
        LpTightener::new(
            vec![vec![1.0], vec![1.0]],
            vec![vec![2.0], vec![0.0]],
            vec![1, 1, 1],
            vec![Bound::new(0.0, 1.0)],
            vec![vec![Bound::new(2.0, 3.0)]],
            MipConfig {
                timeout_secs: 10.0,
                obbt_rounds: rounds,
                ..Default::default()
            },
        )
    };
    let orig = Bound::new(-1.0, 3.0);
    let (tightened, newly_stable) = obbt_tightener(3)
        .tighten_layer(0, &[orig])
        .expect("OBBT tightening should succeed");
    // Same verdict as the single-pass path (test_lp_tighten_proves_stability).
    assert!(tightened[0].lower() >= 1.99, "got {}", tightened[0].lower());
    assert!(tightened[0].upper() <= 3.01, "got {}", tightened[0].upper());
    assert_eq!(newly_stable, 1, "OBBT must also prove the neuron stable");
    // Never widens the original box.
    assert!(tightened[0].lower() >= orig.lower() - 1e-6);
    assert!(tightened[0].upper() <= orig.upper() + 1e-6);
}

/// P3: the OBBT path tightens a COUPLED layer at least as much as the
/// single-pass path (OBBT is round-1 of independent min/max plus fixpoint
/// rounds, so it can only match or beat it — never loosen).
#[test]
#[ntest::timeout(30_000)]
fn test_lp_tighten_obbt_at_least_as_tight() {
    let build = |rounds: usize| {
        let (weights, biases, layer_dims) = small_network();
        LpTightener::new(
            weights,
            biases,
            layer_dims,
            vec![Bound::new(0.0, 1.0), Bound::new(0.0, 1.0)],
            vec![vec![
                Bound::new(-1.0, 1.0),
                Bound::new(-1.0, 1.0),
                Bound::new(-1.0, 1.0),
            ]],
            MipConfig {
                timeout_secs: 10.0,
                obbt_rounds: rounds,
                ..Default::default()
            },
        )
    };
    let start = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
    ];
    let (single, _) = build(0).tighten_layer(0, &start).expect("single pass");
    let (obbt, _) = build(3).tighten_layer(0, &start).expect("obbt pass");
    for (i, (o, s)) in obbt.iter().zip(single.iter()).enumerate() {
        // OBBT never wider than the single pass, and both stay sound (⊆ start).
        assert!(
            o.lower() >= s.lower() - 1e-6,
            "neuron {i}: OBBT lower {} looser than single-pass {}",
            o.lower(),
            s.lower()
        );
        assert!(
            o.upper() <= s.upper() + 1e-6,
            "neuron {i}: OBBT upper {} looser than single-pass {}",
            o.upper(),
            s.upper()
        );
        assert!(o.lower() >= start[i].lower() - 1e-6 && o.upper() <= start[i].upper() + 1e-6);
    }
}

/// `obbt_relaxation_bounds` tightens a COUPLED relaxation and RELAXES binaries.
///
/// Model: x ∈ [0,1]; z is an integer (binary) col; row `y - 2x = 0` couples a
/// free y ∈ [-100,100] to x, and row `y - 10z <= 0` with z relaxed to [0,1].
/// The reachable y is [0,2] (from 2x, x∈[0,1]); OBBT must return that box, far
/// tighter than the [-100,100] the column was declared with — proving the LP
/// coupling flows through `to_ay_model_relaxed` and the binary is relaxed (an
/// UNrelaxed z∈{0,1} would still give y∈[0,2] here, so the test also checks a
/// row that only the relaxation makes non-trivial: `z` itself optimizes to
/// [0,1] continuous, which an integer col would report identically — the load
/// bearing check is the y coupling).
#[test]
fn obbt_relaxation_tightens_coupled_box() {
    use crate::ir::MilpProblem;
    use std::time::{Duration, Instant};

    let mut p = MilpProblem::new();
    let x = p.add_col(0.0, 0.0, 1.0);
    let y = p.add_col(0.0, -100.0, 100.0);
    let z = p.add_integer_col(0.0, 0.0, 1.0);
    // y = 2x
    p.add_row(0.0, 0.0, [(y, 1.0), (x, -2.0)]);
    // y <= 10 z  (relaxed z in [0,1] keeps this a valid bound; with x=1 -> y=2 -> z>=0.2)
    p.add_row(f64::NEG_INFINITY, 0.0, [(y, 1.0), (z, -10.0)]);

    let report = obbt_relaxation_bounds(
        &p,
        &[y, z],
        4,
        Duration::from_secs(5),
        Instant::now() + Duration::from_secs(30),
        8,
    )
    .expect("obbt");
    assert!(!report.infeasible, "coupled relaxation is feasible");
    let (ylo, yhi) = report.bounds[0];
    // y reachable exactly [0,2]; OBBT must prove it (outward-rounded, so allow slack).
    assert!(
        (-1e-6..=1e-6).contains(&ylo),
        "y lower should be ~0, got {ylo}"
    );
    assert!(
        (2.0 - 1e-6..=2.0 + 1e-6).contains(&yhi),
        "y upper should be ~2, got {yhi}"
    );
    // The coupling genuinely tightened y from its declared [-100,100].
    assert!(report.tightened >= 1, "at least y must have tightened");
    // z relaxed to continuous stays within [0,1].
    let (zlo, zhi) = report.bounds[1];
    assert!(
        zlo >= -1e-6 && zhi <= 1.0 + 1e-6,
        "z stays in [0,1], got [{zlo},{zhi}]"
    );
}

/// PROPERTY-CONDITIONED OBBT mechanism: adding an OUTPUT/violation row to the
/// relaxation tightens an INTERMEDIATE column that the same relaxation WITHOUT
/// the row leaves wide. This is the LP-level proof that the whole-net finisher's
/// property-conditioning genuinely reaches back into the intermediates (and is
/// SOUND: the tightened box is still an outer bound over the conditioned region).
///
/// Model: pre ∈ [-2000, 2000] (a big-M-scale intermediate); ReLU triangle
/// post = relu(pre) with those bounds; out = post. Row `post - pre >= 0` (the
/// `y >= x` triangle facet) means `out >= pre`, so the violation row `out <= 5`
/// forces `pre <= 5`. Unconditioned, `pre`'s upper stays ~2000; conditioned it
/// collapses to ~5.
#[test]
fn obbt_conditioning_row_tightens_intermediate() {
    use crate::ir::MilpProblem;
    use std::time::{Duration, Instant};

    // Build the shared triangle relaxation of a single unstable ReLU with
    // pre ∈ [l,u] = [-2000, 2000], post = relu(pre), out = post.
    let build = |with_violation: bool| -> MilpProblem {
        let l = -2000.0;
        let u = 2000.0;
        let mut p = MilpProblem::new();
        let pre = p.add_col(0.0, l, u);
        let post = p.add_col(0.0, 0.0, u); // post in [0, u]
        let out = p.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
        // post >= pre  (triangle facet y >= x)
        p.add_row(0.0, f64::INFINITY, [(post, 1.0), (pre, -1.0)]);
        // post <= u(pre - l)/(u - l)  (upper envelope)
        let slope = u / (u - l);
        p.add_row(f64::NEG_INFINITY, -slope * l, [(post, 1.0), (pre, -slope)]);
        // out = post
        p.add_row(0.0, 0.0, [(out, 1.0), (post, -1.0)]);
        if with_violation {
            // The band-violation row: out <= 5 (the finisher's `Σ coeffs·y <= thr`).
            p.add_row(f64::NEG_INFINITY, 5.0, [(out, 1.0)]);
        }
        p
    };

    let pre_col = Col(0);
    let far = Instant::now() + Duration::from_secs(30);

    // Unconditioned: pre's upper stays ~u (only the forward range).
    let uncond =
        obbt_relaxation_bounds(&build(false), &[pre_col], 4, Duration::from_secs(5), far, 8)
            .expect("obbt uncond");
    let (_, uhi) = uncond.bounds[0];
    assert!(
        uhi > 1000.0,
        "unconditioned pre upper should stay wide, got {uhi}"
    );

    // Conditioned: the violation row propagates back to pre <= ~5.
    let cond = obbt_relaxation_bounds(&build(true), &[pre_col], 4, Duration::from_secs(5), far, 8)
        .expect("obbt cond");
    let (_, chi) = cond.bounds[0];
    assert!(
        chi <= 5.0 + 1e-6,
        "conditioned pre upper must collapse to ~5, got {chi}"
    );
    // SOUNDNESS: the conditioned upper is still an OUTER bound — pre = 5 is
    // reachable under out <= 5 (post = pre = 5, out = 5), so the bound did not
    // cut off a feasible point of the conditioned region.
    assert!(
        chi >= 5.0 - 1e-6,
        "conditioned pre upper must not undercut 5, got {chi}"
    );
}

/// An integer column outside the ReLU-binary contract is refused (fail-closed).
#[test]
fn obbt_relaxation_rejects_non_binary_integer() {
    use crate::ir::MilpProblem;
    use std::time::{Duration, Instant};
    let mut p = MilpProblem::new();
    let _x = p.add_col(0.0, 0.0, 1.0);
    let k = p.add_integer_col(0.0, 0.0, 5.0); // not [0,1], not pinned
    let err = obbt_relaxation_bounds(
        &p,
        &[k],
        1,
        Duration::from_secs(1),
        Instant::now() + Duration::from_secs(5),
        8,
    );
    assert!(err.is_err(), "non-binary integer column must be refused");
}
