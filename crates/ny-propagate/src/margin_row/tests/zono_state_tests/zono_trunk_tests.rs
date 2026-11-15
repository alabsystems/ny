// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#fd-shared-state` (ENGINE_PROGRAM M3) tests: the shared class-independent
//! trunk of the frozen dual state and the `margin_row.frozen_dual_bab_state`
//! tier mode. A child of `zono_state_tests` (its fixtures, exact-rational
//! claims and double-double checker) and a sibling of `zono_bab_tests` (its
//! sweep, problem and option helpers).
//!
//! Bit identity (the default must not move a bit):
//!
//! * `fd_trunk_build_refactor_is_bit_identical` — the refactored
//!   `FrozenDualState::build` against the FROZEN pre-M3 copy
//!   (the test-only `zono_state::build::pre_m3`), every field and every refusal;
//! * `fd_trunk_anchor_is_bit_identical_to_per_class_build` — a trunk on one
//!   class's own slopes reproduces that class's per-class state bit for bit;
//! * `fd_trunk_joint_of_one_class_is_its_own_margin`;
//! * `fd_trunk_per_class_tier_equals_the_direct_search` — the default tier is
//!   build + search, bit for bit;
//! * `fd_trunk_single_class_shared_tier_equals_per_class` — the whole shared
//!   path, end to end, on a one-class tier.
//!
//! Containment of shared-slope states: `fd_trunk_shared_state_contains_
//! exact_rational_samples` (exact rationals, fold-ball extremes) and
//! `fd_trunk_shared_state_contains_dd_samples`.
//!
//! The trunk/tail contract: `fd_trunk_rows_are_shared_across_classes`,
//! `fd_trunk_tail_layers_are_read_by_no_row`, and the mutation pins
//! `fd_trunk_refuses_a_tail_mask_that_a_row_reads`,
//! `fd_trunk_refuses_a_tail_that_changes_a_shared_slope`,
//! `fd_trunk_digest_refuses_corrupted_rows`,
//! `fd_trunk_refuses_a_tail_of_another_class_or_layout`.
//!
//! Fault injection: `fd_trunk_garbage_slopes_are_refused_or_stay_sound`
//! (invalid slopes / mu refused; random valid slopes stay contained and never
//! close a witnessed class).
//!
//! The tier: `fd_trunk_shared_tier_never_closes_a_witnessed_sat_class`,
//! `fd_trunk_shared_tier_planted_violation_is_never_closed`,
//! `fd_trunk_shared_tier_closed_classes_hold_at_sampled_points` (positive
//! control), `fd_trunk_shared_tier_reports_its_observables`,
//! `fd_trunk_shared_tier_is_thread_count_independent`, and the refusals /
//! dark default / byte accounting.

use super::zono_bab_tests::{opts, problem_of, sweep};
use super::*;

use crate::margin_row::bab::root_eval;
use crate::margin_row::spec::TwinOpSpec;
use crate::margin_row::zono_bab::{
    run_tier, search, FrozenDualStateMode, SearchReport, TierPolicy, TierReport,
    FROZEN_DUAL_SHARED_LOSS_TOL,
};
use crate::margin_row::zono_state::build::pre_m3::build_pre_m3;
use crate::margin_row::zono_state::{class_margin, tail_layers, FrozenDualTrunk, TrunkRelax};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Equal length and equal in every bit.
fn same(x: &[f64], y: &[f64]) -> bool {
    x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// The first field (build telemetry excluded) in which two states differ in
/// any bit, or `None`.
fn first_difference(a: &FrozenDualState, b: &FrozenDualState) -> Option<&'static str> {
    if a.t != b.t {
        return Some("t");
    }
    if a.class != b.class {
        return Some("class");
    }
    if a.n_gens != b.n_gens {
        return Some("n_gens");
    }
    if a.n_in_gens != b.n_in_gens {
        return Some("n_in_gens");
    }
    if a.c_obj.to_bits() != b.c_obj.to_bits() {
        return Some("c_obj");
    }
    if !same(&a.d, &b.d) {
        return Some("d");
    }
    if a.r_obj.to_bits() != b.r_obj.to_bits() {
        return Some("r_obj");
    }
    if !same(&a.a, &b.a) {
        return Some("a");
    }
    if !same(&a.c_in, &b.c_in) {
        return Some("c_in");
    }
    if !same(&a.r, &b.r) {
        return Some("r");
    }
    if !same(&a.lam, &b.lam) {
        return Some("lam");
    }
    if !same(&a.mu, &b.mu) {
        return Some("mu");
    }
    if !same(&a.lo, &b.lo) {
        return Some("lo");
    }
    if !same(&a.hi, &b.hi) {
        return Some("hi");
    }
    if a.enc != b.enc {
        return Some("enc");
    }
    if a.layer != b.layer {
        return Some("layer");
    }
    if a.site != b.site {
        return Some("site");
    }
    if a.neuron != b.neuron {
        return Some("neuron");
    }
    None
}

/// The trunk under the JOINT slopes of `classes`.
fn joint_trunk(f: &Fx, frame: &FrozenDualFrame, classes: &[usize]) -> FrozenDualTrunk {
    let slopes = TrunkRelax::joint(&f.net, frame, classes, None).expect("joint slopes");
    FrozenDualTrunk::build(&f.net, frame, slopes, &FrozenDualOptions::default()).expect("trunk")
}

/// The trunk under `class`'s OWN slopes.
fn anchor_trunk(f: &Fx, frame: &FrozenDualFrame, class: usize) -> FrozenDualTrunk {
    let own = class_margin(&f.net, frame, class, None).expect("own margin");
    FrozenDualTrunk::build(
        &f.net,
        frame,
        TrunkRelax::anchor(frame, &own),
        &FrozenDualOptions::default(),
    )
    .expect("anchor trunk")
}

/// A trunk whose every slope on layer `li` is `slope(li)` (a valid slope,
/// `mu` certified from the box).
fn trunk_with_slopes(
    f: &Fx,
    frame: &FrozenDualFrame,
    slope: impl Fn(usize) -> f64,
) -> FrozenDualTrunk {
    FrozenDualTrunk::build(
        &f.net,
        frame,
        TrunkRelax::from_slopes(frame, |li, _| slope(li), None),
        &FrozenDualOptions::default(),
    )
    .expect("a slope in [0, 1] is valid")
}

/// `class`'s state over `trunk`.
fn shared_state(
    f: &Fx,
    frame: &FrozenDualFrame,
    trunk: &FrozenDualTrunk,
    class: usize,
) -> FrozenDualState {
    let tail = trunk
        .relax()
        .tail(&f.net, frame, class, None)
        .expect("tail");
    trunk.state(frame, class, &tail).expect("shared state")
}

/// Layers with an unstable neuron whose slopes the trunk FIXES (some row may
/// read them).
fn shared_unstable_layers(frame: &FrozenDualFrame) -> Vec<usize> {
    let tail = tail_layers(frame);
    frame
        .layers
        .iter()
        .enumerate()
        .filter(|(li, layer)| !tail[*li] && !layer.unst.is_empty())
        .map(|(li, _)| li)
        .collect()
}

fn shared_policy(node_cap: u64) -> TierPolicy {
    TierPolicy::clamped(node_cap, 60.0).with_state(FrozenDualStateMode::Shared)
}

/// The double-double margin `y_t - y_j` at `x`.
fn dd_margin(net: &TwinNet, t: usize, j: usize, x: &[f64]) -> f64 {
    let vals = dd_forward(net, x).expect("dd forward");
    let logits = &vals[net.ops.len()];
    (logits[t] - logits[j]).to_f64()
}

fn midpoint(lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| f64::midpoint(*l, *h))
        .collect()
}

/// `(argmax, runner-up)` of the double-double logits at `x`.
fn top_two(net: &TwinNet, x: &[f64]) -> (usize, usize) {
    let vals = dd_forward(net, x).expect("dd forward");
    let logits: Vec<f64> = vals[net.ops.len()].iter().map(|v| v.to_f64()).collect();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    (idx[0], idx[1])
}

/// Everything a search decided, by bits.
fn search_digest(r: &SearchReport) -> String {
    format!(
        "verdict={:?} nodes={} depth={} prunes={} slack={:016x} far_used={} root={:016x} \
         levels={:?}",
        r.verdict,
        r.nodes,
        r.depth,
        r.certified_prunes,
        r.cert_min_slack.to_bits(),
        r.far_used,
        r.root_certified.to_bits(),
        r.levels
    )
}

/// Every class outcome of a tier, by bits.
fn tier_digest(rep: &TierReport) -> Vec<String> {
    rep.classes
        .iter()
        .map(|c| match &c.result {
            Ok(run) => format!(
                "class={} {} root_g={:016x} loss={:?} state={}",
                c.class,
                search_digest(&run.search),
                run.root_g.to_bits(),
                run.root_loss.map(f64::to_bits),
                run.state.token()
            ),
            Err(why) => format!("class={} refused={why:?}", c.class),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bit identity
// ---------------------------------------------------------------------------

#[test]
fn fd_trunk_build_refactor_is_bit_identical() {
    let mut compared = 0usize;
    let mut rows = 0usize;
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().take(4));
    for f in &fxs {
        let frame = frame_of(f);
        for &j in &f.adv {
            let opts = FrozenDualOptions::default();
            let new = FrozenDualState::build(&f.net, &frame, j, &opts).expect("refactored build");
            let old = build_pre_m3(&f.net, &frame, j, &opts).expect("pre-M3 build");
            assert_eq!(
                first_difference(&new, &old),
                None,
                "{} class {j}: the refactored per-class build moved a bit",
                f.name
            );
            assert_eq!(
                (
                    new.stats.passes,
                    new.stats.columns,
                    new.stats.estimate_bytes
                ),
                (
                    old.stats.passes,
                    old.stats.columns,
                    old.stats.estimate_bytes
                ),
                "{} class {j}: build telemetry",
                f.name
            );
            assert_eq!(new.root_bound().to_bits(), old.root_bound().to_bits());
            compared += 1;
            rows += new.num_unstable();
        }
        // Refusals: the same variant on the same inputs.
        let starved = FrozenDualOptions {
            max_state_bytes: 1,
            ..FrozenDualOptions::default()
        };
        let late = FrozenDualOptions {
            deadline: Some(std::time::Instant::now()),
            ..FrozenDualOptions::default()
        };
        for opts in [starved, late] {
            let new = FrozenDualState::build(&f.net, &frame, f.adv[0], &opts)
                .map(|_| ())
                .map_err(|e| e.token());
            let old = build_pre_m3(&f.net, &frame, f.adv[0], &opts)
                .map(|_| ())
                .map_err(|e| e.token());
            assert!(new.is_err(), "{}: the refusal was not exercised", f.name);
            assert_eq!(new, old, "{}: refusal parity", f.name);
        }
        let new = FrozenDualState::build(&f.net, &frame, f.t, &FrozenDualOptions::default())
            .map(|_| ())
            .map_err(|e| e.token());
        let old = build_pre_m3(&f.net, &frame, f.t, &FrozenDualOptions::default())
            .map(|_| ())
            .map_err(|e| e.token());
        assert!(new.is_err());
        assert_eq!(new, old, "{}: the true class refuses alike", f.name);
    }
    assert!(
        compared > 0 && rows > 0,
        "no state with an unstable row compared: VOID"
    );
}

#[test]
fn fd_trunk_anchor_is_bit_identical_to_per_class_build() {
    let mut compared = 0usize;
    let mut fixed_layers = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        fixed_layers += shared_unstable_layers(&frame).len();
        for &j in &f.adv {
            let reference = state_of(f, &frame, j);
            let own = class_margin(&f.net, &frame, j, None).expect("own margin");
            assert_eq!(own.c_obj.to_bits(), reference.c_obj.to_bits(), "{}", f.name);
            assert!(same(&own.d, &reference.d), "{}: own margin row", f.name);
            assert_eq!(own.r_obj.to_bits(), reference.r_obj.to_bits(), "{}", f.name);
            assert_eq!(
                own.root_bound().to_bits(),
                reference.root_bound().to_bits(),
                "{} class {j}: the own margin's root is the per-class root",
                f.name
            );
            let trunk = anchor_trunk(f, &frame, j);
            let tail = trunk
                .relax()
                .tail(&f.net, &frame, j, None)
                .expect("anchor tail");
            assert_eq!(
                tail.root_bound().to_bits(),
                own.root_bound().to_bits(),
                "{} class {j}: the anchor class loses nothing",
                f.name
            );
            let shared = trunk.state(&frame, j, &tail).expect("anchor state");
            assert_eq!(
                first_difference(&shared, &reference),
                None,
                "{} class {j}: trunk + tail on the class's own slopes is not its per-class state",
                f.name
            );
            assert_eq!(shared.root_bound().to_bits(), tail.root_bound().to_bits());
            compared += 1;
        }
    }
    assert!(compared > 0, "VOID");
    assert!(
        fixed_layers > 0,
        "no fixture has a shared (non-tail) unstable layer, so no slope was ever fixed: VOID"
    );
}

#[test]
fn fd_trunk_joint_of_one_class_is_its_own_margin() {
    let mut compared = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        let j = f.adv[0];
        let own = class_margin(&f.net, &frame, j, None).expect("own margin");
        let joint = TrunkRelax::joint(&f.net, &frame, &[j], None).expect("joint of one");
        let tail = joint.tail(&f.net, &frame, j, None).expect("tail");
        assert_eq!(tail.c_obj.to_bits(), own.c_obj.to_bits(), "{}", f.name);
        assert!(same(&tail.d, &own.d), "{}: margin row", f.name);
        assert_eq!(tail.r_obj.to_bits(), own.r_obj.to_bits(), "{}", f.name);
        let trunk = FrozenDualTrunk::build(&f.net, &frame, joint, &FrozenDualOptions::default())
            .expect("trunk");
        let st = trunk.state(&frame, j, &tail).expect("state");
        assert_eq!(
            first_difference(&st, &state_of(f, &frame, j)),
            None,
            "{}: a one-class joint trunk is not the per-class state",
            f.name
        );
        compared += 1;
    }
    assert!(compared > 0, "VOID");
}

#[test]
fn fd_trunk_per_class_tier_equals_the_direct_search() {
    let mut compared = 0usize;
    let mut searched = 0u64;
    for f in sweep().iter().take(8) {
        let frame = frame_of(f);
        let dj = vec![f64::NAN; f.adv.len()];
        let policy = TierPolicy::clamped(4_000, 60.0);
        assert_eq!(policy.state, FrozenDualStateMode::PerClass);
        let tier = run_tier(
            &f.net, &f.root, f.t, &f.adv, &dj, &f.adv, &f.ybox, None, &policy,
        );
        assert_eq!(tier.state, FrozenDualStateMode::PerClass);
        assert_eq!(
            tier.trunk_ms, 0.0,
            "{}: the per-class tier built a trunk",
            f.name
        );
        assert_eq!(tier.trunk_lambda, "-");
        assert_eq!(tier.shared_root_loss_max, None);
        for report in &tier.classes {
            let run = report.result.as_ref().expect("per-class state built");
            assert_eq!(run.state, FrozenDualStateMode::PerClass);
            assert_eq!(run.root_loss, None);
            let st = state_of(f, &frame, report.class);
            assert_eq!(
                run.root_g.to_bits(),
                st.root_bound().to_bits(),
                "{}",
                f.name
            );
            let direct = search(&problem_of(&st), &opts(4_000, 0));
            assert_eq!(
                search_digest(&run.search),
                search_digest(&direct),
                "{} class {}: the default tier diverged from build + search",
                f.name,
                report.class
            );
            searched += direct.nodes;
            compared += 1;
        }
        assert_eq!(tier.classes_built, tier.classes.len());
    }
    assert!(compared > 0 && searched > 0, "no class was searched: VOID");
}

#[test]
fn fd_trunk_single_class_shared_tier_equals_per_class() {
    let mut compared = 0usize;
    let mut nodes = 0u64;
    for f in sweep().iter().take(8) {
        let dj = vec![f64::NAN; f.adv.len()];
        for &j in &f.adv {
            let open = [j];
            let per = run_tier(
                &f.net,
                &f.root,
                f.t,
                &f.adv,
                &dj,
                &open,
                &f.ybox,
                None,
                &TierPolicy::clamped(4_000, 60.0),
            );
            let shared = run_tier(
                &f.net,
                &f.root,
                f.t,
                &f.adv,
                &dj,
                &open,
                &f.ybox,
                None,
                &shared_policy(4_000),
            );
            assert_eq!(shared.state, FrozenDualStateMode::Shared);
            assert_ne!(
                shared.trunk_lambda, "-",
                "{}: no trunk was built: VOID",
                f.name
            );
            let p = per.classes[0].result.as_ref().expect("per-class run");
            let s = shared.classes[0].result.as_ref().expect("shared run");
            assert_eq!(s.state, FrozenDualStateMode::Shared);
            assert_eq!(
                s.root_loss.map(f64::to_bits),
                Some(0.0f64.to_bits()),
                "{} class {j}: one class shares its own slopes",
                f.name
            );
            assert_eq!(
                p.root_g.to_bits(),
                s.root_g.to_bits(),
                "{} class {j}",
                f.name
            );
            assert_eq!(
                search_digest(&p.search),
                search_digest(&s.search),
                "{} class {j}: the one-class shared tier diverged from the per-class tier",
                f.name
            );
            assert_eq!(per.closed, shared.closed);
            nodes += s.search.nodes;
            compared += 1;
        }
    }
    assert!(
        compared > 0 && nodes > 0,
        "no shared class was searched: VOID"
    );
}

// ---------------------------------------------------------------------------
// Containment of shared-slope states
// ---------------------------------------------------------------------------

#[test]
fn fd_trunk_shared_state_contains_exact_rational_samples() {
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for f in &fixtures() {
        let frame = frame_of(f);
        let joint = joint_trunk(f, &frame, &f.adv);
        let hardest = *f.adv.last().expect("an adversarial class");
        let anchor = anchor_trunk(f, &frame, hardest);
        for (which, trunk) in [("joint", &joint), ("anchor", &anchor)] {
            for &j in f.adv.iter().take(2) {
                let st = shared_state(f, &frame, trunk, j);
                let mut rng = StdRng::seed_from_u64(0x7A11 ^ (j as u64) << 5);
                for s in 0..6 {
                    let x = sample(&mut rng, &f.lo, &f.hi, s);
                    for p in [Pert::Stored, Pert::Up, Pert::Mixed(s as u64 + 3)] {
                        let bad = exact_claims(f, &frame, &st, &x, p);
                        checked += 1;
                        if !bad.is_empty() {
                            failures.push(format!(
                                "{} {which} class {j} sample {s} {p:?}: {bad:?}",
                                f.name
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0, "VOID");
    assert!(
        failures.is_empty(),
        "SHARED-TRUNK CONTAINMENT FAILED (a false-unsat generator): {failures:#?}"
    );
}

#[test]
fn fd_trunk_shared_state_contains_dd_samples() {
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for f in &fixtures() {
        let frame = frame_of(f);
        let trunk = joint_trunk(f, &frame, &f.adv);
        for &j in f.adv.iter().take(3) {
            let st = shared_state(f, &frame, &trunk, j);
            let mut rng = StdRng::seed_from_u64(0xDD5 ^ (j as u64) << 8);
            for s in 0..200 {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                let pc = check_point(&f.net, &frame, &st, &x).expect("check");
                checked += 1;
                if !pc.holds() {
                    failures.push(format!("{} class {j} sample {s}: {pc:?}", f.name));
                }
            }
        }
    }
    assert!(checked > 0, "VOID");
    assert!(
        failures.is_empty(),
        "SHARED-TRUNK CONTAINMENT FAILED (double-double): {failures:#?}"
    );
}

// ---------------------------------------------------------------------------
// The trunk / tail contract
// ---------------------------------------------------------------------------

#[test]
fn fd_trunk_rows_are_shared_across_classes() {
    let mut pairs = 0usize;
    let mut differing_margins = 0usize;
    for f in &fixtures() {
        if f.adv.len() < 2 {
            continue;
        }
        let frame = frame_of(f);
        let tail = tail_layers(&frame);
        let trunk = joint_trunk(f, &frame, &f.adv);
        let states: Vec<FrozenDualState> = f
            .adv
            .iter()
            .map(|&j| shared_state(f, &frame, &trunk, j))
            .collect();
        let first = &states[0];
        assert!(
            same(trunk.rows_for_test(), &first.a),
            "{}: a class state must carry the trunk's rows",
            f.name
        );
        for st in &states[1..] {
            assert!(
                same(&st.a, &first.a) && same(&st.c_in, &first.c_in) && same(&st.r, &first.r),
                "{}: the rows differ between classes",
                f.name
            );
            assert!(same(&st.lo, &first.lo) && same(&st.hi, &first.hi));
            assert_eq!(st.enc, first.enc);
            assert_eq!(st.layer, first.layer);
            assert_eq!(st.neuron, first.neuron);
            for k in 0..st.num_unstable() {
                if !tail[st.layer[k]] {
                    assert_eq!(
                        (st.lam[k].to_bits(), st.mu[k].to_bits()),
                        (first.lam[k].to_bits(), first.mu[k].to_bits()),
                        "{} row {k}: a shared slope differs between classes",
                        f.name
                    );
                }
            }
            if !same(&st.d, &first.d) {
                differing_margins += 1;
            }
            pairs += 1;
        }
    }
    assert!(pairs > 0, "VOID");
    assert!(
        differing_margins > 0,
        "every class got the same margin row: the tails are not per class: VOID"
    );
}

#[test]
fn fd_trunk_tail_layers_are_read_by_no_row() {
    let mut tail_perturbed = 0usize;
    let mut dependence_seen = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        let tail = tail_layers(&frame);
        // The index rule itself.
        for (li, layer) in frame.layers.iter().enumerate() {
            if tail[li] {
                for m in &frame.layers {
                    assert!(
                        m.unst.is_empty() || layer.op >= m.input,
                        "{}: tail layer {li} precedes an unstable layer's input",
                        f.name
                    );
                }
            }
        }
        let base = trunk_with_slopes(f, &frame, |_| 0.375);
        // Every TAIL slope moved: the rows must not move a bit.
        let tail_moved = trunk_with_slopes(f, &frame, |li| if tail[li] { 0.8125 } else { 0.375 });
        assert!(
            same(base.rows_for_test(), tail_moved.rows_for_test()),
            "{}: a tail slope moved a row",
            f.name
        );
        if frame
            .layers
            .iter()
            .enumerate()
            .any(|(li, layer)| tail[li] && !layer.unst.is_empty())
        {
            tail_perturbed += 1;
        }
        // Every SHARED slope moved: some row must move (the check can see a
        // dependence at all).
        let shared_moved = trunk_with_slopes(f, &frame, |li| if tail[li] { 0.375 } else { 0.8125 });
        if !same(base.rows_for_test(), shared_moved.rows_for_test()) {
            dependence_seen += 1;
        }
    }
    assert!(
        tail_perturbed > 0,
        "no fixture has an unstable tail layer: VOID"
    );
    assert!(
        dependence_seen > 0,
        "no row depends on a shared slope, so the invariance proves nothing: VOID"
    );
}

#[test]
fn fd_trunk_refuses_a_tail_mask_that_a_row_reads() {
    // MUTATION PIN of the runtime guard: a mask that calls every layer a tail
    // layer lets each class re-choose slopes the rows were built with; the
    // row passes must refuse the moment one reaches such a layer.
    let mut refused = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        if shared_unstable_layers(&frame).is_empty() {
            continue;
        }
        let own = class_margin(&f.net, &frame, f.adv[0], None).expect("own margin");
        let lying =
            TrunkRelax::anchor(&frame, &own)
                .with_tail_mask_for_test(vec![true; frame.layers.len()]);
        match FrozenDualTrunk::build(&f.net, &frame, lying, &FrozenDualOptions::default()) {
            Err(ZonoRefusal::Internal(msg)) => {
                assert!(msg.contains("tail layer"), "{}: {msg}", f.name);
                refused += 1;
            }
            other => panic!(
                "{}: a trunk whose rows read a layer marked tail must refuse, got {:?}",
                f.name,
                other.map(|t| t.num_unstable())
            ),
        }
    }
    assert!(refused > 0, "VOID");
}

#[test]
fn fd_trunk_refuses_a_tail_that_changes_a_shared_slope() {
    // MUTATION PIN of the assembly check: the rows were built with the trunk's
    // slope, so a margin row with any other slope there is a different
    // relaxation symbol, and the state would not contain the network.
    let mut pinned = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        let Some(&li) = shared_unstable_layers(&frame).first() else {
            continue;
        };
        let trunk = joint_trunk(f, &frame, &f.adv);
        let honest = trunk
            .relax()
            .tail(&f.net, &frame, f.adv[0], None)
            .expect("tail");
        assert!(
            trunk.state(&frame, f.adv[0], &honest).is_ok(),
            "{}: the honest tail must assemble",
            f.name
        );
        let fixed = trunk.relax().slope(li, 0).expect("a shared slope");
        for forged in [1.0 - fixed, f64::from_bits(fixed.to_bits() ^ 1)] {
            if forged.to_bits() == fixed.to_bits() {
                continue;
            }
            let mut lying = honest.clone();
            lying.perturb_slope_for_test(li, 0, forged);
            match trunk.state(&frame, f.adv[0], &lying) {
                Err(ZonoRefusal::Internal(msg)) => {
                    assert!(msg.contains("shared slope"), "{}: {msg}", f.name);
                }
                other => panic!(
                    "{}: a tail that changes shared slope ({li}, 0) must be refused, got {:?}",
                    f.name,
                    other.map(|st| st.class)
                ),
            }
            pinned += 1;
        }
    }
    assert!(pinned > 0, "VOID");
}

#[test]
fn fd_trunk_digest_refuses_corrupted_rows() {
    let mut pinned = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        let mut trunk = joint_trunk(f, &frame, &f.adv);
        if trunk.rows_for_test().is_empty() {
            continue;
        }
        let tail = trunk
            .relax()
            .tail(&f.net, &frame, f.adv[0], None)
            .expect("tail");
        assert!(
            trunk.state(&frame, f.adv[0], &tail).is_ok(),
            "{}: intact trunk",
            f.name
        );
        let idx = trunk.rows_for_test().len() / 2;
        trunk.corrupt_for_test(idx);
        match trunk.state(&frame, f.adv[0], &tail) {
            Err(ZonoRefusal::Internal(msg)) => {
                assert!(msg.contains("changed after"), "{}: {msg}", f.name);
            }
            other => panic!(
                "{}: a trunk whose rows changed must refuse, got {:?}",
                f.name,
                other.map(|st| st.class)
            ),
        }
        pinned += 1;
    }
    assert!(pinned > 0, "VOID");
}

#[test]
fn fd_trunk_refuses_a_tail_of_another_class_or_layout() {
    // MUTATION PINS of the assembly's bindings (adversarial review of M3).
    // (a) The shared tier pairs tails with classes BY POSITION and then
    //     reports, and the cover adapter closes, the class it asked for: a
    //     tail of another class must never assemble as this one, or one
    //     class's search would certify another class's margin.
    // (b) The trunk's rows and a tail's margin row are one state only when
    //     their generator columns name the same symbols: a frame, or a tail
    //     taken on a frame, with another layout must be refused.
    let mut classes = 0usize;
    let mut layouts = 0usize;
    for f in &fixtures() {
        let frame = frame_of(f);
        let trunk = joint_trunk(f, &frame, &f.adv);
        let (&j0, rest) = f.adv.split_first().expect("an adversarial class");
        let tail = trunk.relax().tail(&f.net, &frame, j0, None).expect("tail");
        assert!(
            trunk.state(&frame, j0, &tail).is_ok(),
            "{}: the honest tail must assemble",
            f.name
        );
        for &asked in rest {
            match trunk.state(&frame, asked, &tail) {
                Err(ZonoRefusal::Internal(msg)) => {
                    assert!(msg.contains("offered as class"), "{}: {msg}", f.name);
                }
                res => panic!(
                    "{}: a class-{j0} tail assembled as class {asked}: {:?}",
                    f.name,
                    res.map(|st| st.class)
                ),
            }
            classes += 1;
        }
        // The same shapes, another input centre: the input symbols would
        // mean something else.
        let mut moved = frame.clone();
        let Some(m0) = moved.mid.first_mut() else {
            continue;
        };
        *m0 += 0.5;
        match trunk.state(&moved, j0, &tail) {
            Err(ZonoRefusal::Internal(msg)) => {
                assert!(msg.contains("layout"), "{}: {msg}", f.name);
            }
            res => panic!(
                "{}: a trunk assembled on a frame of another layout: {:?}",
                f.name,
                res.map(|st| st.class)
            ),
        }
        let foreign = trunk
            .relax()
            .tail(&f.net, &moved, j0, None)
            .expect("a tail on the moved frame");
        match trunk.state(&frame, j0, &foreign) {
            Err(ZonoRefusal::Internal(msg)) => {
                assert!(msg.contains("layout"), "{}: {msg}", f.name);
            }
            res => panic!(
                "{}: a tail taken on a frame of another layout assembled: {:?}",
                f.name,
                res.map(|st| st.class)
            ),
        }
        layouts += 1;
    }
    assert!(classes > 0 && layouts > 0, "VOID");
}

// ---------------------------------------------------------------------------
// Fault injection: the slope choice is untrusted
// ---------------------------------------------------------------------------

#[test]
fn fd_trunk_garbage_slopes_are_refused_or_stay_sound() {
    let mut refusals = 0usize;
    let mut states = 0usize;
    let mut points = 0usize;
    let mut failures = Vec::new();
    for f in &fixtures() {
        let frame = frame_of(f);
        let Some(&target) = shared_unstable_layers(&frame).first() else {
            continue;
        };
        // (a) Invalid slopes on a shared layer, and invalid `mu`: refused.
        for bad in [f64::NAN, f64::INFINITY, -0.25, 1.5] {
            let slopes =
                TrunkRelax::from_slopes(&frame, |li, _| if li == target { bad } else { 0.5 }, None);
            assert!(
                FrozenDualTrunk::build(&f.net, &frame, slopes, &FrozenDualOptions::default())
                    .is_err(),
                "{}: slope {bad} accepted",
                f.name
            );
            refusals += 1;
        }
        for bad_mu in [0.0, -1.0, f64::NAN] {
            let slopes = TrunkRelax::from_slopes(&frame, |_, _| 0.5, Some(bad_mu));
            assert!(
                FrozenDualTrunk::build(&f.net, &frame, slopes, &FrozenDualOptions::default())
                    .is_err(),
                "{}: mu {bad_mu} accepted",
                f.name
            );
            refusals += 1;
        }
        // (b) Seeded random slopes in [0, 1]: contained at every sample, and a
        // class with a sampled counterexample never closes.
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(0x6A2B ^ seed);
            let draws: Vec<Vec<f64>> = frame
                .layers
                .iter()
                .map(|layer| {
                    (0..layer.unst.len())
                        .map(|_| rng.random_range(0.0..=1.0))
                        .collect()
                })
                .collect();
            let slopes = TrunkRelax::from_slopes(&frame, |li, p| draws[li][p], None);
            let trunk =
                FrozenDualTrunk::build(&f.net, &frame, slopes, &FrozenDualOptions::default())
                    .expect("every slope in [0, 1] is valid");
            for &j in f.adv.iter().take(3) {
                let st = shared_state(f, &frame, &trunk, j);
                states += 1;
                let mut prng = StdRng::seed_from_u64(0xD00D ^ seed ^ (j as u64) << 7);
                let mut witnessed = false;
                for s in 0..64 {
                    let x = sample(&mut prng, &f.lo, &f.hi, s);
                    let pc = check_point(&f.net, &frame, &st, &x).expect("check");
                    points += 1;
                    if !pc.holds() {
                        failures.push(format!(
                            "{} seed {seed} class {j} sample {s}: {pc:?}",
                            f.name
                        ));
                    }
                    witnessed |= pc.margin < 0.0;
                }
                let rep = search(&problem_of(&st), &opts(4_000, 0));
                if witnessed && rep.closed() {
                    failures.push(format!(
                        "{} seed {seed} class {j}: a sampled counterexample, yet the \
                         random-slope state CLOSED after {} nodes",
                        f.name, rep.nodes
                    ));
                }
            }
        }
    }
    assert!(refusals > 0 && states > 0 && points > 0, "VOID");
    assert!(
        failures.is_empty(),
        "GARBAGE SLOPES BROKE SOUNDNESS (soundness emergency): {failures:#?}"
    );
}

// ---------------------------------------------------------------------------
// The tier in `shared` mode
// ---------------------------------------------------------------------------

#[test]
fn fd_trunk_shared_tier_never_closes_a_witnessed_sat_class() {
    let mut witnessed = 0usize;
    let mut built = 0usize;
    let mut failures = Vec::new();
    let mut fxs = fixtures();
    fxs.extend(sweep());
    for f in &fxs {
        let x0 = midpoint(&f.lo, &f.hi);
        // (a) The swapped target: the runner-up as the "true" class makes the
        // midpoint a counterexample for the argmax class; every other class is
        // open too, so the trunk is shared.
        let (top, second) = top_two(&f.net, &x0);
        let m = dd_margin(&f.net, second, top, &x0);
        if m < 0.0 {
            witnessed += 1;
            let adv: Vec<usize> = (0..f.net.n_out).filter(|&c| c != second).collect();
            let dj = vec![f64::NAN; adv.len()];
            let tier = run_tier(
                &f.net,
                &f.root,
                second,
                &adv,
                &dj,
                &adv,
                &f.ybox,
                None,
                &shared_policy(4_000),
            );
            built += tier.classes_built;
            if tier.closed.contains(&top) {
                failures.push(format!(
                    "{}: the shared tier closed the swapped class {top} (margin {m:e} at the \
                     midpoint)",
                    f.name
                ));
            }
        }
        // (b) The real target: every class with a sampled counterexample, all
        // classes open together.
        let mut sat_classes = Vec::new();
        for &j in &f.adv {
            let mut rng = StdRng::seed_from_u64(0x5A7 ^ (j as u64) << 4);
            let hit = (0..48).any(|s| {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                dd_margin(&f.net, f.t, j, &x) < 0.0
            });
            if hit {
                sat_classes.push(j);
            }
        }
        if sat_classes.is_empty() {
            continue;
        }
        witnessed += sat_classes.len();
        let dj = vec![f64::NAN; f.adv.len()];
        let tier = run_tier(
            &f.net,
            &f.root,
            f.t,
            &f.adv,
            &dj,
            &f.adv,
            &f.ybox,
            None,
            &shared_policy(4_000),
        );
        built += tier.classes_built;
        for j in &sat_classes {
            if tier.closed.contains(j) {
                failures.push(format!(
                    "{} class {j}: a sampled counterexample, yet the shared tier CLOSED it",
                    f.name
                ));
            }
        }
    }
    assert!(
        witnessed >= 4,
        "only {witnessed} witnessed sat classes: the negative control is VOID"
    );
    assert!(built > 0, "no shared tier built a class: VOID");
    assert!(
        failures.is_empty(),
        "SHARED-TRUNK TIER CLOSED A SAT CLASS (soundness emergency): {failures:#?}"
    );
}

#[test]
fn fd_trunk_shared_tier_planted_violation_is_never_closed() {
    let mut planted = 0usize;
    let mut built = 0usize;
    let mut failures = Vec::new();
    for seed in 0..9u64 {
        let half = [0.2, 0.08, 0.03][(seed % 3) as usize];
        let mut rng = StdRng::seed_from_u64(0x9A17 + seed);
        let spec0 = tiny_spec(&mut rng, 0.4);
        let f0 = fx("plant0", spec0.clone(), half, 0x9A17 + seed, false);
        let x0 = midpoint(&f0.lo, &f0.hi);
        let (t, j) = top_two(&f0.net, &x0);
        let m0 = dd_margin(&f0.net, t, j, &x0);
        for delta in [1e-3, 1e-6, 1e-9] {
            let mut spec = spec0.clone();
            let Some(TwinOpSpec::Gemm { bias, .. }) = spec.ops.last_mut() else {
                panic!("the tiny spec ends in its final Gemm");
            };
            bias[j] += m0 + delta;
            let f = fx("plant", spec, half, 0x9A17 + seed, false);
            let m = dd_margin(&f.net, t, j, &x0);
            if m >= 0.0 {
                continue;
            }
            planted += 1;
            let adv: Vec<usize> = (0..f.net.n_out).filter(|&c| c != t).collect();
            let dj = vec![f64::NAN; adv.len()];
            let tier = run_tier(
                &f.net,
                &f.root,
                t,
                &adv,
                &dj,
                &adv,
                &f.ybox,
                None,
                &shared_policy(4_000),
            );
            built += tier.classes_built;
            if tier.closed.contains(&j) {
                failures.push(format!(
                    "seed {seed} delta {delta:e}: planted margin {m:e} at the midpoint, yet the \
                     shared tier CLOSED class {j}"
                ));
            }
        }
    }
    assert!(planted >= 9, "only {planted} planted violations: VOID");
    assert!(built > 0, "no shared tier built a class: VOID");
    assert!(
        failures.is_empty(),
        "SHARED-TRUNK TIER CLOSED A PLANTED VIOLATION (soundness emergency): {failures:#?}"
    );
}

#[test]
fn fd_trunk_shared_tier_closed_classes_hold_at_sampled_points() {
    let mut closed_by_split = 0usize;
    let mut closed_multi = 0usize;
    let mut closed_points = 0usize;
    let mut failures = Vec::new();
    for f in &sweep() {
        let dj = vec![f64::NAN; f.adv.len()];
        // Every class open together (one shared trunk), then each class alone
        // (a one-class trunk IS its per-class state, so the per-class positive
        // control carries over and this cannot go VOID silently).
        let mut runs: Vec<Vec<usize>> = vec![f.adv.clone()];
        runs.extend(f.adv.iter().map(|&j| vec![j]));
        for open in &runs {
            let tier = run_tier(
                &f.net,
                &f.root,
                f.t,
                &f.adv,
                &dj,
                open,
                &f.ybox,
                None,
                &shared_policy(6_000),
            );
            for report in &tier.classes {
                let Ok(run) = &report.result else {
                    continue;
                };
                if !tier.closed.contains(&report.class) {
                    continue;
                }
                if run.search.nodes > 0 {
                    closed_by_split += 1;
                }
                if open.len() > 1 {
                    closed_multi += 1;
                }
                let j = report.class;
                let mut rng = StdRng::seed_from_u64(0xC105 ^ (j as u64) << 3);
                for s in 0..64 {
                    let x = sample(&mut rng, &f.lo, &f.hi, s);
                    let m = dd_margin(&f.net, f.t, j, &x);
                    closed_points += 1;
                    if m <= 0.0 {
                        failures.push(format!(
                            "{} class {j} ({} open): CLOSED ({} nodes) but the point {s} has \
                             margin {m:e}",
                            f.name,
                            open.len(),
                            run.search.nodes
                        ));
                    }
                }
            }
        }
    }
    eprintln!(
        "[fd-shared-state-positive] closed_by_split={closed_by_split} \
         closed_multi={closed_multi} points={closed_points}"
    );
    assert!(
        failures.is_empty(),
        "SHARED-TRUNK TIER CLOSED A CLASS WITH A COUNTEREXAMPLE: {failures:#?}"
    );
    assert!(
        closed_by_split > 0,
        "no class of the sweep closed by splitting in the shared tier: the positive control \
         is VOID"
    );
}

#[test]
fn fd_trunk_shared_tier_reports_its_observables() {
    let mut engaged = 0usize;
    for f in &fixtures() {
        if f.adv.len() < 2 {
            continue;
        }
        let dj = vec![f64::NAN; f.adv.len()];
        let tier = run_tier(
            &f.net,
            &f.root,
            f.t,
            &f.adv,
            &dj,
            &f.adv,
            &f.ybox,
            None,
            &shared_policy(2_000),
        );
        assert_eq!(tier.state, FrozenDualStateMode::Shared);
        assert!(
            tier.trunk_lambda == "joint" || tier.trunk_lambda.starts_with("anchor:"),
            "{}: lambda={:?}",
            f.name,
            tier.trunk_lambda
        );
        assert!(tier.trunk_ms > 0.0, "{}: the trunk took no time", f.name);
        assert_eq!(
            tier.classes_built, tier.objectives,
            "{}: a 60 s shared tier must build every class",
            f.name
        );
        let mut losses = Vec::new();
        for report in &tier.classes {
            let run = report.result.as_ref().expect("built");
            assert_eq!(run.state, FrozenDualStateMode::Shared);
            let loss = run.root_loss.expect("the loss is known on a clean fixture");
            assert!(loss.is_finite());
            losses.push(loss);
        }
        let max = losses.iter().copied().reduce(f64::max);
        assert_eq!(
            tier.shared_root_loss_max.map(f64::to_bits),
            max.map(f64::to_bits),
            "{}: the summary's loss is the classes' max",
            f.name
        );
        engaged += 1;
    }
    assert!(engaged > 0, "VOID");
    assert_eq!(FROZEN_DUAL_SHARED_LOSS_TOL, 0.01, "the program's 0.01 line");
}

#[test]
fn fd_trunk_shared_tier_is_thread_count_independent() {
    let f = &fixture0();
    assert!(f.adv.len() >= 2, "the pin needs several open classes: VOID");
    let dj = vec![f64::NAN; f.adv.len()];
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| {
            run_tier(
                &f.net,
                &f.root,
                f.t,
                &f.adv,
                &dj,
                &f.adv,
                &f.ybox,
                None,
                &shared_policy(3_000),
            )
        })
    };
    let a = run(1);
    assert!(a.classes_built > 0, "VOID: no class was built");
    for threads in [3, 8] {
        let b = run(threads);
        assert_eq!(
            a.trunk_lambda, b.trunk_lambda,
            "{threads} threads: the slope choice"
        );
        assert_eq!(
            tier_digest(&a),
            tier_digest(&b),
            "{threads} threads: the shared tier depends on the thread count"
        );
        assert_eq!(a.closed, b.closed);
        assert_eq!(
            a.shared_root_loss_max.map(f64::to_bits),
            b.shared_root_loss_max.map(f64::to_bits)
        );
    }
    // The trunk itself, built in different pools, is the same trunk.
    let frame = frame_of(f);
    let rows = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| joint_trunk(f, &frame, &f.adv).rows_for_test().to_vec())
    };
    let r1 = rows(1);
    assert!(
        same(&r1, &rows(8)),
        "the trunk's rows depend on the thread count"
    );
}

// ---------------------------------------------------------------------------
// Refusals, the dark default, bytes
// ---------------------------------------------------------------------------

#[test]
fn fd_trunk_refusals_fail_closed() {
    let f = &fixture0();
    let frame = frame_of(f);
    let past = Some(std::time::Instant::now());
    assert!(matches!(
        TrunkRelax::joint(&f.net, &frame, &[], None),
        Err(ZonoRefusal::Internal(_))
    ));
    assert!(matches!(
        TrunkRelax::joint(&f.net, &frame, &[f.t], None),
        Err(ZonoRefusal::Shape(_))
    ));
    assert!(matches!(
        TrunkRelax::joint(&f.net, &frame, &f.adv, past),
        Err(ZonoRefusal::Deadline)
    ));
    assert!(matches!(
        class_margin(&f.net, &frame, f.t, None),
        Err(ZonoRefusal::Shape(_))
    ));
    assert!(matches!(
        class_margin(&f.net, &frame, f.adv[0], past),
        Err(ZonoRefusal::Deadline)
    ));
    let slopes = TrunkRelax::joint(&f.net, &frame, &f.adv, None).expect("joint");
    assert!(matches!(
        slopes.tail(&f.net, &frame, f.adv[0], past),
        Err(ZonoRefusal::Deadline)
    ));
    assert!(matches!(
        slopes.tail(&f.net, &frame, f.t, None),
        Err(ZonoRefusal::Shape(_))
    ));
    let starved = FrozenDualOptions {
        max_state_bytes: 1,
        ..FrozenDualOptions::default()
    };
    assert!(matches!(
        FrozenDualTrunk::build(&f.net, &frame, slopes.clone(), &starved),
        Err(ZonoRefusal::Memory { .. })
    ));
    let late = FrozenDualOptions {
        deadline: past,
        ..FrozenDualOptions::default()
    };
    assert!(matches!(
        FrozenDualTrunk::build(&f.net, &frame, slopes, &late),
        Err(ZonoRefusal::Deadline)
    ));
    // A shared tier whose deadline has passed refuses every class and closes
    // nothing.
    let eng = BackwardEngine::new(&f.net, &f.root);
    let re = root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
    let tier = run_tier(
        &f.net,
        &f.root,
        f.t,
        &f.adv,
        &re.dj,
        &f.adv,
        &re.root_ybox(),
        past,
        &shared_policy(4_000),
    );
    assert!(
        tier.closed.is_empty(),
        "a dead tier closed {:?}",
        tier.closed
    );
    assert_eq!(tier.classes.len(), f.adv.len());
    assert_eq!(tier.classes_built, 0);
}

#[test]
fn fd_trunk_state_mode_is_dark_by_default() {
    use ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE as DECL;
    assert_eq!(DECL.default, ny_levers::DefaultSpec::Text("per_class"));
    assert_eq!(DECL.moat, ny_levers::MoatRisk::High);
    let ny_levers::LeverKind::Enum(members) = DECL.kind else {
        panic!("margin_row.frozen_dual_bab_state must be declared as an Enum");
    };
    assert_eq!(members, &FrozenDualStateMode::TOKENS[..]);
    assert_eq!(
        FrozenDualStateMode::default(),
        FrozenDualStateMode::PerClass
    );
    for token in FrozenDualStateMode::TOKENS {
        assert_eq!(
            FrozenDualStateMode::from_token(token).map(FrozenDualStateMode::token),
            Some(token)
        );
    }
    assert_eq!(FrozenDualStateMode::from_token("Shared"), None);
    assert_eq!(
        TierPolicy::clamped(10, 0.0).state,
        FrozenDualStateMode::PerClass
    );
    assert_eq!(
        TierPolicy::clamped(10, 0.0)
            .with_state(FrozenDualStateMode::Shared)
            .state,
        FrozenDualStateMode::Shared
    );
    // A governed read (not a raw environment access).
    if ny_levers::read(&DECL).source == ny_levers::Source::Default {
        assert_eq!(
            TierPolicy::from_gates().state,
            FrozenDualStateMode::PerClass,
            "with no environment and no preset cell the tier builds per class"
        );
    }
}

#[test]
fn fd_trunk_bytes_cover_the_copied_rows() {
    for f in &fixtures() {
        let frame = frame_of(f);
        let trunk = joint_trunk(f, &frame, &f.adv);
        let tail = trunk
            .relax()
            .tail(&f.net, &frame, f.adv[0], None)
            .expect("tail");
        let st = trunk.state(&frame, f.adv[0], &tail).expect("state");
        assert_eq!(trunk.num_unstable(), st.num_unstable(), "{}", f.name);
        assert!(trunk.bytes() >= (trunk.rows_for_test().len() as u64) * 8);
        assert_eq!(
            st.bytes(),
            trunk.state_copy_bytes(&tail),
            "{}: a class copy includes its objective and flattened slopes",
            f.name
        );
    }
}

#[test]
fn fd_trunk_owned_bytes_include_spare_capacity_without_copying_it() {
    let f = fixture0();
    let frame = frame_of(&f);
    let mut trunk = joint_trunk(&f, &frame, &f.adv);
    let tail = trunk
        .relax()
        .tail(&f.net, &frame, f.adv[0], None)
        .expect("tail");
    let owned = trunk.bytes();
    let copy = trunk.state_copy_bytes(&tail);
    let before = trunk.state(&frame, f.adv[0], &tail).expect("state");
    trunk.reserve_storage_for_test();
    assert!(
        trunk.bytes() > owned + 8 + size_of::<ReluSite>() as u64 + size_of::<Vec<f64>>() as u64 + 8
    );
    assert_eq!(trunk.state_copy_bytes(&tail), copy, "clones copy lengths");
    let after = trunk
        .state(&frame, f.adv[0], &tail)
        .expect("reserved state");
    assert_eq!(after.bytes(), copy);
    assert_eq!(first_difference(&before, &after), None);
}

#[test]
fn fd_trunk_class_copy_admission_accounts_for_all_retained_tails() {
    use crate::margin_row::zono_bab::shared_class_memory_for_test;
    let f = fixture0();
    let frame = frame_of(&f);
    assert!(f.adv.len() >= 2, "the test needs another retained tail");
    let classes = &f.adv[..2];
    let (resident, trunk, copy, result) = shared_class_memory_for_test(&f.net, &frame, classes, 1);
    assert!(resident > trunk, "tails and tier metadata stay live");
    assert!(
        matches!(result, Err(ZonoRefusal::Memory { estimate_bytes, budget_bytes })
        if estimate_bytes == copy && budget_bytes == copy - 1)
    );
    let (same_resident, same_trunk, same_copy, result) =
        shared_class_memory_for_test(&f.net, &frame, classes, 0);
    assert_eq!(
        (same_resident, same_trunk, same_copy),
        (resident, trunk, copy)
    );
    assert_eq!(result.expect("copy fits at the boundary"), copy);
}
