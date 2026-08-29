// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#deadline-parity` — CI GUARD against the "fast path guarded on deadline
//! ABSENCE" defect class.
//!
//! # The defect class
//!
//! A FAST / EXACT / MORE-PRECISE code path is guarded on deadline **absence**
//! (`deadline.is_none()`, `!deadline_is_hard`, `if deadline.is_some() { degraded }`).
//! **Every scored competition run carries a deadline. Tests never set one.**
//! So the degraded arm is taken *always in competition and only in competition*,
//! and the entire class is invisible to an ordinary test suite — every test
//! passes while the shipped binary silently runs the other branch.
//!
//! A 5-way audit on 2026-08-17 found 30 candidates -> 8 confirmed -> 4 root
//! causes and prescribed exactly this guard; it was never built, and within four
//! months the class regenerated at
//! `batched/backward_core.rs` (`dispatch_node_backward`, the `Layer::Linear`
//! arm, introduced by `6f49a660`). This module is that guard.
//!
//! # The experimental design
//!
//! Each lane runs the SAME workload twice:
//!
//! * arm `None`    — no deadline at all (what the test suite has always done)
//! * arm `Some(t)` — a deadline so far away ([`GENEROUS_DEADLINE`]) that no work
//!   can legitimately be skipped for lack of time
//!
//! Because the second arm can never actually expire, **any** observable
//! difference between the arms is a branch-on-deadline-PRESENCE artifact, i.e.
//! the defect class itself. The guard asserts:
//!
//! 1. **Bit-identical bounds** — compared via `f32::to_bits`, so `-0.0`/`+0.0`
//!    and NaN payloads cannot mask a difference. A difference here is a
//!    CORRECTNESS bug (the competition binary computes different numbers than
//!    every test), not a performance one.
//! 2. **Bounded wall-clock ratio** — the deadline arm may pay a small polling
//!    tax, but abandoning a batched/accelerated kernel because a deadline exists
//!    is a multiple-x cliff. See [`MAX_WALLCLOCK_RATIO`].
//!
//! # Adding a lane
//!
//! Add a `DeadlineLane` to [`lanes`]. A lane is any closure that takes
//! `Option<Instant>` and returns a [`LaneFingerprint`]. Keep the workload
//! deterministic and engine-driven; a lane that does not actually enter the
//! code under test is a wrong-lane null and guards nothing.
//!
//! ## Lanes that must NOT assert bit-identity (N24)
//!
//! Bit-identity is the right assertion only where the two arms CLAIM identity.
//! Some deadline-present arms are DELIBERATELY different — the sound-wider
//! certified-f64 conv IBP family (`sound || deadline.is_some()` in
//! `graph_ibp.rs:781/:796/:823/:838` and `conv2d/bound.rs:315/:377/:794/:943`;
//! N24 in `docs/DEADLINE_PRESENCE_AUDIT_2026-08-24.md`). Those arms are a
//! SOUNDNESS choice: the deadline arm is *wider*, on purpose, and "fixing" it
//! would be a moat regression. A lane whose workload routes conv IBP through
//! those arms (any conv-bearing graph via `collect_node_bounds` — N26 routes
//! there too) must either avoid conv layers entirely (as both current lanes
//! do: Linear/ReLU only), or assert an enclosure/tolerance relation instead of
//! `to_bits` equality — the companion guard
//! `tests/deadline_presence_bound_parity.rs` keeps its `1e-4` tolerance for
//! exactly this reason. Do not weaken THIS module's bit assertion to
//! accommodate a conv lane; give the conv lane its own relation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use ny_core::NaiveCpuGemmEngine;

use super::indexed_pending::IndexedPendingLinearBounds;
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearBounds, LinearLayer, ReLULayer};

/// A deadline this far in the future can never expire during the workload, so
/// no branch may legitimately drop work "because time ran out".
const GENEROUS_DEADLINE: Duration = Duration::from_hours(1);

/// Wall-clock ratio (deadline arm / no-deadline arm) above which the deadline
/// arm is judged to have abandoned a fast path rather than merely paying a
/// polling tax.
///
/// Polling between chunks costs a few percent. Dropping a batched or
/// accelerated kernel costs a multiple. `8.0` sits well above realistic CI
/// noise for a workload of this size (each arm is milliseconds, and we take the
/// MIN over [`TIMING_REPEATS`] runs, which is the noise-robust statistic) while
/// still failing hard on the 20x-class degradation this guard exists to catch.
const MAX_WALLCLOCK_RATIO: f64 = 8.0;

/// Repeats per arm for the timing lane; the MIN is used (robust to scheduler
/// noise and to a saturated build box, where the MEAN is worthless).
const TIMING_REPEATS: usize = 5;

/// Bit-exact fingerprint of everything a lane publishes.
///
/// Coefficients, biases AND the certified per-coefficient error are all
/// included: two arms that agree on the coefficients but disagree on the
/// certified error publish DIFFERENT bounds after `concretize_sound`, so
/// omitting the error channel would make this guard blind to exactly the kind
/// of precision split the class produces.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct LaneFingerprint {
    entries: Vec<(String, Vec<u32>)>,
}

impl LaneFingerprint {
    fn push_bounds(&mut self, label: &str, bounds: &LinearBounds) {
        let mut bits: Vec<u32> = Vec::new();
        bits.extend(bounds.lower_a.iter().map(|v| v.to_bits()));
        bits.extend(bounds.lower_b.iter().map(|v| v.to_bits()));
        bits.extend(bounds.upper_a.iter().map(|v| v.to_bits()));
        bits.extend(bounds.upper_b.iter().map(|v| v.to_bits()));
        // `None` (exact) and `Some(all zeros)` are DIFFERENT states downstream:
        // the first says "no error channel", the second says "error channel
        // present and zero". Tag them distinctly.
        match bounds.lower_a_err.as_ref() {
            Some(err) => {
                bits.push(1);
                bits.extend(err.iter().map(|v| v.to_bits()));
            }
            None => bits.push(0),
        }
        match bounds.upper_a_err.as_ref() {
            Some(err) => {
                bits.push(1);
                bits.extend(err.iter().map(|v| v.to_bits()));
            }
            None => bits.push(0),
        }
        self.entries.push((label.to_string(), bits));
    }

    /// First differing entry, as `(label, index, lhs_bits, rhs_bits)`.
    fn first_difference(&self, other: &Self) -> Option<(String, usize, u32, u32)> {
        for (lhs, rhs) in self.entries.iter().zip(other.entries.iter()) {
            if lhs.0 != rhs.0 {
                return Some((format!("{} vs {}", lhs.0, rhs.0), 0, 0, 0));
            }
            for (idx, (a, b)) in lhs.1.iter().zip(rhs.1.iter()).enumerate() {
                if a != b {
                    return Some((lhs.0.clone(), idx, *a, *b));
                }
            }
            if lhs.1.len() != rhs.1.len() {
                return Some((lhs.0.clone(), lhs.1.len().min(rhs.1.len()), 0, 0));
            }
        }
        if self.entries.len() != other.entries.len() {
            return Some(("<entry count>".to_string(), 0, 0, 0));
        }
        None
    }
}

/// One deadline-parity lane: a name, the source site it guards, and the
/// workload.
struct DeadlineLane {
    /// Human-readable lane name.
    name: &'static str,
    /// The `file.rs:line`-style site whose deadline predicate this lane
    /// exercises. Named in the failure message so a regression points straight
    /// at the offending branch instead of at "some bound changed".
    site: &'static str,
    /// Runs the workload under the given deadline authority.
    run: fn(Option<Instant>) -> LaneFingerprint,
}

fn lanes() -> Vec<DeadlineLane> {
    vec![
        DeadlineLane {
            name: "batched-linear-dispatch",
            site: "batched/backward_core.rs dispatch_node_backward -> Layer::Linear arm",
            run: run_batched_linear_lane,
        },
        DeadlineLane {
            name: "batched-linear-relu-chain",
            site: "batched/backward_core.rs dispatch_node_backward (Linear + ReLU families)",
            run: run_linear_relu_chain_lane,
        },
    ]
}

// ---------------------------------------------------------------------------
// Deterministic fixture construction
// ---------------------------------------------------------------------------

/// Small deterministic LCG so the fixture has no all-equal rows (an all-equal
/// or all-zero fixture can make two genuinely different kernels agree by
/// accident, which would silently defang this guard).
fn pseudo(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Spread over roughly [-1, 1) with a non-trivial mantissa.
    ((*state >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

/// Layer weight shape `(rows, cols)`: CROWN backward through it maps an
/// incoming `A` of width `rows` to an outgoing `A` of width `cols`.
fn fixture_layer(rows: usize, cols: usize, state: &mut u32) -> LinearLayer {
    let weight = Array2::from_shape_fn((rows, cols), |_| pseudo(state));
    let bias = Array1::from_shape_fn(rows, |_| pseudo(state));
    LinearLayer::new(weight, Some(bias)).expect("fixture linear layer is well formed")
}

fn fixture_bounds(num_outputs: usize, width: usize, state: &mut u32) -> LinearBounds {
    LinearBounds::new(
        Array2::from_shape_fn((num_outputs, width), |_| pseudo(state)),
        Array1::from_shape_fn(num_outputs, |_| pseudo(state)),
        Array2::from_shape_fn((num_outputs, width), |_| pseudo(state)),
        Array1::from_shape_fn(num_outputs, |_| pseudo(state)),
    )
    .expect("fixture linear bounds are well formed")
}

/// Build an `IndexedPendingLinearBounds` whose plan contains `names`.
fn pending_for(names: &[&str], n_domains: usize) -> IndexedPendingLinearBounds {
    let mut graph = GraphNetwork::new();
    let mut output_name = None;
    for &name in names {
        graph.add_node(GraphNode::from_input(
            name,
            Layer::Linear(
                LinearLayer::new(Array2::from_elem((1, 1), 1.0_f32), Some(Array1::zeros(1)))
                    .expect("plan placeholder layer"),
            ),
        ));
        output_name = Some(name);
    }
    graph.set_output(output_name.expect("plan needs at least one node"));
    let plan = graph.dispatch_plan().expect("dispatch plan should build");
    IndexedPendingLinearBounds::new(plan, n_domains)
}

// ---------------------------------------------------------------------------
// Lane: the batched Linear dispatch (the ninth site)
// ---------------------------------------------------------------------------

/// Domains in the batch. Must be > 1 or the batched arm degenerates to the
/// per-domain arm and the lane guards nothing.
const LANE_DOMAINS: usize = 6;
/// Objective rows carried backward.
const LANE_OUTPUTS: usize = 24;
/// Incoming coefficient width == the layer's output width.
const LANE_ROWS: usize = 96;
/// Outgoing coefficient width == the layer's input width.
const LANE_COLS: usize = 128;

fn run_batched_linear_lane(deadline: Option<Instant>) -> LaneFingerprint {
    let mut state = 0x5eed_1234_u32;
    let layer = fixture_layer(LANE_ROWS, LANE_COLS, &mut state);
    let node = GraphNode::new("lin", Layer::Linear(layer), vec!["src".to_string()]);

    let node_lbs: Vec<Option<LinearBounds>> = (0..LANE_DOMAINS)
        .map(|_| Some(fixture_bounds(LANE_OUTPUTS, LANE_ROWS, &mut state)))
        .collect();

    let constrained_inputs: Vec<BoundedTensor> = (0..LANE_DOMAINS)
        .map(|_| {
            BoundedTensor::new(
                Array1::from_elem(LANE_COLS, -1.0_f32).into_dyn(),
                Array1::from_elem(LANE_COLS, 1.0_f32).into_dyn(),
            )
            .expect("fixture box")
        })
        .collect();
    let empty_cache: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    let caches: Vec<&HashMap<String, Arc<BoundedTensor>>> =
        (0..LANE_DOMAINS).map(|_| &empty_cache).collect();
    let beta_states = vec![None; LANE_DOMAINS];
    let alpha_states = vec![None; LANE_DOMAINS];
    let mut pending = pending_for(&["src"], LANE_DOMAINS);

    super::backward_core::dispatch_node_backward(
        "lin",
        &node,
        node_lbs,
        &constrained_inputs,
        &caches,
        &beta_states,
        &alpha_states,
        &mut pending,
        LANE_DOMAINS,
        LANE_COLS,
        &NaiveCpuGemmEngine,
        deadline,
        None,
        false,
    )
    .expect("batched Linear dispatch must succeed under BOTH deadline arms");

    fingerprint_pending(&pending, &["src"], LANE_DOMAINS)
}

// ---------------------------------------------------------------------------
// Lane: a Linear -> ReLU -> Linear chain (covers more dispatch families)
// ---------------------------------------------------------------------------

fn run_linear_relu_chain_lane(deadline: Option<Instant>) -> LaneFingerprint {
    let mut state = 0x1234_5eed_u32;
    let width = 48;
    let outputs = 12;
    let domains = 4;

    let tail = fixture_layer(width, width, &mut state);
    let head = fixture_layer(width, width, &mut state);

    let tail_node = GraphNode::new("tail", Layer::Linear(tail), vec!["act".to_string()]);
    let relu_node = GraphNode::new("act", Layer::ReLU(ReLULayer), vec!["head".to_string()]);
    let head_node = GraphNode::new("head", Layer::Linear(head), vec!["src".to_string()]);

    // Pre-activation bounds for the ReLU, straddling zero so the relaxation is
    // genuinely exercised (an all-active or all-inactive ReLU is a no-op and
    // would make this lane vacuous).
    // Draw a CENTRE and a positive RADIUS so `lower <= upper` holds by
    // construction; two independent draws can invert.
    let mut pre_state = 0xabcd_0001_u32;
    let centres: Vec<f32> = (0..width).map(|_| pseudo(&mut pre_state) * 0.5).collect();
    let radii: Vec<f32> = (0..width)
        .map(|_| 0.25 + 0.5 * (pseudo(&mut pre_state) + 1.0).abs())
        .collect();
    let pre_lower = Array1::from_shape_fn(width, |i| centres[i] - radii[i]);
    let pre_upper = Array1::from_shape_fn(width, |i| centres[i] + radii[i]);
    let pre = Arc::new(
        BoundedTensor::new(pre_lower.into_dyn(), pre_upper.into_dyn()).expect("pre-activation box"),
    );
    let mut cache: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    cache.insert("head".to_string(), Arc::clone(&pre));

    let constrained_inputs: Vec<BoundedTensor> = (0..domains)
        .map(|_| {
            BoundedTensor::new(
                Array1::from_elem(width, -1.0_f32).into_dyn(),
                Array1::from_elem(width, 1.0_f32).into_dyn(),
            )
            .expect("fixture box")
        })
        .collect();
    let caches: Vec<&HashMap<String, Arc<BoundedTensor>>> = (0..domains).map(|_| &cache).collect();
    let beta_states = vec![None; domains];
    let alpha_states = vec![None; domains];

    let mut pending = pending_for(&["act", "head", "src"], domains);

    let mut current: Vec<Option<LinearBounds>> = (0..domains)
        .map(|_| Some(fixture_bounds(outputs, width, &mut state)))
        .collect();

    for (name, node) in [
        ("tail", &tail_node),
        ("act", &relu_node),
        ("head", &head_node),
    ] {
        super::backward_core::dispatch_node_backward(
            name,
            node,
            current,
            &constrained_inputs,
            &caches,
            &beta_states,
            &alpha_states,
            &mut pending,
            domains,
            width,
            &NaiveCpuGemmEngine,
            deadline,
            None,
            false,
        )
        .unwrap_or_else(|e| panic!("chain dispatch at '{name}' must succeed under both arms: {e}"));

        // Hand this node's output to the next node in the chain.
        let next_name = match name {
            "tail" => "act",
            "act" => "head",
            _ => break,
        };
        current = (0..domains)
            .map(|d| {
                pending
                    .get_name(next_name)
                    .and_then(|per_domain| per_domain.get(d))
                    .and_then(|opt| opt.clone())
            })
            .collect();
    }

    fingerprint_pending(&pending, &["act", "head", "src"], domains)
}

fn fingerprint_pending(
    pending: &IndexedPendingLinearBounds,
    names: &[&str],
    n_domains: usize,
) -> LaneFingerprint {
    let mut fp = LaneFingerprint::default();
    for &name in names {
        for domain in 0..n_domains {
            if let Some(bounds) = pending
                .get_name(name)
                .and_then(|per_domain| per_domain.get(domain))
                .and_then(|opt| opt.as_ref())
            {
                fp.push_bounds(&format!("{name}[domain {domain}]"), bounds);
            }
        }
    }
    assert!(
        !fp.entries.is_empty(),
        "deadline-parity lane produced NO bounds — the lane does not reach the code under test \
         (a wrong-lane null guards nothing)"
    );
    // Guard the guard. A fixture that degenerated to all-zero (or to a handful
    // of tag words) would make the parity assertion pass no matter how badly the
    // arms diverged. This project has lost real measurements to exactly that
    // shape, so require substantive, varied content.
    let total_words: usize = fp.entries.iter().map(|(_, bits)| bits.len()).sum();
    assert!(
        total_words > 64,
        "deadline-parity fingerprint has only {total_words} words — too small to be meaningful"
    );
    let nonzero = fp
        .entries
        .iter()
        .flat_map(|(_, bits)| bits.iter())
        .filter(|&&w| w != 0)
        .count();
    assert!(
        nonzero * 4 > total_words,
        "deadline-parity fingerprint is mostly zeros ({nonzero}/{total_words} non-zero) — the \
         parity assertion would be vacuous"
    );
    fp
}

// ---------------------------------------------------------------------------
// The guard
// ---------------------------------------------------------------------------

/// **The bit-identity half of the guard.**
///
/// A generous deadline cannot expire, so the two arms must agree bit-for-bit.
/// A failure here means the competition binary (always deadline-carrying)
/// computes different numbers than every test in the suite (never
/// deadline-carrying).
#[test]
fn deadline_presence_must_not_change_published_bounds() {
    let mut failures: Vec<String> = Vec::new();

    for lane in lanes() {
        let without = (lane.run)(None);
        let with = (lane.run)(Some(Instant::now() + GENEROUS_DEADLINE));

        if let Some((label, idx, lhs, rhs)) = without.first_difference(&with) {
            failures.push(format!(
                "\n  lane '{}' DIFFERS with vs without a deadline\n    \
                 site: {}\n    \
                 first difference: {label}, word {idx}: \
                 no-deadline=0x{lhs:08x} ({}) vs deadline=0x{rhs:08x} ({})",
                lane.name,
                lane.site,
                f32::from_bits(lhs),
                f32::from_bits(rhs),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "#deadline-parity VIOLATION — a code path branches on deadline PRESENCE and changes \
         the published bound.\n\
         Every scored run carries a deadline; no test does. So the branch these lanes caught is \
         taken ALWAYS in competition and NEVER in the test suite.\n\
         Fix by giving the fast path a cancellation seam (chunk + poll), not by accepting the \
         degraded arm.{}",
        failures.join("")
    );
}

/// **The wall-clock half of the guard.**
///
/// The deadline arm may pay a polling tax; it may not abandon a batched or
/// accelerated kernel. Uses MIN over repeats, which is the statistic that
/// survives a saturated CI box.
#[test]
fn deadline_presence_must_not_blow_up_wallclock() {
    let mut failures: Vec<String> = Vec::new();

    for lane in lanes() {
        // Warm up so first-touch allocation is not attributed to either arm.
        let _ = (lane.run)(None);
        let _ = (lane.run)(Some(Instant::now() + GENEROUS_DEADLINE));

        let mut best_without = Duration::MAX;
        let mut best_with = Duration::MAX;
        for _ in 0..TIMING_REPEATS {
            let t0 = Instant::now();
            let _ = (lane.run)(None);
            best_without = best_without.min(t0.elapsed());

            let t1 = Instant::now();
            let _ = (lane.run)(Some(Instant::now() + GENEROUS_DEADLINE));
            best_with = best_with.min(t1.elapsed());
        }

        let ratio = best_with.as_secs_f64() / best_without.as_secs_f64().max(1e-9);
        // Always report: a ratio well BELOW 1.0 is itself a finding (it means the
        // branch labelled "fast path" is not actually the faster arm), and this
        // number is the only place it is observable.
        println!(
            "#deadline-parity lane '{}': no-deadline {:?}, deadline {:?} => ratio {ratio:.2}x",
            lane.name, best_without, best_with
        );
        if ratio > MAX_WALLCLOCK_RATIO {
            failures.push(format!(
                "\n  lane '{}' is {ratio:.1}x SLOWER with a deadline \
                 ({:?} vs {:?}), limit {MAX_WALLCLOCK_RATIO:.1}x\n    site: {}",
                lane.name, best_with, best_without, lane.site,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "#deadline-parity VIOLATION — a fast path is abandoned because a deadline EXISTS.\n\
         Every scored run carries a deadline, so this cost is paid always in competition and \
         never in the test suite.\n\
         Fix by chunking the fast path and polling between chunks, not by deleting it.{}",
        failures.join("")
    );
}
