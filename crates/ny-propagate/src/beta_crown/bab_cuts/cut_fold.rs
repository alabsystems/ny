// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified Cut-CROWN increment C2 (CPU-first gate): dark λ-fold of
//! multi-neuron cuts into the graph CROWN backward LOWER bound
//! (`docs/CERTIFIED_CUT_CROWN_DESIGN.md` §C2).
//!
//! A registered cut set for a ReLU node contributes
//! `Σ_j λ_j·(Σ_{i∈G_j} cc_i·relu(ẑ_i) − B_j)` to the LOWER objective. Since
//! each cut satisfies `Σ cc·relu(ẑ) − B ≤ 0` on the box and `λ_j ≥ 0`, the
//! folded objective `f + Σ λ_j·g_j ≤ f` everywhere, so any lower bound of the
//! folded objective lower-bounds the true `f` (`cuts_fold_lower_bound` in the
//! Lean schema). Mechanically: `λ_j·cc_i` multiplies the POST-activation
//! `relu(ẑ_i)`, so it is added to the incoming lower-side coefficient of
//! neuron `i` at that ReLU node BEFORE relaxation selection; the constant
//! `−Σ λ_j·B_j` is added to the lower bias. The upper side is untouched (an
//! upper bound of `f` folded with `+λ·g` would need `λ ≤ 0`; we only read the
//! lower margin).
//!
//! DARK GATE: everything here is inert unless the `NY_CUT_FOLD=1` environment
//! variable is set AND an entry is registered for the ReLU node OF THE GRAPH
//! being propagated (entries are keyed by [`CutFoldScope`], the per-
//! `GraphNetwork`-instance token — never by bare node name). The default
//! path is byte-identical to today (empty registry ⇒ `None` ⇒ untouched
//! `LinearBounds`).
//!
//! Rounding note (experiment-grade): the coefficient/bias additions are plain
//! f32 ops with no directed rounding, and the certified `lower_a_err` is not
//! widened for the fold. The production C5 path must derive the fold with
//! outward rounding (the IBP directed-rounding machinery); for the C2 root
//! margin measurement the half-ulp fold error is orders of magnitude below
//! the margins of interest.

use crate::bounds::LinearBounds;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

/// One ReLU node's registered fold data: the summed λ-scaled cut weights per
/// neuron plus the summed lower-bias shift.
#[derive(Debug, Clone, Default)]
pub struct CutFoldEntry {
    /// `(flat neuron index within the ReLU layer, Σ_j λ_j·cc_i(j))`,
    /// added to the LOWER-side post-activation coefficient of every
    /// objective row.
    pub coeffs: Vec<(u32, f32)>,
    /// `−Σ_j λ_j·B_j`, added to every objective row's lower bias
    /// (a per-objective constant; `≤ 0` whenever every `B_j ≥ 0`).
    pub bias_shift: f32,
}

/// Identity of ONE `GraphNetwork` instance for cut-fold registration.
///
/// A cut `Σ cc·relu(ẑ) − B ≤ 0` is derived for one specific graph (and its
/// input box). The registry is process-global, so keying entries by bare node
/// name would fold a cut into ANY same-named node of ANY other graph running
/// in the process — parallel in-process verifications, or property 2 of a
/// sequential multi-property run. Folding a foreign cut can RAISE the lower
/// objective on a graph the cut was never proven for: an unsound bound, not
/// just noise. Every registration and every fold lookup therefore carries the
/// graph's scope token, minted once per `GraphNetwork::new()`.
///
/// `Clone`d graphs share the token while they remain semantically identical (a
/// configured clone has the same model, and BaB sub-domain boxes are subsets of
/// the registration box, so a valid cut stays valid). Every structural graph
/// mutation and output retarget mints a fresh token, preventing a mutated clone
/// from reading entries derived for its source model. Registrants that reuse ONE
/// graph instance across DIFFERENT input boxes must `clear_cut_fold()` and
/// re-derive between boxes — the box is part of the cut's validity, and the
/// scope token cannot see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CutFoldScope(u64);

impl CutFoldScope {
    /// Mint a fresh, process-unique scope token.
    pub fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

type CutFoldRegistry = HashMap<CutFoldScope, HashMap<String, CutFoldEntry>>;

fn registry() -> &'static RwLock<CutFoldRegistry> {
    static REGISTRY: OnceLock<RwLock<CutFoldRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Number of times a fold was actually applied to a backward pass — lets the
/// experiment harness assert the fold site was really exercised.
static APPLIED: AtomicU64 = AtomicU64::new(0);

/// The dark gate: cut folding is active only when `NY_CUT_FOLD=1`.
/// Read per ReLU-node backward step (a handful of times per pass — cheap).
pub fn cut_fold_enabled() -> bool {
    matches!(std::env::var("NY_CUT_FOLD").ok().as_deref(), Some("1"))
}

/// Register (or replace) the fold entry for a ReLU node of ONE graph
/// instance (`scope` = that graph's [`GraphNetwork::cut_fold_scope`]).
pub fn set_cut_fold(scope: CutFoldScope, node_name: &str, entry: CutFoldEntry) {
    if let Ok(mut guard) = registry().write() {
        guard
            .entry(scope)
            .or_default()
            .insert(node_name.to_string(), entry);
    }
}

/// Clear the entire fold registry (back to the byte-identical default path).
pub fn clear_cut_fold() {
    if let Ok(mut guard) = registry().write() {
        guard.clear();
    }
}

/// How many backward passes have had a fold applied since the last reset.
pub fn cut_fold_applied_count() -> u64 {
    APPLIED.load(Ordering::Relaxed)
}

/// Reset the applied-fold counter.
pub fn reset_cut_fold_applied_count() {
    APPLIED.store(0, Ordering::Relaxed);
}

/// Apply the registered fold for `node_name` OF THE GRAPH identified by
/// `scope` to the incoming post-activation `LinearBounds`, LOWER side only,
/// BEFORE ReLU relaxation selection.
///
/// Returns `None` (zero-cost untouched path) unless `NY_CUT_FOLD=1` and an
/// entry is registered for this (graph, node) pair — an entry registered for
/// a same-named node of a different graph never folds here (see
/// [`CutFoldScope`]). On any index out of range the fold is skipped entirely
/// (a mis-registration must never corrupt bounds).
pub fn fold_lower_side(
    scope: CutFoldScope,
    node_name: &str,
    lb: &LinearBounds,
) -> Option<LinearBounds> {
    if !cut_fold_enabled() {
        return None;
    }
    let entry = {
        let guard = registry().read().ok()?;
        if guard.is_empty() {
            return None;
        }
        guard.get(&scope)?.get(node_name)?.clone()
    };
    let num_inputs = lb.num_inputs();
    if entry
        .coeffs
        .iter()
        .any(|&(i, _)| (i as usize) >= num_inputs)
    {
        debug_assert!(
            false,
            "cut fold for '{node_name}' has neuron index out of range (layer width {num_inputs})"
        );
        return None;
    }
    let mut folded = lb.clone();
    // Diagnostic for the C2 magnitude question: a fold of O(λ·cc) is absorbed
    // below the f32 ulp when the incoming coefficients are ≳ 2^24 larger.
    if tracing::enabled!(tracing::Level::DEBUG) {
        let max_abs_coeff = entry
            .coeffs
            .iter()
            .flat_map(|&(i, _)| lb.lower_a().column(i as usize).to_vec())
            .fold(0.0f32, |m, v| m.max(v.abs()));
        let max_abs_bias = lb.lower_b().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        tracing::debug!(
            node = node_name,
            max_abs_lower_coeff_at_cut_neurons = max_abs_coeff,
            max_abs_lower_bias = max_abs_bias,
            fold_coeffs = entry.coeffs.len(),
            bias_shift = entry.bias_shift,
            "cut fold applied"
        );
    }
    {
        let lower_a = folded.lower_a_mut();
        for row in 0..lower_a.nrows() {
            for &(i, c) in &entry.coeffs {
                lower_a[[row, i as usize]] += c;
            }
        }
    }
    {
        let lower_b = folded.lower_b_mut();
        for v in lower_b.iter_mut() {
            *v += entry.bias_shift;
        }
    }
    APPLIED.fetch_add(1, Ordering::Relaxed);
    Some(folded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    fn lb_2x3() -> LinearBounds {
        LinearBounds::new(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            Array1::from(vec![0.5f32, -0.5]),
            array![[7.0f32, 8.0, 9.0], [10.0, 11.0, 12.0]],
            Array1::from(vec![1.5f32, 2.5]),
        )
        .expect("valid test bounds")
    }

    /// Single sequential test: env-var + global-registry manipulation must not
    /// race with itself across parallel test threads. (The registry itself is
    /// scope-keyed, so entries registered here can never fold into another
    /// test's graph even while `NY_CUT_FOLD=1` is set process-wide.)
    #[test]
    fn fold_gate_and_application() {
        // Serialized env scope (clippy env wall): all NY_CUT_FOLD phases run
        // under the blessed editor; pre-test state is restored on exit.
        ny_test_utils::env::with_env_edits(fold_gate_and_application_phases);
    }

    fn fold_gate_and_application_phases(env: &mut ny_test_utils::env::EnvEditor) {
        let scope = CutFoldScope::fresh();
        // Phase 1: default (env unset) ⇒ inert even with an entry registered.
        env.remove("NY_CUT_FOLD");
        set_cut_fold(
            scope,
            "cf_test_node",
            CutFoldEntry {
                coeffs: vec![(0, 0.25), (2, 1.0)],
                bias_shift: -0.75,
            },
        );
        let lb = lb_2x3();
        assert!(fold_lower_side(scope, "cf_test_node", &lb).is_none());

        // Phase 2: enabled but no entry for this node ⇒ inert; and an entry
        // registered under a DIFFERENT scope must never fold here.
        env.set("NY_CUT_FOLD", "1");
        assert!(fold_lower_side(scope, "cf_other_node", &lb).is_none());
        assert!(
            fold_lower_side(CutFoldScope::fresh(), "cf_test_node", &lb).is_none(),
            "an entry registered for one graph scope must be invisible to another scope"
        );

        // Phase 3: enabled + entry ⇒ lower side folded, upper untouched.
        reset_cut_fold_applied_count();
        let folded = fold_lower_side(scope, "cf_test_node", &lb).expect("fold must apply");
        let want_lower_a: Array2<f32> = array![[1.25f32, 2.0, 4.0], [4.25, 5.0, 7.0]];
        assert_eq!(folded.lower_a(), &want_lower_a);
        assert_eq!(
            folded.lower_b(),
            &Array1::from(vec![0.5f32 - 0.75, -0.5 - 0.75])
        );
        assert_eq!(folded.upper_a(), lb.upper_a());
        assert_eq!(folded.upper_b(), lb.upper_b());
        assert_eq!(cut_fold_applied_count(), 1);

        // Phase 4: out-of-range neuron index ⇒ fold refused (release path).
        #[cfg(not(debug_assertions))]
        {
            set_cut_fold(
                scope,
                "cf_test_node",
                CutFoldEntry {
                    coeffs: vec![(99, 1.0)],
                    bias_shift: 0.0,
                },
            );
            assert!(fold_lower_side(scope, "cf_test_node", &lb).is_none());
        }

        // Phase 5: cleared registry ⇒ inert again.
        clear_cut_fold();
        assert!(fold_lower_side(scope, "cf_test_node", &lb).is_none());

        // Phase 6: end-to-end through the graph CROWN spec backward — the
        // MultiReluCutK k=3 genuine-coupling geometry. z1 = x1,
        // z2 = -x1 + 2x2, z3 = -x1 - 2x2 on [-1,1]^2; f = -Σ relu(z_i).
        // True min f = -3 (joint cut bound 3, tight); plain CROWN composes the
        // three upper chords to -4. Folding the proven cut Σ relu(z) ≤ 3 at λ
        // yields -4 + λ (coefficient -1+λ scales the chords, bias -3λ): every
        // λ ∈ (0,1] strictly tightens, λ=1 reaches the true min.
        graph_fold_strictly_tightens(env);
    }

    fn graph_fold_strictly_tightens(env: &mut ny_test_utils::env::EnvEditor) {
        use crate::layers::{Layer, LinearLayer, ReLULayer};
        use crate::network::{GraphNetwork, GraphNode};
        use ndarray::{arr1, arr2};
        use ny_tensor::BoundedTensor;

        let mut graph = GraphNetwork::new();
        let pre = LinearLayer::new(
            arr2(&[[1.0f32, 0.0], [-1.0, 2.0], [-1.0, -2.0]]),
            Some(arr1(&[0.0f32, 0.0, 0.0])),
        )
        .expect("pre layer");
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["pre".to_string()],
        ));
        let head = LinearLayer::new(arr2(&[[-1.0f32, -1.0, -1.0]]), Some(arr1(&[0.0f32])))
            .expect("head layer");
        graph.add_node(GraphNode::new(
            "head",
            Layer::Linear(head),
            vec!["relu".to_string()],
        ));
        graph.set_output("head");

        let input = BoundedTensor::new(
            ndarray::array![-1.0f32, -1.0].into_dyn(),
            ndarray::array![1.0f32, 1.0].into_dyn(),
        )
        .expect("input box");
        let spec = arr2(&[[1.0f32]]);

        let run = |graph: &GraphNetwork| -> f32 {
            let bounds = graph
                .propagate_crown_with_specs_and_engine(&input, &spec, None)
                .expect("spec CROWN");
            bounds.lower().iter().copied().next().expect("one spec row")
        };

        clear_cut_fold();
        env.remove("NY_CUT_FOLD");
        let baseline = run(&graph);
        assert!(
            (baseline + 4.0).abs() < 1e-5,
            "plain CROWN on the k=3 geometry must be -4, got {baseline}"
        );

        // The proven k=3 cut: relu(z1)+relu(z2)+relu(z3) ≤ 3 on the box
        // (MultiReluCutK demo_joint_cut_le; derive_cut_bound reproduces B=3
        // in multi_relu_cut::tests::root_bound_dominates_exact_relu_sum).
        env.set("NY_CUT_FOLD", "1");
        let mut prev = baseline;
        for lambda in [0.25f32, 0.5, 1.0] {
            set_cut_fold(
                graph.cut_fold_scope(),
                "relu",
                CutFoldEntry {
                    coeffs: vec![(0, lambda), (1, lambda), (2, lambda)],
                    bias_shift: -3.0 * lambda,
                },
            );
            reset_cut_fold_applied_count();
            let folded = run(&graph);
            assert_eq!(cut_fold_applied_count(), 1);
            let expected = -4.0 + lambda;
            assert!(
                (folded - expected).abs() < 1e-5,
                "λ={lambda}: expected {expected}, got {folded}"
            );
            assert!(folded > prev, "λ={lambda} must strictly tighten");
            assert!(
                folded <= -3.0 + 1e-5,
                "λ={lambda}: bound {folded} must stay below the true min -3 (sound)"
            );
            prev = folded;
        }
        clear_cut_fold();
    }
}
