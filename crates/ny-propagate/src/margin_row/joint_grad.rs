//! `#joint-grad` — the term ny's root alpha ascent structurally omits.
//!
//! # What is missing, and why it is the whole abc root gap
//!
//! alpha-beta-CROWN's root is a 20-iteration loop in which **every** iteration
//! recomputes **every** intermediate pre-activation box from the current alpha
//! and backpropagates the final loss through that recomputation. Read it at
//! source: `auto_LiRPA/optimized_bounds.py:610` passes
//! `interm_bounds if fix_interm_bounds else None`, and the root call
//! (`beta_CROWN_solver.py:674`, `method='CROWN-Optimized'`) supplies
//! `interm_bounds=None`, so the expression is `None` on every iteration —
//! intermediate bounds are rebuilt each time, under grad.
//!
//! ny does not. `RootGates` derives every `(l, u)` once from the forward
//! tableau (`root.rs`, `concretize_box` + the shrink-only `tighten_layer`
//! intersection), `gates_from_box` turns those into `(alpha, s, c)`, and the
//! ascent then moves `alpha` with `(l, u)` — and therefore the upper chords
//! `s`, `c` — held FROZEN. `supergradient` differentiates the lower lines
//! `alpha_j * z_j` alone: it reads the accumulated coefficient `v` and keeps
//! `v.max(0.0)`, the branch on which the lower line is selected. The discarded
//! negative half is the UPPER chord, whose `(l, u)`-dependence is exactly the
//! term abc gets free from autograd and ny scores as **zero**.
//!
//! `#rebake` attacked this by ALTERNATING — ascend with the frame frozen, bake
//! the optimized alpha into a fresh tableau, ascend again. Measured 2026-09-15:
//! it halts after exactly one accepted round, because a fixed point of an
//! alternating map is not a joint optimum (textbook block-coordinate stalling),
//! and on `resnet_large` it cannot afford even one round (a round costs a root
//! build, ~37 s, against a ~12 s remaining slice). The banked conclusion was
//! "more rounds cannot fix it; only a joint step can." This is that joint step.
//!
//! # The decomposition
//!
//! For an unstable neuron `j` the pass contributes `v_j * line_j` to the bound,
//! with `line_j = alpha_j * z_j` when `v_j > 0` and `line_j = s_j * z_j + c_j`
//! when `v_j < 0`, where `s = u/(u-l)` and `c = -u*l/(u-l)`. Only the second
//! carries an `(l, u)` dependence — `alpha` is a free variable of the ascent,
//! not a function of the box. Differentiating the chord:
//!
//! ```text
//!   ds/dl =  u/(u-l)^2      dc/dl = -u^2/(u-l)^2
//!   ds/du = -l/(u-l)^2      dc/du =  l^2/(u-l)^2
//!
//!   dF/dl_j = v_j * (z_j * ds/dl + dc/dl) = v_j * u_j * (z_j - u_j) / (u_j-l_j)^2
//!   dF/du_j = v_j * (z_j * ds/du + dc/du) = v_j * l_j * (l_j - z_j) / (u_j-l_j)^2
//! ```
//!
//! That is [`box_weights`], and it is exact.
//!
//! What remains is `d(l_j, u_j)/d(alpha_m)` for every earlier layer `m`. Doing
//! this the way abc does — a full Jacobian — is unaffordable here: it is one
//! backward pass per refined row per iteration, i.e. the cost of
//! `backward_interm` (25-36 s) twenty times over. But the ascent does not need
//! a Jacobian. It needs `sum_j w_j * d(z_j)/d(alpha_m)`, a vector-Jacobian
//! product, which is **one reverse sweep** — the same shape as the forward
//! `linearized_walk`, run backwards through `conv_apply_backward`. That is
//! [`joint_adjoint`], and its cost is one trunk sweep per gradient refresh.
//! `mean_supergradient` is called `1 + frozen_improvements` times per ascent,
//! not once per iteration, so the refresh cadence is already the cheap one.
//!
//! # The one approximation, stated plainly
//!
//! `l_j` and `u_j` are the min and max of `z_j` over the input box, so their
//! exact alpha-derivatives are `dz_j/dalpha` evaluated at two different
//! attaining points. This module evaluates both at the single frozen-sign walk
//! point `x*` that `supergradient` already uses, which makes the seed for
//! neuron `j` simply `dF/dl_j + dF/du_j`. It is a surrogate DIRECTION, not a
//! certified quantity.
//!
//! **Soundness is unaffected, by construction and not by care.** Nothing here
//! ever reaches a bound. The gradient only proposes a candidate alpha; every
//! candidate is re-scored by the unchanged certified `eval`, accepted only on
//! `f > best`, and under `cfg.guard` the entire ascent is reverted unless the
//! independently recomputed published root beats `published_before`. A wrong
//! direction costs iterations, never correctness — which is precisely why ny
//! can afford an approximation where abc pays for exactness.

use ndarray::Array2;

use super::net::{conv_apply_backward, TwinNet, TwinOp};

/// Shipped default. `#joint-grad` is an EXPERIMENT constant: disarmed, the
/// ascent's direction is bit-identical to the historical one.
pub(crate) const JOINT_GRAD: bool = false;

/// Compute the joint term and REPORT its magnitude without applying it.
///
/// Splitting probe from apply is what lets one box hold answer two questions
/// at once: with `JOINT_GRAD_PROBE = true` and `JOINT_GRAD = false` the
/// published root bounds must come back BIT-IDENTICAL to the incumbent (the
/// direction is untouched), while `ratio` says whether the omitted term is
/// even large enough to change a direction. A ratio near zero would close this
/// front for the cost of one build instead of two.
pub(crate) const JOINT_GRAD_PROBE: bool = false;

/// Exact sensitivity of the frozen-frame bound to one unstable neuron's own
/// intermediate box, through the upper chord `s*z + c`.
///
/// Returns `(dF/dl, dF/du)`. Both are zero unless the pass selected the UPPER
/// chord for this (neuron, column) — i.e. unless the accumulated coefficient
/// `v` is strictly negative — because the lower line `alpha*z` has no `(l, u)`
/// dependence at all once `alpha` is a free variable.
///
/// Fails soft to `(0, 0)` on a degenerate or non-finite box: a neuron with
/// `u <= l` is not a live relaxation and must not inject an infinity into a
/// direction that is then normalised by its own max.
pub(crate) fn box_weights(v: f64, z: f64, l: f64, u: f64) -> (f64, f64) {
    if v.partial_cmp(&0.0) != Some(std::cmp::Ordering::Less) {
        return (0.0, 0.0);
    }
    let w = u - l;
    // A finite `w` fails `w > 0` exactly when `w <= 0`; NaN/inf fail `is_finite`.
    if !w.is_finite() || w <= 0.0 || !l.is_finite() || !u.is_finite() || !z.is_finite() {
        return (0.0, 0.0);
    }
    let w2 = w * w;
    let dl = v * u * (z - u) / w2;
    let du = v * l * (l - z) / w2;
    if dl.is_finite() && du.is_finite() {
        (dl, du)
    } else {
        (0.0, 0.0)
    }
}

/// Everything [`joint_adjoint`] needs from the root, as plain slices.
///
/// Deliberately NOT `&RootGates`: taking plain data is what lets the reverse
/// sweep be unit-tested against a hand-built net and hand-computed answers,
/// without constructing a root or standing up a benchmark.
///
/// Every field is indexed by trunk relu layer. `alpha` and `s` are per-NEURON
/// (all `n` of them, so stable and split-baked neurons carry their exact fixed
/// lines); `unst`, `seed`, `upper` and `z` are per-UNSTABLE-POSITION and must
/// agree in length with each other for every layer.
pub(crate) struct JointFrame<'a> {
    /// Per layer, per neuron: the lower-line slope (the candidate alpha).
    pub alpha: &'a [Vec<f64>],
    /// Per layer, per neuron: the upper-chord slope.
    pub s: &'a [Vec<f64>],
    /// Per layer: neuron indices of the unstable set, ascending.
    pub unst: &'a [Vec<usize>],
    /// Per layer, per unstable position: `dF/dl + dF/du` from [`box_weights`].
    pub seed: &'a [Vec<f64>],
    /// Per layer, per unstable position: did the pass select the upper chord?
    pub upper: &'a [Vec<bool>],
    /// Per layer, per unstable position: the pre-activation walk value `z`.
    pub z: &'a [Vec<f64>],
}

/// One reverse sweep of the frozen-sign linearized trunk, seeded by each
/// layer's own box weights, accumulating `dF/dalpha` for every EARLIER layer.
///
/// This is strictly the JOINT term. It never seeds the output row, so it cannot
/// double-count the direct term `supergradient` already returns; the two are
/// summed by the caller.
///
/// Returns `None` — never a partial answer — on any structural surprise: a
/// shape disagreement, a tensor the walk cannot transpose, or a `Gemm` at or
/// below the last trunk relu (the trunk is pre-Gemm by construction, and
/// `linearized_walk` refuses the same way).
pub(crate) fn joint_adjoint(net: &TwinNet, frame: &JointFrame) -> Option<Vec<Vec<f64>>> {
    let n_layers = frame.alpha.len();
    if frame.s.len() != n_layers
        || frame.unst.len() != n_layers
        || frame.seed.len() != n_layers
        || frame.upper.len() != n_layers
        || frame.z.len() != n_layers
        || net.trunk_relus.len() != n_layers
    {
        return None;
    }
    for li in 0..n_layers {
        let k = frame.unst[li].len();
        if frame.seed[li].len() != k || frame.upper[li].len() != k || frame.z[li].len() != k {
            return None;
        }
        if frame.s[li].len() != frame.alpha[li].len() {
            return None;
        }
    }
    let last_relu = *net.trunk_relus.last()?;
    if last_relu >= net.ops.len() {
        return None;
    }

    // Adjoint per tensor id; tensor `k + 1` is the output of op `k`.
    let mut adj: Vec<Vec<f64>> = net.tsize.iter().map(|&n| vec![0.0; n]).collect();
    let mut g: Vec<Vec<f64>> = frame.alpha.iter().map(|a| vec![0.0; a.len()]).collect();

    // Descending: when op `k` is reached, every consumer of tensor `k + 1` has
    // already deposited into `adj`, because a consumer's op index is always at
    // least its input tensor id.
    for k in (0..=last_relu).rev() {
        // Move the downstream adjoint row out before touching `adj` again:
        // `adj[input][j] += adj[k + 1][j] * ..` would borrow `adj` mutably and
        // immutably in one expression. Taking it is sound because no op with
        // index below `k` can reference tensor `k + 1` (an op's input tensor
        // ids never exceed its own index), so this row is dead after op `k`.
        let down = std::mem::take(adj.get_mut(k + 1)?);
        match net.ops.get(k)? {
            TwinOp::Relu { input, layer } => {
                let li = *layer;
                // `usize::MAX` marks the head relu, which is not a trunk layer
                // and carries no alpha the trunk ascent may move.
                let Some(alpha_l) = frame.alpha.get(li) else {
                    continue;
                };
                let n = alpha_l.len();
                if net.tsize.get(k + 1)? != &n || net.tsize.get(*input)? != &n || down.len() != n {
                    return None;
                }
                let unst = &frame.unst[li];
                let upper = &frame.upper[li];
                let s_l = &frame.s[li];
                let z_l = &frame.z[li];

                // (a) alpha sensitivity from the adjoint arriving from
                //     DOWNSTREAM box seeds. The lower line is `alpha_j * z_j`,
                //     so `d/dalpha_j = z_j`; the upper chord has no alpha in it.
                for (pos, &j) in unst.iter().enumerate() {
                    if !upper[pos] && j < n {
                        g[li][j] += down[j] * z_l[pos];
                    }
                }

                // (b) push the adjoint through this relu's selected line.
                let mut slope = alpha_l.clone();
                for (pos, &j) in unst.iter().enumerate() {
                    if upper[pos] && j < n {
                        slope[j] = s_l[j];
                    }
                }
                for j in 0..n {
                    adj[*input][j] += down[j] * slope[j];
                }

                // (c) SEED this layer's own box weights onto its
                //     pre-activation. `(l_j, u_j)` are bounds on `z_j`, so
                //     their alpha-sensitivity travels the same reverse path.
                for (pos, &j) in unst.iter().enumerate() {
                    if j < n {
                        adj[*input][j] += frame.seed[li][pos];
                    }
                }
            }
            TwinOp::Conv(c) => {
                let n_out = *net.tsize.get(k + 1)?;
                let n_in = *net.tsize.get(c.input)?;
                let src = Array2::from_shape_vec((n_out, 1), down).ok()?;
                let mut dst = Array2::<f64>::zeros((n_in, 1));
                // Signed transpose: we want the true adjoint, not the
                // absolute-value variant the error lanes use.
                conv_apply_backward(c, &src, &mut dst, false);
                let ds = dst.as_slice()?;
                for j in 0..n_in {
                    adj[c.input][j] += ds[j];
                }
            }
            TwinOp::Add { lhs, rhs } => {
                let n = *net.tsize.get(k + 1)?;
                if net.tsize.get(*lhs)? != &n || net.tsize.get(*rhs)? != &n {
                    return None;
                }
                for j in 0..n {
                    let a = down[j];
                    adj[*lhs][j] += a;
                    adj[*rhs][j] += a;
                }
            }
            TwinOp::Flatten { input } => {
                let n = *net.tsize.get(k + 1)?;
                if net.tsize.get(*input)? != &n {
                    return None;
                }
                for j in 0..n {
                    adj[*input][j] += down[j];
                }
            }
            TwinOp::ChannelAffine { input, scale, .. } => {
                let n = *net.tsize.get(k + 1)?;
                if net.tsize.get(*input)? != &n || scale.len() != n {
                    return None;
                }
                for j in 0..n {
                    adj[*input][j] += down[j] * scale[j];
                }
            }
            // The trunk is pre-Gemm by construction; `linearized_walk` refuses
            // a Gemm below `i_gemm1` for the same reason.
            TwinOp::Gemm { .. } => return None,
        }
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::{box_weights, joint_adjoint, JointFrame, TwinNet, TwinOp, JOINT_GRAD};

    /// The upper chord itself, as `root::gates_from_box` defines it.
    fn chord(l: f64, u: f64, z: f64) -> f64 {
        let s = u / (u - l);
        let c = -u * l / (u - l);
        s * z + c
    }

    #[test]
    fn the_box_weights_match_central_finite_differences_of_the_chord() {
        // The whole mechanism rests on these two derivatives being right, so
        // check them against the chord they claim to differentiate rather than
        // against a re-derivation of the same algebra.
        let (v, z, l, u) = (-1.3f64, 0.7f64, -2.0f64, 3.0f64);
        let h = 1e-6;
        let fd_l = (v * chord(l + h, u, z) - v * chord(l - h, u, z)) / (2.0 * h);
        let fd_u = (v * chord(l, u + h, z) - v * chord(l, u - h, z)) / (2.0 * h);
        let (dl, du) = box_weights(v, z, l, u);
        assert!(
            (dl - fd_l).abs() <= 1e-6 * fd_l.abs().max(1.0),
            "dF/dl {dl} vs finite difference {fd_l}"
        );
        assert!(
            (du - fd_u).abs() <= 1e-6 * fd_u.abs().max(1.0),
            "dF/du {du} vs finite difference {fd_u}"
        );
        assert!(dl != 0.0 && du != 0.0, "the fixture must exercise both");
    }

    #[test]
    fn the_box_weights_hold_across_a_grid_of_live_relaxations() {
        let h = 1e-6;
        for &l in &[-4.0f64, -1.5, -0.25] {
            for &u in &[0.25f64, 1.5, 4.0] {
                for &z in &[l + 0.1, 0.0, u - 0.1] {
                    for &v in &[-0.01f64, -1.0, -9.5] {
                        let (dl, du) = box_weights(v, z, l, u);
                        let fd_l = (v * chord(l + h, u, z) - v * chord(l - h, u, z)) / (2.0 * h);
                        let fd_u = (v * chord(l, u + h, z) - v * chord(l, u - h, z)) / (2.0 * h);
                        assert!(
                            (dl - fd_l).abs() <= 1e-5 * fd_l.abs().max(1.0),
                            "l={l} u={u} z={z} v={v}: dl {dl} vs {fd_l}"
                        );
                        assert!(
                            (du - fd_u).abs() <= 1e-5 * fd_u.abs().max(1.0),
                            "l={l} u={u} z={z} v={v}: du {du} vs {fd_u}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_lower_branch_carries_no_box_dependence() {
        // `v >= 0` selects `alpha * z`, which contains no `l` or `u` at all.
        // Scoring it non-zero would double-count against `supergradient`.
        assert_eq!(box_weights(0.0, 0.7, -2.0, 3.0), (0.0, 0.0));
        assert_eq!(box_weights(4.2, 0.7, -2.0, 3.0), (0.0, 0.0));
    }

    #[test]
    fn a_degenerate_or_non_finite_box_fails_soft_to_zero() {
        // The direction is normalised by its own max, so one infinity would
        // silently zero every other coordinate.
        assert_eq!(box_weights(-1.0, 0.0, 2.0, 2.0), (0.0, 0.0), "u == l");
        assert_eq!(box_weights(-1.0, 0.0, 3.0, 1.0), (0.0, 0.0), "u < l");
        assert_eq!(box_weights(-1.0, 0.0, f64::NEG_INFINITY, 1.0), (0.0, 0.0));
        assert_eq!(box_weights(-1.0, f64::NAN, -1.0, 1.0), (0.0, 0.0));
    }

    fn affine(input: usize, scale: Vec<f64>) -> TwinOp {
        let n = scale.len();
        TwinOp::ChannelAffine {
            input,
            scale,
            shift: vec![0.0; n],
            scale_rel_err: 0.0,
            shift_err: vec![0.0; n],
        }
    }

    /// input -> affine -> relu(layer 0) -> affine -> relu(layer 1)
    fn two_layer_net() -> TwinNet {
        TwinNet {
            n_in: 2,
            ops: vec![
                affine(0, vec![2.0, 3.0]),
                TwinOp::Relu { input: 1, layer: 0 },
                affine(2, vec![1.0, 1.0]),
                TwinOp::Relu { input: 3, layer: 1 },
            ],
            tsize: vec![2, 2, 2, 2, 2],
            i_gemm1: 4,
            trunk_relus: vec![1, 3],
            n_y: 2,
            n_out: 2,
        }
    }

    #[test]
    fn a_downstream_box_seed_reaches_an_upstream_alpha() {
        // THE term ny currently scores as zero: layer 1's box depends on
        // layer 0's alpha, so layer 1's box weight must land on `g[0]`.
        let net = two_layer_net();
        let alpha = vec![vec![0.4, 0.5], vec![0.6, 0.7]];
        let s = vec![vec![0.8, 0.9], vec![1.1, 1.2]];
        let unst = vec![vec![1usize], vec![1usize]];
        let seed = vec![vec![5.0], vec![7.0]];
        let upper = vec![vec![false], vec![true]];
        let z = vec![vec![2.0], vec![3.0]];
        let g = joint_adjoint(
            &net,
            &JointFrame {
                alpha: &alpha,
                s: &s,
                unst: &unst,
                seed: &seed,
                upper: &upper,
                z: &z,
            },
        )
        .expect("the sweep must succeed on a well-formed trunk");

        // adj[t3] = 7 (layer 1 seed) -> through the unit affine -> adj[t2] = 7;
        // layer 0 neuron 1 is on its LOWER line, so dF/dalpha = adj * z = 7*2.
        assert_eq!(
            g[0][1], 14.0,
            "layer 1's box weight must reach layer 0's alpha"
        );
        assert_eq!(g[0][0], 0.0, "neuron 0 is stable here and carries no seed");
        // Layer 1's own neuron sits on the UPPER chord, which contains no
        // alpha, and nothing downstream seeds it.
        assert_eq!(g[1], vec![0.0, 0.0]);
    }

    #[test]
    fn zero_seeds_give_exactly_zero_because_this_is_only_the_joint_term() {
        // The sweep never seeds the output row, so with no box weights it must
        // vanish identically — otherwise it would double-count the direct term
        // `supergradient` already returns.
        let net = two_layer_net();
        let alpha = vec![vec![0.4, 0.5], vec![0.6, 0.7]];
        let s = vec![vec![0.8, 0.9], vec![1.1, 1.2]];
        let unst = vec![vec![1usize], vec![1usize]];
        let seed = vec![vec![0.0], vec![0.0]];
        let upper = vec![vec![false], vec![false]];
        let z = vec![vec![2.0], vec![3.0]];
        let g = joint_adjoint(
            &net,
            &JointFrame {
                alpha: &alpha,
                s: &s,
                unst: &unst,
                seed: &seed,
                upper: &upper,
                z: &z,
            },
        )
        .expect("well-formed");
        assert!(g.iter().flatten().all(|&x| x == 0.0), "got {g:?}");
    }

    #[test]
    fn an_upper_chord_neuron_receives_no_alpha_gradient() {
        let net = two_layer_net();
        let alpha = vec![vec![0.4, 0.5], vec![0.6, 0.7]];
        let s = vec![vec![0.8, 0.9], vec![1.1, 1.2]];
        let unst = vec![vec![1usize], vec![1usize]];
        let seed = vec![vec![5.0], vec![7.0]];
        let z = vec![vec![2.0], vec![3.0]];
        // Same fixture as the reaching test, but layer 0 now sits on its upper
        // chord: `s*z + c` has no alpha in it, so the gradient must vanish even
        // though the seed still arrives.
        let upper = vec![vec![true], vec![true]];
        let g = joint_adjoint(
            &net,
            &JointFrame {
                alpha: &alpha,
                s: &s,
                unst: &unst,
                seed: &seed,
                upper: &upper,
                z: &z,
            },
        )
        .expect("well-formed");
        assert_eq!(g[0][1], 0.0, "upper chord carries no alpha dependence");
    }

    #[test]
    fn an_add_fans_the_adjoint_to_both_of_its_inputs() {
        // input -> relu(0) -> {affine, affine} -> add -> relu(1).
        // Two parallel unit paths, so layer 0's alpha must receive exactly
        // TWICE what a single path delivers. A transpose that dropped one
        // branch would still look plausible without this.
        let net = TwinNet {
            n_in: 2,
            ops: vec![
                affine(0, vec![1.0, 1.0]),
                TwinOp::Relu { input: 1, layer: 0 },
                affine(2, vec![1.0, 1.0]),
                affine(2, vec![1.0, 1.0]),
                TwinOp::Add { lhs: 3, rhs: 4 },
                TwinOp::Relu { input: 5, layer: 1 },
            ],
            tsize: vec![2, 2, 2, 2, 2, 2, 2],
            i_gemm1: 6,
            trunk_relus: vec![1, 5],
            n_y: 2,
            n_out: 2,
        };
        let alpha = vec![vec![0.4, 0.5], vec![0.6, 0.7]];
        let s = vec![vec![0.8, 0.9], vec![1.1, 1.2]];
        let unst = vec![vec![1usize], vec![1usize]];
        let seed = vec![vec![0.0], vec![7.0]];
        let upper = vec![vec![false], vec![true]];
        let z = vec![vec![2.0], vec![3.0]];
        let g = joint_adjoint(
            &net,
            &JointFrame {
                alpha: &alpha,
                s: &s,
                unst: &unst,
                seed: &seed,
                upper: &upper,
                z: &z,
            },
        )
        .expect("well-formed");
        assert_eq!(g[0][1], 28.0, "2 paths * seed 7 * z 2");
    }

    #[test]
    fn a_shape_disagreement_refuses_rather_than_returning_a_partial_answer() {
        let net = two_layer_net();
        let alpha = vec![vec![0.4, 0.5], vec![0.6, 0.7]];
        let s = vec![vec![0.8, 0.9], vec![1.1, 1.2]];
        let unst = vec![vec![1usize], vec![1usize]];
        let upper = vec![vec![false], vec![true]];
        let z = vec![vec![2.0], vec![3.0]];
        // seed shorter than `unst` for layer 1.
        let seed = vec![vec![5.0], vec![]];
        assert!(joint_adjoint(
            &net,
            &JointFrame {
                alpha: &alpha,
                s: &s,
                unst: &unst,
                seed: &seed,
                upper: &upper,
                z: &z
            },
        )
        .is_none());
    }

    #[test]
    fn the_probe_also_ships_off() {
        const {
            assert!(
                !super::JOINT_GRAD_PROBE,
                "the probe costs a reverse sweep per refresh; it is a measurement build only"
            )
        };
    }

    #[test]
    fn joint_grad_ships_off() {
        const {
            assert!(
                !JOINT_GRAD,
                "#joint-grad is an EXPERIMENT constant; shipped default stays false so the \
ascent's direction is bit-identical until it is measured"
            )
        };
    }
}
