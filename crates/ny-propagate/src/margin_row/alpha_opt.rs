// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-opt: projected supergradient optimization of the root trunk alpha
//! (#twinwall).
//!
//! WHY. The lane's lower-line slopes are born binary — `gates_from_box`
//! (`root.rs`) pins `alpha = [u >= -l]` per unstable neuron and nothing ever
//! moves them. alpha-beta-CROWN OPTIMIZES the same free parameter, and on the
//! cifar100 deep band (lane roots -0.61..-1.76) it proves at the root rows this
//! lane starts too deep to close (idx_7704: lane root -0.35, abc INIT=1 with
//! zero BaB). Every row the lane has ever proven started shallower than -0.57,
//! so the root bound is the binding constraint — this module moves it.
//!
//! WHAT. A root-phase loop (before the tree starts, never per-domain):
//! run one certified margin-seeded Lower pass under a candidate alpha, extract
//! a per-neuron supergradient from the SAME pass, step alpha toward a better
//! bound (projected to `[0, 1]`, restricted to unstable neurons), and keep the
//! best CERTIFIED objective ever seen (monotone accept — the committed alpha is
//! never worse than the binary heuristic under the frozen evaluation frame).
//!
//! THE GRADIENT (exact for the frozen-sign linearization; heuristic overall).
//! At trunk relu `li` the Lower-lane transform is `lv = vp*alpha_j + vn*s_j`
//! with `vp = max(v, 0)`, `vn = min(v, 0)` (`engine.rs`), so alpha enters ONLY
//! through the positive part: `d lv / d alpha_j = vp >= 0`. Holding fixed (a)
//! the sign selections the pass made at every other relu and (b) the concretize
//! argmin signs, the final bound is LINEAR in `lv`, and its coefficient is the
//! value at the argmin point `x*` of neuron `j`'s pre-activation under the
//! pass's own selected lines:
//!
//!   d bound_f / d alpha_{li,j} = vp_{li,j,f} * U_{li,j,f},
//!
//! where `x*_i = mid_i - sign(A_{i,f}) * rad_i` (the derivative of the
//! concretize `A@mid - |A|@rad` term is `mid_i - sign(A_i)*rad_i`, i.e. it
//! FLIPS with the coefficient sign — that is exactly what seeding the walk at
//! `x*` accounts for), and `U` is one cheap single-column forward walk of `x*`
//! through the linearized trunk (each earlier relu replaced by the affine line
//! the pass selected for that column: lower `alpha*z` on `v > 0`, upper chord
//! `s*z + c` on `v < 0`). Across sign flips the objective is only piecewise —
//! we do NOT chase the exact Clarke supergradient (alpha-beta-CROWN does not
//! either) — and the certified-slack terms' own alpha-dependence (the E-lane
//! grows with `ms = max(alpha, s)` when alpha exceeds `s`, plus the gamma
//! envelopes) is IGNORED in the direction: the direction is the gradient of
//! the frozen-sign LINEARIZATION, not of the slack-included certified bound.
//! Both approximations are direction-only: a wrong direction wastes one pass
//! and nothing more, because every ACCEPTED iterate is scored by the
//! unchanged certified Outward pass, slack included.
//!
//! SOUNDNESS.
//! * `alpha*y <= relu(y)` holds pointwise on all of R for ANY `alpha in
//!   [0, 1]` — validity of the lower line is box-independent and does not
//!   require binariness (the GPU seam's own docs state the same generality).
//! * The upper chord `(s, c)` depends only on `(l, u)`, which this module
//!   never moves — it is NOT touched (the build's `repair_upper_lines`
//!   certificate stands verbatim).
//! * THE ONE LOAD-BEARING COUPLING: the backward error lane contracts by
//!   `ms = max(alpha, s)` — the Lipschitz constant of `v -> vp*a + vn*s`.
//!   A stale binary-era `ms` under a raised alpha would UNDER-scale the carried
//!   error (a false-UNSAT generator), so `ms` is re-derived from the candidate
//!   alpha on every evaluation and on commit, by the same `max(alpha, s)`
//!   formula the root uses.
//! * The forward tableau is NOT rebuilt: the stored `l`/`u`/`clip_rows` were
//!   produced by a valid relaxation and remain valid enclosures of the true
//!   pre-activation ranges regardless of which (valid) backward alpha is used
//!   later. In particular `apply_gates`' outward widening constant — the one
//!   build-time line whose derivation assumes a binary lower product — never
//!   sees a fractional alpha, because backward-only optimization never
//!   executes it.
//! * Search iterates evaluate through `DomainGates` overrides (the engine's
//!   established injection point); the root struct is mutated ONCE, on accept.
//!   With the gate off, or on refusal/abort/no-improvement, the root is
//!   byte-identical to the build.
//!
//! GATE. `NY_MARGIN_ROW_ALPHA_OPT=1` (exact) arms it; default OFF.
//! `NY_MARGIN_ROW_ALPHA_ITERS` (default 8) caps iterations;
//! `NY_MARGIN_ROW_ALPHA_SECS` (default 20, further capped to 40% of the
//! remaining deadline) caps wall clock. Engagement telemetry
//! (`[alpha-opt] iter=...`) prints unconditionally when armed, so an inert
//! arming is detectable in one log (a null from a lever that never fired is
//! vacuous).
//!
//! BANKED CAVEAT (why this is pass-frugal): on the INTERNAL verifier a working
//! alpha gradient measured 0 conversions because cost-per-iteration was
//! binding. Here one iteration costs ONE backward pass (~600 ms on cifar100)
//! plus a single-column forward walk (milliseconds); the y-pack and head frame
//! are built once and frozen.

use ndarray::Array2;
use std::collections::BTreeMap;
use std::time::Instant;

use super::bounds::{
    compose_viay, head_gates, margin_seed, per_class_direct, row_dots, MarginBatch, YBox,
};
use super::engine::{BackwardEngine, Collect, DomainGates, GateVecs, LaneDir, PassOut};
use super::net::{conv_apply_forward, TwinNet, TwinOp};
use super::root::RootGates;

/// Is the optimizer armed? Exact `"1"` only; default OFF.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_OPT)
            .value
            .as_bool()
    })
}

fn env_iters() -> usize {
    ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_ITERS)
        .value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(8)
        .clamp(1, 64)
}

fn env_secs() -> f64 {
    ny_levers::read(&ny_levers::decls::sound_channel::MARGIN_ROW_ALPHA_SECS)
        .value
        .as_f64()
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(20.0)
}

/// What one optimization run did (telemetry + test surface).
pub struct AlphaOptReport {
    /// Certified frozen-frame objective of the heuristic (binary) alpha.
    pub baseline: f64,
    /// Best certified frozen-frame objective found (>= baseline by monotone
    /// accept; == baseline when nothing improved).
    pub best: f64,
    /// Gradient iterations attempted (excludes the baseline pass).
    pub iters: usize,
    /// Certified backward passes spent (baseline + iterations).
    pub passes: usize,
    /// Was a non-heuristic alpha committed to the root?
    pub accepted: bool,
    /// Neurons whose committed alpha differs from the heuristic.
    pub moved_neurons: usize,
    /// Largest committed |delta alpha|.
    pub max_dalpha: f64,
}

/// Env-gated entry: no-op unless `NY_MARGIN_ROW_ALPHA_OPT=1`.
///
/// Called from `lane_impl` between the root build and the BaB driver, where
/// the root is still owned mutably. Bound-quality only: any internal error
/// aborts the search and leaves the root byte-identical to the build.
pub fn maybe_optimize_root_alpha(
    net: &TwinNet,
    root: &mut RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
) {
    if !enabled() {
        return;
    }
    let _ = optimize_root_alpha(net, root, t, adv, deadline, env_iters(), env_secs());
}

/// The optimizer proper (test-visible; the env gate lives in
/// [`maybe_optimize_root_alpha`]).
pub(crate) fn optimize_root_alpha(
    net: &TwinNet,
    root: &mut RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    max_iters: usize,
    secs_budget: f64,
) -> Option<AlphaOptReport> {
    let t0 = Instant::now();
    if !root.mode.outward() {
        // Parity is the bit-parity oracle against core.py; it must not move.
        eprintln!("[alpha-opt] refuse: parity mode");
        return None;
    }
    if adv.is_empty() || root.layers.iter().all(|l| l.unst.is_empty()) {
        eprintln!(
            "[alpha-opt] refuse: nothing to optimize (no adv classes or no unstable neurons)"
        );
        return None;
    }
    // Never starve the tree search: cap at 40% of the remaining deadline.
    let budget = match deadline {
        Some(dl) => secs_budget.min(0.4 * dl.saturating_duration_since(t0).as_secs_f64()),
        None => secs_budget,
    };
    if budget < 0.5 {
        eprintln!("[alpha-opt] refuse: no budget (deadline too close)");
        return None;
    }
    let sr = search_alpha(net, &*root, t, adv, t0, budget, max_iters)?;
    // COMMIT: strictly-better certified objective only (monotone accept). The
    // heuristic alpha is otherwise retained bit-identically.
    let accepted = sr.best > sr.baseline;
    let mut moved_neurons = 0usize;
    let mut max_dalpha = 0.0f64;
    if accepted {
        for (li, layer) in root.layers.iter_mut().enumerate() {
            let cand = &sr.best_alpha[li];
            // Touch ONLY unstable neurons: stable and split-baked neurons keep
            // their exact fixed lines bit-identically.
            let unst = layer.unst.clone();
            for j in unst {
                let a_new = cand[j].clamp(0.0, 1.0);
                let a_old = layer.alpha[j];
                if a_new != a_old {
                    moved_neurons += 1;
                    max_dalpha = max_dalpha.max((a_new - a_old).abs());
                    layer.alpha[j] = a_new;
                }
                // Re-derive the alpha-coupled certified quantities via the same
                // path the root build uses:
                //  * `s`/`c` are functions of `(l, u)` ONLY (`gates_from_box` +
                //    `repair_upper_lines`), and `(l, u)` are frozen here, so
                //    their build-time certificate stands verbatim — nothing to
                //    recompute.
                //  * `ms = max(alpha, s)` (root.rs) IS alpha-coupled: it is the
                //    error-contraction Lipschitz constant of the backward gate
                //    transform, and a stale value under a raised alpha would
                //    under-scale the carried error (false-UNSAT risk). Recompute.
                layer.ms[j] = layer.alpha[j].max(layer.s[j]);
            }
        }
    }
    eprintln!(
        "[alpha-opt] done iters={} passes={} root_bound_before={:.6} after={:.6} accepted={} \
moved_neurons={} max_dalpha={:.4} secs={:.2}",
        sr.iters,
        sr.passes,
        sr.baseline,
        sr.best,
        accepted,
        moved_neurons,
        max_dalpha,
        t0.elapsed().as_secs_f64()
    );
    Some(AlphaOptReport {
        baseline: sr.baseline,
        best: sr.best,
        iters: sr.iters,
        passes: sr.passes,
        accepted,
        moved_neurons,
        max_dalpha,
    })
}

struct SearchRes {
    best_alpha: Vec<Vec<f64>>,
    baseline: f64,
    best: f64,
    iters: usize,
    passes: usize,
}

/// The search: immutable root, candidates injected as `DomainGates`.
fn search_alpha(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    t0: Instant,
    budget: f64,
    max_iters: usize,
) -> Option<SearchRes> {
    let eng = BackwardEngine::new(net, root);
    let mode = root.mode;
    // ---- Frozen evaluation frame (built ONCE under the heuristic gates) ----
    // y-pack -> y-box -> head gates -> margin seeds -> m1 / m2v. All stay
    // fixed while trunk alpha moves; every quantity is a valid certified bound
    // for ANY valid trunk relaxation, so freezing costs tightness of the
    // OBJECTIVE only, never soundness. The published root bound is recomputed
    // from scratch by `root_eval` under the committed gates.
    let mb_all = match MarginBatch::new(net, t, adv) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[alpha-opt] abort: margin batch: {e}");
            return None;
        }
    };
    let (al, au) = match eng.y_rows(None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[alpha-opt] abort: y-pack: {e}");
            return None;
        }
    };
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, mode);
    let ms_all = margin_seed(&mb_all, &gates, &ybox, mode);
    let al_dots = row_dots(root, &al);
    let au_dots = row_dots(root, &au);
    let m2v_all = compose_viay(&eng, &mb_all, &gates, &al, &au, &al_dots, &au_dots, mode);
    // Mirror root_eval's admission: only classes the frozen head paths do NOT
    // already close get a direct-pass column (same width as the root's own
    // direct pass; outward closure is strict `> 0`).
    let fail: Vec<usize> = (0..adv.len())
        .filter(|&k| {
            let d = ms_all.m1[k].max(m2v_all[k]);
            !(d.is_finite() && d > 0.0)
        })
        .collect();
    if fail.is_empty() {
        eprintln!("[alpha-opt] refuse: all classes close under the frozen head frame");
        return None;
    }
    let fail_classes: Vec<usize> = fail.iter().map(|&k| adv[k]).collect();
    let mbf = match MarginBatch::new(net, t, &fail_classes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[alpha-opt] abort: fail-set margin batch: {e}");
            return None;
        }
    };
    let ms_f = margin_seed(&mbf, &gates, &ybox, mode);
    let m2v_f: Vec<f64> = fail.iter().map(|&k| m2v_all[k]).collect();
    let nf = mbf.nf();

    // One evaluation = one certified Lower pass under the candidate alpha
    // (with the pre-transform coefficient capture the gradient needs) plus the
    // frozen-frame objective: min over classes of max(direct, m1, m2v).
    let eval = |alpha: &[Vec<f64>]| -> Result<(f64, usize, PassOut), String> {
        let dom = gates_override(root, alpha);
        let pass = eng
            .run_collect(
                &ms_f.seed,
                Some(&dom),
                LaneDir::Lower,
                None,
                Collect {
                    unst_abs: false,
                    rows: None,
                    unst_rows: true,
                },
            )
            .map_err(|e| e.to_string())?;
        let direct = per_class_direct(&eng, &pass, &ms_f, 0..nf);
        let mut worst = f64::INFINITY;
        let mut worst_col = 0usize;
        for f in 0..nf {
            let d = direct[f].max(ms_f.m1[f]).max(m2v_f[f]);
            if !d.is_finite() {
                return Err("non-finite objective".into());
            }
            if d < worst {
                worst = d;
                worst_col = f;
            }
        }
        Ok((worst, worst_col, pass))
    };

    let init: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();
    let te = Instant::now();
    let (baseline, mut worst_col, mut best_pass) = match eval(&init) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[alpha-opt] abort: baseline pass: {e}");
            return None;
        }
    };
    let mut pass_secs = te.elapsed().as_secs_f64().max(1e-3);
    eprintln!(
        "[alpha-opt] iter=0 root_bound_before={baseline:.6} after={baseline:.6} accepted=true \
(baseline; frozen-frame objective over {nf} classes)"
    );
    let mut best = baseline;
    let mut best_alpha = init;
    let mut grad = supergradient(net, root, &best_alpha, &best_pass, worst_col);
    let mut eta = 0.25f64;
    let mut passes = 1usize;
    let mut iters = 0usize;
    for it in 1..=max_iters {
        // Leave room for the pass we are about to spend.
        if t0.elapsed().as_secs_f64() + 1.5 * pass_secs > budget {
            break;
        }
        let Some(g) = grad.as_ref() else { break };
        let gmax = g
            .iter()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |m, &x| m.max(x.abs()));
        if !(gmax.is_finite() && gmax > 0.0) {
            break;
        }
        // Projected normalized supergradient step FROM THE BEST iterate.
        let mut cand = best_alpha.clone();
        let mut moved = false;
        for (li, layer) in root.layers.iter().enumerate() {
            for &j in &layer.unst {
                let a2 = (cand[li][j] + eta * g[li][j] / gmax).clamp(0.0, 1.0);
                if a2 != cand[li][j] {
                    cand[li][j] = a2;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
        iters = it;
        let te = Instant::now();
        let (f, fcol, pass) = match eval(&cand) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[alpha-opt] abort: pass failed at iter {it}: {e}");
                break;
            }
        };
        pass_secs = pass_secs.max(te.elapsed().as_secs_f64());
        passes += 1;
        let improved = f > best;
        eprintln!(
            "[alpha-opt] iter={it} root_bound_before={best:.6} after={f:.6} accepted={improved} \
eta={eta:.3}"
        );
        if improved {
            best = f;
            best_alpha = cand;
            best_pass = pass;
            worst_col = fcol;
            grad = supergradient(net, root, &best_alpha, &best_pass, worst_col);
        } else {
            eta *= 0.5;
            if eta < 0.01 {
                break;
            }
        }
    }
    Some(SearchRes {
        best_alpha,
        baseline,
        best,
        iters,
        passes,
    })
}

/// Full-layer gate override for a candidate alpha. `s`/`c` are the root's
/// certified chords (untouched); `ms` is re-derived from the CANDIDATE alpha —
/// the one alpha-coupled certified quantity (see the module doc).
fn gates_override(root: &RootGates, alpha: &[Vec<f64>]) -> DomainGates {
    let mut dg = DomainGates::default();
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.unst.is_empty() {
            continue;
        }
        let ms: Vec<f64> = alpha[li]
            .iter()
            .zip(&layer.s)
            .map(|(a, s)| a.max(*s))
            .collect();
        dg.layers.insert(
            li,
            GateVecs {
                alpha: alpha[li].clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms,
            },
        );
    }
    dg
}

/// Supergradient of the objective column `col` w.r.t. every unstable trunk
/// alpha: `g[li][j] = vp_{li,j,col} * U_{li,j,col}` (see the module doc).
///
/// Heuristic-only: the direction never feeds a verdict — every accepted
/// iterate is re-scored by the unchanged certified pass.
fn supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    col: usize,
) -> Option<Vec<Vec<f64>>> {
    let vsigns = pass.unst_rows.as_ref()?;
    // Concretize argmin of the Lower bound for this column:
    // x*_i = mid_i - sign(A_i)*rad_i (mid on a zero coefficient).
    let n_in = root.mid.len();
    let r = pass.a.ncols();
    if col >= r {
        return None;
    }
    let asl = pass.a.as_slice()?;
    let mut xstar = vec![0.0; n_in];
    for i in 0..n_in {
        let a = asl[i * r + col];
        xstar[i] = if a > 0.0 {
            root.mid[i] - root.rad[i]
        } else if a < 0.0 {
            root.mid[i] + root.rad[i]
        } else {
            root.mid[i]
        };
    }
    let (u, _y) = linearized_walk(net, root, alpha, vsigns, col, &xstar, false)?;
    let mut g: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    for (li, layer) in root.layers.iter().enumerate() {
        let (Some(uvals), Some(vmat)) = (u.get(&li), vsigns.get(&li)) else {
            continue;
        };
        let vs = vmat.as_slice()?;
        let rr = vmat.ncols();
        if uvals.len() != layer.unst.len() || vmat.nrows() != layer.unst.len() || col >= rr {
            return None;
        }
        for (pos, &j) in layer.unst.iter().enumerate() {
            let vp = vs[pos * rr + col].max(0.0);
            g[li][j] = vp * uvals[pos];
        }
    }
    Some(g)
}

/// Forward walk of `x*` through the FROZEN-SIGN linearized trunk: every op is
/// applied exactly (f64, with bias); every relu is replaced per neuron by the
/// affine line the pass selected for column `col` (Lower lane: `alpha*z` on a
/// positive incoming coefficient, `s*z + c` on a negative one; exact fixed
/// lines — stable and split-baked neurons — are sign-independent).
///
/// Returns `(U, y)`: per trunk layer, the walk values at that layer's UNSTABLE
/// neurons captured BEFORE the layer's own transform — the sensitivities `U` —
/// and, when `head` is set, the head pre-activation vector `ŷ(x*)` (the first
/// head Gemm's output; empty when `head` is false).
///
/// `head = false` reproduces the original walk BIT-IDENTICALLY (break at the
/// last trunk relu, no head ops touched). `head = true` (#margin-row-beta C3)
/// additionally applies the LAST trunk relu's selected lines — the same code
/// path as interior relus — and continues through the remaining trunk-tail ops
/// to `net.i_gemm1`, applying the head Gemm there. A Gemm anywhere OTHER than
/// exactly `i_gemm1` still refuses (`None`); relus between the last trunk relu
/// and `i_gemm1` cannot exist (net.rs compile: every relu before gemm1 is a
/// trunk relu).
///
/// `pub(crate)`: #margin-row-beta reuses this walk for its per-split
/// supergradient (`beta::step_scales`) — same frozen-sign linearization, same
/// direction-only contract (nothing verdict-bearing reads the result).
pub(crate) fn linearized_walk(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    vsigns: &BTreeMap<usize, Array2<f64>>,
    col: usize,
    xstar: &[f64],
    head: bool,
) -> Option<(BTreeMap<usize, Vec<f64>>, Vec<f64>)> {
    let last_relu = *net.trunk_relus.last()?;
    let stop = if head { net.i_gemm1 } else { last_relu };
    let mut vals: Vec<Option<Vec<f64>>> = vec![None; net.ops.len() + 1];
    vals[0] = Some(xstar.to_vec());
    let mut u_out: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (k, op) in net.ops.iter().enumerate().take(stop + 1) {
        let out: Vec<f64> = match op {
            TwinOp::Conv(c) => {
                let src = vals[c.input].as_ref()?;
                let n_out = net.tsize[k + 1];
                let src_m = Array2::from_shape_vec((src.len(), 1), src.clone()).ok()?;
                let mut dst = Array2::<f64>::zeros((n_out, 1));
                conv_apply_forward(c, &src_m, &mut dst, false);
                let p = c.oshape.1 * c.oshape.2;
                let mut v = dst.as_slice()?.to_vec();
                for (j, vj) in v.iter_mut().enumerate() {
                    *vj += c.bias[j / p];
                }
                v
            }
            TwinOp::Add { lhs, rhs } => {
                let a = vals[*lhs].as_ref()?;
                let b = vals[*rhs].as_ref()?;
                a.iter().zip(b).map(|(x, y)| x + y).collect()
            }
            TwinOp::Flatten { input } => vals[*input].as_ref()?.clone(),
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                ..
            } => {
                let src = vals[*input].as_ref()?;
                src.iter()
                    .enumerate()
                    .map(|(j, &v)| scale[j] * v + shift[j])
                    .collect()
            }
            TwinOp::Relu { input, layer } => {
                let li = *layer;
                let src = vals[*input].as_ref()?;
                let rec = root.layers.get(li)?;
                if src.len() != rec.n {
                    return None;
                }
                // Sensitivities: pre-transform values at the unstable neurons.
                if !rec.unst.is_empty() {
                    u_out.insert(li, rec.unst.iter().map(|&j| src[j]).collect());
                }
                if k == last_relu && !head {
                    break;
                }
                // Default per-neuron map: the (candidate) lower line `a*z`.
                // For every exact-line neuron — stable active (1,1,0), stable
                // inactive (0,0,0), split-baked — this IS the selected map
                // (candidates only move unstable alphas). Unstable neurons are
                // overwritten below with the pass's per-column selection.
                let mut out: Vec<f64> = src
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| alpha[li][j] * v)
                    .collect();
                if let Some(vmat) = vsigns.get(&li) {
                    let vs = vmat.as_slice()?;
                    let rr = vmat.ncols();
                    if vmat.nrows() != rec.unst.len() || col >= rr {
                        return None;
                    }
                    for (pos, &j) in rec.unst.iter().enumerate() {
                        // Negative incoming coefficient -> the pass selected
                        // the upper chord for this (neuron, column). A zero
                        // coefficient contributes nothing to the bound either
                        // way; keep the lower line (matches `vp = 0`).
                        if vs[pos * rr + col] < 0.0 {
                            out[j] = rec.s[j] * src[j] + rec.c[j];
                        }
                    }
                }
                out
            }
            TwinOp::Gemm {
                input,
                weight,
                bias,
                shape,
            } => {
                // #margin-row-beta C3: the FIRST head Gemm maps the trunk
                // output to the pre-head-relu vector y — exactly the layer the
                // head supergradient needs. Allowed only at exactly `i_gemm1`
                // in head mode; any other Gemm refuses as before.
                if !(head && k == net.i_gemm1) {
                    return None; // structurally pre-gemm otherwise
                }
                let src = vals[*input].as_ref()?;
                let (no, ni) = *shape;
                if src.len() != ni {
                    return None;
                }
                (0..no)
                    .map(|j| {
                        let row = &weight[j * ni..(j + 1) * ni];
                        row.iter().zip(src).map(|(w, v)| w * v).sum::<f64>() + bias[j]
                    })
                    .collect()
            }
        };
        vals[k + 1] = Some(out);
    }
    let y = if head {
        vals[net.i_gemm1 + 1].take()?
    } else {
        Vec::new()
    };
    Some((u_out, y))
}
