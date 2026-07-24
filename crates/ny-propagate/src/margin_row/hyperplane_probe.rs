// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! READ-ONLY diagnostic probe (#hyperplane-probe).
//!
//! Decides a research go/no-go for a "correlation-directed hyperplane
//! branching" bet on deep-resnet UNSAT. It computes the singular-value
//! spectrum of the trunk->head Jacobian
//!
//! ```text
//!   J = diag(|w_margin|) * (d y_head / d z_trunkrelu)
//! ```
//!
//! where `y_head` is the head Gemm pre-activation (the twin-net seeds here, so
//! `net.n_y` == the head Gemm output width == `Gemm_56`), `z_trunkrelu` is a
//! trunk ReLU's OUTPUT (each layer is probed), and `w_margin` is the binding
//! output-margin row projected onto the head neurons
//! (`gemm2[t,:] - gemm2[j*,:]`, default `j* = 55`).
//!
//! It is STRICTLY read-only: it builds its OWN seed + retained-row spec, runs an
//! extra backward pass through the *unchanged* engine, computes numbers, logs
//! them to stderr (prefix `[hyperplane-probe]`), and returns. It never mutates a
//! bound, a gate, or a verdict. When `NY_HYPERPLANE_PROBE` is unset [`run`] is
//! never called (see the guarded call site in `mod.rs::lane_impl`), so behavior
//! is byte-identical to a build without this file.
//!
//! Two spectra are reported per trunk relu layer:
//!   * PRIMARY (task formula) `diag(|w_margin|) * dHead/dRelu_out` — the
//!     intrinsic rank of the class-weighted trunk->head map.
//!   * WIDTH-FAITHFUL `diag(|w_margin|) * dHead/dRelu_out * diag(radius)` —
//!     each direction weighted by the relu OUTPUT box radius, i.e. its actual
//!     contribution to the head-box WIDTH. Dead (fully inactive) layers have
//!     radius 0 and contribute nothing to the width.
//!
//! Overrides (optional, diagnostic-only):
//!   * `NY_HYPERPLANE_LAYER=<li>`  force the PRIMARY-verdict layer.
//!   * `NY_HYPERPLANE_CLASS=<j>`   margin class for the weighting (default 55).

use ndarray::Array2;

use super::bounds::{head_gates, margin_seed, per_class_direct, MarginBatch, MarginSeed, YBox};
use super::engine::{domain_gates, BackwardEngine, Collect, LaneDir, Seed};
use super::net::TwinNet;
use super::root::{RetainedLayer, RetainedRows, RootGates};
use super::rounding::{next_down, next_up};

const PFX: &str = "[hyperplane-probe]";

/// Is the probe enabled? Only an explicit truthy `NY_HYPERPLANE_PROBE` arms it.
pub fn enabled() -> bool {
    matches!(
        std::env::var("NY_HYPERPLANE_PROBE").as_deref(),
        Ok("1") | Ok("true") | Ok("on") | Ok("yes")
    )
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.trim().parse().ok())
}

/// Entry point. Read-only; any failure just logs and returns.
pub fn run(net: &TwinNet, root: &RootGates, t: usize, adv: &[usize]) {
    if let Err(msg) = run_inner(net, root, t, adv) {
        eprintln!("{PFX} aborted (read-only, no effect on verdict): {msg}");
    }
}

/// One trunk relu layer's computed spectra summary.
struct LayerReport {
    li: usize,
    width: usize,
    unstable: usize,
    /// PRIMARY (class-weighted) spectrum eigenvalues (= sigma^2), sorted desc.
    eig_primary: Vec<f64>,
    /// WIDTH-FAITHFUL (class-weighted x radius) eigenvalues, sorted desc.
    eig_width: Vec<f64>,
}

fn run_inner(net: &TwinNet, root: &RootGates, t: usize, adv: &[usize]) -> Result<(), String> {
    // ---- topology dump (so the chosen nodes are interpretable) -------------
    let n_y = net.n_y;
    let (_, _, (g1_rows, g1_cols)) = net.gemm1();
    let (w2, _, (g2_rows, g2_cols)) = net.gemm2();
    eprintln!(
        "{PFX} net: n_in={} n_out(classes)={} n_y(head/Gemm_56 width)={} i_gemm1(op)={} \
         trunk_relu_layers={}",
        net.n_in,
        net.n_out,
        n_y,
        net.i_gemm1,
        root.layers.len()
    );
    eprintln!(
        "{PFX} gemm1(head, Gemm_56) shape=(out={g1_rows}, in={g1_cols})  \
         gemm2(final) shape=(out={g2_rows}, in={g2_cols})"
    );
    for (li, lg) in root.layers.iter().enumerate() {
        let (mut lmin, mut umax) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..lg.n {
            lmin = lmin.min(lg.l[i]);
            umax = umax.max(lg.u[i]);
        }
        let dead = lg.unst.is_empty() && umax <= 0.0;
        eprintln!(
            "{PFX}   trunk_relu[{li}] op={} width={} unstable={} preact_box=[{lmin:.3},{umax:.3}]{}",
            lg.op,
            lg.n,
            lg.unst.len(),
            if dead { " DEAD(all-inactive->output==0)" } else { "" }
        );
    }
    if root.layers.is_empty() || n_y == 0 {
        return Err("degenerate net (no trunk relus or zero head width)".into());
    }

    let eng = BackwardEngine::new(net, root);

    // ---- head pre-activation box + binding margin (sanity + class pick) ----
    // This is the box whose width the bet is about. NOTE: the margin-row engine
    // uses a tight forward-DeepPoly (M/D) tableau, so its head box is MUCH
    // narrower than the loose backward-CROWN box (~419) the bet references.
    let (al, au) = eng.y_rows(None).map_err(|e| e.to_string())?;
    let ybox = YBox::from_rows(&eng, &al, &au);
    let (mut box_w, mut box_arg) = (f64::NEG_INFINITY, 0usize);
    for i in 0..n_y {
        let w = ybox.uy[i] - ybox.ly[i];
        if w > box_w {
            box_w = w;
            box_arg = i;
        }
    }
    eprintln!(
        "{PFX} head(Gemm_56) pre-act box (margin-row DeepPoly relaxation): max width={box_w:.4} \
         at head neuron {box_arg}  [bet references a ~419-wide backward-CROWN box]"
    );

    // Per-margin ROOT lower bounds -> the binding (worst) adversarial class.
    let gates = head_gates(&ybox, root.mode);
    let mb = MarginBatch::new(net, t, adv).map_err(|e| e.to_string())?;
    let ms = margin_seed(&mb, &gates, &ybox, root.mode);
    let mpass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .map_err(|e| e.to_string())?;
    let lbs = per_class_direct(&eng, &mpass, &ms, 0..mb.nf());
    let (mut binding_row, mut binding_lb) = (0usize, f64::INFINITY);
    for (r0, &lb) in lbs.iter().enumerate() {
        if lb < binding_lb {
            binding_lb = lb;
            binding_row = r0;
        }
    }
    let binding_class = adv.get(binding_row).copied().unwrap_or(usize::MAX);
    eprintln!(
        "{PFX} spec: t(true class)={t} adv_count={} | binding margin class={binding_class} \
         root_lower_bound={binding_lb:.5} (margin<0 => not yet UNSAT at root)",
        adv.len()
    );

    // Margin class for the weighting: default 55 if present, else the binding.
    let want = env_usize("NY_HYPERPLANE_CLASS").unwrap_or(55);
    let j_star = if adv.contains(&want) {
        want
    } else {
        binding_class
    };
    eprintln!(
        "{PFX} weighting margin class j*={j_star} (requested {want}, binding {binding_class}); \
         w_margin = |gemm2[t,:] - gemm2[j*,:]| over the {n_y} head neurons"
    );
    let w_margin: Vec<f64> = (0..n_y)
        .map(|k| (w2[t * n_y + k] - w2[j_star * n_y + k]).abs())
        .collect();

    // ---- capture the Jacobian block for EVERY trunk relu in ONE pass -------
    // Seed identity at the head pre-activation (n_y one-hots) and retain ALL
    // neurons of ALL layers. The engine's Tier-0 capture stores, at each layer
    // (BEFORE its own gate transform), the coefficient of every seed row on that
    // relu's OUTPUT -> exactly (d y_head / d z_relu_out)^T over the current
    // (root) relaxation. Two lanes are captured because for layers with LIVE
    // relus between them and the head the coefficient is relaxation- and
    // direction-dependent; they agree when only linear ops intervene.
    let layers: Vec<RetainedLayer> = root
        .layers
        .iter()
        .map(|lg| RetainedLayer {
            idx: (0..lg.n).collect(),
            unst_pos: Vec::new(),
            a_l: Vec::new(),
            a_u: Vec::new(),
            naug: net.n_in + 1,
        })
        .collect();
    let retained = RetainedRows { layers };
    let seed = eng.identity_seed();
    let cap = eng
        .run_collect(
            &seed,
            None,
            LaneDir::Lower,
            None,
            Collect {
                unst_abs: false,
                rows: Some(&retained),
            },
        )
        .map_err(|e| e.to_string())?;
    let coll = cap
        .coll_rows
        .as_ref()
        .ok_or("engine returned no captured coefficient blocks")?;

    // ---- per-layer spectra -------------------------------------------------
    let mut reports: Vec<LayerReport> = Vec::new();
    for (li, lg) in root.layers.iter().enumerate() {
        let n_t = lg.n;
        let m = match coll.get(&li) {
            Some(m) if m.nrows() == n_t && m.ncols() == n_y => m,
            _ => continue, // layer not on the backward path / shape mismatch
        };
        let msl = m.as_slice().expect("standard layout");
        // relu OUTPUT box radius^2 per feature: rad_i = (relu(u)-relu(l))/2.
        let rad2: Vec<f64> = (0..n_t)
            .map(|i| {
                let r = (lg.u[i].max(0.0) - lg.l[i].max(0.0)) * 0.5;
                r * r
            })
            .collect();
        let eig_primary = spectrum(msl, n_t, n_y, &w_margin, None);
        let eig_width = spectrum(msl, n_t, n_y, &w_margin, Some(&rad2));
        reports.push(LayerReport {
            li,
            width: n_t,
            unstable: lg.unst.len(),
            eig_primary,
            eig_width,
        });
    }

    // Compact per-layer summary.
    eprintln!("{PFX} --- per-trunk-relu spectra (top8 energy fraction) ---");
    for r in &reports {
        let (p8, p_frob, p_er) = summarize(&r.eig_primary);
        let (w8, w_frob, w_er) = summarize(&r.eig_width);
        eprintln!(
            "{PFX}   li={} width={} unst={} | PRIMARY top8={:.3} effrank={:.2} Frob^2={:.3e} \
             | WIDTH-FAITHFUL top8={:.3} effrank={:.2} Frob^2={:.3e}{}",
            r.li,
            r.width,
            r.unstable,
            p8,
            p_er,
            p_frob,
            w8,
            w_er,
            w_frob,
            if w_frob == 0.0 {
                " [dead: no width]"
            } else {
                ""
            }
        );
    }

    // ---- primary layer choice + full spectrum ------------------------------
    // Default: the LAST trunk relu that actually carries head-box width (last
    // layer with a non-degenerate WIDTH-FAITHFUL Frobenius) — the best proxy for
    // "Relu_31, the last conv-block relu before the head". Overridable.
    let default_li = reports
        .iter()
        .rev()
        .find(|r| summarize(&r.eig_width).1 > 0.0)
        .or_else(|| reports.last())
        .map(|r| r.li);
    let li_t = env_usize("NY_HYPERPLANE_LAYER").or(default_li);
    let chosen = li_t.and_then(|li| reports.iter().find(|r| r.li == li));
    let Some(rep) = chosen else {
        return Err("no trunk relu layer available for the primary verdict".into());
    };
    eprintln!(
        "{PFX} === PRIMARY layer li={} (width={}, unstable={}) — last width-carrying trunk relu \
         / best Relu_31 proxy ===",
        rep.li, rep.width, rep.unstable
    );
    let verdict_top8 = report_full(
        "class-weighted (task formula, PRIMARY)",
        &rep.eig_primary,
        n_y,
    );
    let width_top8 = report_full(
        "class-weighted x relu-radius (width-faithful)",
        &rep.eig_width,
        n_y,
    );

    // ---- verdict -----------------------------------------------------------
    let verdict = |top8: f64| {
        if top8 >= 0.70 {
            "GREENLIGHT"
        } else if top8 < 0.30 {
            "KILL"
        } else {
            "AMBIGUOUS"
        }
    };
    eprintln!(
        "{PFX} VERDICT (layer li={}, margin class {j_star}): task-formula top8={verdict_top8:.3} \
         => {}  |  width-faithful top8={width_top8:.3} => {}",
        rep.li,
        verdict(verdict_top8),
        verdict(width_top8)
    );

    // ---- METRIC B: hyperplane Lagrangian cut vs best single-neuron axis -----
    // Does a hyperplane split actually beat an axis-aligned single-neuron split
    // at MOVING the binding-margin bound? Read-only A/B (measure, never decide).
    if let Some(m) = coll.get(&rep.li) {
        let blb = lbs.get(binding_row).copied().unwrap_or(f64::NAN);
        // Weight the cut by the BINDING margin (the bound metric B moves), so
        // the A/B is self-consistent even when class 55 is not binding.
        let w_bind: Vec<f64> = (0..n_y)
            .map(|k| (w2[t * n_y + k] - w2[binding_class * n_y + k]).abs())
            .collect();
        if let Err(e) = metric_b(
            &eng,
            root,
            &mb,
            &ms,
            binding_row,
            blb,
            &ybox,
            rep.li,
            m,
            &w_bind,
            binding_class,
        ) {
            eprintln!("{PFX} metric-B aborted (read-only, no effect on verdict): {e}");
        }
    }
    Ok(())
}

/// Top LEFT singular vector `u` (head-pre-activation space, len `n_y`) of the
/// class-weighted Jacobian `J = diag(|w|)·Mᵀ` at the primary layer, plus its
/// singular value. `u` is the head-space direction of maximal trunk-driven,
/// class-weighted variance — the correlation-directed hyperplane normal.
fn top_singular_vec(m: &[f64], n_t: usize, n_y: usize, w: &[f64]) -> (Vec<f64>, f64) {
    // G = J Jᵀ  (n_y x n_y), G[a,b] = w[a] w[b] sum_i m[i,a] m[i,b].
    let mut g = vec![0.0f64; n_y * n_y];
    for i in 0..n_t {
        let row = &m[i * n_y..(i + 1) * n_y];
        for a in 0..n_y {
            let va = row[a];
            if va == 0.0 {
                continue;
            }
            let dst = &mut g[a * n_y..a * n_y + n_y];
            for (b, gv) in dst.iter_mut().enumerate() {
                *gv += va * row[b];
            }
        }
    }
    for a in 0..n_y {
        for b in 0..n_y {
            g[a * n_y + b] *= w[a] * w[b];
        }
    }
    let (vals, vecs) = jacobi_eig(g, n_y);
    let mut kmax = 0usize;
    for k in 1..n_y {
        if vals[k] > vals[kmax] {
            kmax = k;
        }
    }
    let u: Vec<f64> = (0..n_y).map(|r| vecs[r * n_y + kmax]).collect();
    (u, vals[kmax].max(0.0).sqrt())
}

/// Run one backward pass on a single-column seed and return the concretized
/// scalar (lower or upper lane, root box, engine directed rounding).
fn run_col(
    eng: &BackwardEngine<'_>,
    col: &[f64],
    ecol: Option<&[f64]>,
    lower: bool,
) -> Result<f64, String> {
    let n = col.len();
    let s = Array2::from_shape_vec((n, 1), col.to_vec()).map_err(|e| e.to_string())?;
    let e = match ecol {
        Some(ec) => Some(Array2::from_shape_vec((n, 1), ec.to_vec()).map_err(|x| x.to_string())?),
        None => None,
    };
    let dir = if lower {
        LaneDir::Lower
    } else {
        LaneDir::Upper
    };
    let pass = eng
        .run(&Seed { s, e }, None, dir, None, false)
        .map_err(|x| x.to_string())?;
    let v = if lower {
        eng.concretize_lower(&pass)
    } else {
        eng.concretize_upper(&pass)
    };
    Ok(v[0])
}

/// Maximize a concave `f(β)` over β>=0: geometric grid then ternary refine.
fn max_concave(betas: &[f64], f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let (mut bbest, mut ybest, mut bi) = (0.0f64, f(0.0), 0usize);
    for (i, &b) in betas.iter().enumerate() {
        let y = f(b);
        if y > ybest {
            ybest = y;
            bbest = b;
            bi = i;
        }
    }
    let mut lo = if bi > 0 { betas[bi - 1] } else { 0.0 };
    let mut hi = if bi + 1 < betas.len() {
        betas[bi + 1]
    } else {
        betas[bi] * 2.0 + 1e-9
    };
    for _ in 0..24 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) < f(m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    let bs = f64::midpoint(lo, hi);
    let ys = f(bs);
    if ys > ybest {
        (bs, ys)
    } else {
        (bbest, ybest)
    }
}

#[allow(clippy::too_many_arguments)]
fn metric_b(
    eng: &BackwardEngine<'_>,
    root: &RootGates,
    mb: &MarginBatch,
    ms: &MarginSeed,
    binding_row: usize,
    binding_lb: f64,
    ybox: &YBox,
    li: usize,
    m: &Array2<f64>,
    w_margin: &[f64],
    j_star: usize,
) -> Result<(), String> {
    let n_y = mb.n_y;
    let n_t = m.nrows();
    if m.ncols() != n_y {
        return Err("captured block width != n_y".into());
    }
    let msl = m.as_slice().expect("standard layout");
    let outward = root.mode.outward();

    // Top singular direction (head-pre-activation space).
    let (u, sigma1) = top_singular_vec(msl, n_t, n_y, w_margin);
    eprintln!(
        "{PFX} --- METRIC B: hyperplane cut vs best single-neuron axis split (layer li={li}) ---"
    );
    eprintln!(
        "{PFX}   cut = top-1 singular direction of class-{j_star}-weighted J, in HEAD pre-act \
         space: s = u·head (sigma1={sigma1:.4}). [A Relu_{li}-feature cut v·z is its dual; the \
         head cut is engine-native and directly tightens the head box.]"
    );

    // Interval of s = u·head over the relaxation (engine directed rounding).
    let a = run_col(eng, &u, None, true)?;
    let b = run_col(eng, &u, None, false)?;
    let tmid = f64::midpoint(a, b);
    eprintln!(
        "{PFX}   s=u·head interval=[{a:.5},{b:.5}] span={:.5} split t={tmid:.5}",
        b - a
    );

    // Binding-margin seed column (head-gate-composed) + its error column.
    let sc: Vec<f64> = (0..n_y).map(|i| ms.seed.s[[i, binding_row]]).collect();
    let se: Option<Vec<f64>> = ms
        .seed
        .e
        .as_ref()
        .map(|e| (0..n_y).map(|i| e[[i, binding_row]]).collect());
    let cst = ms.cst[binding_row];
    let cst_err = ms.cst_err[binding_row];
    // marg LB from a concretized coefficient (mirrors per_class_direct).
    let marg_lb = |low0: f64| -> f64 {
        let v = low0 + cst;
        if outward {
            next_down(next_down(v - next_up(cst_err)))
        } else {
            v
        }
    };

    // β grid scaled to the u·head span (margin folds are cheap single-column
    // passes); L(β) is concave so grid + ternary is ample.
    let span = (b - a).abs().max(1e-9);
    let betas: Vec<f64> = std::iter::once(0.0)
        .chain((0..11).map(|k| (0.03125 * 2f64.powi(k)) / span))
        .collect();

    // Hyperplane Lagrangian on side σ: {σ(u·head − t) >= 0}. σ=+1 => {u·head>=t}
    // folds −β(u·head−t); σ=−1 => {u·head<=t} folds −β(t−u·head).
    let solve_side = |sigma: f64| -> (f64, f64) {
        let mut f = |beta: f64| -> f64 {
            let col: Vec<f64> = (0..n_y).map(|i| sc[i] - sigma * beta * u[i]).collect();
            match run_col(eng, &col, se.as_deref(), true) {
                Ok(low0) => marg_lb(low0) + sigma * beta * tmid,
                Err(_) => f64::NEG_INFINITY,
            }
        };
        max_concave(&betas, &mut f)
    };
    let (bstar_ge, lstar_ge) = solve_side(1.0);
    let (bstar_le, lstar_le) = solve_side(-1.0);
    // Best side (matches the axis "better branch" comparison).
    let (side_sign, bstar, lstar) = if lstar_ge >= lstar_le {
        (1.0, bstar_ge, lstar_ge)
    } else {
        (-1.0, bstar_le, lstar_le)
    };
    let dlb_hyper = lstar - binding_lb;

    // Head-box tightening on the best side: batched over all 100 head neurons,
    // a few β values (100-column passes are the expensive part, keep it small).
    let (mut ly2, mut uy2) = (ybox.ly.clone(), ybox.uy.clone());
    let box_betas = [bstar.max(0.25 / span), 1.0 / span, 4.0 / span];
    for &beta in &box_betas {
        if beta <= 0.0 {
            continue;
        }
        let mut sl = Array2::<f64>::zeros((n_y, n_y));
        let mut su = Array2::<f64>::zeros((n_y, n_y));
        for a in 0..n_y {
            for i in 0..n_y {
                let ident = if i == a { 1.0 } else { 0.0 };
                sl[[i, a]] = ident - side_sign * beta * u[i];
                su[[i, a]] = ident + side_sign * beta * u[i];
            }
        }
        let cl = {
            let p = eng
                .run(&Seed { s: sl, e: None }, None, LaneDir::Lower, None, false)
                .map_err(|x| x.to_string())?;
            eng.concretize_lower(&p)
        };
        let cu = {
            let p = eng
                .run(&Seed { s: su, e: None }, None, LaneDir::Upper, None, false)
                .map_err(|x| x.to_string())?;
            eng.concretize_upper(&p)
        };
        for a in 0..n_y {
            ly2[a] = ly2[a].max(cl[a] + side_sign * beta * tmid);
            uy2[a] = uy2[a].min(cu[a] - side_sign * beta * tmid);
        }
    }
    let (w_before_sum, w_before_max) = box_widths(&ybox.ly, &ybox.uy);
    let (w_after_sum, w_after_max) = box_widths(&ly2, &uy2);
    let shrink_sum = pct_shrink(w_before_sum, w_after_sum);
    let shrink_max = pct_shrink(w_before_max, w_after_max);

    eprintln!(
        "{PFX}   HYPERPLANE (best side u·head{}t, β*={bstar:.4}): Δlb={dlb_hyper:+.6} \
         (lb {binding_lb:.5} -> {lstar:.5}); [≥t Δlb={:+.6}, ≤t Δlb={:+.6}]",
        if side_sign > 0.0 { ">=" } else { "<=" },
        lstar_ge - binding_lb,
        lstar_le - binding_lb
    );
    eprintln!(
        "{PFX}   HYPERPLANE head-box width: sum {w_before_sum:.4}->{w_after_sum:.4} \
         ({shrink_sum:.1}% shrink), max {w_before_max:.4}->{w_after_max:.4} ({shrink_max:.1}% shrink)"
    );

    // ---- axis baseline: best single-neuron pin at the same layer -----------
    // Each pin needs a full head-box refresh (y_rows) + margin pass = expensive,
    // so cap at the top-scored unstable neurons (what the brancher would pick).
    let cap = env_usize("NY_HYPERPLANE_AXIS_MAX").unwrap_or(4);
    let lg = &root.layers[li];
    // score unstable positions by relaxation slack c*(u-l) (standard split score).
    let mut order: Vec<usize> = (0..lg.unst.len()).collect();
    order.sort_by(|&p, &q| {
        let sp = lg.c[lg.unst[p]] * (lg.u[lg.unst[p]] - lg.l[lg.unst[p]]);
        let sq = lg.c[lg.unst[q]] * (lg.u[lg.unst[q]] - lg.l[lg.unst[q]]);
        sq.partial_cmp(&sp).unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(cap);
    let mut axis_best_lb = f64::NEG_INFINITY;
    let mut axis_best_desc = (usize::MAX, 0i8);
    let mut axis_best_box = (w_before_sum, w_before_max);
    let mut evaluated = 0usize;
    for &p in &order {
        for d in [1i8, -1i8] {
            let dom = domain_gates(root, &[(li, p, d)]);
            let (al2, au2) = match eng.y_rows(Some(&dom)) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let ybox2 = YBox::from_rows(eng, &al2, &au2);
            if ybox2.is_empty() {
                continue; // infeasible branch (neuron effectively stable)
            }
            let gates2 = head_gates(&ybox2, root.mode);
            let ms2 = margin_seed(mb, &gates2, &ybox2, root.mode);
            let pass2 = match eng.run(&ms2.seed, Some(&dom), LaneDir::Lower, None, false) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let lb2 = per_class_direct(eng, &pass2, &ms2, binding_row..binding_row + 1)[0];
            evaluated += 1;
            if lb2 > axis_best_lb {
                axis_best_lb = lb2;
                axis_best_desc = (lg.unst[p], d);
                axis_best_box = box_widths(&ybox2.ly, &ybox2.uy);
            }
        }
    }
    let dlb_axis = if axis_best_lb.is_finite() {
        axis_best_lb - binding_lb
    } else {
        0.0
    };
    let axis_shrink_sum = pct_shrink(w_before_sum, axis_best_box.0);
    eprintln!(
        "{PFX}   BEST-AXIS (neuron {} branch {}, {evaluated} pins evaluated of {} unstable): \
         Δlb={dlb_axis:+.5} (lb -> {axis_best_lb:.5}); head-box sum {w_before_sum:.4}->{:.4} \
         ({axis_shrink_sum:.1}% shrink)",
        axis_best_desc.0,
        if axis_best_desc.1 > 0 {
            "active"
        } else {
            "inactive"
        },
        lg.unst.len(),
        axis_best_box.0,
    );

    // ---- A/B verdict -------------------------------------------------------
    let beats_axis = dlb_hyper > dlb_axis;
    let verdict = if shrink_sum < 2.0 || !beats_axis {
        "KILL"
    } else if shrink_sum >= 10.0 && dlb_hyper >= 0.1 && beats_axis {
        "GREENLIGHT"
    } else {
        "AMBIGUOUS"
    };
    eprintln!(
        "{PFX}   METRIC-B VERDICT: Δlb_hyperplane={dlb_hyper:+.5} vs Δlb_best-axis={dlb_axis:+.5} \
         (hyperplane {} axis) | head-box shrink {shrink_sum:.1}% => {verdict}",
        if beats_axis { "BEATS" } else { "<=" }
    );
    Ok(())
}

/// (sum of per-neuron widths, max width) of a box.
fn box_widths(ly: &[f64], uy: &[f64]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut mx = 0.0f64;
    for (l, u) in ly.iter().zip(uy) {
        let w = (u - l).max(0.0);
        sum += w;
        mx = mx.max(w);
    }
    (sum, mx)
}

fn pct_shrink(before: f64, after: f64) -> f64 {
    if before <= 0.0 {
        0.0
    } else {
        100.0 * (before - after) / before
    }
}

/// Cyclic Jacobi with eigenVECTORS. Returns `(eigenvalues[n], V[n*n] row-major)`
/// where column `k` of `V` is the eigenvector for `eigenvalues[k]`.
fn jacobi_eig(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n == 0 {
        return (Vec::new(), v);
    }
    let at = |i: usize, j: usize| i * n + j;
    let mut scale = 0.0f64;
    for k in 0..n {
        scale += a[at(k, k)].abs();
    }
    let tol = 1e-20 * (scale * scale + 1.0);
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[at(p, q)] * a[at(p, q)];
            }
        }
        if off <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[at(p, q)];
                if apq == 0.0 {
                    continue;
                }
                let (app, aqq) = (a[at(p, p)], a[at(q, q)]);
                let theta = (aqq - app) / (2.0 * apq);
                let trot = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (trot * trot + 1.0).sqrt();
                let s = trot * c;
                for k in 0..n {
                    if k != p && k != q {
                        let (akp, akq) = (a[at(k, p)], a[at(k, q)]);
                        let npk = c * akp - s * akq;
                        let nqk = s * akp + c * akq;
                        a[at(k, p)] = npk;
                        a[at(p, k)] = npk;
                        a[at(k, q)] = nqk;
                        a[at(q, k)] = nqk;
                    }
                }
                let npp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                let nqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[at(p, p)] = npp;
                a[at(q, q)] = nqq;
                a[at(p, q)] = 0.0;
                a[at(q, p)] = 0.0;
                for k in 0..n {
                    let (vkp, vkq) = (v[at(k, p)], v[at(k, q)]);
                    v[at(k, p)] = c * vkp - s * vkq;
                    v[at(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    let vals = (0..n).map(|i| a[at(i, i)]).collect();
    (vals, v)
}

/// (top8 energy fraction, Frobenius^2, participation eff-rank) for a spectrum.
fn summarize(eig: &[f64]) -> (f64, f64, f64) {
    let total: f64 = eig.iter().map(|e| e.max(0.0)).sum();
    if total <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let top8: f64 = eig.iter().take(8).map(|e| e.max(0.0)).sum::<f64>() / total;
    let sumsq: f64 = eig.iter().map(|e| e.max(0.0).powi(2)).sum();
    let er = if sumsq > 0.0 {
        total * total / sumsq
    } else {
        0.0
    };
    (top8, total, er)
}

/// Log one spectrum in full; returns the cumulative top-8 energy fraction.
fn report_full(label: &str, eig: &[f64], n_y: usize) -> f64 {
    let total: f64 = eig.iter().map(|e| e.max(0.0)).sum();
    if total <= 0.0 {
        eprintln!("{PFX} [{label}] degenerate spectrum (||J||_F^2 = 0)");
        return 0.0;
    }
    let sig: Vec<f64> = eig.iter().map(|e| e.max(0.0).sqrt()).collect();
    let cum = |k: usize| eig.iter().take(k).map(|e| e.max(0.0)).sum::<f64>() / total;
    let sumsq: f64 = eig.iter().map(|e| e.max(0.0).powi(2)).sum();
    let part_ratio = if sumsq > 0.0 {
        total * total / sumsq
    } else {
        0.0
    };
    let shannon = {
        let mut h = 0.0;
        for e in eig {
            let p = e.max(0.0) / total;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h.exp()
    };
    let show = sig.len().min(16);
    let head: Vec<String> = sig[..show].iter().map(|s| format!("{s:.4}")).collect();
    eprintln!(
        "{PFX} [{label}] rank<= {n_y}  ||J||_F^2={total:.6e}  top{show} singular values: [{}]",
        head.join(", ")
    );
    eprintln!(
        "{PFX} [{label}] cum energy: top1={:.3} top2={:.3} top4={:.3} top8={:.3} top16={:.3}  \
         eff_rank(participation)={part_ratio:.2} eff_rank(shannon)={shannon:.2}",
        cum(1),
        cum(2),
        cum(4),
        cum(8),
        cum(16)
    );
    cum(8)
}

/// Eigenvalues (= squared singular values) of the weighted Jacobian's Gram
/// matrix `G = J J^T`, sorted descending. `J[a,i] = w[a] * m[i,a] * rw[i]`, so
/// `G[a,b] = w[a] w[b] * sum_i rw2[i] * m[i,a] * m[i,b]`. `rw2 = None` => rw==1.
fn spectrum(m: &[f64], n_t: usize, n_y: usize, w: &[f64], rw2: Option<&[f64]>) -> Vec<f64> {
    let mut g = vec![0.0f64; n_y * n_y];
    for i in 0..n_t {
        let scale = rw2.map_or(1.0, |r| r[i]);
        if scale == 0.0 {
            continue;
        }
        let row = &m[i * n_y..(i + 1) * n_y];
        for a in 0..n_y {
            let va = scale * row[a];
            if va == 0.0 {
                continue;
            }
            let dst = &mut g[a * n_y..a * n_y + n_y];
            for (b, gv) in dst.iter_mut().enumerate() {
                *gv += va * row[b];
            }
        }
    }
    for a in 0..n_y {
        for b in 0..n_y {
            g[a * n_y + b] *= w[a] * w[b];
        }
    }
    let mut eig = jacobi_eigenvalues(g, n_y);
    eig.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eig
}

/// Cyclic Jacobi eigenvalue algorithm for a symmetric `n x n` matrix (row-major,
/// consumed in place). Returns the eigenvalues (diagonal after convergence).
/// Values-only (no eigenvectors); ample for a spectrum diagnostic.
fn jacobi_eigenvalues(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let at = |i: usize, j: usize| i * n + j;
    let mut scale = 0.0f64;
    for k in 0..n {
        scale += a[at(k, k)].abs();
    }
    let tol = 1e-18 * (scale * scale + 1.0);
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[at(p, q)] * a[at(p, q)];
            }
        }
        if off <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[at(p, q)];
                if apq == 0.0 {
                    continue;
                }
                let app = a[at(p, p)];
                let aqq = a[at(q, q)];
                let theta = (aqq - app) / (2.0 * apq);
                // t = sign(theta)/(|theta| + sqrt(theta^2+1)); theta==0 => t=1.
                let trot = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (trot * trot + 1.0).sqrt();
                let s = trot * c;
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[at(k, p)];
                        let akq = a[at(k, q)];
                        let npk = c * akp - s * akq;
                        let nqk = s * akp + c * akq;
                        a[at(k, p)] = npk;
                        a[at(p, k)] = npk;
                        a[at(k, q)] = nqk;
                        a[at(q, k)] = nqk;
                    }
                }
                let npp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                let nqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[at(p, p)] = npp;
                a[at(q, q)] = nqq;
                a[at(p, q)] = 0.0;
                a[at(q, p)] = 0.0;
            }
        }
    }
    (0..n).map(|i| a[at(i, i)]).collect()
}

#[cfg(test)]
mod tests {
    use super::jacobi_eigenvalues;

    #[test]
    fn jacobi_matches_known_diag_and_symmetric() {
        // Diagonal matrix: eigenvalues are the diagonal.
        let d = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 7.0];
        let mut e = jacobi_eigenvalues(d, 3);
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e[0] - 1.0).abs() < 1e-9);
        assert!((e[1] - 3.0).abs() < 1e-9);
        assert!((e[2] - 7.0).abs() < 1e-9);

        // 2x2 [[2,1],[1,2]] -> eigenvalues 1 and 3.
        let s = vec![2.0, 1.0, 1.0, 2.0];
        let mut e2 = jacobi_eigenvalues(s, 2);
        e2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((e2[0] - 1.0).abs() < 1e-9);
        assert!((e2[1] - 3.0).abs() < 1e-9);

        // Rank-1 PSD w w^T with w=[1,2,2]: eigenvalues 9,0,0; trace preserved.
        let w = [1.0f64, 2.0, 2.0];
        let mut g = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                g[i * 3 + j] = w[i] * w[j];
            }
        }
        let e3 = jacobi_eigenvalues(g, 3);
        let total: f64 = e3.iter().sum();
        let mx = e3.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((total - 9.0).abs() < 1e-9);
        assert!((mx - 9.0).abs() < 1e-9); // rank-1 => all energy in top-1
    }
}
