// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #margin-row-beta (`NY_MARGIN_ROW_BETA=1`, default OFF): β-CROWN split
//! Lagrangians for the margin-row lane (#twinwall).
//!
//! WHY. The lane's splits only piece-FIX a neuron's gate
//! (`engine.rs::domain_gates_split_only`): the split constraint
//! `z >= 0` / `z <= 0` never enters the bound as a dual term, which is why
//! children barely improve on parents (measured frontier explosion: idx_8600
//! 18 → 415 open domains at depth 30 while idx_6659 drains and proves).
//! alpha-beta-CROWN attaches one `beta_j >= 0` per split; this module does the
//! same for THIS lane's backward pass.
//!
//! THE MATH (weak duality; also at [`super::engine::BetaSplit`]). A domain's
//! region is `box ∩_j {s_j * z_j(x) >= 0}` with `s_j` the split sign. For any
//! `beta >= 0`:
//!
//! ```text
//!   min_{region} f >= min_{region} [f - Σ_j beta_j s_j z_j]
//!                  >= certified-lower-bound_{box}( relax(f - Σ beta_j s_j z_j) )
//! ```
//!
//! The engine realizes the extra terms as coefficient shifts of `-s_j*beta_j`
//! on `z_j` right after that relu's gate transform (Lower lane; `+s_j*beta_j`
//! on the Upper lane), so the ONE certified concretize remains the scorer:
//! every accepted β is accepted because the unchanged certified pass said its
//! bound is better. A bad β proposal costs one pass; it cannot move a verdict.
//!
//! THE OPTIMIZER (heuristic, direction-only). Projected supergradient ascent
//! reusing the machinery `alpha_opt` already validated: under the frozen-sign
//! linearization, `d bound / d beta_j = -s_j * ẑ_j(x*)`, where `ẑ_j(x*)` is
//! the linearized walk value of split `j`'s pre-activation at the concretize
//! argmin `x*` of the worst class row. Positive gradient means the relaxed
//! minimizer VIOLATES the split constraint, so penalizing it (raising β) can
//! lift the bound. The step rule ([`polyak_step`], default) is the classic
//! known-target Polyak subgradient step against the target τ = 0 (closure
//! needs `b > 0`): `t = λ·(0 − b_direct)/Σ_movable g²`, sized to close the
//! DIRECT-PATH gap of the worst column in one move; the concave scorer +
//! monotone accept absorbs overshoot. `NY_MARGIN_ROW_BETA_POLYAK=0` restores
//! the legacy sign-only step `±η·|v_k|` so the A/B lives in ONE binary.
//! Children inherit their parent's accepted β (valid: the child region is a
//! subset of the parent's, so every parent constraint still holds) and the
//! fresh split starts at β = 0 (the term vanishes ⇒ trivially valid).
//!
//! HEAD TERMS (#margin-row-beta C3). Head splits `s_i·y_i >= 0` live at the
//! exact layer the margin seed multiplies, so their Lagrangian
//! `f − Σ_i βʰ_i s_i y_i` needs NO engine change: [`seed_with_head_terms`]
//! shifts the seed coefficient by `−s_i·βʰ_i` per column and charges the one
//! f64 add into the seed's certified error lane. Their supergradient
//! `gʰ_i = −s_i·ŷ_i(x*)` comes from the SAME linearized walk, extended one
//! layer higher (through the first head Gemm).
//!
//! PER-COLUMN β (#margin-row-beta-percol, `NY_MARGIN_ROW_BETA_PERCOL=1`,
//! default OFF). The shared vector above is scored against the single worst
//! column: every OTHER column's split-constraint violations stay unpriced,
//! which is exactly why sibling objectives stay open while the worst one
//! tightens (measured: shared-β closes ~2× the children but converts
//! nothing). abc optimizes β PER OBJECTIVE; this arm does the same for the K
//! worst failing columns (`NY_MARGIN_ROW_BETA_COLS`, default 8): each carries
//! its own trunk+head β (inherited like the shared vector), its own Polyak
//! gap (`−per_j[c]`) and its own supergradient (one linearized walk per
//! column — the walk's x* and frozen-sign selections are column-dependent).
//! ONE certified pass still scores ALL columns simultaneously per trial;
//! acceptance is MONOTONE PER COLUMN: `per_best[c] = max` over scored trials
//! of column c's certified direct value — sound because EVERY trial's β pack
//! is a valid per-column Lagrangian (β >= 0 per column), so each scored value
//! is a valid bound for its column regardless of what other columns' β did,
//! and the domain bound `min_c max(per_best[c], m1[c], m2v[c])` is a min of
//! valid column bounds. Engine terms ride `DomainGates::beta_pc`
//! (`apply_beta_terms` restricted to `col..col+1` — the stacked arm's
//! column-range anatomy on the uniform path); head terms ride
//! [`seed_with_head_terms_pc`]. OFF ⇒ bit-identical shared-vector behavior.
//!
//! COST when armed: one extra certified pass per accepted/rejected trial
//! (`NY_MARGIN_ROW_BETA_ITERS`, default 1) per expanded domain with splits,
//! plus one cheap single-column linearized walk (K walks per iteration in
//! per-column mode). Disarmed: zero — the `DomainGates::beta` map stays
//! empty, the seed is the untouched base margin seed, and the engine's
//! application site is never entered (bit-identical passes).

use super::bab::{HeadFix, TrunkSplit};
use super::engine::{BetaSplit, DomainGates, PassOut, PcBetaCol, Seed};
use super::net::TwinNet;
use super::root::RootGates;
use super::rounding::{slack16, UNIT};

/// #beta-close-eps: the Polyak target sits THIS far past zero, because the
/// lane's closure test is STRICT (`b > 0`) and a zero-target ascent converges
/// to the threshold without crossing it (measured: a domain stalled at
/// -2.5e-5 with step 1.7e-12). Sized well above f64 noise and well below any
/// bound scale that matters (the smallest banked closing margin is ~7e-4).
pub(crate) const CLOSE_EPS: f64 = 1e-4;

/// Is the lane's β-CROWN armed? Exact `"1"` only; default OFF.
///
/// Declared as `decls::margin_row::MARGIN_ROW_BETA`; the chokepoint pins the
/// same exact-`"1"` arming rule this read had before the migration, and records
/// a near-miss token (`"true"`, `" 1"`) as a REJECTION instead of losing it.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BETA)
            .value
            .as_bool();
        if on {
            // Engagement telemetry (R9): an armed run announces itself once
            // even if no domain ever carries a split (e.g. closed at root),
            // so an inert arming is detectable in one log.
            eprintln!("[beta] armed (NY_MARGIN_ROW_BETA=1)");
        }
        on
    })
}

/// Relative step size (units of the split neuron's incoming |coefficient|).
/// LEGACY sign-step only (`NY_MARGIN_ROW_BETA_POLYAK=0`); the Polyak rule
/// ignores it.
pub(crate) fn eta() -> f64 {
    static E: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        // The declaration carries `trim().parse::<f64>()` + `is_finite()` and a
        // CLOSED lower bound at zero; the `> 0.0` half of the legacy filter
        // stays HERE on purpose, so an explicit `0` remains distinguishable
        // from absence at the chokepoint (both still resolve to 0.5).
        ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BETA_ETA)
            .value
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.5)
    })
}

/// Ascent trials per domain evaluation (each = one certified pass).
pub(crate) fn iters() -> usize {
    static I: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *I.get_or_init(|| {
        // `clamp(1, 8)` stays at the reader: the chokepoint hands back an
        // explicit `0` rather than swallowing it, and "explicitly zero" and
        // "absent" are different facts for the receipt even though both end
        // up running one trial.
        ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BETA_ITERS)
            .value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(1)
            .clamp(1, 8)
    })
}

/// Polyak relaxation factor λ (`NY_MARGIN_ROW_BETA_LAMBDA`, default 1.0):
/// the fraction of the direct-path gap the step is sized to close.
pub(crate) fn lambda() -> f64 {
    static L: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        // The `> 0.0` half of the legacy filter stays here: the chokepoint hands
        // an explicit `0` back rather than swallowing it, so the receipt can
        // still distinguish "explicitly zero" from "absent". Both land on 1.0.
        ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BETA_LAMBDA)
            .value
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

/// Recovery cap for the per-node λ memory: accepts grow λ by 1.5× back up to
/// this, so a user-raised `NY_MARGIN_ROW_BETA_LAMBDA` is never clawed back
/// below its configured value.
pub(crate) fn lambda_cap() -> f64 {
    lambda().max(1.0)
}

/// λ floor: the ascent loop stops halving below 2⁻⁴ (a step this small is the
/// measured-epsilon regime the fix exists to escape).
pub(crate) const LAMBDA_MIN: f64 = 0.0625;

/// Gap-targeted Polyak step rule (`NY_MARGIN_ROW_BETA_POLYAK`): default ON
/// when β is armed; exactly `"0"` restores the legacy sign-step so the A/B
/// lives in ONE binary.
pub(crate) fn polyak() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // OPT-OUT: declared `Bool(true)`, so absence and any non-`0` token keep it
    // engaged exactly as `!= Some("0")` did.
    *P.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BETA_POLYAK)
            .value
            .as_bool()
    })
}

/// Head-split β terms (`NY_MARGIN_ROW_BETA_HEADS`): default ON when β is
/// armed; exactly `"0"` = trunk-only (the heads attribution arm).
pub(crate) fn heads_on() -> bool {
    static H: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_BETA_HEADS)
            .value
            .as_bool()
    })
}

/// #margin-row-beta-percol (`NY_MARGIN_ROW_BETA_PERCOL=1`, default OFF): each
/// of the K worst FAILING columns carries and ascends its OWN β vector instead
/// of one shared vector scored against the single worst column. Exact `"1"`
/// only; OFF preserves the shared-vector lane bit-identically. Only meaningful
/// with `NY_MARGIN_ROW_BETA=1`; the per-column ascent always uses the Polyak
/// rule (the legacy sign-step has no per-column gap to target).
pub(crate) fn percol() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        let on = std::env::var("NY_MARGIN_ROW_BETA_PERCOL").ok().as_deref() == Some("1");
        if on {
            // Engagement telemetry (R9): announce arming once even if no
            // domain ever reaches the ascent.
            eprintln!(
                "[beta-pc] armed (NY_MARGIN_ROW_BETA_PERCOL=1, cols={})",
                beta_cols()
            );
        }
        on
    })
}

/// #margin-row-beta-percol: how many worst failing columns get their own β
/// (`NY_MARGIN_ROW_BETA_COLS`, default 8, clamped to `[1, 32]`). The honest
/// budget: each selected column costs one single-column linearized walk per
/// ascent iteration (milliseconds); the certified pass — the dominant cost —
/// stays ONE per iteration regardless of K.
pub(crate) fn beta_cols() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("NY_MARGIN_ROW_BETA_COLS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(8)
            .clamp(1, 32)
    })
}

/// Install `betas` (aligned with `trunk`) as the domain's engine terms.
/// Zero and non-finite entries are dropped (a `beta = 0` term must not even
/// perturb a `-0.0` coefficient bit-wise; the engine guards this too).
pub(crate) fn set_terms(
    dom: &mut DomainGates,
    root: &RootGates,
    trunk: &[TrunkSplit],
    betas: &[f64],
) {
    dom.beta.clear();
    for (k, &(li, pos, sign)) in trunk.iter().enumerate() {
        let beta = betas.get(k).copied().unwrap_or(0.0);
        if !(beta.is_finite() && beta > 0.0) || sign == 0 {
            continue;
        }
        let Some(&neuron) = root.layers.get(li).and_then(|rec| rec.unst.get(pos)) else {
            continue;
        };
        dom.beta
            .entry(li)
            .or_default()
            .push(BetaSplit { neuron, sign, beta });
    }
}

/// #margin-row-beta-percol: install PER-COLUMN trunk terms as the domain's
/// `beta_pc` engine terms. `cols` pairs a seed column with its β vector
/// (aligned with `trunk`, exactly as [`set_terms`]). Filters are identical to
/// [`set_terms`]: zero/non-finite β and sign-0 terms are dropped (a zero term
/// must not perturb a `-0.0` coefficient bit-wise), unknown layers/positions
/// are skipped (loosening-only).
///
/// SOUNDNESS: weak duality holds PER COLUMN — column `c`'s objective on the
/// region `box ∩_j {s_j·z_j >= 0}` satisfies
/// `min f_c >= certified-lower-bound(relax(f_c − Σ_j β_{j,c}·s_j·z_j))` for
/// ANY `β_{·,c} >= 0`, independently of every other column's multipliers.
/// The engine applies each column's terms to that column only (same certified
/// arithmetic as the shared map, column-range-restricted; see
/// `engine::PcBetaCol`), composing ADDITIVELY with any shared terms (a sum of
/// valid multipliers is a valid multiplier).
pub(crate) fn set_terms_pc(
    dom: &mut DomainGates,
    root: &RootGates,
    trunk: &[TrunkSplit],
    cols: &[(usize, &[f64])],
) {
    dom.beta_pc.clear();
    for &(col, betas) in cols {
        for (k, &(li, pos, sign)) in trunk.iter().enumerate() {
            let beta = betas.get(k).copied().unwrap_or(0.0);
            if !(beta.is_finite() && beta > 0.0) || sign == 0 {
                continue;
            }
            let Some(&neuron) = root.layers.get(li).and_then(|rec| rec.unst.get(pos)) else {
                continue;
            };
            let layer = dom.beta_pc.entry(li).or_default();
            let term = BetaSplit { neuron, sign, beta };
            match layer.iter_mut().find(|pc| pc.col == col) {
                Some(pc) => pc.terms.push(term),
                None => layer.push(PcBetaCol {
                    col,
                    terms: vec![term],
                }),
            }
        }
    }
}

/// #margin-row-beta C3: the head-split Lagrangian, realized as a SEED SHIFT —
/// no engine change. For head fixes `(i, s_i)` with multipliers `βʰ_i >= 0`:
///
/// ```text
///   seed'.s[[i, col]] = seed.s[[i, col]] − s_i·βʰ_i          for all col
///   seed'.e[[i, col]] = slack16(seed.e[[i, col]] + 2·UNIT·|seed'.s[[i, col]]|)
/// ```
///
/// No bias term: the split point is 0 (mirrors `apply_beta_terms`,
/// engine.rs). Note the `s_i = −1` branch: the head gate zeroes that seed row
/// entirely (bounds.rs: alpha = s = 0 on the inactive fix), so today the
/// direct pass carries NO information from `y_i <= 0` beyond the gate —
/// head-β re-introduces it as `+βʰ·y_i`, a term the bound currently cannot
/// express at all.
///
/// SOUNDNESS (the design's one new proof obligation). On the child region
/// `s_i·y_i >= 0` holds by construction of the head split (the ybox is
/// clamped the same way), so for any `βʰ >= 0`,
/// `f − Σ βʰ_i s_i y_i <= f` pointwise on the region, and the certified
/// box-lower-bound of the relaxed shifted objective lower-bounds `f` on the
/// region — the identical weak-duality argument as [`BetaSplit`], applied one
/// layer higher. `β >= 0` is enforced by the same projection upstream. Error
/// lane: the shift is an additive constant per seed entry — an exact
/// `β·(±1.0)` f64 multiply feeding ONE f64 add — charged `2·UNIT·|result|`
/// via `slack16`, the same invariant `apply_beta_terms` documents: an
/// additive constant does not change any downstream Lipschitz constant, so
/// the carried error needs no rescaling. The parallel bound routes `m1`/`m2v`
/// never see the shift and remain independently valid; `max` of valid lower
/// bounds is valid.
///
/// `seed.e == None` handling (adversarial must-fix #4): the error matrix
/// exists only in outward mode (bounds.rs: `mode.outward().then(...)`), and —
/// VERIFIED — the β path does NOT refuse parity mode upstream (`beta_armed`
/// checks only env + splits; `apply_beta_terms` likewise skips its charge
/// when the lane matrix has no error plane). So:
/// * outward + `e == None`: REFUSE (`None`) — a certified lane may never take
///   an uncharged shift. Unreachable today by construction; defensive.
/// * parity (`!outward`) + `e == None`: shift charge-free — the parity lane
///   carries no error anywhere by design (Python-parity testing only, never a
///   competition lane), exactly matching the trunk-β site's behavior there.
///
/// Skips `β <= 0` / non-finite / sign-0 terms exactly as [`set_terms`] (a
/// zero term must not perturb a `-0.0` coefficient bit-wise). Returns `None`
/// on any shape drift — the caller falls back to no-head-β for this eval.
pub(crate) fn seed_with_head_terms(
    base: &Seed,
    heads: &[HeadFix],
    head_betas: &[f64],
    outward: bool,
) -> Option<Seed> {
    if outward && base.e.is_none() {
        return None;
    }
    let mut s = base.s.clone();
    let mut e = base.e.clone();
    {
        let n_y = s.nrows();
        let nf = s.ncols();
        let ss = s.as_slice_mut()?;
        let mut es = e.as_mut().and_then(|m| m.as_slice_mut());
        for (k, &(i, sign)) in heads.iter().enumerate() {
            let beta = head_betas.get(k).copied().unwrap_or(0.0);
            if !(beta.is_finite() && beta > 0.0) || sign == 0 {
                continue;
            }
            if i >= n_y {
                return None;
            }
            // Lower lane bounds `f − Σ βʰ s y` from below ⇒ shift `−s·βʰ`.
            let delta = -f64::from(sign.signum()) * beta;
            for c0 in 0..nf {
                let v = ss[i * nf + c0] + delta;
                ss[i * nf + c0] = v;
                if let Some(es) = es.as_deref_mut() {
                    es[i * nf + c0] = slack16(es[i * nf + c0] + 2.0 * UNIT * v.abs());
                }
            }
        }
    }
    Some(Seed { s, e })
}

/// #margin-row-beta-percol: PER-COLUMN head-split seed shift. Identical
/// per-entry math and error charge as [`seed_with_head_terms`], restricted to
/// each pair's OWN column:
///
/// ```text
///   seed'.s[[i, col]] = seed.s[[i, col]] − s_i·βʰ_{i,col}
///   seed'.e[[i, col]] = slack16(seed.e[[i, col]] + 2·UNIT·|seed'.s[[i, col]]|)
/// ```
///
/// `cols` pairs a seed column with its head-β vector (aligned with `heads`).
/// SOUNDNESS is the [`seed_with_head_terms`] argument applied per column:
/// column `col`'s functional is relaxed by ITS OWN valid Lagrangian
/// (`βʰ >= 0`, `s_i·y_i >= 0` on the region), other columns' seed entries are
/// untouched. The additive-constant E-lane invariant holds per touched entry
/// (one exact `β·(±1.0)` multiply feeding ONE f64 add, charged
/// `2·UNIT·|result|`; no downstream Lipschitz constant moves). Composes with
/// the SHARED shift by chaining (apply the shared shift first, then this on
/// the result: effective multiplier = shared + per-column >= 0, each add
/// charged). Refusals (`None`) as in [`seed_with_head_terms`], plus
/// out-of-range columns.
pub(crate) fn seed_with_head_terms_pc(
    base: &Seed,
    heads: &[HeadFix],
    cols: &[(usize, &[f64])],
    outward: bool,
) -> Option<Seed> {
    if outward && base.e.is_none() {
        return None;
    }
    let mut s = base.s.clone();
    let mut e = base.e.clone();
    {
        let n_y = s.nrows();
        let nf = s.ncols();
        let ss = s.as_slice_mut()?;
        let mut es = e.as_mut().and_then(|m| m.as_slice_mut());
        for &(col, head_betas) in cols {
            if col >= nf {
                return None;
            }
            for (k, &(i, sign)) in heads.iter().enumerate() {
                let beta = head_betas.get(k).copied().unwrap_or(0.0);
                if !(beta.is_finite() && beta > 0.0) || sign == 0 {
                    continue;
                }
                if i >= n_y {
                    return None;
                }
                // Lower lane bounds `f_col − Σ βʰ s y` from below ⇒ `−s·βʰ`.
                let delta = -f64::from(sign.signum()) * beta;
                let v = ss[i * nf + col] + delta;
                ss[i * nf + col] = v;
                if let Some(es) = es.as_deref_mut() {
                    es[i * nf + col] = slack16(es[i * nf + col] + 2.0 * UNIT * v.abs());
                }
            }
        }
    }
    Some(Seed { s, e })
}

/// Ascent direction: `(g, legacy_scale)` per trunk split then per head split.
/// The Polyak rule consumes only `g`; the legacy sign-step consumes both.
pub(crate) struct BetaDir {
    /// Per trunk split, aligned with the domain's `trunk` list.
    pub trunk: Vec<(f64, f64)>,
    /// Per head split, aligned with the domain's `heads` list (empty when the
    /// caller requested trunk-only).
    pub heads: Vec<(f64, f64)>,
}

/// Per-split ascent direction and scale from the CURRENT pass.
///
/// Returns, per trunk split, `g_k = -s_k * ẑ_k(x*)` (the frozen-sign
/// linearization's supergradient; module docs) and `scale_k` the incoming
/// |coefficient| at the split neuron for the worst column (β's natural unit);
/// and per HEAD split (`heads` non-empty ⇒ the walk is extended through the
/// first head Gemm), `gʰ_i = -s_i * ŷ_i(x*)` with legacy scale
/// `|seed.s[[i, col]]|`. NOTE (adversarial must-fix #6): for a `s = -1` head
/// the base seed row is gated to ZERO (bounds.rs: alpha = s = 0), so the
/// legacy head scale is ~0 and the legacy sign-step cannot move head β —
/// POLYAK=0 + HEADS=1 is NOT a meaningful attribution arm; heads are
/// attributed POLYAK=1 HEADS=0 vs POLYAK=1 HEADS=1 (runbook arm C).
///
/// `None` when the pass carries no capture or any shape drifts — the caller
/// simply skips the trial (fail-open, never a verdict).
///
/// Heuristic-only: nothing here feeds a verdict; every proposal is re-scored
/// by the unchanged certified pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_scales(
    net: &TwinNet,
    root: &RootGates,
    dom: &DomainGates,
    trunk: &[TrunkSplit],
    heads: &[HeadFix],
    seed: &Seed,
    pass: &PassOut,
    col: usize,
) -> Option<BetaDir> {
    let vsigns = pass.unst_rows.as_ref()?;
    // Concretize argmin of the Lower bound for this column:
    // x*_i = mid_i - sign(A_i) * rad_i (mid on a zero coefficient).
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
    // Walk under the DOMAIN's alphas where overridden (piece-fixes and clip
    // rebuilds included); the chord side inside the walk stays the root's —
    // a direction-only approximation, exactly as in alpha_opt.
    let alpha: Vec<Vec<f64>> = root
        .layers
        .iter()
        .enumerate()
        .map(|(li, layer)| {
            dom.layers
                .get(&li)
                .map_or_else(|| layer.alpha.clone(), |gv| gv.alpha.clone())
        })
        .collect();
    let want_head = !heads.is_empty();
    let (u_out, yhat) =
        super::alpha_opt::linearized_walk(net, root, &alpha, vsigns, col, &xstar, want_head)?;
    let mut trunk_out = Vec::with_capacity(trunk.len());
    for &(li, pos, sign) in trunk {
        let zhat = u_out.get(&li).and_then(|v| v.get(pos)).copied();
        let vmag = vsigns.get(&li).and_then(|m| {
            let rr = m.ncols();
            let vs = m.as_slice()?;
            (pos < m.nrows() && col < rr).then(|| vs[pos * rr + col].abs())
        });
        let (Some(zhat), Some(vmag)) = (zhat, vmag) else {
            return None;
        };
        if !(zhat.is_finite() && vmag.is_finite()) {
            return None;
        }
        let g = -f64::from(sign.signum()) * zhat;
        // Scale floor: a zero incoming coefficient still deserves a step when
        // the gradient says the minimizer violates the constraint.
        trunk_out.push((g, vmag.max(1e-6)));
    }
    let mut heads_out = Vec::with_capacity(heads.len());
    if want_head {
        let nf = seed.s.ncols();
        let n_y = seed.s.nrows();
        let ss = seed.s.as_slice()?;
        if col >= nf {
            return None;
        }
        for &(i, sign) in heads {
            if i >= n_y || sign == 0 {
                return None;
            }
            let yh = yhat.get(i).copied()?;
            if !yh.is_finite() {
                return None;
            }
            // Head supergradient: the exact trunk analogue one layer higher.
            let g = -f64::from(sign.signum()) * yh;
            heads_out.push((g, ss[i * nf + col].abs().max(1e-6)));
        }
    }
    Some(BetaDir {
        trunk: trunk_out,
        heads: heads_out,
    })
}

/// #margin-row-beta C1: the gap-targeted Polyak step (classic known-target
/// subgradient rule, target τ = 0). Over the CONCATENATION trunk-then-heads:
///
/// ```text
///   movable(k):  g_k > 0  OR  (g_k < 0 AND β_k > 0)
///   S = Σ_{movable} g_k²
///   t = λ · gap / S            (gap = 0 − b_direct > 0)
///   β_k ← max(0, β_k + t·g_k)  for all k
/// ```
///
/// `gap` MUST be the ascended function's own gap — the direct-path value
/// `per_j[worst_col]`, NOT the composite `max(direct, m1, m2v)` (adversarial
/// must-fix #3: when m1/m2v carries the max, the composite's gap is not the
/// gap of the function the duals ascend). First-order predicted gain is
/// `t·S = λ·gap` — sized to close the gap in one move; the concave scorer +
/// monotone accept absorbs overshoot.
///
/// A `g < 0` coordinate at `β = 0` is pinned by the projection and excluded
/// from `S` so it cannot dilute the step. Refusals (`None`): length mismatch,
/// non-positive/non-finite `gap` or `λ`, `S <= 0` or non-finite, `t <= 0`
/// after the `[0, 1e6]` clamp, any non-finite `β`, or nothing moves.
/// Returns `(new_betas, t)` — `t` feeds the `[beta]` telemetry.
///
/// SOUNDNESS: tightness-only. This proposes WHICH `β >= 0` vector is scored;
/// `max(0, ·)` keeps the single weak-duality obligation, the certified pass
/// remains the unchanged scorer, and acceptance stays monotone. A wrong `t`
/// costs a pass, never a verdict.
pub(crate) fn polyak_step(
    betas: &[f64],
    dir: &[(f64, f64)],
    gap: f64,
    lambda: f64,
) -> Option<(Vec<f64>, f64)> {
    if betas.len() != dir.len()
        || !(gap.is_finite() && gap > 0.0)
        || !(lambda.is_finite() && lambda > 0.0)
    {
        return None;
    }
    let mut s2 = 0.0f64;
    for (&b0, &(g, _)) in betas.iter().zip(dir) {
        if g > 0.0 || (g < 0.0 && b0 > 0.0) {
            s2 += g * g;
        }
    }
    if !(s2.is_finite() && s2 > 0.0) {
        return None;
    }
    let t = (lambda * gap / s2).clamp(0.0, 1e6);
    if !(t.is_finite() && t > 0.0) {
        return None;
    }
    let mut out = Vec::with_capacity(betas.len());
    let mut moved = false;
    for (&b0, &(g, _)) in betas.iter().zip(dir) {
        let b1 = (b0 + t * g).max(0.0);
        if !b1.is_finite() {
            return None;
        }
        moved |= b1 != b0;
        out.push(b1);
    }
    moved.then_some((out, t))
}

/// LEGACY projected step (`NY_MARGIN_ROW_BETA_POLYAK=0` A/B arm):
/// `beta_k <- max(0, beta_k + eta * scale_k * sign(g_k))`.
/// `None` when nothing moves (all gradients zero / steps projected away).
pub(crate) fn apply_step(betas: &[f64], dir: &[(f64, f64)], eta: f64) -> Option<Vec<f64>> {
    if betas.len() != dir.len() {
        return None;
    }
    let mut out = Vec::with_capacity(betas.len());
    let mut moved = false;
    for (&b0, &(g, scale)) in betas.iter().zip(dir) {
        let b1 = if g > 0.0 {
            b0 + eta * scale
        } else if g < 0.0 {
            (b0 - eta * scale).max(0.0)
        } else {
            b0
        };
        if !b1.is_finite() {
            return None;
        }
        moved |= b1 != b0;
        out.push(b1);
    }
    moved.then_some(out)
}

/// Engagement telemetry (R9): rate-limited so a scored run stays readable but
/// an inert arming is detectable in one log. The fix's own observables live
/// here: calls from depth 1–2 (head gate), `heads=` > 0 (head terms carried),
/// `t=` > 1 on early accepts (Polyak actually sizing steps), `lam=` (per-node
/// step memory; = legacy η when POLYAK=0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn report(
    depth: usize,
    splits: usize,
    b0: f64,
    b1: f64,
    accepted: bool,
    betas: &[f64],
    head_betas: &[f64],
    t: f64,
    lam: f64,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static ACC: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if accepted {
        ACC.fetch_add(1, Ordering::Relaxed);
    }
    if n < 8 || n.is_multiple_of(128) {
        let max_beta = betas
            .iter()
            .chain(head_betas.iter())
            .copied()
            .fold(0.0f64, f64::max);
        let heads = head_betas.iter().filter(|b| **b > 0.0).count();
        eprintln!(
            "[beta] call={n} depth={depth} splits={splits} b0={b0:.6} b1={b1:.6} \
accepted={accepted} max_beta={max_beta:.3e} t={t:.3e} lam={lam:.3e} heads={heads} \
accepted_total={}",
            ACC.load(Ordering::Relaxed)
        );
    }
}

/// #margin-row-beta-percol engagement telemetry (R9): the per-column arm's
/// own observables — `cols=` (columns selected for ascent this eval),
/// `accepted=` (column-accept events across the eval's trials), `closed_cols=`
/// (columns whose composite bound crossed the closure threshold DURING the
/// ascent). Rate-limited like `[beta]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn report_pc(
    depth: usize,
    splits: usize,
    b0: f64,
    b1: f64,
    cols: usize,
    accepted: usize,
    closed_cols: usize,
    t: f64,
    lam: f64,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    static ACC: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    ACC.fetch_add(accepted, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(128) {
        eprintln!(
            "[beta-pc] call={n} depth={depth} splits={splits} b0={b0:.6} b1={b1:.6} \
cols={cols} accepted={accepted} closed_cols={closed_cols} t={t:.3e} lam={lam:.3e} \
accepted_total={}",
            ACC.load(Ordering::Relaxed)
        );
    }
}

/// Refusal telemetry (R9: every refusal path prints). Rate-limited globally;
/// the site tag makes a systematically-refusing lane visible in one log.
pub(crate) fn refuse(site: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 || n.is_multiple_of(256) {
        eprintln!("[beta] refuse={site} n={n}");
    }
}

/// Convenience for tests: the per-layer term map a `betas` vector produces.
#[cfg(test)]
pub(crate) fn terms_for_test(
    root: &RootGates,
    trunk: &[TrunkSplit],
    betas: &[f64],
) -> std::collections::BTreeMap<usize, Vec<BetaSplit>> {
    let mut dom = DomainGates::default();
    set_terms(&mut dom, root, trunk, betas);
    dom.beta
}
