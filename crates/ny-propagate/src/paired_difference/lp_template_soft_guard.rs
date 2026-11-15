// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Source-derived soft guards pay for an ENTIRE closed root-box cover.
//!
//! For either mask s, e=r-s*z>=0 and G=(2s-1)z+e>=0. Authored
//! prefix composition gives G=g(root)+sum a_t e_t. If a cell certifies
//! g<=-tau<0, then sum max(a_t,0)e_t>=tau. A nonnegative beta with
//! B+beta*tau>=M therefore pays a root-output bound M on that cell.
//! Taking the componentwise MAX of all cell charges gives one unconditional
//! output row. No mask is a phase premise. Every open leaf must close; a
//! partial/refused cover supplies no row. Coefficient intervals repair every
//! source-prefix operation, and beta payment is checked with LOWER rounding
//! before its coefficient charges are formed with UPPER rounding.

use super::*;

const MAX_INPUTS: usize = 5;
const MAX_OUTPUTS: usize = 5;
const MAX_GUARDS: usize = 8_192;
const MAX_NODES: usize = 255;
const TARGET: f64 = 0.04;

pub(in super::super) struct SoftGuardLeaf {
    pub(in super::super) domain: BoxDomain,
    /// Prior ReLU stage order, F neurons then G neurons at each stage.
    pub(in super::super) guard: Option<usize>,
    pub(in super::super) beta: f64,
    pub(in super::super) root_upper: f64,
    pub(in super::super) guard_root_upper: Option<f64>,
}

pub(in super::super) struct SoftGuardCover {
    pub(in super::super) output: usize,
    pub(in super::super) sign: i8,
    /// Own source-center mask portfolio, even when old portfolios deduplicate.
    pub(in super::super) portfolio: usize,
    pub(in super::super) upper: f64,
    pub(in super::super) nodes: usize,
    pub(in super::super) leaves: Vec<SoftGuardLeaf>,
}

struct Guard {
    root: Expr,
    g: bool,
    /// A positive endpoint per same-tower slack, including the own unit.
    positive: Vec<f64>,
    /// Selection only: rounded sum never enters a proof endpoint.
    magnitude: f64,
}

fn collect_guards(
    source: &[Expr],
    mask: &[u8],
    g: bool,
    guards: &mut Vec<Guard>,
    live: &mut usize,
    deadline: Option<Instant>,
) -> Option<()> {
    if source.len() != mask.len() || source.is_empty() || mask.iter().any(|&s| s > 1) {
        return None;
    }
    let old = source.first()?.slack.len();
    let total = old.checked_add(source.len())?;
    if total > MAX_SLACKS || guards.len().checked_add(source.len())? > MAX_GUARDS {
        return None;
    }
    for (j, e) in source.iter().enumerate() {
        if !alive(deadline) || e.slack.len() != old {
            return None;
        }
        *live = live.checked_add(total.checked_add(e.root.len())?.checked_add(1)?)?;
        if *live > MAX_LIVE_COEFFICIENTS {
            return None;
        }
        let eta = if mask[j] == 1 { 1.0 } else { -1.0 };
        let mut positive = vec![0.0; total];
        for (k, a) in e.slack.iter().copied().enumerate() {
            if !alive(deadline) {
                return None;
            }
            positive[k] = a.scale(eta)?.hi.max(0.0);
        }
        // The own slack is new and has no predecessor coefficient. Hence
        // this exact source unit does not require a rounded coefficient sum.
        positive[old + j] = 1.0;
        let magnitude = positive.iter().sum();
        guards.push(Guard {
            root: Expr {
                root: e
                    .root
                    .iter()
                    .copied()
                    .map(|a| a.scale(eta))
                    .collect::<Option<_>>()?,
                bias: e.bias.scale(eta)?,
                slack: Vec::new(),
            },
            g,
            positive,
            magnitude,
        });
    }
    Some(())
}

/// No approximate quotient is authority: its exact-f64 nonnegative beta
/// must pay the complete excess when its product/add are rounded DOWN.
fn pays(upper: f64, target: f64, guard_upper: f64, beta: f64) -> bool {
    if ![upper, target, guard_upper, beta]
        .into_iter()
        .all(admissible)
        || guard_upper >= 0.0
        || beta < 0.0
    {
        return false;
    }
    Interval::point(beta)
        .product(Interval::point(-guard_upper))
        .and_then(|payment| payment.add(Interval::point(target)))
        .is_some_and(|payment| payment.lo >= upper)
}

fn repaired_beta(upper: f64, target: f64, guard_upper: f64) -> Option<f64> {
    if upper <= target || guard_upper >= 0.0 || !admissible(guard_upper) {
        return None;
    }
    let excess = Interval::point(upper).sub(Interval::point(target))?.hi;
    let mut beta = (excess / -guard_upper).next_up();
    for _ in 0..16 {
        if !admissible(beta) || beta < 0.0 {
            return None;
        }
        if pays(upper, target, guard_upper, beta) {
            return Some(beta);
        }
        beta = beta.next_up();
    }
    None
}

/// The guard search stopped (deadline, or a root value that would not
/// evaluate): the whole cover gives up, as distinct from `Ok(None)`, "no guard
/// pays here".
struct GuardSearchAborted;

fn select_guard(
    guards: &[Guard],
    domain: &BoxDomain,
    upper: f64,
    target: f64,
    deadline: Option<Instant>,
) -> Result<Option<(usize, f64, f64)>, GuardSearchAborted> {
    let mut best: Option<(usize, f64, f64, f64)> = None;
    for (i, guard) in guards.iter().enumerate() {
        if !alive(deadline) {
            return Err(GuardSearchAborted);
        }
        let gu = root_value(&guard.root, domain, deadline)
            .ok_or(GuardSearchAborted)?
            .hi;
        let Some(beta) = repaired_beta(upper, target, gu) else {
            continue;
        };
        // Own midpoint slacks are zero in the proposal's floating evaluation.
        // L1 is only a deterministic tie-breaking proposal, not a proof value.
        let score = beta * guard.magnitude;
        if best
            .as_ref()
            .is_none_or(|&(j, _, _, old)| score.total_cmp(&old).then_with(|| i.cmp(&j)).is_lt())
        {
            best = Some((i, beta, gu, score));
        }
    }
    Ok(best.map(|(i, b, gu, _)| (i, b, gu)))
}

struct Cell {
    domain: BoxDomain,
    upper: f64,
    depth: usize,
}

fn charged_cover(
    f: &[Expr],
    g: &[Expr],
    wf: &[Interval],
    wg: &[Interval],
    bias: Interval,
    guards: &[Guard],
    nf: usize,
    ng: usize,
    domain: &BoxDomain,
    max_nodes: usize,
    target: f64,
    deadline: Option<Instant>,
) -> Option<(Vec<f64>, usize, Vec<SoftGuardLeaf>)> {
    if max_nodes == 0
        || max_nodes > MAX_NODES
        || !admissible(target)
        || !alive(deadline)
        || guards.len() > MAX_GUARDS
        || guards
            .iter()
            .any(|a| a.positive.len() > if a.g { ng } else { nf })
    {
        return None;
    }
    let width = nf.checked_add(ng)?;
    if width > MAX_SLACKS.checked_mul(2)? {
        return None;
    }
    let first = output::cell_upper(f, g, wf, wg, bias, domain, deadline)?;
    let sensitivity = output::sensitivities(f, g, wf, wg);
    if sensitivity.len() != domain.lo.len() {
        return None;
    }
    let mut frontier = vec![Cell {
        domain: domain.clone(),
        upper: first,
        depth: 0,
    }];
    let mut closed = Vec::new();
    let mut extra = vec![0.0_f64; width];
    let mut nodes = 1usize;
    let mut charge_terms = 0usize;
    while !frontier.is_empty() {
        if !alive(deadline) {
            return None;
        }
        let index = frontier
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.upper.total_cmp(&b.upper).then_with(|| ib.cmp(ia)))?
            .0;
        let cell = &frontier[index];
        let charge = if cell.upper > target {
            select_guard(guards, &cell.domain, cell.upper, target, deadline).ok()?
        } else {
            None
        };
        if cell.upper <= target || charge.is_some() {
            if let Some((i, beta, gu)) = charge {
                if !pays(cell.upper, target, gu, beta) {
                    return None;
                }
                let guard = &guards[i];
                charge_terms = charge_terms.checked_add(guard.positive.len())?;
                if charge_terms > MAX_TERMS {
                    return None;
                }
                let offset = if guard.g { nf } else { 0 };
                for (j, &a) in guard.positive.iter().enumerate() {
                    if !alive(deadline) || !admissible(a) || a < 0.0 {
                        return None;
                    }
                    let value = Interval::point(beta).product(Interval::point(a))?.hi;
                    extra[offset + j] = extra[offset + j].max(value);
                }
            }
            let cell = frontier.remove(index);
            closed.push(SoftGuardLeaf {
                domain: cell.domain,
                guard: charge.map(|c| c.0),
                beta: charge.map_or(0.0, |c| c.1),
                root_upper: cell.upper,
                guard_root_upper: charge.map(|c| c.2),
            });
            continue;
        }
        // An unpaid/open frontier is never returned as a certificate.
        if nodes.checked_add(2)? > max_nodes {
            return None;
        }
        let axis = (0..domain.lo.len())
            .filter(|&j| {
                let m = 0.5 * cell.domain.lo[j] + 0.5 * cell.domain.hi[j];
                m.is_finite() && cell.domain.lo[j] < m && m < cell.domain.hi[j]
            })
            .max_by(|&a, &b| {
                let score = |j: usize| {
                    let width = cell.domain.hi[j] - cell.domain.lo[j];
                    let value = if cell.depth % 4 == 3 {
                        width / (domain.hi[j] - domain.lo[j])
                    } else {
                        width * sensitivity[j]
                    };
                    if value.is_nan() {
                        0.0
                    } else {
                        value
                    }
                };
                score(a).total_cmp(&score(b)).then_with(|| b.cmp(&a))
            })?;
        let middle = 0.5 * cell.domain.lo[axis] + 0.5 * cell.domain.hi[axis];
        let depth = cell.depth.checked_add(1)?;
        let mut left = cell.domain.clone();
        let mut right = cell.domain.clone();
        left.hi[axis] = middle;
        right.lo[axis] = middle;
        // Bound BOTH closed children before replacing their covering parent.
        let lu = output::cell_upper(f, g, wf, wg, bias, &left, deadline)?;
        let ru = output::cell_upper(f, g, wf, wg, bias, &right, deadline)?;
        frontier.remove(index);
        frontier.push(Cell {
            domain: left,
            upper: lu,
            depth,
        });
        frontier.push(Cell {
            domain: right,
            upper: ru,
            depth,
        });
        nodes += 2;
    }
    alive(deadline).then_some((extra, nodes, closed))
}

fn emit(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    stage: usize,
    f: &[Expr],
    g: &[Expr],
    fs: &[Slack],
    gs: &[Slack],
    guards: &[Guard],
    reference_portfolio: usize,
    reference_covers: &[OutputCover],
    deadline: Option<Instant>,
) -> Option<(Vec<LinearRow>, Vec<SoftGuardCover>)> {
    let tf = output::tail(chain, stage, f.len(), false, deadline)?;
    let tg = output::tail(chain, stage, g.len(), true, deadline)?;
    let outputs = bindings.last()?;
    let mut rows = Vec::new();
    let mut covers = Vec::new();
    let mut terms_used = 0usize;
    for j in 0..chain.output_dim {
        for sign in [1i8, -1i8] {
            if !alive(deadline) {
                return None;
            }
            if !reference_covers.iter().any(|c| {
                c.output == j
                    && c.sign == sign
                    && c.portfolio == reference_portfolio
                    && c.upper > TARGET
            }) {
                continue;
            }
            let wf: Vec<_> = tf[j]
                .root
                .iter()
                .copied()
                .map(|a| a.scale(sign as f64))
                .collect::<Option<_>>()?;
            let wg: Vec<_> = tg[j]
                .root
                .iter()
                .copied()
                .map(|a| a.scale(sign as f64))
                .collect::<Option<_>>()?;
            let bias = tf[j].bias.sub(tg[j].bias)?.scale(sign as f64)?;
            let fp = output::penalty(f, &wf, fs, deadline)?;
            let ngw: Vec<_> = wg.iter().copied().map(Interval::neg).collect();
            let gp = output::penalty(g, &ngw, gs, deadline)?;
            let Some((extra, nodes, leaves)) = charged_cover(
                f,
                g,
                &wf,
                &wg,
                bias,
                guards,
                fs.len(),
                gs.len(),
                domain,
                MAX_NODES,
                TARGET,
                deadline,
            ) else {
                if !alive(deadline) {
                    return None;
                }
                continue;
            };
            if !leaves.iter().any(|leaf| leaf.guard.is_some()) {
                continue;
            }
            let mut terms = vec![(outputs.0[j], -(sign as f64)), (outputs.1[j], sign as f64)];
            for (k, (slack, base)) in fs
                .iter()
                .copied()
                .zip(fp)
                .chain(gs.iter().copied().zip(gp))
                .enumerate()
            {
                if !alive(deadline) {
                    return None;
                }
                let coefficient = Interval::point(base).add(Interval::point(extra[k]))?.hi;
                append_slack(&mut terms, slack, coefficient)?;
            }
            let next = terms_used.checked_add(terms.len())?;
            if next > MAX_TERMS || rows.len() >= 10 {
                continue;
            }
            terms_used = next;
            rows.push(LinearRow {
                terms,
                rhs: -TARGET,
            });
            covers.push(SoftGuardCover {
                output: j,
                sign,
                portfolio: 2,
                upper: TARGET,
                nodes,
                leaves,
            });
        }
    }
    alive(deadline).then_some((rows, covers))
}

/// Recompose only the own-mask prefix after all old rows were produced.
/// This transparent optional pass keeps the old API's arithmetic and row
/// ordering unchanged; no external rows/guards/bounds enter this path.
pub(super) fn rows(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    masks: &[Mask],
    reference_portfolio: usize,
    reference_covers: &[OutputCover],
    deadline: Option<Instant>,
) -> Option<(Vec<LinearRow>, Vec<SoftGuardCover>)> {
    if !alive(deadline) || !suitable_fp_environment() {
        return None;
    }
    chain.validate().ok()?;
    chain.input_state(domain)?;
    valid_bindings(chain, bindings)?;
    if chain.input_dim > MAX_INPUTS || chain.output_dim > MAX_OUTPUTS || masks.is_empty() {
        return Some((Vec::new(), Vec::new()));
    }
    let expected: Vec<_> = chain
        .stages
        .iter()
        .enumerate()
        .filter_map(|(s, stage)| {
            if let Stage::Relu { width } = stage {
                Some((s, *width))
            } else {
                None
            }
        })
        .collect();
    if expected.len() != masks.len()
        || expected.iter().zip(masks).any(|(&(s, width), mask)| {
            mask.stage != s
                || mask.f.len() != width
                || mask.g.len() != width
                || mask.f.iter().chain(&mask.g).any(|&s| s > 1)
        })
    {
        return None;
    }
    let identity = || {
        (0..chain.input_dim)
            .map(|j| {
                let mut root = vec![Interval::point(0.0); chain.input_dim];
                root[j] = Interval::point(1.0);
                Expr {
                    root,
                    bias: Interval::point(0.0),
                    slack: Vec::new(),
                }
            })
            .collect::<Vec<_>>()
    };
    let mut f = identity();
    let mut g = identity();
    let mut fs = Vec::new();
    let mut gs = Vec::new();
    let mut guards = Vec::new();
    let mut guard_live = 0usize;
    let mut index = 0usize;
    for (s, stage) in chain.stages.iter().enumerate() {
        if !alive(deadline) {
            return None;
        }
        match stage {
            Stage::Affine {
                rows,
                cols,
                wf,
                wg,
                bf,
                bg,
            } => {
                let old = count(&f)?.checked_add(count(&g)?)?;
                let nf = rows.checked_mul(
                    chain
                        .input_dim
                        .checked_add(f.first()?.slack.len())?
                        .checked_add(1)?,
                )?;
                let ng = rows.checked_mul(
                    chain
                        .input_dim
                        .checked_add(g.first()?.slack.len())?
                        .checked_add(1)?,
                )?;
                if old
                    .checked_add(nf)?
                    .checked_add(ng)?
                    .checked_add(guard_live)?
                    > MAX_LIVE_COEFFICIENTS
                {
                    return None;
                }
                f = authored_affine(&f, wf, bf, *rows, *cols, deadline)?;
                g = authored_affine(&g, wg, bg, *rows, *cols, deadline)?;
            }
            Stage::Relu { width } => {
                let mask = masks.get(index)?;
                index += 1;
                if f.len() != *width || g.len() != *width {
                    return None;
                }
                if index == masks.len() {
                    return emit(
                        chain,
                        domain,
                        bindings,
                        s,
                        &f,
                        &g,
                        &fs,
                        &gs,
                        &guards,
                        reference_portfolio,
                        reference_covers,
                        deadline,
                    );
                }
                // The guard is authored BEFORE applying the mask, so its
                // coefficients contain only strictly earlier slack variables.
                collect_guards(&f, &mask.f, false, &mut guards, &mut guard_live, deadline)?;
                collect_guards(&g, &mask.g, true, &mut guards, &mut guard_live, deadline)?;
                let required = width
                    .checked_mul(
                        chain
                            .input_dim
                            .checked_add(fs.len())?
                            .checked_add(*width)?
                            .checked_add(1)?,
                    )?
                    .checked_add(
                        width.checked_mul(
                            chain
                                .input_dim
                                .checked_add(gs.len())?
                                .checked_add(*width)?
                                .checked_add(1)?,
                        )?,
                    )?;
                if required
                    .checked_add(count(&f)?)?
                    .checked_add(count(&g)?)?
                    .checked_add(guard_live)?
                    > MAX_LIVE_COEFFICIENTS
                {
                    return None;
                }
                f = apply_mask(f, &mask.f, deadline)?;
                g = apply_mask(g, &mask.g, deadline)?;
                for j in 0..*width {
                    fs.push(Slack {
                        pre: bindings[s].0[j],
                        post: bindings[s + 1].0[j],
                        mask: mask.f[j],
                    });
                    gs.push(Slack {
                        pre: bindings[s].1[j],
                        post: bindings[s + 1].1[j],
                        mask: mask.g[j],
                    });
                }
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "lp_template_soft_guard_tests.rs"]
mod tests;
