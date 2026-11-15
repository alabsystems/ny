// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whole-output supports of a causal template, over a COMPLETE root cover.
//!
//! For any real q, rho(H+sum K_i e_i) multiplied by q is at most
//! q*rho(H)+sum max(q*K_i,0)*e_i, since e_i>=0. Compose the authored
//! affine output head with interval arithmetic and apply this inequality to
//! every final ReLU. The root-only sum is bounded jointly: stable affine
//! contributions are added BEFORE box maximization. Remaining paired terms
//! use WG*(rho(HF)-rho(HG))+(WF-WG)*rho(HF), met with component bounds.
//! A finite closed box cover certifies its maximum. The adaptive target and
//! split score are proposals only; the complete frontier remains covered
//! even when the node cap prevents further refinement.

use super::*;

const MAX_INPUTS: usize = 5;
const MAX_OUTPUTS: usize = 5;
const MAX_NODES: usize = 255;
const PROPOSAL_TARGET: f64 = 0.04;

struct Cell {
    domain: BoxDomain,
    upper: f64,
}

fn identity(width: usize) -> Option<Vec<Expr>> {
    if width == 0 || width.checked_mul(width.checked_add(1)?)? > MAX_LIVE_COEFFICIENTS {
        return None;
    }
    Some(
        (0..width)
            .map(|j| {
                let mut root = vec![Interval::point(0.0); width];
                root[j] = Interval::point(1.0);
                Expr {
                    root,
                    bias: Interval::point(0.0),
                    slack: Vec::new(),
                }
            })
            .collect(),
    )
}

pub(super) fn tail(
    chain: &PairedChain,
    stage: usize,
    width: usize,
    g: bool,
    deadline: Option<Instant>,
) -> Option<Vec<Expr>> {
    let mut forms = identity(width)?;
    for authored in chain.stages.get(stage.checked_add(1)?..)? {
        if !alive(deadline) {
            return None;
        }
        let Stage::Affine {
            rows,
            cols,
            wf,
            wg,
            bf,
            bg,
        } = authored
        else {
            return None;
        };
        forms = authored_affine(
            &forms,
            if g { wg } else { wf },
            if g { bg } else { bf },
            *rows,
            *cols,
            deadline,
        )?;
    }
    (forms.len() == chain.output_dim && alive(deadline)).then_some(forms)
}

fn rho(x: Interval) -> Interval {
    Interval {
        lo: x.lo.max(0.0),
        hi: x.hi.max(0.0),
    }
}

fn root_pair(
    f: &Expr,
    g: &Expr,
    rf: Interval,
    rg: Interval,
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<Interval> {
    let delta = root_difference(f, g, domain, deadline)?;
    let components = rho(rf).sub(rho(rg))?;
    let result = Interval {
        lo: delta.lo.min(0.0).max(components.lo),
        hi: delta.hi.max(0.0).min(components.hi),
    };
    result.valid().then_some(result)
}

pub(super) fn cell_upper(
    f: &[Expr],
    g: &[Expr],
    wf: &[Interval],
    wg: &[Interval],
    bias: Interval,
    domain: &BoxDomain,
    deadline: Option<Instant>,
) -> Option<f64> {
    if f.len() != g.len() || wf.len() != f.len() || wg.len() != g.len() {
        return None;
    }
    let mut affine = Expr {
        root: vec![Interval::point(0.0); domain.lo.len()],
        bias,
        slack: Vec::new(),
    };
    for i in 0..f.len() {
        if !alive(deadline) || !wf[i].valid() || !wg[i].valid() {
            return None;
        }
        let rf = root_value(&f[i], domain, deadline)?;
        let rg = root_value(&g[i], domain, deadline)?;
        let sf = if rf.lo >= 0.0 {
            Some(true)
        } else if rf.hi <= 0.0 {
            Some(false)
        } else {
            None
        };
        let sg = if rg.lo >= 0.0 {
            Some(true)
        } else if rg.hi <= 0.0 {
            Some(false)
        } else {
            None
        };
        if let (Some(sf), Some(sg)) = (sf, sg) {
            for (source, weight, active) in [(&f[i], wf[i], sf), (&g[i], wg[i].neg(), sg)] {
                if active {
                    affine.bias = affine.bias.add(source.bias.product(weight)?)?;
                    for j in 0..affine.root.len() {
                        affine.root[j] = affine.root[j].add(source.root[j].product(weight)?)?;
                    }
                }
            }
        } else {
            let difference = root_pair(&f[i], &g[i], rf, rg, domain, deadline)?;
            let paired = wg[i]
                .product(difference)?
                .add(wf[i].sub(wg[i])?.product(rho(rf))?)?;
            let independent = wf[i].product(rho(rf))?.sub(wg[i].product(rho(rg))?)?;
            let upper = paired.hi.min(independent.hi);
            affine.bias = affine.bias.add(Interval::point(upper))?;
        }
    }
    let result = root_value(&affine, domain, deadline)?.hi;
    (admissible(result) && alive(deadline)).then_some(result)
}

/// Selection-only scores. Rounded values never enter a proof endpoint.
pub(super) fn sensitivities(f: &[Expr], g: &[Expr], wf: &[Interval], wg: &[Interval]) -> Vec<f64> {
    (0..f.first().map_or(0, |e| e.root.len()))
        .map(|j| {
            (0..f.len()).fold(0.0, |sum, i| {
                sum + wf[i].lo.abs().max(wf[i].hi.abs())
                    * f[i].root[j].lo.abs().max(f[i].root[j].hi.abs())
                    + wg[i].lo.abs().max(wg[i].hi.abs())
                        * g[i].root[j].lo.abs().max(g[i].root[j].hi.abs())
            })
        })
        .collect()
}

fn complete_cover(
    f: &[Expr],
    g: &[Expr],
    wf: &[Interval],
    wg: &[Interval],
    bias: Interval,
    domain: &BoxDomain,
    max_nodes: usize,
    target: f64,
    deadline: Option<Instant>,
) -> Option<(f64, usize, Vec<BoxDomain>)> {
    if max_nodes == 0 || max_nodes > MAX_NODES || !admissible(target) || !alive(deadline) {
        return None;
    }
    let upper = cell_upper(f, g, wf, wg, bias, domain, deadline)?;
    let sensitivity = sensitivities(f, g, wf, wg);
    let mut frontier = vec![Cell {
        domain: domain.clone(),
        upper,
    }];
    let mut nodes = 1usize;
    loop {
        if !alive(deadline) {
            return None;
        }
        let index = frontier
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.upper.total_cmp(&b.upper).then_with(|| ib.cmp(ia)))?
            .0;
        if frontier[index].upper <= target || nodes.checked_add(2)? > max_nodes {
            break;
        }
        let parent = &frontier[index];
        let axis = (0..domain.lo.len())
            .filter(|&j| parent.domain.lo[j] < parent.domain.hi[j])
            .max_by(|&a, &b| {
                let score = |j: usize| {
                    let value = (parent.domain.hi[j] - parent.domain.lo[j]) * sensitivity[j];
                    if value.is_nan() {
                        0.0
                    } else {
                        value
                    }
                };
                score(a).total_cmp(&score(b)).then_with(|| b.cmp(&a))
            });
        let Some(axis) = axis else { break };
        let middle = 0.5 * parent.domain.lo[axis] + 0.5 * parent.domain.hi[axis];
        if !middle.is_finite()
            || middle <= parent.domain.lo[axis]
            || middle >= parent.domain.hi[axis]
        {
            break;
        }
        let mut left = parent.domain.clone();
        let mut right = parent.domain.clone();
        left.hi[axis] = middle;
        right.lo[axis] = middle;
        // Both CLOSED children are bounded before replacing their parent.
        // If a deadline/refusal occurs here, no partial cover is returned.
        let left_upper = cell_upper(f, g, wf, wg, bias, &left, deadline)?;
        let right_upper = cell_upper(f, g, wf, wg, bias, &right, deadline)?;
        frontier.remove(index);
        frontier.push(Cell {
            domain: left,
            upper: left_upper,
        });
        frontier.push(Cell {
            domain: right,
            upper: right_upper,
        });
        nodes += 2;
    }
    let upper = frontier
        .iter()
        .map(|cell| cell.upper)
        .fold(f64::NEG_INFINITY, f64::max);
    (admissible(upper) && alive(deadline)).then(|| {
        (
            upper,
            nodes,
            frontier.into_iter().map(|cell| cell.domain).collect(),
        )
    })
}

pub(super) fn penalty(
    source: &[Expr],
    weights: &[Interval],
    slacks: &[Slack],
    deadline: Option<Instant>,
) -> Option<Vec<f64>> {
    if source.len() != weights.len() || source.iter().any(|e| e.slack.len() != slacks.len()) {
        return None;
    }
    let mut result = vec![0.0; slacks.len()];
    for (e, weight) in source.iter().zip(weights) {
        for (j, coefficient) in e.slack.iter().copied().enumerate() {
            if !alive(deadline) {
                return None;
            }
            let p = weight.product(coefficient)?.hi.max(0.0);
            result[j] = Interval::point(result[j]).add(Interval::point(p))?.hi;
        }
    }
    Some(result)
}

pub(super) fn rows(
    chain: &PairedChain,
    domain: &BoxDomain,
    bindings: &[(Vec<usize>, Vec<usize>)],
    stage: usize,
    f: &[Expr],
    g: &[Expr],
    fs: &[Slack],
    gs: &[Slack],
    portfolio: usize,
    deadline: Option<Instant>,
) -> Option<(Vec<LinearRow>, Vec<OutputCover>)> {
    if domain.lo.len() > MAX_INPUTS || chain.output_dim > MAX_OUTPUTS {
        return Some((Vec::new(), Vec::new()));
    }
    if f.is_empty() || f.len() != g.len() || !alive(deadline) {
        return None;
    }
    let tf = tail(chain, stage, f.len(), false, deadline)?;
    let tg = tail(chain, stage, g.len(), true, deadline)?;
    let outputs = bindings.last()?;
    let mut result = Vec::new();
    let mut covers = Vec::new();
    for output in 0..chain.output_dim {
        for sign in [1i8, -1i8] {
            if !alive(deadline) {
                return None;
            }
            let wf: Vec<_> = tf[output]
                .root
                .iter()
                .copied()
                .map(|w| w.scale(sign as f64))
                .collect::<Option<_>>()?;
            let wg: Vec<_> = tg[output]
                .root
                .iter()
                .copied()
                .map(|w| w.scale(sign as f64))
                .collect::<Option<_>>()?;
            let bias = tf[output].bias.sub(tg[output].bias)?.scale(sign as f64)?;
            let fp = penalty(f, &wf, fs, deadline)?;
            let neg_gw: Vec<_> = wg.iter().copied().map(Interval::neg).collect();
            let gp = penalty(g, &neg_gw, gs, deadline)?;
            let (upper, nodes, leaves) = complete_cover(
                f,
                g,
                &wf,
                &wg,
                bias,
                domain,
                MAX_NODES,
                PROPOSAL_TARGET,
                deadline,
            )?;
            let mut terms = vec![
                (outputs.0[output], -(sign as f64)),
                (outputs.1[output], sign as f64),
            ];
            for (slack, coefficient) in fs.iter().copied().zip(fp).chain(gs.iter().copied().zip(gp))
            {
                append_slack(&mut terms, slack, coefficient)?;
            }
            if terms.len() > MAX_TERMS || !alive(deadline) {
                return None;
            }
            result.push(LinearRow { terms, rhs: -upper });
            covers.push(OutputCover {
                output,
                sign,
                portfolio,
                upper,
                nodes,
                leaves,
            });
        }
    }
    alive(deadline).then_some((result, covers))
}

#[cfg(test)]
#[path = "lp_template_output_tests.rs"]
mod tests;
