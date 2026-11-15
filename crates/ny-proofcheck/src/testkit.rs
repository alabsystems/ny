// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Test-only entry points into the checker's arithmetic (hidden from docs).
//! Nothing in ny calls these outside tests.

use std::collections::HashMap;

use crate::arith::Iv;
use crate::backsub::{backsub, AlphaMap, BetaTerm, Ctx, Seeds, SiteView};
use crate::fd::{self, Form, Frame, LambdaMap, RowRef, Seed};
use crate::ibp::{forward, magnitudes, Bounds};
use crate::lower::{lower, Net};
use crate::onnx_wire::parse_model;

/// Decode and lower an ONNX byte string.
pub fn net_from_onnx(bytes: &[u8]) -> Result<Net, String> {
    let m = parse_model(bytes).map_err(|e| e.0)?;
    lower(&m).map_err(|e| e.text().to_string())
}

struct View {
    b: Bounds,
}

impl SiteView for View {
    fn site(&self, t: usize) -> (&[f64], &[f64]) {
        (&self.b.lo[t], &self.b.hi[t])
    }
}

/// A certified lower bound on `Σ obj_j y_j − d_up` over the region
/// `box ∩ literals`, using PURE IBP site bounds over `box` clamped by the
/// literals `(tensor name, index, pos)`, the given β (aligned with the
/// literals) and α. Returns `(L, linear form, constant)` or `None` when the
/// region is certified empty.
#[allow(clippy::too_many_arguments)]
pub fn lower_bound(
    net: &Net,
    lo: &[f64],
    hi: &[f64],
    lits: &[(&str, usize, bool)],
    beta: &[f64],
    alpha: Option<&AlphaMap>,
    obj: &[(usize, f64)],
    d_up: f64,
) -> Option<(f64, Vec<f64>, f64)> {
    let mut b = forward(net, lo, hi);
    let mags = magnitudes(net, &b);
    let mut betas = Vec::new();
    for (k, &(name, i, pos)) in lits.iter().enumerate() {
        let t = *net.by_name.get(name)?;
        if pos {
            b.lo[t][i] = b.lo[t][i].max(0.0);
        } else {
            b.hi[t][i] = b.hi[t][i].min(0.0);
        }
        if let Some(&bv) = beta.get(k) {
            betas.push(BetaTerm {
                tensor: t,
                index: i,
                coef: if pos { -bv } else { bv },
            });
        }
    }
    let view = View { b };
    let seeds = Seeds {
        tensor: net.output,
        rows: vec![obj.to_vec()],
        kappa0: vec![-d_up],
        nterms: 1,
    };
    let ctx = Ctx {
        net,
        mags: &mags,
        view: &view,
        alpha,
        betas: &betas,
        box_lo: lo,
        box_hi: hi,
        chunk_rows: 16,
        want_lin: true,
    };
    let r = backsub(&ctx, &seeds).pop()?;
    if r.empty {
        return None;
    }
    let (lin, c) = r.lin?;
    Some((r.lb, lin, c))
}

/// An α map from `(tensor name, neuron, alpha)` triples.
#[must_use]
pub fn alpha_map(net: &Net, entries: &[(&str, usize, f64)]) -> AlphaMap {
    let mut m: AlphaMap = HashMap::new();
    for &(name, i, a) in entries {
        if let Some(&t) = net.by_name.get(name) {
            m.entry(t).or_default().insert(i, a);
        }
    }
    m
}

/// B2.3: the checker's own frozen-dual state over the box `[lo, hi]` with
/// PURE IBP root bounds, the hint slopes `lambda` (`(tensor name, neuron,
/// λ)`), one objective form for `Σ obj_j y_j - d` (`d ∈ [d_lo, d_hi]`) and
/// one row per named site `(tensor name, neuron)`. Returns the frame, the
/// objective form, the rows (in `sites` order) and the IBP bounds.
#[allow(clippy::too_many_arguments)]
pub fn fd_state(
    net: &Net,
    lo: &[f64],
    hi: &[f64],
    lambda: &[(&str, usize, f64)],
    obj: &[(usize, f64)],
    d_lo: f64,
    d_hi: f64,
    sites: &[(&str, usize)],
) -> Option<(Frame, Form, Vec<Form>, Bounds)> {
    let b = forward(net, lo, hi);
    let mags = magnitudes(net, &b);
    let mut lam: LambdaMap = HashMap::new();
    for &(name, i, v) in lambda {
        lam.entry(*net.by_name.get(name)?).or_default().insert(i, v);
    }
    let frame = Frame::new(net, &b, lo, hi, Some(&lam)).ok()?;
    let mut seeds = vec![Seed {
        tensor: net.output,
        coeffs: obj.to_vec(),
        k0: Iv::new(-d_hi, -d_lo),
    }];
    for &(name, i) in sites {
        seeds.push(fd::site_seed(*net.by_name.get(name)?, i));
    }
    let mut forms = fd::forms(net, &mags, &frame, &seeds);
    let rows = forms.split_off(1);
    let margin = forms.pop()?;
    Some((frame, margin, rows, b))
}

/// B2.3: the certified fd bound of a node over `fd_state`'s outputs: the
/// rows with their sides (`true`: `z >= 0`) and the dual point.
#[must_use]
pub fn fd_node_bound(
    net: &Net,
    frame: &Frame,
    margin: &Form,
    rows: &[Form],
    sites: &[(&str, usize, bool)],
    l0: &[f64],
    l1: &[f64],
) -> f64 {
    let refs: Vec<RowRef<'_>> = rows
        .iter()
        .zip(sites)
        .map(|(form, &(name, i, on))| RowRef {
            form,
            relax: net
                .by_name
                .get(name)
                .map_or(fd::Relu::Bad, |&t| frame.relu(t, i)),
            on,
        })
        .collect();
    fd::node_bound(margin, &refs, l0, l1)
}
