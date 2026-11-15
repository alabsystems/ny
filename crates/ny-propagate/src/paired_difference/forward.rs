// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fast certified forward affine forms in common/difference coordinates.
//! Every stored form has exact f64 coefficients and a directionally repaired
//! constant. Coefficient dot-product rounding is charged against root input
//! magnitudes. Scalar common-mode sums/differences have error-free TwoSum
//! residuals charged against rigorous source value boxes.

use super::*;

#[derive(Clone)]
pub(super) struct Form {
    pub(super) a: Vec<f64>,
    pub(super) b: f64,
}

#[derive(Clone)]
pub(super) struct Forms {
    pub(super) lower: Form,
    pub(super) upper: Form,
}

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

pub(super) fn scalar_sum_error(a: f64, b: f64, sum: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || !sum.is_finite() {
        return None;
    }
    let virtual_b = sum - a;
    let error = (a - (sum - virtual_b)) + (b - virtual_b);
    if !error.is_finite() {
        None
    } else if error == 0.0 {
        Some(0.0)
    } else {
        Some(error.abs().next_up())
    }
}

fn valid(form: &Form) -> bool {
    form.b.is_finite() && form.a.iter().all(|v| v.is_finite())
}

pub(super) fn concretize(form: &Form, domain: &BoxDomain, lower: bool) -> Option<f64> {
    let mut result = form.b;
    for j in 0..form.a.len() {
        let c = form.a[j];
        let endpoint = if (c >= 0.0) == lower {
            domain.lo[j]
        } else {
            domain.hi[j]
        };
        result = if lower {
            add_down(result, mul_down(c, endpoint))
        } else {
            add_up(result, mul_up(c, endpoint))
        };
    }
    result.is_finite().then_some(result)
}

/// Substitute exact stored scalar coefficients into already certified forms.
/// The rounded coefficient sums are repaired in the constant by a Higham
/// dot-product error bound, including its absolute subnormal allowance.
pub(super) fn combine(
    terms: &[(f64, &Forms)],
    bias: f64,
    domain: &BoxDomain,
    lower: bool,
) -> Option<Form> {
    let dim = domain.lo.len();
    let mut result = Form {
        a: vec![0.0; dim],
        b: bias,
    };
    let mut absolute = vec![0.0; dim];
    let mut count = 0;
    for &(c, pair) in terms {
        if c == 0.0 {
            continue;
        }
        if !c.is_finite() {
            return None;
        }
        let form = if (c >= 0.0) == lower {
            &pair.lower
        } else {
            &pair.upper
        };
        result.b = if lower {
            add_down(result.b, mul_down(c, form.b))
        } else {
            add_up(result.b, mul_up(c, form.b))
        };
        for j in 0..dim {
            result.a[j] += c * form.a[j];
            absolute[j] = add_up(absolute[j], mul_up(c.abs(), form.a[j].abs()));
        }
        count += 1;
    }
    if count > 0 {
        let mut charge = 0.0;
        for j in 0..dim {
            charge = add_up(
                charge,
                mul_up(
                    dot_error(count, absolute[j]),
                    magnitude(domain.lo[j], domain.hi[j]),
                ),
            );
        }
        result.b = if lower {
            add_down(result.b, -charge)
        } else {
            add_up(result.b, charge)
        };
    }
    valid(&result).then_some(result)
}

fn pair(terms: &[(f64, &Forms)], lo_bias: f64, hi_bias: f64, domain: &BoxDomain) -> Option<Forms> {
    Some(Forms {
        lower: combine(terms, lo_bias, domain, true)?,
        upper: combine(terms, hi_bias, domain, false)?,
    })
}

fn bounds(forms: &[Forms], domain: &BoxDomain) -> Option<(Vec<f64>, Vec<f64>)> {
    let lo = forms
        .iter()
        .map(|p| concretize(&p.lower, domain, true))
        .collect::<Option<Vec<_>>>()?;
    let hi = forms
        .iter()
        .map(|p| concretize(&p.upper, domain, false))
        .collect::<Option<Vec<_>>>()?;
    if lo
        .iter()
        .zip(&hi)
        .any(|(&l, &h)| l > h || magnitude(l, h) > 1e150)
    {
        return None;
    }
    Some((lo, hi))
}

fn relu_forms(input: &Forms, l: f64, u: f64, domain: &BoxDomain) -> Option<Forms> {
    let ls = relu_slope(1.0, l, u);
    let us = relu_slope(-1.0, l, u);
    let lb = independent_intercept(1.0, ls, l, u, 0.0, 0.0, 0.0, 0.0);
    let negative_ub = independent_intercept(-1.0, -us, l, u, 0.0, 0.0, 0.0, 0.0);
    Some(Forms {
        lower: combine(&[(ls, input)], lb, domain, true)?,
        upper: combine(&[(us, input)], -negative_ub, domain, false)?,
    })
}

fn transported_relu_difference(
    f: &Forms,
    g: &Forms,
    d: &Forms,
    ranges: PairedReluRanges,
    hull: &PairedReluHull,
    domain: &BoxDomain,
    lower: bool,
    deadline: Option<Instant>,
) -> Option<Form> {
    let sign = if lower { 1.0 } else { -1.0 };
    let mut best: Option<(Form, f64)> = None;
    for (pf, pg) in ranges.candidate_slopes(sign, -sign) {
        if !alive(deadline) {
            return None;
        }
        let Some(pk) = hull.certified_intercept(sign, -sign, pf, pg) else {
            continue;
        };
        // The upper form is obtained by negating a lower support of -delta.
        let af = sign * pf;
        let ag = sign * pg;
        let k = sign * pk;
        let common = af + ag;
        let error = scalar_sum_error(af, ag, common)?;
        let f_charge = mul_up(error, magnitude(ranges.lf, ranges.uf));
        let g_charge = mul_up(error, magnitude(ranges.lg, ranges.ug));
        let fb = if lower {
            add_down(k, -f_charge)
        } else {
            add_up(k, f_charge)
        };
        let gb = if lower {
            add_down(k, -g_charge)
        } else {
            add_up(k, g_charge)
        };
        // Both identities are exact before common's scalar rounding repair:
        // af*f+ag*g = (af+ag)*f-ag*d = (af+ag)*g+af*d.
        let proposals = [
            combine(&[(common, f), (-ag, d)], fb, domain, lower),
            combine(&[(common, g), (af, d)], gb, domain, lower),
            combine(&[(af, f), (ag, g)], k, domain, lower),
        ];
        for form in proposals.into_iter().flatten() {
            let score = concretize(&form, domain, lower)?;
            if best
                .as_ref()
                .is_none_or(|(_, old)| if lower { score > *old } else { score < *old })
            {
                best = Some((form, score));
            }
        }
    }
    best.map(|(form, _)| form)
}

/// Return one certified enclosure before each stage and one at its output.
/// This is linear in network size for a fixed small input dimension.
pub(super) fn states(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
) -> Option<Vec<Enclosure>> {
    states_with_reference(chain, domain, deadline, paired, None)
}

/// References originate only from a checked refinement of this same chain
/// over the search root. Subboxes inherit those enclosures by set inclusion.
/// The private caller owns that provenance; no external bounds are imported.
pub(super) fn states_with_reference(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
    reference: Option<&[Enclosure]>,
) -> Option<Vec<Enclosure>> {
    states_impl(chain, domain, deadline, paired, reference, false)
}

/// Opt-in affine-input upper supports. Existing callers keep the exact old
/// forward path; only this entry snapshots preceding source forms and builds
/// additional repaired supports. References retain the same private provenance.
pub(super) fn states_strong_with_reference(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
    reference: Option<&[Enclosure]>,
) -> Option<Vec<Enclosure>> {
    states_impl(chain, domain, deadline, paired, reference, true)
}

#[cfg(test)]
pub(super) fn states_strong(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
) -> Option<Vec<Enclosure>> {
    states_strong_with_reference(chain, domain, deadline, paired, None)
}

fn states_impl(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
    reference: Option<&[Enclosure]>,
    strong: bool,
) -> Option<Vec<Enclosure>> {
    if strong {
        chain.validate().ok()?;
        if !lp_certificate::suitable_fp_environment() || !alive(deadline) {
            return None;
        }
    }
    if reference.is_some_and(|r| r.len() != chain.stages.len() + 1) {
        return None;
    }
    let input = chain.input_state(domain)?;
    let zero = Form {
        a: vec![0.0; chain.input_dim],
        b: 0.0,
    };
    let mut f = Vec::new();
    for j in 0..chain.input_dim {
        let mut a = vec![0.0; chain.input_dim];
        a[j] = 1.0;
        let form = Form { a, b: 0.0 };
        f.push(Forms {
            lower: form.clone(),
            upper: form,
        });
    }
    let mut g = f.clone();
    let mut d = vec![
        Forms {
            lower: zero.clone(),
            upper: zero
        };
        chain.input_dim
    ];
    let mut result = vec![input];
    // The snapshot index is the input state of the complete following affine
    // run. Both source boxes and root forms belong to that same private state.
    let mut affine_source = strong.then(|| (0usize, f.clone(), g.clone(), d.clone()));
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
                let source = &result[s];
                let mut nf = Vec::with_capacity(*rows);
                let mut ng = Vec::with_capacity(*rows);
                let mut nd = Vec::with_capacity(*rows);
                for i in 0..*rows {
                    if !alive(deadline) {
                        return None;
                    }
                    let ft: Vec<_> = (0..*cols).map(|j| (wf[i * cols + j], &f[j])).collect();
                    let gt: Vec<_> = (0..*cols).map(|j| (wg[i * cols + j], &g[j])).collect();
                    nf.push(pair(&ft, bf[i], bf[i], domain)?);
                    ng.push(pair(&gt, bg[i], bg[i], domain)?);
                    // Both paired and plain controls transport the exact
                    // affine difference. They differ only at ReLU supports.
                    let mut dt = Vec::with_capacity(2 * cols);
                    let mut charge = 0.0;
                    for j in 0..*cols {
                        let a = wf[i * cols + j];
                        let b = wg[i * cols + j];
                        let dw = a - b;
                        let error = scalar_sum_error(a, -b, dw)?;
                        charge = add_up(
                            charge,
                            mul_up(error, magnitude(source.f_lo[j], source.f_hi[j])),
                        );
                        dt.push((dw, &f[j]));
                        dt.push((b, &d[j]));
                    }
                    let db = bf[i] - bg[i];
                    let be = scalar_sum_error(bf[i], -bg[i], db)?;
                    charge = add_up(charge, be);
                    nd.push(pair(
                        &dt,
                        add_down(db, -charge),
                        add_up(db, charge),
                        domain,
                    )?);
                }
                f = nf;
                g = ng;
                d = nd;
                let (f_lo, f_hi) = bounds(&f, domain)?;
                let (g_lo, g_hi) = bounds(&g, domain)?;
                let (d_lo, d_hi) = bounds(&d, domain)?;
                let mut target = Enclosure {
                    f_lo,
                    f_hi,
                    g_lo,
                    g_hi,
                    d_lo,
                    d_hi,
                    hulls: None,
                };
                forward_meet::affine_meet(stage, source, &mut target)?;
                meet_reference(&mut target, reference.and_then(|r| r.get(s + 1)))?;
                result.push(target);
            }
            Stage::Relu { width } => {
                let source = &result[s];
                let ranges: Vec<_> = (0..*width)
                    .map(|j| PairedReluRanges {
                        lf: source.f_lo[j],
                        uf: source.f_hi[j],
                        lg: source.g_lo[j],
                        ug: source.g_hi[j],
                        dl: source.d_lo[j],
                        du: source.d_hi[j],
                    })
                    .collect();
                let hulls = ranges
                    .iter()
                    .map(|r| r.prepare())
                    .collect::<Option<Vec<_>>>()?;
                let mut nf = Vec::with_capacity(*width);
                let mut ng = Vec::with_capacity(*width);
                let mut nd = Vec::with_capacity(*width);
                for j in 0..*width {
                    if !alive(deadline) {
                        return None;
                    }
                    nf.push(relu_forms(&f[j], ranges[j].lf, ranges[j].uf, domain)?);
                    ng.push(relu_forms(&g[j], ranges[j].lg, ranges[j].ug, domain)?);
                    if paired && ranges[j].lf >= 0.0 && ranges[j].lg >= 0.0 {
                        nd.push(d[j].clone());
                    } else if paired && ranges[j].uf <= 0.0 && ranges[j].ug <= 0.0 {
                        let zero = Form {
                            a: vec![0.0; domain.lo.len()],
                            b: 0.0,
                        };
                        nd.push(Forms {
                            lower: zero.clone(),
                            upper: zero,
                        });
                    } else if paired && ranges[j].lf >= 0.0 && ranges[j].ug <= 0.0 {
                        nd.push(f[j].clone());
                    } else if paired && ranges[j].uf <= 0.0 && ranges[j].lg >= 0.0 {
                        nd.push(pair(&[(-1.0, &g[j])], 0.0, 0.0, domain)?);
                    } else if paired {
                        nd.push(Forms {
                            lower: transported_relu_difference(
                                &f[j], &g[j], &d[j], ranges[j], &hulls[j], domain, true, deadline,
                            )?,
                            upper: transported_relu_difference(
                                &f[j], &g[j], &d[j], ranges[j], &hulls[j], domain, false, deadline,
                            )?,
                        });
                    } else {
                        nd.push(pair(&[(1.0, &nf[j]), (-1.0, &ng[j])], 0.0, 0.0, domain)?);
                    }
                }
                let old_upper = if strong {
                    let old_f = bounds(&nf, domain)?.1;
                    let old_g = bounds(&ng, domain)?.1;
                    let old_d = bounds(&nd, domain)?;
                    let (start, previous_f, previous_g, previous_d) = affine_source.as_ref()?;
                    forward_affine_hull::strengthen_tower(
                        chain,
                        *start,
                        s,
                        &result[*start],
                        source,
                        previous_f,
                        &mut nf,
                        domain,
                        false,
                        deadline,
                    )?;
                    forward_affine_hull::strengthen_tower(
                        chain,
                        *start,
                        s,
                        &result[*start],
                        source,
                        previous_g,
                        &mut ng,
                        domain,
                        true,
                        deadline,
                    )?;
                    // Source-paired affine-input supports propagate at every
                    // strong ROOT ReLU. Parallel child search uses the old
                    // forward entry with a certified root reference.
                    if paired {
                        forward_coupled_hull::strengthen(
                            chain, domain, &result, s, *start, previous_d, previous_g, &mut nd,
                            deadline,
                        )?;
                    }
                    Some((old_f, old_g, old_d))
                } else {
                    None
                };
                let (mut f_lo, mut f_hi) = bounds(&nf, domain)?;
                let (mut g_lo, mut g_hi) = bounds(&ng, domain)?;
                let (mut d_lo, mut d_hi) = bounds(&nd, domain)?;
                for j in 0..*width {
                    // Intersect with cheap independent and signed monotonicity
                    // facts. Forms remain valid when boxes become stronger.
                    f_lo[j] = f_lo[j].max(ranges[j].lf.max(0.0));
                    f_hi[j] = f_hi[j].min(ranges[j].uf.max(0.0));
                    g_lo[j] = g_lo[j].max(ranges[j].lg.max(0.0));
                    g_hi[j] = g_hi[j].min(ranges[j].ug.max(0.0));
                    if let Some((old_f, old_g, (old_d_lo, old_d_hi))) = &old_upper {
                        // Tie selection may trade an upper form for a sparser
                        // valid one. Keep the original scalar enclosure too.
                        f_hi[j] = f_hi[j].min(old_f[j]);
                        g_hi[j] = g_hi[j].min(old_g[j]);
                        d_lo[j] = d_lo[j].max(old_d_lo[j]);
                        d_hi[j] = d_hi[j].min(old_d_hi[j]);
                    }
                    if paired {
                        d_lo[j] = d_lo[j].max(ranges[j].dl.min(0.0));
                        d_hi[j] = d_hi[j].min(ranges[j].du.max(0.0));
                    }
                    d_lo[j] = d_lo[j].max((f_lo[j] - g_hi[j]).next_down());
                    d_hi[j] = d_hi[j].min((f_hi[j] - g_lo[j]).next_up());
                    if f_lo[j] > f_hi[j] || g_lo[j] > g_hi[j] || d_lo[j] > d_hi[j] {
                        return None;
                    }
                }
                result[s].hulls = Some(hulls);
                let mut target = Enclosure {
                    f_lo,
                    f_hi,
                    g_lo,
                    g_hi,
                    d_lo,
                    d_hi,
                    hulls: None,
                };
                forward_meet::consistency(&mut target)?;
                meet_reference(&mut target, reference.and_then(|r| r.get(s + 1)))?;
                result.push(target);
                f = nf;
                g = ng;
                d = nd;
                if strong {
                    affine_source = Some((s + 1, f.clone(), g.clone(), d.clone()));
                }
            }
        }
    }
    alive(deadline).then_some(result)
}

fn meet_reference(target: &mut Enclosure, reference: Option<&Enclosure>) -> Option<()> {
    let Some(reference) = reference else {
        return Some(());
    };
    for (lo, hi, rl, ru) in [
        (
            &mut target.f_lo,
            &mut target.f_hi,
            &reference.f_lo,
            &reference.f_hi,
        ),
        (
            &mut target.g_lo,
            &mut target.g_hi,
            &reference.g_lo,
            &reference.g_hi,
        ),
        (
            &mut target.d_lo,
            &mut target.d_hi,
            &reference.d_lo,
            &reference.d_hi,
        ),
    ] {
        if lo.len() != rl.len() || hi.len() != ru.len() || lo.len() != hi.len() {
            return None;
        }
        for j in 0..lo.len() {
            if !rl[j].is_finite() || !ru[j].is_finite() || rl[j] > ru[j] {
                return None;
            }
            lo[j] = lo[j].max(rl[j]);
            hi[j] = hi[j].min(ru[j]);
        }
    }
    forward_meet::consistency(target)
}
