// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One root-wide sequential refinement of intermediate boxes and signed strips.
//!
//! This is intended for an LP scaffold built once, rather than every input box.
//! At state s, every native backward objective reads ONLY states [0,s), so
//! newly tightened target faces cannot justify themselves. Before a following
//! ReLU, both signs of each tower are refined as well as their signed difference.
//! Earlier refined boxes and strips certify new ReLU planes before later
//! objectives read them. Induction thus proves every new enclosure over the
//! full original input box. Original forward affine forms remain valid.

use super::*;

fn alive(deadline: Option<Instant>) -> bool {
    deadline.is_none_or(|d| Instant::now() < d)
}

fn tower_bounds(
    chain: &PairedChain,
    before: usize,
    neuron: usize,
    width: usize,
    g: bool,
    states: &[Enclosure],
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
) -> Option<(f64, f64)> {
    let mut cf = vec![0.0; width];
    let mut cg = vec![0.0; width];
    if g {
        cg[neuron] = 1.0;
    } else {
        cf[neuron] = 1.0;
    }
    let lower = chain
        .back(
            before,
            cf.clone(),
            cg.clone(),
            0.0,
            states,
            domain,
            paired,
            deadline,
        )?
        .0;
    if g {
        cg[neuron] = -1.0;
    } else {
        cf[neuron] = -1.0;
    }
    let upper = -chain
        .back(before, cf, cg, 0.0, states, domain, paired, deadline)?
        .0;
    (alive(deadline) && lower.is_finite() && upper.is_finite() && lower <= upper)
        .then_some((lower, upper))
}

fn valid_interval(lower: f64, upper: f64) -> bool {
    lower.is_finite() && upper.is_finite() && lower <= upper && magnitude(lower, upper) <= 1e150
}

pub(super) fn states(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
) -> Option<Vec<Enclosure>> {
    states_impl(chain, domain, deadline, paired, false, None)
}

pub(super) fn states_strong(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
) -> Option<Vec<Enclosure>> {
    states_impl(chain, domain, deadline, paired, true, None)
}

/// A private source-owned baseline may seed the stronger pass. Keeping its
/// certified faces also preserves the old LP variable allocation and rows.
pub(super) fn states_strong_from_reference(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
    reference: &[Enclosure],
) -> Option<Vec<Enclosure>> {
    states_impl(chain, domain, deadline, paired, true, Some(reference))
}

fn states_impl(
    chain: &PairedChain,
    domain: &BoxDomain,
    deadline: Option<Instant>,
    paired: bool,
    strong: bool,
    reference: Option<&[Enclosure]>,
) -> Option<Vec<Enclosure>> {
    chain.validate().ok()?;
    if !alive(deadline) {
        return None;
    }
    let mut states = if strong {
        if let Some(reference) = reference {
            forward::states_strong_with_reference(chain, domain, deadline, paired, Some(reference))?
        } else {
            let original = forward::states(chain, domain, deadline, paired)?;
            forward::states_strong_with_reference(chain, domain, deadline, paired, Some(&original))?
        }
    } else {
        forward::states(chain, domain, deadline, paired)?
    };
    if states.len() != chain.stages.len() + 1 {
        return None;
    }
    for s in 1..states.len() {
        if !alive(deadline) {
            return None;
        }
        let width = states[s].f_lo.len();
        if states[s].f_hi.len() != width
            || states[s].g_lo.len() != width
            || states[s].g_hi.len() != width
            || states[s].d_lo.len() != width
            || states[s].d_hi.len() != width
        {
            return None;
        }
        let before_relu = matches!(chain.stages.get(s), Some(Stage::Relu { .. }));
        for j in 0..width {
            if !alive(deadline) {
                return None;
            }
            if before_relu {
                let (fl, fh) =
                    tower_bounds(chain, s, j, width, false, &states, domain, deadline, paired)?;
                let (gl, gh) =
                    tower_bounds(chain, s, j, width, true, &states, domain, deadline, paired)?;
                let target = &mut states[s];
                target.f_lo[j] = target.f_lo[j].max(fl);
                target.f_hi[j] = target.f_hi[j].min(fh);
                target.g_lo[j] = target.g_lo[j].max(gl);
                target.g_hi[j] = target.g_hi[j].min(gh);
                if !valid_interval(target.f_lo[j], target.f_hi[j])
                    || !valid_interval(target.g_lo[j], target.g_hi[j])
                {
                    return None;
                }
            }
            let mut cf = vec![0.0; width];
            let mut cg = vec![0.0; width];
            cf[j] = 1.0;
            cg[j] = -1.0;
            let lower = chain
                .back(
                    s,
                    cf.clone(),
                    cg.clone(),
                    0.0,
                    &states,
                    domain,
                    paired,
                    deadline,
                )?
                .0;
            cf[j] = -1.0;
            cg[j] = 1.0;
            let upper = -chain
                .back(s, cf, cg, 0.0, &states, domain, paired, deadline)?
                .0;
            if !alive(deadline) || !lower.is_finite() || !upper.is_finite() || lower > upper {
                return None;
            }
            let state = &mut states[s];
            // Each meet is the intersection of three independently rigorous
            // enclosures: the original transport, native backward, and the
            // independently certified tower boxes.
            state.d_lo[j] = state.d_lo[j]
                .max(lower)
                .max((state.f_lo[j] - state.g_hi[j]).next_down());
            state.d_hi[j] = state.d_hi[j]
                .min(upper)
                .min((state.f_hi[j] - state.g_lo[j]).next_up());
            if !valid_interval(state.d_lo[j], state.d_hi[j]) {
                return None;
            }
        }
        // Each contractor face follows from the exact identity f=g+delta
        // and other already certified faces. It cannot remove a reachable
        // root evaluation, including when newly tightened faces are reused.
        forward_meet::consistency(&mut states[s])?;
        // state[s] is the INPUT of stage[s]. Only a following ReLU consumes
        // hulls at this index, and its next backward users must see the new
        // certified strip rather than the original prepared polygon.
        if before_relu {
            let state = &mut states[s];
            let mut hulls = Vec::with_capacity(width);
            for j in 0..width {
                if !alive(deadline) {
                    return None;
                }
                hulls.push(
                    PairedReluRanges {
                        lf: state.f_lo[j],
                        uf: state.f_hi[j],
                        lg: state.g_lo[j],
                        ug: state.g_hi[j],
                        dl: state.d_lo[j],
                        du: state.d_hi[j],
                    }
                    .prepare()?,
                );
            }
            state.hulls = Some(hulls);
        }
    }
    alive(deadline).then_some(states)
}

#[cfg(test)]
#[path = "refine_tests.rs"]
mod tests;
