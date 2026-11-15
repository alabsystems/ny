// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Shared-input corner floors for the authenticated signed replay.
//!
//! For each component, the minimum of all compatible affine masks is concave.
//! Their sum plus the global affine term therefore attains its minimum over
//! the selected input box at a corner. Omitted coordinates are minimized
//! separately, which can only weaken the floor. At a child, each stored unary
//! phase minimum bounds the same fully compatible-mask envelope from below;
//! taking their maximum at each corner remains safe by that common envelope.
//! The computed maximum itself need not be concave.
//!
//! The wrapper supplies admission, original Source/Plan authority, both replay
//! barriers, deadlines and one-shot use. This numerical data alone is not a
//! certificate. `super::super` is the signed module.
use super::super::certificate::I;
use super::super::{add_down, finite, require_hull_environment};

pub(in crate::single_hidden_hull::mip_cuts::signed) const MAX_D: usize = 32;
pub(in crate::single_hidden_hull::mip_cuts::signed) const MAX_K: usize = 7;
const MAX_CORNERS: usize = 1 << MAX_K;
const MAX_C: usize = 128;
const MAX_MASKS: usize = 4096;

fn valid_i(x: I) -> bool {
    x.lo.is_finite() && x.hi.is_finite() && x.lo <= x.hi
}
fn sub_down(a: f64, b: f64) -> Option<f64> {
    finite(add_down(a, -b))
}
fn gate(poll: &mut dyn FnMut() -> bool) -> Option<()> {
    require_hull_environment().ok()?;
    poll().then_some(())
}

#[derive(Debug)]
struct Layout {
    bounds: Vec<[f64; 2]>,
    bits: Vec<u8>,
    k: usize,
}
impl Layout {
    fn new(lower: &[f64], upper: &[f64], bits: &[u8], k: usize) -> Option<Self> {
        let d = lower.len();
        if d == 0
            || d > MAX_D
            || upper.len() != d
            || k == 0
            || k > MAX_K
            || k > d
            || bits.is_empty()
            || bits.len() > MAX_C
            || bits.iter().any(|&n| n == 0 || n > 8)
        {
            return None;
        }
        let masks = bits
            .iter()
            .try_fold(0usize, |v, &n| v.checked_add(1usize << n))?;
        if masks > MAX_MASKS {
            return None;
        }
        let mut bounds = Vec::with_capacity(d);
        for (&l, &u) in lower.iter().zip(upper) {
            if !l.is_finite() || !u.is_finite() || l > u {
                return None;
            }
            bounds.push([l, u]);
        }
        Some(Self {
            bounds,
            bits: bits.to_vec(),
            k,
        })
    }
    fn coefficients(&self, coeff: &[I]) -> Option<()> {
        (coeff.len() == self.bounds.len() && coeff.iter().copied().all(valid_i)).then_some(())
    }
    fn rows(&self) -> usize {
        self.bits.iter().map(|&n| 1 + 2 * n as usize).sum()
    }
}

/// Canonical sequential coverage is sufficient and needs no per-mask bitset.
/// Any duplicate, skipped mask, reordered component or failed poll is terminal.
#[derive(Debug, Default)]
struct Cursor {
    component: usize,
    mask: usize,
    failed: bool,
}
impl Cursor {
    fn accept(&mut self, layout: &Layout, component: usize, mask: usize) -> Option<()> {
        if self.failed
            || component != self.component
            || mask != self.mask
            || component >= layout.bits.len()
        {
            self.failed = true;
            return None;
        }
        Some(())
    }
    fn advance(&mut self, layout: &Layout) -> bool {
        self.mask += 1;
        if self.mask == 1usize << layout.bits[self.component] {
            self.component += 1;
            self.mask = 0;
            true
        } else {
            false
        }
    }
    fn complete(&self, layout: &Layout) -> bool {
        !self.failed && self.component == layout.bits.len() && self.mask == 0
    }
}

/// Private, move-only advisory axes. The wrapper may publish this value only
/// after the FIRST original replay equality/clock barrier has passed.
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Axes {
    layout: Layout,
    axes: Vec<usize>,
}
impl Axes {
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn selected(&self) -> &[usize] {
        &self.axes
    }
}

#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Ranker {
    layout: Layout,
    cursor: Cursor,
    local: Vec<[f64; 2]>,
    total: Vec<[f64; 2]>,
    raw_minima: Vec<f64>,
}
impl Ranker {
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn new(
        lower: &[f64],
        upper: &[f64],
        bits: &[u8],
        k: usize,
    ) -> Option<Self> {
        require_hull_environment().ok()?;
        let layout = Layout::new(lower, upper, bits, k)?;
        Some(Self {
            local: vec![[f64::INFINITY; 2]; layout.bounds.len()],
            total: vec![[0.; 2]; layout.bounds.len()],
            raw_minima: vec![f64::INFINITY; layout.bits.len()],
            layout,
            cursor: Cursor::default(),
        })
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn push(
        &mut self,
        component: usize,
        mask: usize,
        coeff: &[I],
        raw_lower: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let result = self.push_inner(component, mask, coeff, raw_lower, poll);
        if result.is_none() {
            self.cursor.failed = true;
        }
        result
    }
    fn push_inner(
        &mut self,
        component: usize,
        mask: usize,
        coeff: &[I],
        raw_lower: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        self.cursor.accept(&self.layout, component, mask)?;
        gate(poll)?;
        self.layout.coefficients(coeff)?;
        finite(raw_lower)?;
        self.raw_minima[component] = self.raw_minima[component].min(raw_lower);
        for (i, &q) in coeff.iter().enumerate() {
            let [l, u] = self.layout.bounds[i];
            // Recompute exactly the SAME stored raw mask summand, using the
            // same I::minimum and verified floating-point environment.
            let removed = sub_down(raw_lower, q.minimum(l, u)?)?;
            for (side, x) in [l, u].into_iter().enumerate() {
                let value = finite(add_down(removed, q.scale(x)?.lo))?;
                self.local[i][side] = self.local[i][side].min(value);
            }
        }
        if self.cursor.advance(&self.layout) {
            for (sum, local) in self.total.iter_mut().zip(&mut self.local) {
                for side in 0..2 {
                    sum[side] = finite(add_down(sum[side], local[side]))?;
                    local[side] = f64::INFINITY;
                }
            }
        }
        gate(poll)
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn finish(
        self,
        checked_constant: f64,
        source_d: f64,
        checked_g: &[f64],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Axes> {
        gate(poll)?;
        if !self.cursor.complete(&self.layout) || checked_g.len() != self.layout.bounds.len() {
            return None;
        }
        let offset = offset(checked_constant, source_d, &self.raw_minima)?;
        let mut scores = [f64::NEG_INFINITY; MAX_D];
        let mut order = [0usize; MAX_D];
        for axis in 0..self.layout.bounds.len() {
            gate(poll)?;
            let mut rest = offset;
            for (i, &g) in checked_g.iter().enumerate() {
                let q = I::point(g)?;
                if i != axis {
                    let [l, u] = self.layout.bounds[i];
                    rest = finite(add_down(rest, q.minimum(l, u)?))?;
                }
            }
            let q = I::point(checked_g[axis])?;
            let mut value = f64::INFINITY;
            for (side, &x) in self.layout.bounds[axis].iter().enumerate() {
                let at = finite(add_down(rest, q.scale(x)?.lo))?;
                value = value.min(finite(add_down(at, self.total[axis][side]))?);
            }
            scores[axis] = finite(value)?;
            order[axis] = axis;
        }
        let d = self.layout.bounds.len();
        // Deterministic advice; no optimal-subset or joint-optimum claim.
        order[..d].sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let axes = order[..self.layout.k].to_vec();
        gate(poll)?;
        Some(Axes {
            layout: self.layout,
            axes,
        })
    }
}

fn offset(checked_constant: f64, source_d: f64, raw_minima: &[f64]) -> Option<f64> {
    let mut value = finite(checked_constant)?;
    for &minimum in raw_minima {
        value = sub_down(value, finite(minimum)?)?;
    }
    // d appears exactly once. The old priced global-g rounding residual stays.
    finite(add_down(value, finite(source_d)?))
}

#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Collector {
    axes: Axes,
    cursor: Cursor,
    row: usize,
    values: Vec<f64>,
    raw_minima: Vec<f64>,
}
impl Collector {
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn new(axes: Axes) -> Option<Self> {
        require_hull_environment().ok()?;
        let count = axes.layout.rows().checked_mul(1usize << axes.layout.k)?;
        Some(Self {
            values: vec![f64::INFINITY; count],
            raw_minima: vec![f64::INFINITY; axes.layout.bits.len()],
            axes,
            cursor: Cursor::default(),
            row: 0,
        })
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn push(
        &mut self,
        component: usize,
        mask: usize,
        coeff: &[I],
        raw_lower: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let result = self.push_inner(component, mask, coeff, raw_lower, poll);
        if result.is_none() {
            self.cursor.failed = true;
        }
        result
    }
    fn push_inner(
        &mut self,
        component: usize,
        mask: usize,
        coeff: &[I],
        raw_lower: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let layout = &self.axes.layout;
        self.cursor.accept(layout, component, mask)?;
        gate(poll)?;
        layout.coefficients(coeff)?;
        let bits = layout.bits[component] as usize;
        let count = 1usize << layout.k;
        let mut value = finite(raw_lower)?;
        self.raw_minima[component] = self.raw_minima[component].min(raw_lower);
        // Remove raw summands before adding endpoints. Removing a strengthened
        // Cell/Ray/LP value here would be unsound; the wrapper must pass RAW L.
        for &axis in &self.axes.axes {
            let [l, u] = layout.bounds[axis];
            value = sub_down(value, coeff[axis].minimum(l, u)?)?;
        }
        let mut delta = [I::zero(); MAX_K];
        for (bit, &axis) in self.axes.axes.iter().enumerate() {
            let [l, u] = layout.bounds[axis];
            let low = coeff[axis].scale(l)?;
            let high = coeff[axis].scale(u)?;
            value = finite(add_down(value, low.lo))?;
            // Do not replace this with scale(rounded(u-l)).
            delta[bit] = high.sub(low)?;
        }
        // One paid 128-word scratch replaces repeated bit/side extraction,
        // row-index arithmetic and polling at every (corner, negative-bit).
        let mut corners = [0.; MAX_CORNERS];
        for step in 0..count {
            if step % 32 == 0 {
                gate(poll)?;
            }
            let corner = step ^ (step >> 1);
            if step != 0 {
                let bit = step.trailing_zeros() as usize;
                let change = if corner & (1 << bit) != 0 {
                    delta[bit].lo
                } else {
                    -delta[bit].hi
                };
                value = finite(add_down(value, change))?;
            }
            corners[corner] = value;
        }
        gate(poll)?;
        let start = self.row * count;
        for (out, &value) in self.values[start..start + count]
            .iter_mut()
            .zip(&corners[..count])
        {
            *out = out.min(value);
        }
        for bit in 0..bits {
            gate(poll)?;
            let side = (mask >> bit) & 1;
            let start = (self.row + 1 + 2 * bit + side) * count;
            for (out, &value) in self.values[start..start + count]
                .iter_mut()
                .zip(&corners[..count])
            {
                *out = out.min(value);
            }
        }
        if self.cursor.advance(layout) {
            self.row += 1 + 2 * bits;
        }
        gate(poll)
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn finish(
        self,
        checked_constant: f64,
        source_d: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Data> {
        gate(poll)?;
        if !self.cursor.complete(&self.axes.layout)
            || self.row != self.axes.layout.rows()
            || self.values.iter().any(|x| !x.is_finite())
        {
            return None;
        }
        let offset = offset(checked_constant, source_d, &self.raw_minima)?;
        gate(poll)?;
        Some(Data {
            axes: self.axes,
            values: self.values,
            offset,
        })
    }
}

/// Opaque, immutable numerical product. NOT a freestanding Source certificate.
/// The wrapper seals words and authenticates original Source/Plan plus both
/// completed replays before exposing it through a BoundCache extension.
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Data {
    axes: Axes,
    values: Vec<f64>,
    offset: f64,
}
/// Borrowed COMPLETE target envelope; the wrapper authenticates its origin.
pub(in crate::single_hidden_hull::mip_cuts::signed) struct RootReplacement<'a> {
    pub target: usize,
    pub axes: &'a [usize],
    pub values: &'a [f64],
}
impl Data {
    /// Separately prepaid root-only traversal. Original lower() stays unchanged.
    /// This family substitutes the target point in a complete sum and minimizes
    /// separately; it never changes or refunds the original raw-minimum offset.
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn replacement_root(
        &self,
        global_coeff: &[I],
        replacement: RootReplacement<'_>,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        gate(poll)?;
        let layout = &self.axes.layout;
        layout.coefficients(global_coeff)?;
        if replacement.target >= layout.bits.len()
            || replacement.axes.is_empty()
            || replacement.axes.len() > layout.k
            || &self.axes.axes[..replacement.axes.len()] != replacement.axes
            || replacement.values.len() != 1usize << replacement.axes.len()
            || replacement.values.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        let mut selected = [false; MAX_D];
        for &axis in &self.axes.axes {
            selected[axis] = true;
        }
        let mut global = finite(self.offset)?;
        for (i, &q) in global_coeff.iter().enumerate() {
            if !selected[i] {
                let [l, u] = layout.bounds[i];
                global = finite(add_down(global, q.minimum(l, u)?))?;
            }
        }
        let mut delta = [I::zero(); MAX_K];
        for (bit, &axis) in self.axes.axes.iter().enumerate() {
            let [l, u] = layout.bounds[axis];
            let low = global_coeff[axis].scale(l)?;
            let high = global_coeff[axis].scale(u)?;
            global = finite(add_down(global, low.lo))?;
            delta[bit] = high.sub(low)?;
        }
        let count = 1usize << layout.k;
        let mut best = f64::INFINITY;
        for step in 0..count {
            gate(poll)?;
            let corner = step ^ (step >> 1);
            if step != 0 {
                let bit = step.trailing_zeros() as usize;
                let change = if corner & (1usize << bit) != 0 {
                    delta[bit].lo
                } else {
                    -delta[bit].hi
                };
                global = finite(add_down(global, change))?;
            }
            let mut value = global;
            let mut row = 0usize;
            for (component, &n) in layout.bits.iter().enumerate() {
                let part = if component == replacement.target {
                    replacement.values[corner & (replacement.values.len() - 1)]
                } else {
                    self.values[row * count + corner]
                };
                value = finite(add_down(value, part))?;
                row += 1 + 2 * n as usize;
            }
            best = best.min(value);
        }
        gate(poll)?;
        finite(best)
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn selected(&self) -> &[usize] {
        &self.axes.axes
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn negative_bits(&self) -> &[u8] {
        &self.axes.layout.bits
    }
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn words(
        &self,
    ) -> impl Iterator<Item = u64> + '_ {
        [
            1,
            self.axes.layout.bounds.len() as u64,
            self.axes.layout.bits.len() as u64,
            self.axes.axes.len() as u64,
            self.values.len() as u64,
            self.offset.to_bits(),
        ]
        .into_iter()
        .chain(
            self.axes
                .layout
                .bounds
                .iter()
                .flat_map(|b| b.iter().map(|v| v.to_bits())),
        )
        .chain(self.axes.layout.bits.iter().map(|&v| v as u64))
        .chain(self.axes.axes.iter().map(|&v| v as u64))
        .chain(self.values.iter().map(|v| v.to_bits()))
    }
    /// `global_coeff` is original exported point g, or g plus legal child-sign
    /// beta terms. `extra_constant` is ONLY the corresponding beta bias term;
    /// old component gains/intercepts/d must never be added a second time.
    /// Existing authenticated head mapping constructs disjoint `on`/`off`.
    pub(in crate::single_hidden_hull::mip_cuts::signed) fn lower(
        &self,
        global_coeff: &[I],
        extra_constant: I,
        on: &[usize],
        off: &[usize],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        gate(poll)?;
        let layout = &self.axes.layout;
        layout.coefficients(global_coeff)?;
        if !valid_i(extra_constant) || on.len() != layout.bits.len() || off.len() != on.len() {
            return None;
        }
        for ((&n, &a), &b) in layout.bits.iter().zip(on).zip(off) {
            let allowed = (1usize << n) - 1;
            if a & b != 0 || (a | b) & !allowed != 0 {
                return None;
            }
        }
        let mut selected = [false; MAX_D];
        for &axis in &self.axes.axes {
            selected[axis] = true;
        }
        let mut global = finite(add_down(self.offset, extra_constant.lo))?;
        for (i, &q) in global_coeff.iter().enumerate() {
            if !selected[i] {
                let [l, u] = layout.bounds[i];
                global = finite(add_down(global, q.minimum(l, u)?))?;
            }
        }
        let mut delta = [I::zero(); MAX_K];
        for (bit, &axis) in self.axes.axes.iter().enumerate() {
            let [l, u] = layout.bounds[axis];
            let low = global_coeff[axis].scale(l)?;
            let high = global_coeff[axis].scale(u)?;
            global = finite(add_down(global, low.lo))?;
            delta[bit] = high.sub(low)?;
        }
        let count = 1usize << layout.k;
        let mut best = f64::INFINITY;
        for step in 0..count {
            gate(poll)?;
            let corner = step ^ (step >> 1);
            if step != 0 {
                let bit = step.trailing_zeros() as usize;
                let change = if corner & (1 << bit) != 0 {
                    delta[bit].lo
                } else {
                    -delta[bit].hi
                };
                global = finite(add_down(global, change))?;
            }
            let mut value = global;
            let mut row = 0usize;
            for (component, &n) in layout.bits.iter().enumerate() {
                let mut part = self.values[row * count + corner];
                for bit in 0..n as usize {
                    let phase = 1 << bit;
                    if on[component] & phase != 0 || off[component] & phase != 0 {
                        let side = usize::from(on[component] & phase != 0);
                        part = part.max(self.values[(row + 1 + 2 * bit + side) * count + corner]);
                    }
                }
                value = finite(add_down(value, part))?;
                row += 1 + 2 * n as usize;
            }
            best = best.min(value);
        }
        gate(poll)?;
        finite(best)
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_core_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_core_edge_tests.rs"]
mod edge_tests;
