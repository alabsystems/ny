// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded adaptive endpoint-face cover for one authenticated affine family.
//!
//! The minimum over all mask affines is concave. A complete binary endpoint
//! cover contains every original box vertex, even though its faces need not
//! contain every interior point. All leaves therefore use ONE unchanged
//! affine family. Nominal advice never removes a leaf from the final proof.
use super::super::certificate::I;
use super::super::{add_down, finite, require_hull_environment};
use super::core::Ranker;

const MAX_D: usize = 32;
const MAX_C: usize = 128;
const MAX_MASKS: usize = 4096;
const MAX_SPLITS: usize = 64;
const MAX_DEPTH: usize = 16;
const ADVICE_CAP: usize = 7_500_000;
const LEAF: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Quote {
    pub d: usize,
    pub k: usize,
    pub masks: usize,
    pub components: usize,
    pub max_splits: usize,
    pub max_depth: usize,
    pub advice_work: usize,
    pub work: usize,
    pub bytes: usize,
}
/// Work excludes wrapper setup/ownership and ordinary-cache entry credits.
/// No quotation is a promise that the live owner can afford this mode.
pub(super) fn quote(d: usize, bits: &[u8], k: usize) -> Option<Quote> {
    if fixed_scratch_bytes() > 8192 || size_of::<Node>() > 8 || size_of::<Face>() > 32 {
        return None;
    }
    if !(1..=MAX_D).contains(&d)
        || !(1..=7).contains(&k)
        || k > d
        || bits.is_empty()
        || bits.len() > MAX_C
        || bits.iter().any(|&b| b == 0 || b > 8)
    {
        return None;
    }
    let c = bits.len();
    let m: usize = bits.iter().map(|&b| 1usize << b).sum();
    if m > MAX_MASKS {
        return None;
    }
    let z = 1usize << k;
    let e = if k == d { 0 } else { MAX_SPLITS };
    let depth = d.min(MAX_DEPTH);
    let leaves = z + e;
    let nodes = z + 2 * e;
    let rank = 4096 + 32 * m * (d + 1) + 32 * c * d + 32 * d * d + 32 * (c + d + k + 1);
    let capture = 8 * m * (d + 1);
    let initial = 4096 + 16 * m * z + 16 * m * k + 32 * c * z;
    let per_split = 4 * m * depth + 24 * m + 8 * (c + 1) * d + 512 + 64 * c;
    let advice_work = (initial + e * per_split).min(ADVICE_CAP);
    if initial > advice_work {
        return None;
    }
    // Original I validation; every potentially selected coordinate's interval
    // endpoint products/removal; Gray seed traversal; extra binary tree nodes;
    // complete per-component accumulation and final original-offset pricing.
    let proof = m * (8 * d + 64 * d + 64)
        + 20 * m * z
        + 20 * m * e
        + 32 * c * leaves
        + 32 * (d + c + d + 1);
    let finish = 64 * (d + c + k + leaves + 1) + 32 * leaves * (k + d + 1);
    let work = rank + capture + advice_work + proof + finish;
    // Stage 1: nominal deltas+rawL, Ranker and box/count copies, leaf state,
    // per-leaf component winner indices, tree, and bounded temporary slices.
    // Stage 2: the nominal affine allocation has been dropped; retain tree,
    // two leaf arrays, original raw minima, boxes and bounded stack scratch.
    let first =
        8 * m * (d + 1) + 32 * leaves + 2 * c * leaves + 8 * nodes + 128 * d + 32 * c + 8192;
    let second = 8 * nodes + 16 * leaves + 64 * d + 32 * c + 8192;
    Some(Quote {
        d,
        k,
        masks: m,
        components: c,
        max_splits: e,
        max_depth: depth,
        advice_work,
        work,
        bytes: first.max(second),
    })
}
fn gate(poll: &mut dyn FnMut() -> bool) -> Option<()> {
    require_hull_environment().ok()?;
    poll().then_some(())
}
fn sub(a: f64, b: f64) -> Option<f64> {
    finite(add_down(a, -b))
}
fn valid(q: I) -> bool {
    q.lo.is_finite() && q.hi.is_finite() && q.lo <= q.hi
}
fn offset(constant: f64, d: f64, minima: &[f64]) -> Option<f64> {
    let mut v = finite(constant)?;
    for &m in minima {
        v = sub(v, finite(m)?)?;
    }
    finite(add_down(v, finite(d)?))
}
#[derive(Debug, Default)]
struct Cursor {
    component: usize,
    mask: usize,
    failed: bool,
}
impl Cursor {
    fn accept(&mut self, bits: &[u8], c: usize, m: usize) -> Option<()> {
        if self.failed || self.component != c || self.mask != m || c >= bits.len() {
            self.failed = true;
            return None;
        }
        Some(())
    }
    fn next(&mut self, bits: &[u8]) -> bool {
        self.mask += 1;
        if self.mask == 1usize << bits[self.component] {
            self.component += 1;
            self.mask = 0;
            true
        } else {
            false
        }
    }
    fn complete(&self, bits: &[u8]) -> bool {
        !self.failed && self.component == bits.len() && self.mask == 0
    }
}
#[derive(Debug, Clone, Copy)]
struct Node {
    axis: u8,
    low: u16,
    high: u16,
    leaf: u16,
}
impl Node {
    fn leaf(index: usize) -> Self {
        Self {
            axis: LEAF,
            low: 0,
            high: 0,
            leaf: index as u16,
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct Face {
    assigned: u32,
    upper: u32,
    value: f64,
    node: usize,
    depth: usize,
}

/// Advisory capture. It has no certificate authority and is never attached.
#[derive(Debug)]
pub(super) struct Capture {
    q: Quote,
    bits: Vec<u8>,
    bounds: Vec<[f64; 2]>,
    ranker: Ranker,
    values: Vec<f64>,
    raw_minima: Vec<f64>,
    cursor: Cursor,
}
impl Capture {
    pub(super) fn new(lower: &[f64], upper: &[f64], bits: &[u8], q: Quote) -> Option<Self> {
        gate(&mut || true)?;
        let expected = quote(lower.len(), bits, q.k)?;
        if expected != q || upper.len() != q.d || bits.len() != q.components {
            return None;
        }
        let ranker = Ranker::new(lower, upper, bits, q.k)?;
        let mut bounds = Vec::with_capacity(q.d);
        for (&l, &u) in lower.iter().zip(upper) {
            if !l.is_finite() || !u.is_finite() || l > u {
                return None;
            }
            bounds.push([l, u]);
        }
        Some(Self {
            q,
            bits: bits.to_vec(),
            bounds,
            ranker,
            values: Vec::with_capacity(q.masks * (q.d + 1)),
            raw_minima: vec![f64::INFINITY; q.components],
            cursor: Cursor::default(),
        })
    }
    pub(super) fn push(
        &mut self,
        c: usize,
        m: usize,
        coeff: &[I],
        raw: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let r = self.push_inner(c, m, coeff, raw, poll);
        if r.is_none() {
            self.cursor.failed = true;
        }
        r
    }
    fn push_inner(
        &mut self,
        c: usize,
        m: usize,
        coeff: &[I],
        raw: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        self.cursor.accept(&self.bits, c, m)?;
        gate(poll)?;
        if coeff.len() != self.q.d || !coeff.iter().copied().all(valid) {
            return None;
        }
        finite(raw)?;
        self.ranker.push(c, m, coeff, raw, poll)?;
        self.raw_minima[c] = self.raw_minima[c].min(raw);
        self.values.push(raw);
        for (&q, &[l, u]) in coeff.iter().zip(&self.bounds) {
            // Only advice uses rounded width and an interval endpoint. The
            // second replay independently prices original interval products.
            self.values.push(finite(q.lo * finite(u - l)?)?);
        }
        self.cursor.next(&self.bits);
        gate(poll)
    }
    pub(super) fn finish(
        self,
        constant: f64,
        d: f64,
        g: &[f64],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Cover> {
        if !self.cursor.complete(&self.bits)
            || self.values.len() != self.q.masks * (self.q.d + 1)
            || g.len() != self.q.d
        {
            return None;
        }
        gate(poll)?;
        let axes = self
            .ranker
            .finish(constant, d, g, poll)?
            .selected()
            .to_vec();
        let mut base = offset(constant, d, &self.raw_minima)?;
        let mut delta = [0.; MAX_D];
        for (i, (&v, &[l, u])) in g.iter().zip(&self.bounds).enumerate() {
            let v = finite(v)?;
            base = finite(base + (v * l).min(v * u))?;
            delta[i] = finite(v * finite(u - l)?)?;
        }
        let q = self.q;
        let advice = Advice {
            q,
            bits: self.bits,
            bounds: self.bounds,
            axes,
            values: self.values,
            global_base: base,
            global_delta: delta,
            spent: 0,
            faces: Vec::with_capacity((1 << q.k) + q.max_splits),
            winners: vec![0; ((1 << q.k) + q.max_splits) * q.components],
            nodes: Vec::with_capacity((1 << q.k) + 2 * q.max_splits),
            splits: 0,
        };
        advice.run(poll)
        // `advice` is consumed: nominal affine values/winner arrays are dropped
        // before the second replay can allocate its directed leaf buffers.
    }
}

struct Advice {
    q: Quote,
    bits: Vec<u8>,
    bounds: Vec<[f64; 2]>,
    axes: Vec<usize>,
    values: Vec<f64>,
    global_base: f64,
    global_delta: [f64; MAX_D],
    spent: usize,
    faces: Vec<Face>,
    winners: Vec<u16>,
    nodes: Vec<Node>,
    splits: usize,
}
fn correction(delta: f64, side: bool) -> f64 {
    if side {
        delta.max(0.)
    } else {
        (-delta).max(0.)
    }
}
impl Advice {
    fn charge(&mut self, amount: usize) -> Option<()> {
        let next = self.spent.checked_add(amount)?;
        if next > self.q.advice_work {
            return None;
        }
        self.spent = next;
        Some(())
    }
    fn row(&self, index: usize) -> &[f64] {
        &self.values[index * (self.q.d + 1)..(index + 1) * (self.q.d + 1)]
    }
    fn run(mut self, poll: &mut dyn FnMut() -> bool) -> Option<Cover> {
        gate(poll)?;
        let z = 1usize << self.q.k;
        self.charge(
            4096 + 16 * self.q.masks * z
                + 16 * self.q.masks * self.q.k
                + 32 * self.q.components * z,
        )?;
        let mut assigned = 0u32;
        for &i in &self.axes {
            assigned |= 1u32 << i;
        }
        for corner in 0..z {
            let mut value = self.global_base;
            let mut upper = 0u32;
            for (bit, &axis) in self.axes.iter().enumerate() {
                let side = corner & (1 << bit) != 0;
                if side {
                    upper |= 1u32 << axis;
                }
                value = finite(value + correction(self.global_delta[axis], side))?;
            }
            self.faces.push(Face {
                assigned,
                upper,
                value,
                node: corner,
                depth: self.q.k,
            });
            self.nodes.push(Node::leaf(corner));
        }
        let mut start = 0;
        let mut best = [f64::INFINITY; 128];
        let mut mask_best = [0u16; 128];
        for component in 0..self.q.components {
            best[..z].fill(f64::INFINITY);
            for mask in 0..1usize << self.bits[component] {
                gate(poll)?;
                let row = self.row(start + mask);
                let mut v = row[0];
                for &axis in &self.axes {
                    v = finite(v + correction(row[axis + 1], false))?;
                }
                for step in 0..z {
                    let corner = step ^ (step >> 1);
                    if step != 0 {
                        let bit = step.trailing_zeros() as usize;
                        let axis = self.axes[bit];
                        v = finite(
                            v + if corner & (1 << bit) != 0 {
                                row[axis + 1]
                            } else {
                                -row[axis + 1]
                            },
                        )?;
                    }
                    if v < best[corner] {
                        best[corner] = v;
                        mask_best[corner] = mask as u16;
                    }
                }
            }
            for corner in 0..z {
                self.faces[corner].value = finite(self.faces[corner].value + best[corner])?;
                self.winners[corner * self.q.components + component] = mask_best[corner];
            }
            start += 1usize << self.bits[component];
        }
        let mut stop = Stop::SplitLimit;
        while self.splits < self.q.max_splits {
            gate(poll)?;
            let index = (0..self.faces.len()).min_by(|&a, &b| {
                self.faces[a]
                    .value
                    .total_cmp(&self.faces[b].value)
                    .then(a.cmp(&b))
            })?;
            let face = self.faces[index];
            if face.value > 0. {
                stop = Stop::PositiveAdvice;
                break;
            }
            if face.depth >= self.q.max_depth || face.depth == self.q.d {
                stop = Stop::DepthLimit;
                break;
            }
            let cost = 4 * self.q.masks * face.depth
                + 24 * self.q.masks
                + 8 * (self.q.components + 1) * (self.q.d - face.depth)
                + 512
                + 64 * self.q.components;
            if self.charge(cost).is_none() {
                stop = Stop::AdviceLimit;
                break;
            }
            let mut scores = [f64::NEG_INFINITY; MAX_D];
            for axis in 0..self.q.d {
                if face.assigned & (1u32 << axis) != 0 {
                    continue;
                }
                let mut total = self.global_delta[axis];
                let mut separate = total.min(0.);
                let mut start = 0;
                for component in 0..self.q.components {
                    let mask = self.winners[index * self.q.components + component] as usize;
                    let delta = self.row(start + mask)[axis + 1];
                    total = finite(total + delta)?;
                    separate = finite(separate + delta.min(0.))?;
                    start += 1usize << self.bits[component];
                }
                scores[axis] = finite(total.min(0.) - separate)?;
            }
            let axis = (0..self.q.d)
                .filter(|&i| face.assigned & (1u32 << i) == 0)
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))?;
            let mut child = [self.global_base; 2];
            for i in 0..self.q.d {
                if face.assigned & (1u32 << i) != 0 {
                    let v = correction(self.global_delta[i], face.upper & (1u32 << i) != 0);
                    child[0] = finite(child[0] + v)?;
                    child[1] = finite(child[1] + v)?;
                }
            }
            for side in 0..2 {
                child[side] = finite(child[side] + correction(self.global_delta[axis], side != 0))?;
            }
            let new_index = self.faces.len();
            let mut start = 0;
            // Decode this face once. Each nominal parent item then performs
            // one delta load, sign multiplication, max and addition; finiteness
            // is checked once after its bounded sum, not once per coordinate.
            let mut dimensions = [0usize; MAX_DEPTH];
            let mut signs = [0.; MAX_DEPTH];
            let mut used = 0;
            for dim in 0..self.q.d {
                if face.assigned & (1u32 << dim) != 0 {
                    dimensions[used] = dim;
                    signs[used] = if face.upper & (1u32 << dim) != 0 {
                        1.
                    } else {
                        -1.
                    };
                    used += 1;
                }
            }
            if used != face.depth {
                return None;
            }
            for component in 0..self.q.components {
                let mut best = [f64::INFINITY; 2];
                let mut winner = [0u16; 2];
                for mask in 0..1usize << self.bits[component] {
                    let row = self.row(start + mask);
                    let mut value = row[0];
                    for (&dim, &sign) in dimensions[..used].iter().zip(&signs[..used]) {
                        value += (sign * row[dim + 1]).max(0.);
                    }
                    finite(value)?;
                    for side in 0..2 {
                        let v = finite(value + correction(row[axis + 1], side != 0))?;
                        if v < best[side] {
                            best[side] = v;
                            winner[side] = mask as u16;
                        }
                    }
                }
                for side in 0..2 {
                    child[side] = finite(child[side] + best[side])?;
                }
                self.winners[index * self.q.components + component] = winner[0];
                self.winners[new_index * self.q.components + component] = winner[1];
                start += 1usize << self.bits[component];
                gate(poll)?;
            }
            let left = self.nodes.len();
            let right = left + 1;
            self.nodes[face.node] = Node {
                axis: axis as u8,
                low: left as u16,
                high: right as u16,
                leaf: 0,
            };
            self.nodes.push(Node::leaf(index));
            self.nodes.push(Node::leaf(new_index));
            let assigned = face.assigned | (1u32 << axis);
            self.faces[index] = Face {
                assigned,
                upper: face.upper,
                value: child[0],
                node: left,
                depth: face.depth + 1,
            };
            self.faces.push(Face {
                assigned,
                upper: face.upper | (1u32 << axis),
                value: child[1],
                node: right,
                depth: face.depth + 1,
            });
            self.splits += 1;
        }
        gate(poll)?;
        let cover = Cover {
            q: self.q,
            bits: self.bits,
            bounds: self.bounds,
            axes: self.axes,
            nodes: self.nodes,
            leaves: self.faces.len(),
            splits: self.splits,
            advice_spent: self.spent,
            stop,
        };
        cover.validate()?;
        gate(poll)?;
        Some(cover)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stop {
    PositiveAdvice,
    SplitLimit,
    DepthLimit,
    AdviceLimit,
}
/// Opaque complete endpoint cover. It is advisory until the second replay.
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct Cover {
    q: Quote,
    bits: Vec<u8>,
    bounds: Vec<[f64; 2]>,
    axes: Vec<usize>,
    nodes: Vec<Node>,
    leaves: usize,
    splits: usize,
    advice_spent: usize,
    stop: Stop,
}
impl Cover {
    pub(super) fn matches_quote(&self, q: Quote, bits: &[u8]) -> bool {
        self.q == q && self.bits == bits
    }
    pub(super) fn matches_bounds(&self, lower: &[f64], upper: &[f64]) -> bool {
        lower.len() == self.bounds.len()
            && upper.len() == self.bounds.len()
            && self
                .bounds
                .iter()
                .zip(lower.iter().zip(upper))
                .all(|(b, (&l, &u))| b[0].to_bits() == l.to_bits() && b[1].to_bits() == u.to_bits())
    }
    fn validate(&self) -> Option<()> {
        require_hull_environment().ok()?;
        if quote(self.q.d, &self.bits, self.q.k)? != self.q
            || self.advice_spent > self.q.advice_work
            || self
                .bounds
                .iter()
                .any(|b| !b[0].is_finite() || !b[1].is_finite() || b[0] > b[1])
        {
            return None;
        }
        if self.axes.len() != self.q.k
            || self.bits.len() != self.q.components
            || self.bounds.len() != self.q.d
            || self.splits > self.q.max_splits
            || self.leaves != (1 << self.q.k) + self.splits
            || self.nodes.len() != (1 << self.q.k) + 2 * self.splits
        {
            return None;
        }
        let mut seed = 0u32;
        for &i in &self.axes {
            if i >= self.q.d || seed & (1u32 << i) != 0 {
                return None;
            }
            seed |= 1u32 << i;
        }
        let mut seen = vec![false; self.nodes.len()];
        let mut leaves = vec![false; self.leaves];
        for root in 0..1usize << self.q.k {
            let mut stack = [(0usize, 0u32, 0usize); MAX_D + 1];
            let mut n = 1;
            stack[0] = (root, seed, self.q.k);
            while n != 0 {
                n -= 1;
                let (index, used, depth) = stack[n];
                if index >= self.nodes.len() || seen[index] {
                    return None;
                }
                seen[index] = true;
                let node = self.nodes[index];
                if node.axis == LEAF {
                    let leaf = node.leaf as usize;
                    if leaf >= self.leaves || leaves[leaf] {
                        return None;
                    }
                    leaves[leaf] = true;
                } else {
                    let axis = node.axis as usize;
                    if axis >= self.q.d
                        || used & (1u32 << axis) != 0
                        || depth >= self.q.max_depth
                        || n + 2 > stack.len()
                    {
                        return None;
                    }
                    let used = used | (1u32 << axis);
                    stack[n] = (node.low as usize, used, depth + 1);
                    stack[n + 1] = (node.high as usize, used, depth + 1);
                    n += 2;
                }
            }
        }
        (seen.into_iter().all(|x| x) && leaves.into_iter().all(|x| x)).then_some(())
    }
}

/// A root lower bound can be obtained only by consuming a complete collector.
#[derive(Debug)]
pub(in crate::single_hidden_hull::mip_cuts::signed) struct CheckedRoot {
    q: Quote,
    lower: f64,
    leaves: usize,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Stored in the charged checked-root layout; inspected by conformance tests."
        )
    )]
    splits: usize,
    #[expect(
        dead_code,
        reason = "Retain the charged checked-root telemetry layout."
    )]
    advice_work: usize,
    #[expect(
        dead_code,
        reason = "Retain the charged checked-root telemetry layout."
    )]
    stop: Stop,
}
impl CheckedRoot {
    pub(super) fn matches_quote(&self, q: Quote) -> bool {
        self.q == q
    }
    pub(super) fn lower(&self) -> f64 {
        self.lower
    }
    pub(super) fn leaves(&self) -> usize {
        self.leaves
    }
    #[cfg(test)]
    pub(super) fn splits(&self) -> usize {
        self.splits
    }
}
#[derive(Debug)]
pub(super) struct Collector {
    cover: Cover,
    cursor: Cursor,
    local: Vec<f64>,
    total: Vec<f64>,
    raw_minima: Vec<f64>,
}
impl Collector {
    pub(super) fn new(cover: Cover) -> Option<Self> {
        cover.validate()?;
        let l = cover.leaves;
        let c = cover.q.components;
        Some(Self {
            cover,
            cursor: Cursor::default(),
            local: vec![f64::INFINITY; l],
            total: vec![0.; l],
            raw_minima: vec![f64::INFINITY; c],
        })
    }
    pub(super) fn push(
        &mut self,
        c: usize,
        m: usize,
        coeff: &[I],
        raw: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let r = self.push_inner(c, m, coeff, raw, poll);
        if r.is_none() {
            self.cursor.failed = true;
        }
        r
    }
    fn push_inner(
        &mut self,
        c: usize,
        m: usize,
        coeff: &[I],
        raw: f64,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        self.cursor.accept(&self.cover.bits, c, m)?;
        gate(poll)?;
        finite(raw)?;
        self.raw_minima[c] = self.raw_minima[c].min(raw);
        traverse(&self.cover, coeff, raw, poll, &mut |leaf, value| {
            self.local[leaf] = self.local[leaf].min(value);
            Some(())
        })?;
        if self.cursor.next(&self.cover.bits) {
            for (out, local) in self.total.iter_mut().zip(&mut self.local) {
                *out = finite(add_down(*out, finite(*local)?))?;
                *local = f64::INFINITY;
            }
        }
        gate(poll)
    }
    pub(super) fn finish(
        self,
        constant: f64,
        d: f64,
        g: &[f64],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<CheckedRoot> {
        gate(poll)?;
        if !self.cursor.complete(&self.cover.bits) || g.len() != self.cover.q.d {
            return None;
        }
        let mut base = offset(constant, d, &self.raw_minima)?;
        let mut coeff = [I::zero(); MAX_D];
        for (i, (&v, &[l, u])) in g.iter().zip(&self.cover.bounds).enumerate() {
            coeff[i] = I::point(v)?;
            base = finite(add_down(base, coeff[i].minimum(l, u)?))?;
        }
        let mut lower = f64::INFINITY;
        traverse(
            &self.cover,
            &coeff[..g.len()],
            base,
            poll,
            &mut |leaf, value| {
                lower = lower.min(finite(add_down(value, self.total[leaf]))?);
                Some(())
            },
        )?;
        gate(poll)?;
        Some(CheckedRoot {
            q: self.cover.q,
            lower: finite(lower)?,
            leaves: self.cover.leaves,
            splits: self.cover.splits,
            advice_work: self.cover.advice_spent,
            stop: self.cover.stop,
        })
    }
}
fn traverse(
    cover: &Cover,
    coeff: &[I],
    raw: f64,
    poll: &mut dyn FnMut() -> bool,
    emit: &mut dyn FnMut(usize, f64) -> Option<()>,
) -> Option<()> {
    if coeff.len() != cover.q.d || !coeff.iter().copied().all(valid) {
        return None;
    }
    let mut correction = [[0.; 2]; MAX_D];
    let mut delta = [I::zero(); MAX_D];
    for (i, (&q, &[l, u])) in coeff.iter().zip(&cover.bounds).enumerate() {
        let low = q.scale(l)?;
        let high = q.scale(u)?;
        let minimum = q.minimum(l, u)?;
        correction[i] = [sub(low.lo, minimum)?, sub(high.lo, minimum)?];
        delta[i] = high.sub(low)?;
    }
    let mut value = finite(raw)?;
    for &axis in &cover.axes {
        value = finite(add_down(value, correction[axis][0]))?;
    }
    let mut stack = [(0usize, 0.); MAX_D + 1];
    for step in 0..1usize << cover.q.k {
        if step % 16 == 0 {
            gate(poll)?;
        }
        let root = step ^ (step >> 1);
        if step != 0 {
            let bit = step.trailing_zeros() as usize;
            let axis = cover.axes[bit];
            value = finite(add_down(
                value,
                if root & (1 << bit) != 0 {
                    delta[axis].lo
                } else {
                    -delta[axis].hi
                },
            ))?;
        }
        let mut n = 1;
        stack[0] = (root, value);
        while n != 0 {
            n -= 1;
            let (index, v) = stack[n];
            let node = cover.nodes[index];
            if node.axis == LEAF {
                emit(node.leaf as usize, v)?;
            } else {
                let axis = node.axis as usize;
                stack[n] = (node.low as usize, finite(add_down(v, correction[axis][0]))?);
                stack[n + 1] = (
                    node.high as usize,
                    finite(add_down(v, correction[axis][1]))?,
                );
                n += 2;
            }
        }
    }
    gate(poll)
}

/// The explicit 4096 covers the largest bounded local arrays in any stage:
/// initial 128-value advice scratch, decoded face, tree validation, interval
/// corrections and depth-first proof stack. Heap payload is quoted separately.
fn fixed_scratch_bytes() -> usize {
    size_of::<Capture>()
        + size_of::<Collector>()
        + size_of::<Cover>()
        + size_of::<CheckedRoot>()
        + size_of::<Advice>()
        + 4096
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_adaptive_tests.rs"]
mod tests;
