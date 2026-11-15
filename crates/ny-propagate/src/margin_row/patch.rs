// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bi-patch: PER-COLUMN RECEPTIVE-FIELD ("patch mode") prefix passes for the
//! margin-row `#backward-interm` phase and `#joint-refresh`.
//!
//! WHY. Every trunk-seeded column (a `#backward-interm` / `#interm-alpha` /
//! `#joint-refresh` prefix pair) is an identity row seeded at ONE spatial
//! position of a trunk pre-activation. Its coefficients are exactly zero
//! outside that position's receptive field, yet the dense engine walks every
//! row of every prefix tensor, and `#bi-window` walks a tile rectangle about
//! 2.5x the single-column field. On cct narrow CNN7 the per-column signed
//! product costs (patch vs dense) 0.052 M vs 39.5 M MACs at relu_b, 0.449 M vs
//! 58.4 M at relu_c, 2.76 M vs 96.1 M at relu_d and 8.07 M vs 133.9 M at
//! relu_e; a full-coverage refresh on idx448 is ~19 GMAC/lane against ~700
//! dense and ~62 for the measured crop (design: geometry script `pm_cost.py`).
//!
//! WHAT. For one seeded trunk relu, [`Geometry`] assigns every prefix tensor a
//! [`Frame`]: a per-column rectangle `[Y*sy+oy, +ph) x [X*sx+ox, +pw)` over all
//! channels, in VIRTUAL coordinates (not clamped at the image border). Every
//! column of a pass shares the frame's shape; only its anchor differs. The
//! walk carries each tensor's state as the engine's own [`LaneMat`] over the
//! frame's `c * ph * pw` local rows (channel-major, row-major, the `R` columns
//! contiguous), so:
//!
//! * a conv runs as a VIRTUAL `ConvOp` over the frame (same weights, bias,
//!   fold charges; zero padding; the natural input frame
//!   `sy_in = sy*s, oy_in = oy*s - pad, ph_in = (ph-1)*s + k`, so every tap of
//!   every virtual output lands inside it) through the UNCHANGED conv kernel
//!   and conv route; its `k_bwd` is the FULL op's, so every Higham envelope
//!   is instantiated at the dense size (sound: a real row's nonzero terms are
//!   a subset of the full op's; virtual-padding terms are exact zeros);
//! * relu / channel-affine read their per-neuron gates and parameters at the
//!   ABSOLUTE neuron index of each (local row, column);
//! * rows outside the image (virtual padding) are MASKED: skipped by every
//!   elementwise arm and forced to exact zero (coefficient AND error) whenever
//!   a state is consumed, so they carry nothing into a conv;
//! * a residual `Add` copies to both operands, each EMBEDDED (exact zero
//!   extension at a fixed per-column offset) into its tensor's frame, which is
//!   the UNION of the frames its consumers deliver (equal anchor strides are
//!   required, else the plan is refused); contributions merge with the
//!   unchanged `merge_into` in the dense walk's arrival order;
//! * CROSSOVER: walking backward, the first tensor whose frame exceeds
//!   `ratio * n` (and every tensor below it) switches to a FULL frame — the
//!   whole tensor, anchor `(0, 0)` for every column — where the virtual op is
//!   the real op; the boundary scatter is an exact embed. On a full frame the
//!   walk IS the dense walk (same rows, same order, same expressions);
//! * the input box is concretized per column over the real rows of the input
//!   frame with the unchanged `concretize` expression and the FULL `n_in`.
//!
//! A Lower and an Upper pass over the same columns are FUSED into one walk of
//! `2R` columns (the relu arm picks each column's lane rule); every column's
//! arithmetic is its own direction's pass, column-local, so the fused pair is
//! bit-identical to two passes.
//!
//! SOUNDNESS (the patch pass is the dense pass RESTRICTED TO ITS SUPPORT).
//! * Support lemma: a coefficient seeded at `(c, Y, X)` of the seed tensor can
//!   be nonzero on tensor `t` only at REAL positions inside `Frame_t(Y, X)`
//!   (induction over the reverse walk: conv receptive field, elementwise
//!   relu/affine, Add union). The exact-real recursion applied to the stored
//!   upstream values is zero there as well, so an exact zero with error 0 is a
//!   valid lane state outside the frame.
//! * Coefficients: every kept coefficient and bias is the dense pass's f64
//!   value up to the sign of zero. The virtual conv's back taps at a real input
//!   row are the dense taps restricted to the output frame, in the same
//!   `(ky, kx)` then `oc` order; dropped taps carry exact `±0` terms. Local row
//!   order preserves absolute order, so every per-column accumulation (bias,
//!   concretize) adds the same terms in the same order, minus exact zeros.
//! * Error lane: `E_patch <= E_dense` entrywise and `eb_patch <= eb_dense`.
//!   Every error update (`slack16`, `certify_up`, `combined_err`, `|W|^T comb`,
//!   the merge) is monotone in its inputs, the gamma sizes are the DENSE sizes,
//!   and the patch's bias charges (including the running-accumulator
//!   `2u|b|` carry, charged on real frame rows only) are an ordered SUBSET of
//!   the dense charges at the same running values; fl-addition of nonnegative
//!   terms is monotone. Hence `lb_patch >= lb_dense` and `ub_patch <= ub_dense`
//!   EXACTLY, and the patch pass is also a certified pass in its own right:
//!   every charge covers every rounding it actually performs.
//! * Pinned by `patch_tests` (child of `tests::gpu_conv_tests`): coefficient
//!   and error-lane dominance against the dense pass, never-looser bounds,
//!   exact-rational containment of the true pre-activations at sampled box
//!   points (point boxes included), frame coverage of the dense support, the
//!   crossover, the fused pair, the refusals and the cost model.
//!
//! FAIL-CLOSED. Anything the planner cannot express — a transposed conv, a
//! Flatten/Gemm inside the prefix, an unpinned shape, a union of unequal
//! anchor strides, a seed whose own tensor converts (`no_gain`), a virtual
//! conv with a tap outside its input frame (`tripwire`) — refuses the plan and
//! the caller runs the crop or the dense pass. At walk time any structural
//! disagreement (a beta / per-column beta / `infeasible` gate override, a
//! gate-length mismatch, an embed whose real row would leave its target frame)
//! returns `None` (the same fallback); only the non-finite firewall returns
//! `Some(Err)`, which the phase treats exactly like a dense refusal. There is
//! no verdict surface: every output is an intermediate box the unchanged
//! shrink-only intersection, `gates_from_box` and `repair_upper_lines`
//! consume.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use super::support_program::{PreparedPrefix, PreparedPrefixFrame};
use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkRequest};
use ndarray::Array2;
use ny_core::{NyError, Result};
use std::time::Instant;

use super::engine::{
    prepared_lower_bound_values, CrossingLowerPolicy, DomainGates, FrozenGateFrame, LaneMat,
    ZeroCrossingLowerFrame,
};
use super::gpu_conv::{ConvLane, GpuConvRoute};
use super::input_rows::{InputRowPair, InputRowScope, OriginalRowDemand};
use super::net::{
    compile_conv, nominal_product_dust, underflow_product_up, ConvOp, TwinNet, TwinOp,
};
use super::root::RootGates;
use super::rounding::{certify_up, gamma_n, next_down, next_up, slack16, UNIT};
use super::spec::{TwinOpSpec, TwinSpec};

#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
const INPUT_PRODUCT_BYTES_CAP: usize = super::spec::MAX_ROOT_TABLEAU_ENTRIES * 16;

macro_rules! patch_try {
    ($value:expr) => {
        match $value {
            Ok(value) => value,
            Err(error) => return Some(Err(error)),
        }
    };
}

/// `(C, H, W)` of one tensor.
pub(crate) type Shape = (usize, usize, usize);

/// Why a plan was refused (the `refused=` field of the plan line).
pub(crate) type Refusal = &'static str;

/// Declaration default of `margin_row.bi_patch_ratio`: convert to the dense
/// tail once a frame exceeds half of its tensor (padding waste plus gather
/// overhead make a larger patch a loss).
pub(crate) const DEFAULT_RATIO: f64 = 0.5;
/// Closed admission range of the crossover ratio (the reader clamps to it).
pub(crate) const RATIO_MIN: f64 = 0.05;
/// See [`RATIO_MIN`].
pub(crate) const RATIO_MAX: f64 = 1.0;
/// Declaration default of `margin_row.bi_patch_mb`: the per-pass working set
/// (MiB) that caps a patch chunk's width.
pub(crate) const DEFAULT_MB: usize = 256;

/// Frozen patch-mode configuration of one phase (or one joint refresh).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PatchCfg {
    /// Crossover ratio `rho`, clamped to `[RATIO_MIN, RATIO_MAX]`.
    pub(crate) ratio: f64,
    /// Working-set cap per pass, in MiB (at least 1).
    pub(crate) mb: usize,
}

impl PatchCfg {
    /// Clamp a resolved `(ratio, mb)` pair into the admissible range. A
    /// non-finite ratio (unreachable through the typed reader) takes the
    /// declaration default.
    pub(crate) fn new(ratio: f64, mb: usize) -> Self {
        let ratio = if ratio.is_finite() {
            ratio.clamp(RATIO_MIN, RATIO_MAX)
        } else {
            DEFAULT_RATIO
        };
        Self {
            ratio,
            mb: mb.max(1),
        }
    }
}

/// Geometry-only view of one op, shared by the compiled-net plan
/// ([`PatchPlan::build`]) and the spec-level cost proxy (`auto_alloc`), so
/// the two can never price different frames.
///
/// `Conv` is 137 bytes against the next variant's 40, but this is a `Copy`
/// value view: boxing its geometry (the lint's remedy) would drop `Copy` and
/// allocate on every copy of a plan's per-op view, which is the opposite of
/// what clippy::large_enum_variant guards against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "Copy geometry view: a Box cannot be Copy, and boxing would allocate per copy"
)]
pub(crate) enum GeoOp {
    /// Convolution (`transposed`: a compiled `ConvTranspose`, refused).
    Conv {
        input: usize,
        kernel: (usize, usize, usize, usize),
        stride: (usize, usize),
        pads: (usize, usize, usize, usize),
        ishape: Shape,
        oshape: Shape,
        transposed: bool,
    },
    /// ReLU (elementwise, shape-preserving).
    Relu { input: usize },
    /// Per-channel affine (elementwise; the spec carries its shape).
    Affine { input: usize, shape: Option<Shape> },
    /// Residual add.
    Add { lhs: usize, rhs: usize },
    /// Flatten / Gemm: never inside a trunk prefix.
    Other,
}

/// The geometry of a compiled net.
pub(crate) fn geo_of_net(net: &TwinNet) -> Vec<GeoOp> {
    net.ops
        .iter()
        .map(|op| match op {
            TwinOp::Conv(cv) => GeoOp::Conv {
                input: cv.input,
                kernel: cv.kernel,
                stride: cv.stride,
                pads: cv.pads,
                ishape: cv.ishape,
                oshape: cv.oshape,
                transposed: cv.transposed,
            },
            TwinOp::Relu { input, .. } => GeoOp::Relu { input: *input },
            TwinOp::ChannelAffine { input, .. } => GeoOp::Affine {
                input: *input,
                shape: None,
            },
            TwinOp::Add { lhs, rhs } => GeoOp::Add {
                lhs: *lhs,
                rhs: *rhs,
            },
            TwinOp::Flatten { .. } | TwinOp::Gemm { .. } => GeoOp::Other,
        })
        .collect()
}

/// The geometry of an uncompiled spec (the `#auto-alloc` proxy's input).
pub(crate) fn geo_of_spec(spec: &TwinSpec) -> Vec<GeoOp> {
    spec.ops
        .iter()
        .map(|op| match op {
            TwinOpSpec::Conv {
                input,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => GeoOp::Conv {
                input: *input,
                kernel: *kernel,
                stride: *stride,
                pads: *pads,
                ishape: *ishape,
                oshape: *oshape,
                transposed: false,
            },
            TwinOpSpec::ConvTranspose {
                input,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => GeoOp::Conv {
                input: *input,
                kernel: *kernel,
                stride: *stride,
                pads: *pads,
                ishape: *ishape,
                oshape: *oshape,
                transposed: true,
            },
            TwinOpSpec::Relu { input } => GeoOp::Relu { input: *input },
            TwinOpSpec::ChannelAffine { input, shape, .. } => GeoOp::Affine {
                input: *input,
                shape: Some(*shape),
            },
            TwinOpSpec::Add { lhs, rhs } => GeoOp::Add {
                lhs: *lhs,
                rhs: *rhs,
            },
            TwinOpSpec::Flatten { .. } | TwinOpSpec::Gemm { .. } => GeoOp::Other,
        })
        .collect()
}

/// One tensor's per-column patch.
///
/// Column `r`, seeded at spatial position `(Y, X)` of the seed tensor, owns
/// the rectangle `[Y*sy + oy, +ph) x [X*sx + ox, +pw)` over all `c` channels.
/// A FULL frame is the whole tensor for every column (`sy = sx = 0`,
/// `oy = ox = 0`, `ph = h`, `pw = w`): the dense tail after the crossover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Frame {
    /// Tensor channels.
    pub(crate) c: usize,
    /// Tensor height.
    pub(crate) h: usize,
    /// Tensor width.
    pub(crate) w: usize,
    /// Anchor stride (rows / cols of this tensor per seed position).
    pub(crate) sy: isize,
    pub(crate) sx: isize,
    /// Anchor offset (virtual coordinates; may be negative).
    pub(crate) oy: isize,
    pub(crate) ox: isize,
    /// Patch extent.
    pub(crate) ph: usize,
    pub(crate) pw: usize,
    /// The whole tensor, shared by every column.
    pub(crate) full: bool,
}

impl Frame {
    fn seed((c, h, w): Shape) -> Self {
        Self {
            c,
            h,
            w,
            sy: 1,
            sx: 1,
            oy: 0,
            ox: 0,
            ph: 1,
            pw: 1,
            full: false,
        }
    }

    fn full((c, h, w): Shape) -> Self {
        Self {
            c,
            h,
            w,
            sy: 0,
            sx: 0,
            oy: 0,
            ox: 0,
            ph: h,
            pw: w,
            full: true,
        }
    }

    /// Local rows per column (`c * ph * pw`).
    pub(crate) fn rows(&self) -> usize {
        self.c * self.ph * self.pw
    }

    fn plane(&self) -> usize {
        self.ph * self.pw
    }

    fn shape(&self) -> Shape {
        (self.c, self.h, self.w)
    }

    /// Absolute top-left (virtual) of the column seeded at `(y, x)`.
    fn anchor(&self, (y, x): (usize, usize)) -> (isize, isize) {
        (
            y as isize * self.sy + self.oy,
            x as isize * self.sx + self.ox,
        )
    }

    /// The smallest frame containing both (same tensor, equal anchor
    /// strides), or `None` when the strides disagree.
    fn union(self, other: Self) -> Option<Self> {
        if self.shape() != other.shape() {
            return None;
        }
        if self.full || other.full {
            return Some(Self::full(self.shape()));
        }
        if self.sy != other.sy || self.sx != other.sx {
            return None;
        }
        let oy = self.oy.min(other.oy);
        let ox = self.ox.min(other.ox);
        let ey = (self.oy + self.ph as isize).max(other.oy + other.ph as isize);
        let ex = (self.ox + self.pw as isize).max(other.ox + other.pw as isize);
        Some(Self {
            oy,
            ox,
            ph: (ey - oy) as usize,
            pw: (ex - ox) as usize,
            ..self
        })
    }
}

/// One column's view of one frame: its anchor and the local range of REAL
/// (in-image) rows `[y0, y1) x [x0, x1)`.
#[derive(Debug, Clone, Copy)]
struct ColGeo {
    ay: isize,
    ax: isize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    interior: bool,
}

fn col_geo(f: &Frame, pos: (usize, usize)) -> ColGeo {
    let (ay, ax) = f.anchor(pos);
    let range = |a: isize, p: usize, n: usize| -> (usize, usize) {
        let p = p as isize;
        let lo = (-a).clamp(0, p);
        let hi = (n as isize - a).clamp(0, p);
        (lo as usize, hi.max(lo) as usize)
    };
    let (y0, y1) = range(ay, f.ph, f.h);
    let (x0, x1) = range(ax, f.pw, f.w);
    ColGeo {
        ay,
        ax,
        y0,
        y1,
        x0,
        x1,
        interior: y0 == 0 && y1 == f.ph && x0 == 0 && x1 == f.pw,
    }
}

fn col_geos(f: &Frame, pos: &[(usize, usize)]) -> Vec<ColGeo> {
    pos.iter().map(|&xy| col_geo(f, xy)).collect()
}

/// Absolute flat index of local row `(ch, iy, ix)` for a column, or `None`
/// when the row is virtual padding (outside the image) for that column.
#[inline]
fn abs_index(f: &Frame, g: &ColGeo, ch: usize, iy: usize, ix: usize) -> Option<usize> {
    if iy < g.y0 || iy >= g.y1 || ix < g.x0 || ix >= g.x1 {
        return None;
    }
    let y = (g.ay + iy as isize) as usize;
    let x = (g.ax + ix as isize) as usize;
    Some(ch * f.h * f.w + y * f.w + x)
}

/// `(ch, iy, ix)` of a local row.
#[inline]
fn split_row(f: &Frame, row: usize) -> (usize, usize, usize) {
    let plane = f.plane();
    let rem = row % plane;
    (row / plane, rem / f.pw, rem % f.pw)
}

fn set_shape(shapes: &mut [Option<Shape>], t: usize, s: Shape) -> bool {
    match shapes.get(t).copied() {
        Some(None) => {
            shapes[t] = Some(s);
            true
        }
        Some(Some(prev)) => prev == s,
        None => false,
    }
}

/// `(C, H, W)` of every tensor `0..=upto` that the conv geometry (and a
/// spec-level channel affine) pins, carried through the shape-preserving
/// relu / affine / add ops in both directions.
fn pin_shapes(
    ops: &[GeoOp],
    tsize: &[usize],
    upto: usize,
) -> std::result::Result<Vec<Option<Shape>>, Refusal> {
    let tensors = upto + 1;
    if ops.len() < upto || tsize.len() < tensors {
        return Err("shape");
    }
    let mut shapes: Vec<Option<Shape>> = vec![None; tensors];
    for (k, op) in ops[..upto].iter().enumerate() {
        let pinned = match *op {
            GeoOp::Conv {
                input,
                ishape,
                oshape,
                ..
            } => set_shape(&mut shapes, input, ishape) && set_shape(&mut shapes, k + 1, oshape),
            GeoOp::Affine {
                input,
                shape: Some(shape),
            } => set_shape(&mut shapes, input, shape) && set_shape(&mut shapes, k + 1, shape),
            _ => true,
        };
        if !pinned {
            return Err("shape");
        }
    }
    for _ in 0..tensors {
        let mut changed = false;
        for (k, op) in ops[..upto].iter().enumerate() {
            let group: [Option<usize>; 3] = match *op {
                GeoOp::Relu { input } | GeoOp::Affine { input, .. } => {
                    [Some(k + 1), Some(input), None]
                }
                GeoOp::Add { lhs, rhs } => [Some(k + 1), Some(lhs), Some(rhs)],
                _ => continue,
            };
            let mut known: Option<Shape> = None;
            for t in group.into_iter().flatten() {
                let Some(slot) = shapes.get(t).copied() else {
                    return Err("shape");
                };
                match (known, slot) {
                    (None, Some(s)) => known = Some(s),
                    (Some(a), Some(b)) if a != b => return Err("shape"),
                    _ => {}
                }
            }
            if let Some(s) = known {
                for t in group.into_iter().flatten() {
                    if shapes[t].is_none() {
                        shapes[t] = Some(s);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (t, s) in shapes.iter().enumerate() {
        if let Some((c, h, w)) = *s {
            if c.checked_mul(h).and_then(|v| v.checked_mul(w)) != Some(tsize[t]) {
                return Err("shape");
            }
        }
    }
    Ok(shapes)
}

fn union_into(
    frames: &mut [Option<Frame>],
    t: usize,
    f: Frame,
) -> std::result::Result<(), Refusal> {
    let slot = frames.get_mut(t).ok_or("shape")?;
    *slot = Some(match *slot {
        None => f,
        Some(g) => g.union(f).ok_or("stride_union")?,
    });
    Ok(())
}

/// The frames, crossover and cost of one seeded trunk relu's prefix walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Geometry {
    /// The seeded trunk relu op.
    pub(crate) relu_op: usize,
    /// Its input tensor (the seed rows live there).
    pub(crate) seed_tensor: usize,
    /// Per tensor id `0..=relu_op`: its frame when the walk visits it.
    pub(crate) frames: Vec<Option<Frame>>,
    /// Per op `0..relu_op`: the frame its contribution(s) are produced in
    /// (a conv: its natural input frame; relu / affine / add: the output
    /// frame), delivered into the target tensor's frame by an exact embed.
    pub(crate) contrib: Vec<Option<Frame>>,
    /// Tensor id at which the walk switched to full frames (every tensor at
    /// or below it is full); `None`: the whole walk is patches.
    pub(crate) convert_at: Option<usize>,
    /// Signed-product conv MACs per column of the patch walk, counted as the
    /// virtual kernel performs them (a virtual op has no border taps, so this
    /// is `rows_out * ci * kh * kw` exactly; a full-frame op is priced at its
    /// nominal `n_out * ci * kh * kw`).
    pub(crate) macs_col_patch: u64,
    /// Nominal `n_out * ci * kh * kw` per column of the dense walk (the
    /// `#auto-alloc` dense proxy's unit).
    pub(crate) macs_col_dense: u64,
    /// Rows per column the patch walk visits, and the dense walk.
    pub(crate) rows_col_patch: usize,
    pub(crate) rows_col_dense: usize,
    /// Largest frame (tensor or contribution) of the walk, in rows.
    pub(crate) virt_rows_max: usize,
}

impl Geometry {
    /// Plan the frames of trunk relu op `relu_op`'s prefix walk.
    ///
    /// `ratio` is the crossover `rho`: walking backward, the first tensor
    /// whose frame holds more than `ratio * n` rows (and every tensor below
    /// it) runs full. A non-finite or huge ratio never converts (tests use it
    /// to pin the all-patch walk).
    pub(crate) fn build(
        ops: &[GeoOp],
        tsize: &[usize],
        relu_op: usize,
        ratio: f64,
    ) -> std::result::Result<Self, Refusal> {
        let Some(GeoOp::Relu { input: seed_tensor }) = ops.get(relu_op).copied() else {
            return Err("not_relu");
        };
        if seed_tensor > relu_op || tsize.len() <= relu_op {
            return Err("not_relu");
        }
        let shapes = pin_shapes(ops, tsize, relu_op)?;
        let shape_of = |t: usize| -> std::result::Result<Shape, Refusal> {
            shapes.get(t).copied().flatten().ok_or("shape")
        };
        let too_big = |f: &Frame, n: usize| !f.full && (f.rows() as f64) > ratio * (n as f64);
        let mut frames: Vec<Option<Frame>> = vec![None; relu_op + 1];
        let mut contrib: Vec<Option<Frame>> = vec![None; relu_op];
        frames[seed_tensor] = Some(Frame::seed(shape_of(seed_tensor)?));
        let mut convert_at: Option<usize> = None;
        let (mut macs_patch, mut macs_dense) = (0u64, 0u64);
        for k in (0..relu_op).rev() {
            let t_out = k + 1;
            let Some(mut fout) = frames[t_out] else {
                continue;
            };
            if convert_at.is_none() && too_big(&fout, tsize[t_out]) {
                convert_at = Some(t_out);
            }
            if convert_at.is_some() && !fout.full {
                fout = Frame::full(fout.shape());
                frames[t_out] = Some(fout);
            }
            match ops[k] {
                GeoOp::Conv {
                    transposed: true, ..
                } => return Err("transposed"),
                GeoOp::Conv {
                    input,
                    kernel: (co, ci, kh, kw),
                    stride: (s_h, s_w),
                    pads,
                    ishape,
                    oshape,
                    ..
                } => {
                    if input >= t_out
                        || oshape != fout.shape()
                        || shape_of(input)? != ishape
                        || ci != ishape.0
                        || co != oshape.0
                        || s_h == 0
                        || s_w == 0
                    {
                        return Err("shape");
                    }
                    let fin = if fout.full {
                        Frame::full(ishape)
                    } else {
                        Frame {
                            c: ishape.0,
                            h: ishape.1,
                            w: ishape.2,
                            sy: fout.sy * s_h as isize,
                            sx: fout.sx * s_w as isize,
                            oy: fout.oy * s_h as isize - pads.0 as isize,
                            ox: fout.ox * s_w as isize - pads.1 as isize,
                            ph: (fout.ph - 1) * s_h + kh,
                            pw: (fout.pw - 1) * s_w + kw,
                            full: false,
                        }
                    };
                    let per_row = (ci * kh * kw) as u64;
                    macs_patch += fout.rows() as u64 * per_row;
                    macs_dense += tsize[t_out] as u64 * per_row;
                    contrib[k] = Some(fin);
                    union_into(&mut frames, input, fin)?;
                }
                GeoOp::Relu { input } | GeoOp::Affine { input, .. } => {
                    if input >= t_out || shape_of(input)? != fout.shape() {
                        return Err("shape");
                    }
                    contrib[k] = Some(fout);
                    union_into(&mut frames, input, fout)?;
                }
                GeoOp::Add { lhs, rhs } => {
                    for t in [lhs, rhs] {
                        if t >= t_out || shape_of(t)? != fout.shape() {
                            return Err("shape");
                        }
                    }
                    contrib[k] = Some(fout);
                    union_into(&mut frames, lhs, fout)?;
                    union_into(&mut frames, rhs, fout)?;
                }
                GeoOp::Other => return Err("not_trunk"),
            }
        }
        let Some(mut f0) = frames[0] else {
            return Err("no_input");
        };
        if convert_at.is_none() && too_big(&f0, tsize[0]) {
            convert_at = Some(0);
        }
        if convert_at.is_some() && !f0.full {
            f0 = Frame::full(f0.shape());
            frames[0] = Some(f0);
        }
        // The conversion fell on the FIRST op (the seed tensor itself runs
        // full): the walk is the dense walk plus gathers — no gain.
        if frames[seed_tensor].is_some_and(|f| f.full) {
            return Err("no_gain");
        }
        let (mut rows_patch, mut rows_dense, mut virt_max) = (0usize, 0usize, 0usize);
        for (t, f) in frames.iter().enumerate() {
            if let Some(f) = f {
                rows_patch += f.rows();
                rows_dense += tsize[t];
                virt_max = virt_max.max(f.rows());
            }
        }
        for f in contrib.iter().flatten() {
            virt_max = virt_max.max(f.rows());
        }
        Ok(Self {
            relu_op,
            seed_tensor,
            frames,
            contrib,
            convert_at,
            macs_col_patch: macs_patch,
            macs_col_dense: macs_dense,
            rows_col_patch: rows_patch,
            rows_col_dense: rows_dense,
            virt_rows_max: virt_max,
        })
    }

    /// Tensors the walk visits.
    pub(crate) fn visited(&self) -> usize {
        self.frames.iter().flatten().count()
    }
}

/// The compiled patch plan of one seeded trunk relu: its [`Geometry`] and one
/// virtual conv per patch-output conv op. Built once per layer, shared by
/// every chunk and every `#interm-alpha` candidate of that layer.
pub(crate) struct PatchPlan<'net> {
    /// Cached virtual weights/tables can only run against this immutable net.
    source_net: &'net TwinNet,
    /// Diagnostic callers can inspect/mutate `geo`; product authority retains
    /// the exact original geometry independently and refuses such mutations.
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    product_geometry: Geometry,
    pub(crate) geo: Geometry,
    /// Per op `0..relu_op`: the virtual conv over the op's frames (`None`: not
    /// a conv, or a full-frame conv that runs the real op).
    vops: Vec<Option<ConvOp>>,
    /// Bytes per pass column (coefficient + error lane) of the live states
    /// and conv temporaries: the chunk-width governor's unit.
    bytes_per_col: usize,
}

/// Created only by the complete Outward scoped product entry. Diagnostic
/// PatchWalk arrays alone cannot be promoted into pointwise proof products.
pub(super) struct PatchInputRowsAuthority(());

/// One completed walk supplies both scalar bounds and pointwise row products.
#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
pub(super) struct PatchProducts<'a> {
    pub(super) bounds: PatchPair,
    pub(super) rows: InputRowPair<'a>,
}

/// One fused Lower+Upper patch pair, concretized over the root box.
pub(crate) struct PatchPair {
    /// Certified lower bound per requested column (request order).
    pub(crate) lb: Vec<f64>,
    /// Certified upper bound per requested column (request order).
    pub(crate) ub: Vec<f64>,
    /// Rows per column the patch walk visits, and the dense walk.
    pub(crate) rows_patch: usize,
    pub(crate) rows_dense: usize,
    /// Signed conv MACs of the pair (both lanes), patch and dense.
    pub(crate) macs_patch: f64,
    pub(crate) macs_dense: f64,
    /// The walk switched to full frames somewhere.
    pub(crate) converted: bool,
    /// Pass columns of the fused walk (`2 x requested`).
    pub(crate) fused_cols: usize,
}

/// The input-tensor state of one patch walk (the test surface of
/// [`PatchPlan::walk`]; [`PatchPlan::concretize`] consumes it).
pub(crate) struct PatchWalk {
    /// Frame of the input tensor.
    pub(crate) frame: Frame,
    /// Seed position `(Y, X)` per pass column.
    pub(crate) pos: Vec<(usize, usize)>,
    /// Coefficients over the input frame's local rows (`rows x P`).
    pub(crate) a: Array2<f64>,
    /// Certified coefficient error (outward mode only).
    pub(crate) e: Option<Array2<f64>>,
    /// Bias per pass column.
    pub(crate) b: Vec<f64>,
    /// Certified bias error per pass column.
    pub(crate) eb: Vec<f64>,
}

/// Per-neuron gate vectors of one prefix relu (root or override).
#[derive(Clone, Copy)]
struct Gates<'g> {
    alpha: &'g [f64],
    s: &'g [f64],
    c: &'g [f64],
    ms: &'g [f64],
    lower_policy: CrossingLowerPolicy,
    original: &'g super::root::LayerGates,
}

impl<'net> PatchPlan<'net> {
    /// Plan trunk relu op `relu_op`'s prefix walk on `net` (see
    /// [`Geometry::build`] for `ratio`). `Err` names the refusal.
    pub(crate) fn build(
        net: &'net TwinNet,
        relu_op: usize,
        ratio: f64,
    ) -> std::result::Result<Self, Refusal> {
        if relu_op >= net.i_gemm1 || relu_op >= net.ops.len() {
            return Err("not_trunk");
        }
        if !matches!(net.ops[relu_op], TwinOp::Relu { .. }) {
            return Err("not_relu");
        }
        let geo = Geometry::build(&geo_of_net(net), &net.tsize, relu_op, ratio)?;
        let mut vops: Vec<Option<ConvOp>> = Vec::with_capacity(relu_op);
        for k in 0..relu_op {
            let vop = match (&net.ops[k], geo.contrib[k], geo.frames[k + 1]) {
                (TwinOp::Conv(cv), Some(fin), Some(fout)) if !fout.full => {
                    let (co, ci, _, _) = cv.kernel;
                    let mut vop = compile_conv(
                        cv.input,
                        &cv.wmat,
                        &cv.bias,
                        &cv.bias_err,
                        cv.weight_rel_err,
                        cv.kernel,
                        cv.stride,
                        (0, 0, 0, 0),
                        (ci, fin.ph, fin.pw),
                        (co, fout.ph, fout.pw),
                    );
                    // TRIPWIRE: every tap of every virtual output must land
                    // inside the natural input frame (it does by construction;
                    // a silent truncation here would be unsound).
                    if vop.gather.contains(&usize::MAX) {
                        return Err("tripwire");
                    }
                    // Every envelope at the DENSE accumulation length. This
                    // is SOUND, not just convenient: at a REAL input row the
                    // virtual taps are the full op's taps restricted to the
                    // output frame plus taps from VIRTUAL-PADDING outputs,
                    // whose coefficient and error are forced to exact zero
                    // before the product (the mask), and an exact-zero term
                    // adds no rounding (`0 * w = ±0`, `fl(s + 0) = s`). So a
                    // real row's nonzero-term count never exceeds the full
                    // op's, whatever the virtual table's own maximum (which
                    // can be larger when a frame extends past a small
                    // tensor). Virtual input rows are masked and discarded.
                    vop.k_bwd = cv.k_bwd;
                    Some(vop)
                }
                _ => None,
            };
            vops.push(vop);
        }
        let live_rows = geo.rows_col_patch + 3 * geo.virt_rows_max;
        let bytes_per_col = 16usize.saturating_mul(live_rows).max(1);
        Ok(Self {
            source_net: net,
            product_geometry: geo.clone(),
            geo,
            vops,
            bytes_per_col,
        })
    }

    /// Requested columns per chunk that keep one FUSED pair's working set
    /// (two pass columns per requested column) inside `mb` MiB; at least 1.
    pub(crate) fn chunk_cap(&self, mb: usize) -> usize {
        let budget = mb.max(1).saturating_mul(1 << 20);
        (budget / self.bytes_per_col.saturating_mul(2).max(1)).max(1)
    }

    /// Working-set bytes of ONE requested column of a fused pair (two pass
    /// columns), the unit [`Self::chunk_cap`] divides by. `#root-box-source`
    /// sizes its concurrent chunks with it (scheduling only: every column's
    /// arithmetic is column-local, so no width changes a published bit).
    pub(crate) fn bytes_per_fused_col(&self) -> usize {
        self.bytes_per_col.saturating_mul(2).max(1)
    }

    /// The per-layer `[bi-patch] plan` observable (fixed schema).
    pub(crate) fn plan_line(&self, layer: usize, cols: usize, rho: f64) -> String {
        let geo = &self.geo;
        format!(
            "[bi-patch] plan layer={layer} op={} cols={cols} frames={} virt_rows_max={} \
macs_col_patch={} macs_col_dense={} ratio={:.1} rho={rho:.3} convert_at={} refused=-",
            geo.relu_op,
            geo.visited(),
            geo.virt_rows_max,
            geo.macs_col_patch,
            geo.macs_col_dense,
            if geo.macs_col_patch > 0 {
                geo.macs_col_dense as f64 / geo.macs_col_patch as f64
            } else {
                0.0
            },
            geo.convert_at
                .map_or_else(|| "none".to_string(), |t| format!("t{t}")),
        )
    }

    /// The refusal form of the plan line (same schema).
    pub(crate) fn refused_line(
        layer: usize,
        relu_op: usize,
        cols: usize,
        rho: f64,
        why: Refusal,
    ) -> String {
        format!(
            "[bi-patch] plan layer={layer} op={relu_op} cols={cols} frames=0 virt_rows_max=0 \
macs_col_patch=0 macs_col_dense=0 ratio=0.0 rho={rho:.3} convert_at=none refused={why}"
        )
    }

    /// One certified FUSED Lower+Upper pair for `cols` (flat indices on the
    /// seeded relu's input) under `root`'s prefix gates, optionally overridden
    /// by `dom` (`#interm-alpha` candidates).
    ///
    /// `None`: the plan cannot express this request (the caller runs the crop
    /// or the dense pair). `Some(Err)`: the non-finite firewall refused the
    /// walk (the caller fails open exactly as for a dense refusal).
    ///
    /// `route` (#bi-gpu): the conv backward route of the walk's virtual ops;
    /// `None` is `BackwardEngine::new`'s own.
    pub(crate) fn prefix_bounds(
        &self,
        net: &TwinNet,
        root: &RootGates,
        cols: &[usize],
        dom: Option<&DomainGates>,
        route: Option<&GpuConvRoute>,
    ) -> Option<Result<PatchPair>> {
        let r = cols.len();
        if r == 0 {
            return None;
        }
        let pass_cols: Vec<usize> = cols.iter().chain(cols.iter()).copied().collect();
        let lowers: Vec<bool> = (0..2 * r).map(|i| i < r).collect();
        let walk = match self.walk(net, root, &pass_cols, &lowers, dom, route)? {
            Ok(walk) => walk,
            Err(error) => return Some(Err(error)),
        };
        let mut bounds = Self::concretize(root, &walk, &lowers);
        let ub = bounds.split_off(r);
        let fused = 2 * r;
        Some(Ok(PatchPair {
            lb: bounds,
            ub,
            rows_patch: self.geo.rows_col_patch,
            rows_dense: self.geo.rows_col_dense,
            macs_patch: self.geo.macs_col_patch as f64 * fused as f64,
            macs_dense: self.geo.macs_col_dense as f64 * fused as f64,
            converted: self.geo.convert_at.is_some(),
            fused_cols: fused,
        }))
    }

    /// Inherited-deadline CPU product entry. `resident_bytes` includes every
    /// already-live model/root/plan allocation; the working-set reservation also
    /// includes both lanes, source/destination conversion, embeddings, residual
    /// copies, metadata and parallel kernel scratch. No full input tableau or
    /// diagnostic/GPU row is promoted. Default source admission stays unchanged.
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn prefix_products<'a>(
        &self,
        net: &'a TwinNet,
        root: &'a RootGates,
        cols: &'a [usize],
        deadline: Instant,
        resident_bytes: usize,
        budget_bytes: usize,
    ) -> Option<Result<PatchProducts<'a>>> {
        patch_try!(super::net::check_deadline(deadline));
        if !std::ptr::eq(self.source_net, net)
            || !root.mode.outward()
            || self.geo != self.product_geometry
        {
            return None;
        }
        let run = || -> Result<PatchProducts<'a>> {
            let deadline = root
                .head_group_source
                .as_ref()
                .ok_or_else(|| product_bad("missing original unsplit builder authority"))?
                .input_rows_deadline(deadline)?;
            let p = cols
                .len()
                .checked_mul(2)
                .ok_or_else(|| product_bad("column overflow"))?;
            if p == 0 || p > 4096 || self.geo.relu_op > 4096 {
                return Err(product_bad("product column/op governor exceeded"));
            }
            let floor = self.product_resident_floor(root, deadline)?;
            let required = self.product_required_bytes(resident_bytes.max(floor), p)?;
            if budget_bytes > INPUT_PRODUCT_BYTES_CAP
                || required > budget_bytes
                || resident_bytes < floor
            {
                return Err(NyError::CpuMemoryExceeded {
                    required_bytes: required,
                    budget_bytes: budget_bytes.min(INPUT_PRODUCT_BYTES_CAP),
                    site: "margin-row patch input products total live",
                });
            }
            let scope = InputRowScope::new(net, root, self.geo.relu_op, cols, deadline)?;
            let deadline = deadline.min(scope.deadline());
            for (i, &n) in net.tsize[..=self.geo.relu_op].iter().enumerate() {
                poll_optional_deadline(Some(deadline), i)?;
                if n.checked_mul(8)
                    .and_then(|n| n.checked_add(16))
                    .is_none_or(|n| gamma_n(n) > 0.25)
                {
                    return Err(product_bad("proof accumulation-length governor exceeded"));
                }
            }
            for qualified in rayon::broadcast(|_| {
                super::net::check_deadline(deadline)?;
                ny_core::require_f64_interval_proof_environment()
            }) {
                qualified?;
            }
            let pass_cols: Vec<usize> = cols.iter().chain(cols.iter()).copied().collect();
            let lowers: Vec<bool> = (0..p).map(|i| i < cols.len()).collect();
            let walk = self
                .walk_inner(
                    net,
                    root,
                    &pass_cols,
                    &lowers,
                    None,
                    &GpuConvRoute::Off,
                    Some(deadline),
                )
                .ok_or_else(|| product_bad("patch structural refusal"))??;
            let mut lb = Self::concretize_checked(root, &walk, &lowers, Some(deadline), true)?;
            let ub = lb.split_off(cols.len());
            let seed_frame = self.geo.frames[self.geo.seed_tensor]
                .ok_or_else(|| product_bad("missing original seed frame"))?;
            let rows = InputRowPair::from_patch(
                scope,
                walk,
                seed_frame.shape(),
                PatchInputRowsAuthority(()),
                deadline,
            )?;
            super::net::check_deadline(deadline)?;
            Ok(PatchProducts {
                bounds: PatchPair {
                    lb,
                    ub,
                    rows_patch: self.geo.rows_col_patch,
                    rows_dense: self.geo.rows_col_dense,
                    macs_patch: self.geo.macs_col_patch as f64 * p as f64,
                    macs_dense: self.geo.macs_col_dense as f64 * p as f64,
                    converted: self.geo.convert_at.is_some(),
                    fused_cols: p,
                },
                rows,
            })
        };
        Some(run())
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn product_required_bytes(&self, resident_bytes: usize, p: usize) -> Result<usize> {
        let rows = self
            .geo
            .rows_col_patch
            .checked_add(
                self.geo
                    .virt_rows_max
                    .checked_mul(4)
                    .ok_or_else(|| product_bad("working-set overflow"))?,
            )
            .ok_or_else(|| product_bad("working-set overflow"))?;
        let matrices = rows
            .checked_mul(p)
            .and_then(|n| n.checked_mul(16))
            .ok_or_else(|| product_bad("working-set overflow"))?;
        let max_ci = self
            .vops
            .iter()
            .flatten()
            .map(|v| v.kernel.1)
            .chain(
                self.source_net.ops[..self.geo.relu_op]
                    .iter()
                    .filter_map(|op| match op {
                        TwinOp::Conv(c) => Some(c.kernel.1),
                        _ => None,
                    }),
            )
            .max()
            .unwrap_or(0);
        let kernel = max_ci
            .checked_mul(p)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_mul(rayon::current_num_threads()))
            .ok_or_else(|| product_bad("parallel kernel scratch overflow"))?;
        let metadata = self
            .geo
            .relu_op
            .checked_add(1)
            .and_then(|n| n.checked_mul(2048))
            .and_then(|n| p.checked_mul(256).and_then(|m| n.checked_add(m)))
            .ok_or_else(|| product_bad("metadata overflow"))?;
        resident_bytes
            .checked_add(matrices)
            .and_then(|n| n.checked_add(kernel))
            .and_then(|n| n.checked_add(metadata))
            .ok_or_else(|| product_bad("total-live overflow"))
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn product_resident_floor(
        &self,
        root: &RootGates,
        deadline: Instant,
    ) -> Result<usize> {
        super::engine::input_product_resident_floor_with_deadline(self.source_net, root, deadline)?
            .checked_add(self.plan_resident_bytes(deadline)?)
            .ok_or_else(|| product_bad("source/plan capacity overflow"))
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    fn plan_resident_bytes(&self, deadline: Instant) -> Result<usize> {
        super::net::check_deadline(deadline)?;
        let mut bytes = size_of::<Self>();
        for n in [
            self.geo.frames.capacity(),
            self.geo.contrib.capacity(),
            self.product_geometry.frames.capacity(),
            self.product_geometry.contrib.capacity(),
        ] {
            bytes = bytes
                .checked_add(
                    n.checked_mul(size_of::<Option<Frame>>())
                        .ok_or_else(|| product_bad("plan capacity overflow"))?,
                )
                .ok_or_else(|| product_bad("plan capacity overflow"))?;
        }
        bytes = bytes
            .checked_add(
                self.vops
                    .capacity()
                    .checked_mul(size_of::<Option<ConvOp>>())
                    .ok_or_else(|| product_bad("virtual plan capacity overflow"))?,
            )
            .ok_or_else(|| product_bad("virtual plan capacity overflow"))?;
        for cv in self.vops.iter().flatten() {
            super::net::check_deadline(deadline)?;
            for n in [
                cv.wmat.capacity(),
                cv.wt.capacity(),
                cv.bias.capacity(),
                cv.bias_err.capacity(),
                cv.gather.capacity(),
            ] {
                bytes = bytes
                    .checked_add(
                        n.checked_mul(8)
                            .ok_or_else(|| product_bad("virtual arrays overflow"))?,
                    )
                    .ok_or_else(|| product_bad("virtual arrays overflow"))?;
            }
            bytes = bytes
                .checked_add(
                    cv.back_taps
                        .capacity()
                        .checked_mul(size_of::<Vec<(usize, usize, usize)>>())
                        .ok_or_else(|| product_bad("virtual taps overflow"))?,
                )
                .ok_or_else(|| product_bad("virtual taps overflow"))?;
            for (i, taps) in cv.back_taps.iter().enumerate() {
                poll_optional_deadline(Some(deadline), i)?;
                bytes = bytes
                    .checked_add(
                        taps.capacity()
                            .checked_mul(size_of::<(usize, usize, usize)>())
                            .ok_or_else(|| product_bad("virtual taps overflow"))?,
                    )
                    .ok_or_else(|| product_bad("virtual taps overflow"))?;
            }
        }
        super::net::check_deadline(deadline)?;
        Ok(bytes)
    }

    /// The certified patch walk: pass column `i` is the identity row of
    /// neuron `cols[i]` on the seed tensor, run in the Lower lane when
    /// `lowers[i]`, else the Upper lane. Returns the input tensor's state.
    pub(crate) fn walk(
        &self,
        net: &TwinNet,
        root: &RootGates,
        cols: &[usize],
        lowers: &[bool],
        dom: Option<&DomainGates>,
        route: Option<&GpuConvRoute>,
    ) -> Option<Result<PatchWalk>> {
        let eng = super::bi_gpu::engine(net, root, route);
        self.walk_inner(net, root, cols, lowers, dom, eng.gpu_conv_route(), None)
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_inner(
        &self,
        net: &TwinNet,
        root: &dyn FrozenGateFrame,
        cols: &[usize],
        lowers: &[bool],
        dom: Option<&DomainGates>,
        route: &GpuConvRoute,
        deadline: Option<Instant>,
    ) -> Option<Result<PatchWalk>> {
        PatchKernel {
            source_net: self.source_net,
            geo: &self.geo,
            vops: &self.vops,
        }
        .walk_inner(net, root, cols, lowers, dom, route, deadline)
    }

    /// Concretize a walk over the root box: per pass column the unchanged
    /// `concretize` expression, accumulated over the input frame's REAL rows
    /// in ascending absolute order, with the FULL `n_in` in `gamma_n(n_in+16)`.
    pub(crate) fn concretize(root: &RootGates, walk: &PatchWalk, lowers: &[bool]) -> Vec<f64> {
        Self::concretize_checked(root, walk, lowers, None, false)
            .expect("legacy patch concretization has no deadline")
    }

    fn concretize_checked(
        root: &dyn FrozenGateFrame,
        walk: &PatchWalk,
        lowers: &[bool],
        deadline: Option<Instant>,
        original_domain: bool,
    ) -> Result<Vec<f64>> {
        check_optional_deadline(deadline)?;
        root.checkpoint()?;
        let callback = || root.checkpoint();
        let checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)> =
            root.cooperative().then_some(&callback);
        let f = &walk.frame;
        let p = walk.pos.len();
        let n_in = root.mid().len();
        let cg = col_geos_checked(f, &walk.pos, deadline, checkpoint)?;
        let asl = walk.a.as_slice().expect("standard layout");
        let esl = walk
            .e
            .as_ref()
            .map(|e| e.as_slice().expect("standard layout"));
        let mut v = patch_filled(p, 0.0f64, deadline, checkpoint)?;
        let mut rr = patch_filled(p, 0.0f64, deadline, checkpoint)?;
        let mut tabs = patch_filled(p, 0.0f64, deadline, checkpoint)?;
        let mut pen = patch_filled(p, 0.0f64, deadline, checkpoint)?;
        let outward = root.mode().outward();
        let mut dust = if outward {
            patch_filled(p, 0.0f64, deadline, checkpoint)?
        } else {
            Vec::new()
        };
        for row in 0..f.rows() {
            let (ch, iy, ix) = split_row(f, row);
            for col in 0..p {
                poll_patch(deadline, checkpoint, row * p + col)?;
                if (row * p + col).is_multiple_of(256) {
                    root.checkpoint()?;
                }
                let Some(i) = abs_index(f, &cg[col], ch, iy, ix) else {
                    continue;
                };
                let av = asl[row * p + col];
                let (m, rd) = (root.mid()[i], root.rad()[i]);
                let xa = if original_domain {
                    root.lo()[i]
                        .abs()
                        .max(root.hi()[i].abs())
                        .max(next_up(m.abs() + rd))
                } else {
                    root.xabs()[i]
                };
                v[col] += av * m;
                let aa = av.abs();
                rr[col] += aa * rd;
                tabs[col] += error_mul(aa, xa, outward);
                if outward {
                    for charge in [nominal_product_dust(av, m), nominal_product_dust(aa, rd)] {
                        if charge != 0.0 {
                            dust[col] = next_up(dust[col] + charge);
                        }
                    }
                }
                if let Some(es) = esl {
                    pen[col] += error_mul(es[row * p + col], xa, outward);
                }
            }
        }
        let gam = gamma_n(n_in + 16);
        let bounds = (0..p)
            .map(|col| {
                let lower = lowers.get(col).copied().unwrap_or(true);
                let base = v[col] + walk.b[col];
                let raw = if lower {
                    base - rr[col]
                } else {
                    base + rr[col]
                };
                if !root.mode().outward() {
                    return raw;
                }
                let mut slack = next_up(
                    certify_up(pen[col] + walk.eb[col], 1e-14)
                        + next_up(error_mul(
                            gam,
                            tabs[col] + walk.b[col].abs() + pen[col] + walk.eb[col],
                            true,
                        )),
                );
                if dust[col] != 0.0 {
                    slack = next_up(slack + certify_up(dust[col], gam));
                }
                if lower {
                    next_down(next_down(raw - slack))
                } else {
                    next_up(next_up(raw + slack))
                }
            })
            .collect();
        check_patch(deadline, checkpoint)?;
        Ok(bounds)
    }

    /// Test surface: the walk's input-tensor coefficients (and error lane)
    /// scattered to the dense `n_in x P` layout (virtual rows dropped; they
    /// are exact zeros after the mask).
    #[cfg(test)]
    pub(crate) fn scatter_input(
        walk: &PatchWalk,
        n_in: usize,
    ) -> (Array2<f64>, Option<Array2<f64>>) {
        let f = &walk.frame;
        let p = walk.pos.len();
        let cg = col_geos(f, &walk.pos);
        let mut a = Array2::<f64>::zeros((n_in, p));
        let mut e = walk.e.as_ref().map(|_| Array2::<f64>::zeros((n_in, p)));
        for row in 0..f.rows() {
            let (ch, iy, ix) = split_row(f, row);
            for col in 0..p {
                if let Some(i) = abs_index(f, &cg[col], ch, iy, ix) {
                    a[[i, col]] = walk.a[[row, col]];
                    if let (Some(dst), Some(src)) = (e.as_mut(), walk.e.as_ref()) {
                        dst[[i, col]] = src[[row, col]];
                    }
                }
            }
        }
        (a, e)
    }

    /// Test surface: the virtual conv of op `k` (`None`: not a patch conv).
    #[cfg(test)]
    pub(crate) fn virtual_conv(&self, k: usize) -> Option<&ConvOp> {
        self.vops.get(k).and_then(Option::as_ref)
    }

    /// Test surface: every real local row of tensor `t`'s frame for the
    /// column seeded at flat neuron `j` of the seed tensor, as absolute
    /// indices (`None`: the walk does not visit `t`).
    #[cfg(test)]
    pub(crate) fn real_rows(&self, t: usize, j: usize) -> Option<Vec<usize>> {
        let f = self.geo.frames.get(t).copied().flatten()?;
        let fseed = self
            .geo
            .frames
            .get(self.geo.seed_tensor)
            .copied()
            .flatten()?;
        let hw = fseed.h * fseed.w;
        let sp = j % hw;
        let g = col_geo(&f, (sp / fseed.w, sp % fseed.w));
        let mut rows = Vec::new();
        for row in 0..f.rows() {
            let (ch, iy, ix) = split_row(&f, row);
            if let Some(i) = abs_index(&f, &g, ch, iy, ix) {
                rows.push(i);
            }
        }
        Some(rows)
    }
}

/// Force every virtual-padding row of every border column to exact zero
/// (coefficient and error). Full frames and interior columns have none.
fn mask(
    cur: &mut LaneMat,
    f: &Frame,
    cg: &[ColGeo],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_patch(deadline, checkpoint)?;
    if f.full {
        return Ok(());
    }
    let mut border = patch_reserved(cg.len(), checkpoint)?;
    for (i, g) in cg.iter().enumerate() {
        poll_patch(deadline, checkpoint, i)?;
        if !g.interior {
            border.push(i);
        }
    }
    if border.is_empty() {
        return Ok(());
    }
    let p = cg.len();
    let ls = cur.l.as_slice_mut().expect("standard layout");
    let mut es = cur
        .e
        .as_mut()
        .map(|e| e.as_slice_mut().expect("standard layout"));
    let mut processed = 0usize;
    for row in 0..f.rows() {
        let (_, iy, ix) = split_row(f, row);
        for &col in &border {
            poll_patch(deadline, checkpoint, processed)?;
            processed += 1;
            let g = &cg[col];
            if iy < g.y0 || iy >= g.y1 || ix < g.x0 || ix >= g.x1 {
                ls[row * p + col] = 0.0;
                if let Some(es) = es.as_deref_mut() {
                    es[row * p + col] = 0.0;
                }
            }
        }
    }
    check_patch(deadline, checkpoint)
}

/// Deliver an op's contribution from the frame it was produced in to its
/// target tensor's frame: identity when equal, else an EXACT per-column
/// embed of the real rows (zero extension; to a full frame this is the
/// crossover scatter). `None` (TRIPWIRE) when a real row would leave the
/// target frame — a planner bug, never a silent truncation.
fn deliver(
    lm: LaneMat,
    from: &Frame,
    to: &Frame,
    pos: &[(usize, usize)],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Option<LaneMat>> {
    check_patch(deadline, checkpoint)?;
    if from == to {
        return Ok(Some(lm));
    }
    if from.shape() != to.shape() {
        return Ok(None);
    }
    let p = pos.len();
    let mut l2 = super::engine::zeros_with_checkpoint((to.rows(), p), deadline, checkpoint)?;
    let mut e2 = match lm.e.as_ref() {
        Some(_) => Some(super::engine::zeros_with_checkpoint(
            (to.rows(), p),
            deadline,
            checkpoint,
        )?),
        None => None,
    };
    {
        let src_l =
            lm.l.as_slice()
                .ok_or_else(|| product_bad("embed source layout"))?;
        let src_e = match lm.e.as_ref() {
            Some(e) => Some(
                e.as_slice()
                    .ok_or_else(|| product_bad("embed error layout"))?,
            ),
            None => None,
        };
        let dst_l = l2
            .as_slice_mut()
            .ok_or_else(|| product_bad("embed destination layout"))?;
        let mut dst_e = match e2.as_mut() {
            Some(e) => Some(
                e.as_slice_mut()
                    .ok_or_else(|| product_bad("embed error destination layout"))?,
            ),
            None => None,
        };
        let (fplane, tplane) = (from.plane(), to.plane());
        let mut processed = 0usize;
        for (col, &xy) in pos.iter().enumerate() {
            poll_patch(deadline, checkpoint, col)?;
            let g = col_geo(from, xy);
            let (ty, tx) = to.anchor(xy);
            let (dy, dx) = (g.ay - ty, g.ax - tx);
            for ch in 0..from.c {
                for iy in g.y0..g.y1 {
                    let yy = iy as isize + dy;
                    if yy < 0 || yy >= to.ph as isize {
                        return Ok(None);
                    }
                    for ix in g.x0..g.x1 {
                        poll_patch(deadline, checkpoint, processed)?;
                        processed += 1;
                        let xx = ix as isize + dx;
                        if xx < 0 || xx >= to.pw as isize {
                            return Ok(None);
                        }
                        let s = (ch * fplane + iy * from.pw + ix) * p + col;
                        let d = (ch * tplane + yy as usize * to.pw + xx as usize) * p + col;
                        dst_l[d] = src_l[s];
                        if let (Some(se), Some(de)) = (src_e, dst_e.as_deref_mut()) {
                            de[d] = se[s];
                        }
                    }
                }
            }
        }
    }
    check_patch(deadline, checkpoint)?;
    Ok(Some(LaneMat {
        l: l2,
        e: e2,
        env: None,
    }))
}

/// The conv arm's bias fold `b += L . brep` with its certified fold and
/// bias-error charges — the dense arm's expressions, per column, over the
/// output frame's REAL rows in ascending order, with the dense `n_out`.
#[allow(clippy::too_many_arguments)]
fn conv_bias(
    cur: &LaneMat,
    cv: &ConvOp,
    f: &Frame,
    cg: &[ColGeo],
    n_out: usize,
    outward: bool,
    b: &mut [f64],
    eb: &mut [f64],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_patch(deadline, checkpoint)?;
    let p = cg.len();
    let ls = cur.l.as_slice().expect("standard layout");
    let es = cur
        .e
        .as_ref()
        .map(|e| e.as_slice().expect("standard layout"));
    let gam_b = gamma_n(n_out + 2);
    for row in 0..f.rows() {
        poll_patch(deadline, checkpoint, row)?;
        let (ch, iy, ix) = split_row(f, row);
        let bias = cv.bias[ch];
        let berr = cv.bias_err[ch];
        if bias == 0.0 && berr == 0.0 {
            continue;
        }
        for col in 0..p {
            poll_patch(deadline, checkpoint, row * p + col)?;
            if abs_index(f, &cg[col], ch, iy, ix).is_none() {
                continue;
            }
            let v = ls[row * p + col];
            if bias != 0.0 {
                b[col] += bias * v;
            }
            if let Some(es) = es {
                let er = es[row * p + col];
                let mag = v.abs();
                let ab = bias.abs();
                let charge = error_mul(er + error_mul(gam_b, mag, outward), ab, outward)
                    + error_mul(mag + er, berr, outward);
                eb[col] += add_nominal_dust(charge, bias, v, outward);
                // Running-accumulator carry, as the dense arm (charged on
                // real rows only: an ordered subset of the dense charges).
                if bias != 0.0 {
                    eb[col] += error_mul(2.0 * UNIT, b[col].abs(), outward);
                }
            }
        }
    }
    if outward {
        let gc = gamma_n(8 * n_out + 16);
        for v in eb.iter_mut() {
            *v = certify_up(*v, gc);
        }
    }
    check_patch(deadline, checkpoint)
}

/// The relu arm: bias accumulation against `c` (before the transform), then
/// the in-place gate transform with its error contraction — the dense arm's
/// expressions at each (real row, column), gates read at the ABSOLUTE neuron.
#[allow(clippy::too_many_arguments)]
fn relu_step(
    cur: &mut LaneMat,
    f: &Frame,
    cg: &[ColGeo],
    g: Gates<'_>,
    n: usize,
    lowers: &[bool],
    outward: bool,
    b: &mut [f64],
    eb: &mut [f64],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_patch(deadline, checkpoint)?;
    let p = cg.len();
    let gam_b = gamma_n(n + 2);
    {
        let ls = cur.l.as_slice().expect("standard layout");
        let es = cur
            .e
            .as_ref()
            .map(|e| e.as_slice().expect("standard layout"));
        for row in 0..f.rows() {
            let (ch, iy, ix) = split_row(f, row);
            for col in 0..p {
                poll_patch(deadline, checkpoint, row * p + col)?;
                let Some(j) = abs_index(f, &cg[col], ch, iy, ix) else {
                    continue;
                };
                let cj = g.c[j];
                if cj == 0.0 {
                    continue;
                }
                let v = ls[row * p + col];
                let part = if lowers[col] { v.min(0.0) } else { v.max(0.0) };
                b[col] += part * cj;
                if let Some(es) = es {
                    let mag = v.abs();
                    let charge = error_mul(
                        es[row * p + col] + error_mul(gam_b, mag, outward),
                        cj,
                        outward,
                    ) + error_mul(2.0 * UNIT, b[col].abs(), outward);
                    eb[col] += add_nominal_dust(charge, part, cj, outward);
                }
            }
        }
        if outward {
            let gc = gamma_n(4 * n + 16);
            for v in eb.iter_mut() {
                *v = certify_up(*v, gc);
            }
        }
    }
    let ls = cur.l.as_slice_mut().expect("standard layout");
    let mut es = cur
        .e
        .as_mut()
        .map(|e| e.as_slice_mut().expect("standard layout"));
    for row in 0..f.rows() {
        let (ch, iy, ix) = split_row(f, row);
        for col in 0..p {
            poll_patch(deadline, checkpoint, row * p + col)?;
            let Some(j) = abs_index(f, &cg[col], ch, iy, ix) else {
                continue;
            };
            let idx = row * p + col;
            let v = ls[idx];
            let vp = v.max(0.0);
            let vn = v - vp;
            let alpha = g.lower_policy.alpha(g.original, j, g.alpha[j]);
            let lv = if lowers[col] {
                vp * alpha + vn * g.s[j]
            } else {
                vp * g.s[j] + vn * alpha
            };
            ls[idx] = lv;
            if let Some(es) = es.as_deref_mut() {
                let charge = error_mul(
                    es[idx] + error_mul(4.0 * UNIT, v.abs(), outward),
                    g.ms[j],
                    outward,
                ) + error_mul(4.0 * UNIT, lv.abs(), outward);
                let (positive, negative) = if lowers[col] {
                    (alpha, g.s[j])
                } else {
                    (g.s[j], alpha)
                };
                es[idx] = slack16(add_nominal_dust(
                    add_nominal_dust(charge, vp, positive, outward),
                    vn,
                    negative,
                    outward,
                ));
            }
        }
    }
    check_patch(deadline, checkpoint)
}

/// Per-neuron channel-affine parameters (standalone BN).
#[derive(Clone, Copy)]
struct Affine<'a> {
    scale: &'a [f64],
    shift: &'a [f64],
    rel: f64,
    shift_err: &'a [f64],
}

/// The channel-affine arm: bias fold `sum v_j t_j` with its certified
/// parameter error, then the scale with the fixed-gate error algebra — the
/// dense arm's expressions at each (real row, column).
fn affine_step(
    cur: &mut LaneMat,
    f: &Frame,
    cg: &[ColGeo],
    a: Affine<'_>,
    outward: bool,
    b: &mut [f64],
    eb: &mut [f64],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_patch(deadline, checkpoint)?;
    let p = cg.len();
    let n = a.scale.len();
    let gam_b = gamma_n(n + 2);
    {
        let ls = cur.l.as_slice().expect("standard layout");
        let es = cur
            .e
            .as_ref()
            .map(|e| e.as_slice().expect("standard layout"));
        for row in 0..f.rows() {
            let (ch, iy, ix) = split_row(f, row);
            for col in 0..p {
                poll_patch(deadline, checkpoint, row * p + col)?;
                let Some(j) = abs_index(f, &cg[col], ch, iy, ix) else {
                    continue;
                };
                let tj = a.shift[j];
                let te = a.shift_err[j];
                if tj == 0.0 && te == 0.0 {
                    continue;
                }
                let v = ls[row * p + col];
                if tj != 0.0 {
                    b[col] += tj * v;
                }
                if let Some(es) = es {
                    let er = es[row * p + col];
                    let mag = v.abs();
                    let charge = error_mul(er + error_mul(gam_b, mag, outward), tj.abs(), outward)
                        + error_mul(mag + er, te, outward);
                    eb[col] += add_nominal_dust(charge, tj, v, outward);
                    if tj != 0.0 {
                        eb[col] += error_mul(2.0 * UNIT, b[col].abs(), outward);
                    }
                }
            }
        }
        if outward {
            let gc = gamma_n(8 * n + 16);
            for v in eb.iter_mut() {
                *v = certify_up(*v, gc);
            }
        }
    }
    let rel = a.rel;
    let ls = cur.l.as_slice_mut().expect("standard layout");
    let mut es = cur
        .e
        .as_mut()
        .map(|e| e.as_slice_mut().expect("standard layout"));
    for row in 0..f.rows() {
        let (ch, iy, ix) = split_row(f, row);
        for col in 0..p {
            poll_patch(deadline, checkpoint, row * p + col)?;
            let Some(j) = abs_index(f, &cg[col], ch, iy, ix) else {
                continue;
            };
            let idx = row * p + col;
            let v = ls[idx];
            let sj = a.scale[j];
            let lnew = v * sj;
            if let Some(es) = es.as_deref_mut() {
                let sa = sj.abs();
                let charge = error_mul(
                    es[idx]
                        + error_mul(rel, v.abs(), outward)
                        + error_mul(4.0 * UNIT, v.abs(), outward),
                    sa,
                    outward,
                ) + error_mul(rel + 4.0 * UNIT, lnew.abs(), outward);
                es[idx] = slack16(add_nominal_dust(charge, v, sj, outward));
            }
            ls[idx] = lnew;
        }
    }
    check_patch(deadline, checkpoint)
}

/// Route-only dispatch shared by borrowed and prepared gate readers. The
/// arithmetic kernel and certification policy are unchanged; a gate reader
/// itself has no row-publication authority.
#[allow(clippy::too_many_arguments)]
fn patch_conv_backward(
    route: &GpuConvRoute,
    cv: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    lane: ConvLane,
    outward: bool,
) {
    super::engine::BackwardEngine::conv_backward(route, cv, src, dst, abs_w, lane, || outward);
}

fn product_bad(message: &str) -> NyError {
    NyError::InvalidSpec(format!("margin-row patch input products: {message}"))
}

fn check_optional_deadline(deadline: Option<Instant>) -> Result<()> {
    match deadline {
        Some(at) => super::net::check_deadline(at),
        None => Ok(()),
    }
}

#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
fn poll_optional_deadline(deadline: Option<Instant>, index: usize) -> Result<()> {
    if index.is_multiple_of(4096) {
        check_optional_deadline(deadline)?;
    }
    Ok(())
}

fn error_mul(a: f64, b: f64, outward: bool) -> f64 {
    if outward {
        underflow_product_up(a, b)
    } else {
        a * b
    }
}

fn add_nominal_dust(charge: f64, a: f64, b: f64, outward: bool) -> f64 {
    let dust = if outward {
        nominal_product_dust(a, b)
    } else {
        0.0
    };
    if dust == 0.0 {
        charge
    } else {
        next_up(charge + dust)
    }
}

/// Arithmetic-only view shared by legacy and source-sealed prepared plans.
struct PatchKernel<'a> {
    source_net: &'a TwinNet,
    geo: &'a Geometry,
    vops: &'a [Option<ConvOp>],
}
impl PatchKernel<'_> {
    #[allow(clippy::too_many_arguments)]
    fn walk_inner(
        &self,
        net: &TwinNet,
        root: &dyn FrozenGateFrame,
        cols: &[usize],
        lowers: &[bool],
        dom: Option<&DomainGates>,
        route: &GpuConvRoute,
        deadline: Option<Instant>,
    ) -> Option<Result<PatchWalk>> {
        patch_try!(check_optional_deadline(deadline));
        patch_try!(root.checkpoint());
        let callback = || root.checkpoint();
        let checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)> =
            root.cooperative().then_some(&callback);
        let geo = &self.geo;
        let relu_op = geo.relu_op;
        let p = cols.len();
        if !std::ptr::eq(self.source_net, net)
            || p == 0
            || lowers.len() != p
            || net.ops.len() <= relu_op
            || net.tsize.len() <= relu_op
            || self.vops.len() != relu_op
        {
            return None;
        }
        match &net.ops[relu_op] {
            TwinOp::Relu { input, .. } if *input == geo.seed_tensor => {}
            _ => return None,
        }
        if root.mid().len() != net.n_in
            || root.rad().len() != net.n_in
            || root.xabs().len() != net.n_in
        {
            return None;
        }
        // Split-Lagrangian terms break locality (a `delta` at a split neuron
        // outside a column's field creates a coefficient outside its frame)
        // and no prefix pass carries them: refuse, the dense pass runs.
        if let Some(d) = dom {
            if !d.beta.is_empty() || !d.beta_pc.is_empty() || d.infeasible {
                return None;
            }
            for (li, gv) in &d.layers {
                if let Ok(rec) = root.layer(*li) {
                    let n = rec.n;
                    if gv.alpha.len() != n || gv.s.len() != n || gv.c.len() != n || gv.ms.len() != n
                    {
                        return None;
                    }
                }
            }
        }
        let fseed = geo.frames.get(geo.seed_tensor).copied().flatten()?;
        let hw = fseed.h * fseed.w;
        if hw == 0 || fseed.full {
            return None;
        }
        let n_seed = fseed.c * hw;
        let mut pos: Vec<(usize, usize)> = patch_try!(patch_reserved(p, checkpoint));
        let mut seed = patch_try!(super::engine::zeros_with_checkpoint(
            (fseed.rows(), p),
            deadline,
            checkpoint
        ));
        for (col, &j) in cols.iter().enumerate() {
            patch_try!(poll_patch(deadline, checkpoint, col));
            if j >= n_seed {
                return None;
            }
            let sp = j % hw;
            pos.push((sp / fseed.w, sp % fseed.w));
            // The identity row in local coordinates (the seed frame is 1x1):
            // exact, so the error lane starts at zero.
            seed[[j / hw, col]] = 1.0;
        }
        let outward = root.mode().outward();
        let mut state: Vec<Option<LaneMat>> = patch_try!(patch_reserved(relu_op + 1, checkpoint));
        if checkpoint.is_some() {
            for i in 0..=relu_op {
                patch_try!(poll_patch(deadline, checkpoint, i));
                state.push(None);
            }
        } else {
            state.resize_with(relu_op + 1, || None);
        }
        state[geo.seed_tensor] = Some(LaneMat {
            e: if outward {
                Some(patch_try!(super::engine::zeros_with_checkpoint(
                    seed.dim(),
                    deadline,
                    checkpoint
                )))
            } else {
                None
            },
            l: seed,
            env: None,
        });
        let mut b = patch_try!(patch_filled(p, 0.0f64, deadline, checkpoint));
        let mut eb = patch_try!(patch_filled(p, 0.0f64, deadline, checkpoint));
        for k in (0..relu_op).rev() {
            patch_try!(check_optional_deadline(deadline));
            patch_try!(root.checkpoint());
            let Some(mut cur) = state[k + 1].take() else {
                continue;
            };
            let fout = geo.frames[k + 1]?;
            let fin = geo.contrib[k]?;
            let cg = patch_try!(col_geos_checked(&fout, &pos, deadline, checkpoint));
            patch_try!(mask(&mut cur, &fout, &cg, deadline, checkpoint));
            match &net.ops[k] {
                TwinOp::Conv(cv) => {
                    let n_out = net.tsize[k + 1];
                    if fout.rows() != cur.l.nrows()
                        || (!fout.full && n_out != fout.c * fout.h * fout.w)
                        || cv.bias.len() != fout.c
                        || cv.bias_err.len() != fout.c
                    {
                        return None;
                    }
                    patch_try!(conv_bias(
                        &cur, cv, &fout, &cg, n_out, outward, &mut b, &mut eb, deadline, checkpoint
                    ));
                    let vop: &ConvOp = match self.vops[k].as_ref() {
                        Some(v) => v,
                        None => cv,
                    };
                    let rows_in = fin.rows();
                    let (vci, vih, viw) = vop.ishape;
                    let (vco, voh, vow) = vop.oshape;
                    if vci * vih * viw != rows_in || vco * voh * vow != fout.rows() {
                        return None;
                    }
                    let mut lnew = patch_try!(super::engine::zeros_with_checkpoint(
                        (rows_in, p),
                        deadline,
                        checkpoint
                    ));
                    if let Some(at) = deadline {
                        if root.cooperative() {
                            let checkpoint = || root.checkpoint();
                            patch_try!(super::net::conv_apply_backward_with_checkpoint(
                                vop,
                                &cur.l,
                                &mut lnew,
                                false,
                                at,
                                &checkpoint
                            ));
                        } else {
                            patch_try!(super::net::conv_apply_backward_with_deadline(
                                vop, &cur.l, &mut lnew, false, at
                            ));
                        }
                    } else {
                        patch_conv_backward(
                            route,
                            vop,
                            &cur.l,
                            &mut lnew,
                            false,
                            ConvLane::Signed,
                            outward,
                        );
                    }
                    let enew = if outward {
                        // The dense arm's charges, at the dense `k_bwd`.
                        let gk = gamma_n(vop.k_bwd + 2);
                        let g = next_up(
                            gk + cv.weight_rel_err + error_mul(gk, cv.weight_rel_err, true),
                        );
                        let comb = patch_try!(super::engine::combined_err_with_checkpoint(
                            &cur.l,
                            cur.e.as_ref(),
                            g,
                            deadline,
                            true,
                            checkpoint
                        ));
                        let mut en = patch_try!(super::engine::zeros_with_checkpoint(
                            (rows_in, p),
                            deadline,
                            checkpoint
                        ));
                        if let Some(at) = deadline {
                            if root.cooperative() {
                                let checkpoint = || root.checkpoint();
                                patch_try!(super::net::error_conv_with_checkpoint(
                                    vop,
                                    &comb,
                                    &cur.l,
                                    &mut en,
                                    at,
                                    &checkpoint
                                ));
                            } else {
                                patch_try!(super::net::error_conv_with_deadline(
                                    vop, &comb, &cur.l, &mut en, at
                                ));
                            }
                        } else if route.is_off() {
                            patch_try!(super::net::error_conv_without_deadline(
                                vop, &comb, &cur.l, &mut en
                            ));
                        } else {
                            patch_conv_backward(
                                route,
                                vop,
                                &comb,
                                &mut en,
                                true,
                                ConvLane::Abs,
                                outward,
                            );
                        }
                        let g2 = gamma_n(vop.k_bwd + 8);
                        for (i, v) in en.iter_mut().enumerate() {
                            patch_try!(poll_patch(deadline, checkpoint, i));
                            *v = certify_up(*v, g2);
                        }
                        Some(en)
                    } else {
                        None
                    };
                    let target = geo.frames.get(cv.input).copied().flatten()?;
                    let delivered = patch_try!(deliver(
                        LaneMat {
                            l: lnew,
                            e: enew,
                            env: None,
                        },
                        &fin,
                        &target,
                        &pos,
                        deadline,
                        checkpoint,
                    ))?;
                    patch_try!(super::engine::merge_step_with_checkpoint(
                        &mut state, cv.input, delivered, outward, None, k, &eb, deadline,
                        checkpoint
                    ));
                }
                TwinOp::Relu { input, layer } => {
                    let rec = root.layer(*layer).ok()?;
                    let (alpha, s, c, ms) = match dom.and_then(|d| d.layers.get(layer)) {
                        Some(gv) => (&gv.alpha[..], &gv.s[..], &gv.c[..], &gv.ms[..]),
                        None => (&rec.alpha[..], &rec.s[..], &rec.c[..], &rec.ms[..]),
                    };
                    let n = rec.n;
                    if fout.c * fout.h * fout.w != n
                        || fout.rows() != cur.l.nrows()
                        || alpha.len() != n
                        || s.len() != n
                        || c.len() != n
                        || ms.len() != n
                    {
                        return None;
                    }
                    patch_try!(relu_step(
                        &mut cur,
                        &fout,
                        &cg,
                        Gates {
                            alpha,
                            s,
                            c,
                            ms,
                            lower_policy: root.crossing_lower_policy(),
                            original: rec
                        },
                        n,
                        lowers,
                        outward,
                        &mut b,
                        &mut eb,
                        deadline,
                        checkpoint,
                    ));
                    let target = geo.frames.get(*input).copied().flatten()?;
                    let delivered =
                        patch_try!(deliver(cur, &fin, &target, &pos, deadline, checkpoint))?;
                    patch_try!(super::engine::merge_step_with_checkpoint(
                        &mut state, *input, delivered, outward, None, k, &eb, deadline, checkpoint
                    ));
                }
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                } => {
                    let n = scale.len();
                    if fout.c * fout.h * fout.w != n
                        || fout.rows() != cur.l.nrows()
                        || shift.len() != n
                        || shift_err.len() != n
                    {
                        return None;
                    }
                    patch_try!(affine_step(
                        &mut cur,
                        &fout,
                        &cg,
                        Affine {
                            scale,
                            shift,
                            rel: *scale_rel_err,
                            shift_err,
                        },
                        outward,
                        &mut b,
                        &mut eb,
                        deadline,
                        checkpoint,
                    ));
                    let target = geo.frames.get(*input).copied().flatten()?;
                    let delivered =
                        patch_try!(deliver(cur, &fin, &target, &pos, deadline, checkpoint))?;
                    patch_try!(super::engine::merge_step_with_checkpoint(
                        &mut state, *input, delivered, outward, None, k, &eb, deadline, checkpoint
                    ));
                }
                TwinOp::Add { lhs, rhs } => {
                    let copy = LaneMat {
                        l: patch_try!(super::engine::clone_with_checkpoint(
                            &cur.l, deadline, checkpoint
                        )),
                        e: match cur.e.as_ref() {
                            Some(e) => Some(patch_try!(super::engine::clone_with_checkpoint(
                                e, deadline, checkpoint
                            ))),
                            None => None,
                        },
                        env: None,
                    };
                    let to_lhs = geo.frames.get(*lhs).copied().flatten()?;
                    let to_rhs = geo.frames.get(*rhs).copied().flatten()?;
                    let left = patch_try!(deliver(cur, &fin, &to_lhs, &pos, deadline, checkpoint))?;
                    patch_try!(super::engine::merge_step_with_checkpoint(
                        &mut state, *lhs, left, outward, None, k, &eb, deadline, checkpoint
                    ));
                    let right =
                        patch_try!(deliver(copy, &fin, &to_rhs, &pos, deadline, checkpoint))?;
                    patch_try!(super::engine::merge_step_with_checkpoint(
                        &mut state, *rhs, right, outward, None, k, &eb, deadline, checkpoint
                    ));
                }
                // Never inside a trunk prefix (the planner refuses it); the
                // dense engine errors on a trunk Gemm. Fall back.
                TwinOp::Flatten { .. } | TwinOp::Gemm { .. } => return None,
            }
        }
        let mut fin0 = state[0].take()?;
        let f0 = geo.frames[0]?;
        if fin0.l.nrows() != f0.rows() {
            return None;
        }
        let cg0 = patch_try!(col_geos_checked(&f0, &pos, deadline, checkpoint));
        patch_try!(mask(&mut fin0, &f0, &cg0, deadline, checkpoint));
        // Fail-closed NaN/Inf firewall (the dense pass's, over the rows this
        // walk publishes): verdict math must never see non-finite.
        if deadline.is_some() {
            patch_try!(super::engine::finite_with_checkpoint(
                fin0.l.as_slice().expect("standard layout"),
                deadline,
                checkpoint
            ));
            if let Some(e) = fin0.e.as_ref() {
                patch_try!(super::engine::finite_with_checkpoint(
                    e.as_slice().expect("standard layout"),
                    deadline,
                    checkpoint
                ));
            }
            patch_try!(super::engine::finite_with_checkpoint(
                &b, deadline, checkpoint
            ));
            patch_try!(super::engine::finite_with_checkpoint(
                &eb, deadline, checkpoint
            ));
        }
        let finite = deadline.is_some()
            || (fin0.l.iter().all(|v| v.is_finite())
                && fin0
                    .e
                    .as_ref()
                    .is_none_or(|e| e.iter().all(|v| v.is_finite()))
                && b.iter().all(|v| v.is_finite())
                && eb.iter().all(|v| v.is_finite()));
        if !finite {
            return Some(Err(NyError::NumericalInstability(
                "margin_row: non-finite #bi-patch rows".into(),
            )));
        }
        patch_try!(check_optional_deadline(deadline));
        patch_try!(root.checkpoint());
        Some(Ok(PatchWalk {
            frame: f0,
            pos,
            a: fin0.l,
            e: fin0.e,
            b,
            eb,
        }))
    }
}

fn check_patch(
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_optional_deadline(deadline)?;
    if let Some(checkpoint) = checkpoint {
        checkpoint()?;
    }
    Ok(())
}
fn poll_patch(
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
    index: usize,
) -> Result<()> {
    if index.is_multiple_of(256) {
        check_patch(deadline, checkpoint)?;
    }
    Ok(())
}
fn patch_reserved<T>(
    n: usize,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Vec<T>> {
    if checkpoint.is_none() {
        return Ok(Vec::with_capacity(n));
    }
    n.checked_mul(size_of::<T>())
        .ok_or_else(|| product_bad("allocation overflow"))?;
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| product_bad("allocation refused"))?;
    if v.capacity() > n {
        return Err(product_bad("allocator exceeded prepared admission"));
    }
    if let Some(checkpoint) = checkpoint {
        checkpoint()?;
    }
    Ok(v)
}
fn patch_filled<T: Copy>(
    n: usize,
    value: T,
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Vec<T>> {
    if checkpoint.is_none() {
        return Ok(vec![value; n]);
    }
    let mut v = patch_reserved(n, checkpoint)?;
    for i in 0..n {
        poll_patch(deadline, checkpoint, i)?;
        v.push(value);
    }
    check_patch(deadline, checkpoint)?;
    Ok(v)
}
fn col_geos_checked(
    f: &Frame,
    pos: &[(usize, usize)],
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<Vec<ColGeo>> {
    if checkpoint.is_none() {
        return Ok(col_geos(f, pos));
    }
    let mut v = patch_reserved(pos.len(), checkpoint)?;
    for (i, &xy) in pos.iter().enumerate() {
        poll_patch(deadline, checkpoint, i)?;
        v.push(col_geo(f, xy));
    }
    check_patch(deadline, checkpoint)?;
    Ok(v)
}

fn pin_shapes_prepared(
    ops: &[GeoOp],
    tsize: &[usize],
    upto: usize,
    checkpoint: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Vec<Option<Shape>>> {
    let tensors = upto
        .checked_add(1)
        .ok_or_else(|| product_bad("geometry overflow"))?;
    checkpoint()?;
    if ops.len() < upto || tsize.len() < tensors {
        return Err(product_bad("shape"));
    }
    let mut shapes: Vec<Option<Shape>> = patch_filled(tensors, None, None, Some(checkpoint))?;
    for (k, op) in ops[..upto].iter().enumerate() {
        if k.is_multiple_of(64) {
            checkpoint()?;
        }
        let pinned = match *op {
            GeoOp::Conv {
                input,
                ishape,
                oshape,
                ..
            } => set_shape(&mut shapes, input, ishape) && set_shape(&mut shapes, k + 1, oshape),
            GeoOp::Affine {
                input,
                shape: Some(shape),
            } => set_shape(&mut shapes, input, shape) && set_shape(&mut shapes, k + 1, shape),
            _ => true,
        };
        if !pinned {
            return Err(product_bad("shape"));
        }
    }
    for _ in 0..tensors {
        checkpoint()?;
        let mut changed = false;
        for (k, op) in ops[..upto].iter().enumerate() {
            if k.is_multiple_of(64) {
                checkpoint()?;
            }
            let group: [Option<usize>; 3] = match *op {
                GeoOp::Relu { input } | GeoOp::Affine { input, .. } => {
                    [Some(k + 1), Some(input), None]
                }
                GeoOp::Add { lhs, rhs } => [Some(k + 1), Some(lhs), Some(rhs)],
                _ => continue,
            };
            let mut known: Option<Shape> = None;
            for t in group.into_iter().flatten() {
                let Some(slot) = shapes.get(t).copied() else {
                    return Err(product_bad("shape"));
                };
                match (known, slot) {
                    (None, Some(s)) => known = Some(s),
                    (Some(a), Some(b)) if a != b => return Err(product_bad("shape")),
                    _ => {}
                }
            }
            if let Some(s) = known {
                for t in group.into_iter().flatten() {
                    if shapes[t].is_none() {
                        shapes[t] = Some(s);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (t, s) in shapes.iter().enumerate() {
        if t.is_multiple_of(64) {
            checkpoint()?;
        }
        if let Some((c, h, w)) = *s {
            if c.checked_mul(h).and_then(|v| v.checked_mul(w)) != Some(tsize[t]) {
                return Err(product_bad("shape"));
            }
        }
    }
    Ok(shapes)
}

fn prepared_geometry(
    ops: &[GeoOp],
    tsize: &[usize],
    relu_op: usize,
    ratio: f64,
    checkpoint: &(dyn Fn() -> Result<()> + Sync),
) -> Result<Geometry> {
    let Some(GeoOp::Relu { input: seed_tensor }) = ops.get(relu_op).copied() else {
        return Err(product_bad("not_relu"));
    };
    if seed_tensor > relu_op || tsize.len() <= relu_op {
        return Err(product_bad("not_relu"));
    }
    let shapes = pin_shapes_prepared(ops, tsize, relu_op, checkpoint)?;
    let shape_of = |t: usize| -> Result<Shape> {
        shapes
            .get(t)
            .copied()
            .flatten()
            .ok_or_else(|| product_bad("shape"))
    };
    let too_big = |f: &Frame, n: usize| !f.full && (f.rows() as f64) > ratio * (n as f64);
    let mut frames: Vec<Option<Frame>> = patch_filled(relu_op + 1, None, None, Some(checkpoint))?;
    let mut contrib: Vec<Option<Frame>> = patch_filled(relu_op, None, None, Some(checkpoint))?;
    frames[seed_tensor] = Some(Frame::seed(shape_of(seed_tensor)?));
    let mut convert_at: Option<usize> = None;
    let (mut macs_patch, mut macs_dense) = (0u64, 0u64);
    for k in (0..relu_op).rev() {
        checkpoint()?;
        let t_out = k + 1;
        let Some(mut fout) = frames[t_out] else {
            continue;
        };
        validate_prepared_frame(fout, shape_of(seed_tensor)?)?;
        if convert_at.is_none() && too_big(&fout, tsize[t_out]) {
            convert_at = Some(t_out);
        }
        if convert_at.is_some() && !fout.full {
            fout = Frame::full(fout.shape());
            frames[t_out] = Some(fout);
        }
        match ops[k] {
            GeoOp::Conv {
                transposed: true, ..
            } => return Err(product_bad("transposed")),
            GeoOp::Conv {
                input,
                kernel: (co, ci, kh, kw),
                stride: (s_h, s_w),
                pads,
                ishape,
                oshape,
                ..
            } => {
                if input >= t_out
                    || oshape != fout.shape()
                    || shape_of(input)? != ishape
                    || ci != ishape.0
                    || co != oshape.0
                    || s_h == 0
                    || s_w == 0
                {
                    return Err(product_bad("shape"));
                }
                let fin = if fout.full {
                    Frame::full(ishape)
                } else {
                    Frame {
                        c: ishape.0,
                        h: ishape.1,
                        w: ishape.2,
                        sy: fout
                            .sy
                            .checked_mul(to_signed(s_h)?)
                            .ok_or_else(|| product_bad("anchor overflow"))?,
                        sx: fout
                            .sx
                            .checked_mul(to_signed(s_w)?)
                            .ok_or_else(|| product_bad("anchor overflow"))?,
                        oy: fout
                            .oy
                            .checked_mul(to_signed(s_h)?)
                            .and_then(|n| n.checked_sub(to_signed(pads.0).ok()?))
                            .ok_or_else(|| product_bad("anchor overflow"))?,
                        ox: fout
                            .ox
                            .checked_mul(to_signed(s_w)?)
                            .and_then(|n| n.checked_sub(to_signed(pads.1).ok()?))
                            .ok_or_else(|| product_bad("anchor overflow"))?,
                        ph: fout
                            .ph
                            .checked_sub(1)
                            .and_then(|n| n.checked_mul(s_h))
                            .and_then(|n| n.checked_add(kh))
                            .ok_or_else(|| product_bad("extent overflow"))?,
                        pw: fout
                            .pw
                            .checked_sub(1)
                            .and_then(|n| n.checked_mul(s_w))
                            .and_then(|n| n.checked_add(kw))
                            .ok_or_else(|| product_bad("extent overflow"))?,
                        full: false,
                    }
                };
                validate_prepared_frame(fin, shape_of(seed_tensor)?)?;
                let per_row = u64::try_from(
                    ci.checked_mul(kh)
                        .and_then(|n| n.checked_mul(kw))
                        .ok_or_else(|| product_bad("cost overflow"))?,
                )
                .map_err(|_| product_bad("cost overflow"))?;
                macs_patch = macs_patch
                    .checked_add(
                        (fout.rows() as u64)
                            .checked_mul(per_row)
                            .ok_or_else(|| product_bad("cost overflow"))?,
                    )
                    .ok_or_else(|| product_bad("cost overflow"))?;
                macs_dense = macs_dense
                    .checked_add(
                        (tsize[t_out] as u64)
                            .checked_mul(per_row)
                            .ok_or_else(|| product_bad("cost overflow"))?,
                    )
                    .ok_or_else(|| product_bad("cost overflow"))?;
                contrib[k] = Some(fin);
                prepared_union_into(&mut frames, input, fin)?;
            }
            GeoOp::Relu { input } | GeoOp::Affine { input, .. } => {
                if input >= t_out || shape_of(input)? != fout.shape() {
                    return Err(product_bad("shape"));
                }
                contrib[k] = Some(fout);
                prepared_union_into(&mut frames, input, fout)?;
            }
            GeoOp::Add { lhs, rhs } => {
                for t in [lhs, rhs] {
                    if t >= t_out || shape_of(t)? != fout.shape() {
                        return Err(product_bad("shape"));
                    }
                }
                contrib[k] = Some(fout);
                prepared_union_into(&mut frames, lhs, fout)?;
                prepared_union_into(&mut frames, rhs, fout)?;
            }
            GeoOp::Other => return Err(product_bad("not_trunk")),
        }
    }
    let Some(mut f0) = frames[0] else {
        return Err(product_bad("no_input"));
    };
    validate_prepared_frame(f0, shape_of(seed_tensor)?)?;
    if convert_at.is_none() && too_big(&f0, tsize[0]) {
        convert_at = Some(0);
    }
    if convert_at.is_some() && !f0.full {
        f0 = Frame::full(f0.shape());
        frames[0] = Some(f0);
    }
    // The conversion fell on the FIRST op (the seed tensor itself runs
    // full): the walk is the dense walk plus gathers — no gain.
    if frames[seed_tensor].is_some_and(|f| f.full) {
        return Err(product_bad("no_gain"));
    }
    let (mut rows_patch, mut rows_dense, mut virt_max) = (0usize, 0usize, 0usize);
    for (t, f) in frames.iter().enumerate() {
        if t.is_multiple_of(64) {
            checkpoint()?;
        }
        if let Some(f) = f {
            validate_prepared_frame(*f, shape_of(seed_tensor)?)?;
            rows_patch = rows_patch
                .checked_add(f.rows())
                .ok_or_else(|| product_bad("row overflow"))?;
            rows_dense = rows_dense
                .checked_add(tsize[t])
                .ok_or_else(|| product_bad("row overflow"))?;
            virt_max = virt_max.max(f.rows());
        }
    }
    for f in contrib.iter().flatten() {
        checkpoint()?;
        validate_prepared_frame(*f, shape_of(seed_tensor)?)?;
        virt_max = virt_max.max(f.rows());
    }
    Ok(Geometry {
        relu_op,
        seed_tensor,
        frames,
        contrib,
        convert_at,
        macs_col_patch: macs_patch,
        macs_col_dense: macs_dense,
        rows_col_patch: rows_patch,
        rows_col_dense: rows_dense,
        virt_rows_max: virt_max,
    })
}

fn to_signed(n: usize) -> Result<isize> {
    isize::try_from(n).map_err(|_| product_bad("signed geometry overflow"))
}
fn validate_prepared_frame(f: Frame, seed: Shape) -> Result<()> {
    if [f.c, f.h, f.w, f.ph, f.pw].contains(&0)
        || f.c
            .checked_mul(f.h)
            .and_then(|n| n.checked_mul(f.w))
            .is_none()
        || f.c
            .checked_mul(f.ph)
            .and_then(|n| n.checked_mul(f.pw))
            .is_none()
    {
        return Err(product_bad("invalid prepared frame extent"));
    }
    for (stride, offset, extent, maximum, image) in [
        (f.sy, f.oy, f.ph, seed.1, f.h),
        (f.sx, f.ox, f.pw, seed.2, f.w),
    ] {
        let end = to_signed(
            maximum
                .checked_sub(1)
                .ok_or_else(|| product_bad("empty seed"))?,
        )?
        .checked_mul(stride)
        .and_then(|n| n.checked_add(offset))
        .ok_or_else(|| product_bad("prepared anchor overflow"))?;
        for anchor in [offset, end] {
            if anchor.unsigned_abs() > (isize::MAX as usize) / 4
                || extent > (isize::MAX as usize) / 4
                || image > (isize::MAX as usize) / 4
            {
                return Err(product_bad("prepared embedding coordinate range"));
            }
            anchor
                .checked_neg()
                .ok_or_else(|| product_bad("mask overflow"))?;
            to_signed(image)?
                .checked_sub(anchor)
                .ok_or_else(|| product_bad("mask overflow"))?;
            anchor
                .checked_add(to_signed(extent)?)
                .ok_or_else(|| product_bad("frame end overflow"))?;
        }
    }
    Ok(())
}
fn prepared_union_into(frames: &mut [Option<Frame>], t: usize, f: Frame) -> Result<()> {
    let slot = frames.get_mut(t).ok_or_else(|| product_bad("shape"))?;
    *slot = Some(match *slot {
        None => f,
        Some(g) if f.shape() != g.shape() => return Err(product_bad("shape")),
        Some(g) if f.full || g.full => Frame::full(f.shape()),
        Some(g) => {
            if f.sy != g.sy || f.sx != g.sx {
                return Err(product_bad("stride_union"));
            }
            let oy = f.oy.min(g.oy);
            let ox = f.ox.min(g.ox);
            let ey =
                f.oy.checked_add(to_signed(f.ph)?)
                    .and_then(|a| g.oy.checked_add(to_signed(g.ph).ok()?).map(|b| a.max(b)))
                    .ok_or_else(|| product_bad("union overflow"))?;
            let ex =
                f.ox.checked_add(to_signed(f.pw)?)
                    .and_then(|a| g.ox.checked_add(to_signed(g.pw).ok()?).map(|b| a.max(b)))
                    .ok_or_else(|| product_bad("union overflow"))?;
            Frame {
                oy,
                ox,
                ph: usize::try_from(
                    ey.checked_sub(oy)
                        .ok_or_else(|| product_bad("union overflow"))?,
                )
                .map_err(|_| product_bad("union overflow"))?,
                pw: usize::try_from(
                    ex.checked_sub(ox)
                        .ok_or_else(|| product_bad("union overflow"))?,
                )
                .map_err(|_| product_bad("union overflow"))?,
                ..f
            }
        }
    });
    Ok(())
}

/// Source-sealed immutable patch geometry. Model and gate ownership is retained
/// separately from each produced row tile; no references point back into this plan.
pub(super) struct PreparedPatchPlan {
    prefix: PreparedPrefix,
    geo: Geometry,
    vops: Vec<Option<ConvOp>>,
    geometry_resident: PersistentReservation,
    virtual_resident: PersistentReservation,
    max_ci: usize,
    source_resident: usize,
    deadline: Instant,
}
impl PreparedPatchPlan {
    pub(super) fn build(
        prefix: &PreparedPrefix,
        ratio: f64,
        requested: Option<Instant>,
    ) -> Result<Self> {
        let deadline = prefix
            .deadline(requested)?
            .ok_or_else(|| product_bad("prepared patch requires finite inherited deadline"))?;
        let root = prefix.root();
        let budget = root.budget();
        let net = root.model().net();
        if root.policy().err_lane != super::errlane::ErrLane::PerCol {
            return Err(product_bad(
                "prepared patch requires frozen per-column policy",
            ));
        }
        let relu_op = *net
            .trunk_relus
            .get(prefix.layer_count())
            .ok_or_else(|| product_bad("no next original trunk ReLU"))?;
        prefix.for_relu(relu_op)?;
        if !ratio.is_finite() || !(RATIO_MIN..=RATIO_MAX).contains(&ratio) {
            return Err(product_bad("invalid prepared crossover ratio"));
        }
        let n = relu_op
            .checked_add(1)
            .ok_or_else(|| product_bad("geometry overflow"))?;
        let geometry_bytes = n
            .checked_mul(2 * size_of::<Option<Frame>>() + size_of::<Option<ConvOp>>())
            .and_then(|v| v.checked_add(4096))
            .ok_or_else(|| product_bad("geometry admission overflow"))?;
        let scratch = n
            .checked_mul(size_of::<GeoOp>() + size_of::<Option<Shape>>())
            .and_then(|v| v.checked_add(16384))
            .ok_or_else(|| product_bad("geometry scratch overflow"))?;
        let request = WorkRequest {
            persistent_bytes: geometry_bytes,
            scratch_bytes: scratch,
            work: u64::try_from(
                n.checked_mul(n)
                    .and_then(|v| v.checked_mul(8))
                    .ok_or_else(|| product_bad("geometry work overflow"))?,
            )
            .map_err(|_| product_bad("geometry work overflow"))?,
            ..WorkRequest::default()
        };
        let source_bytes = prefix.resident_bytes(Some(deadline))?;
        prepared_peak(source_bytes, budget, &request, 0, 0)?;
        let mut geometry_lease = budget
            .lease_for(LeaseClass::Fallback, request)
            .map_err(patch_refusal)?;
        let deadline = geometry_lease
            .deadline()
            .map_or(deadline, |d| d.min(deadline));
        let check = || {
            root.deadline(Some(deadline))?;
            Ok(())
        };
        let mut ops = patch_reserved(n, Some(&check))?;
        let mut max_ci = 0;
        for (k, op) in net.ops[..n].iter().enumerate() {
            poll_patch(Some(deadline), Some(&check), k)?;
            // Preserve the existing proof accumulation-length firewall before
            // any coefficient/error arithmetic on this original tensor.
            if net.tsize[k]
                .checked_mul(8)
                .and_then(|n| n.checked_add(16))
                .is_none_or(|n| gamma_n(n) > 0.25)
            {
                return Err(product_bad("proof accumulation-length governor exceeded"));
            }
            if let TwinOp::Conv(c) = op {
                max_ci = max_ci.max(c.kernel.1);
            }
            ops.push(match op {
                TwinOp::Conv(c) => GeoOp::Conv {
                    input: c.input,
                    kernel: c.kernel,
                    stride: c.stride,
                    pads: c.pads,
                    ishape: c.ishape,
                    oshape: c.oshape,
                    transposed: c.transposed,
                },
                TwinOp::Relu { input, .. } => GeoOp::Relu { input: *input },
                TwinOp::ChannelAffine { input, .. } => GeoOp::Affine {
                    input: *input,
                    shape: None,
                },
                TwinOp::Add { lhs, rhs } => GeoOp::Add {
                    lhs: *lhs,
                    rhs: *rhs,
                },
                _ => GeoOp::Other,
            });
        }
        geometry_lease.mark_materialized().map_err(patch_refusal)?;
        let geo = prepared_geometry(&ops, &net.tsize, relu_op, ratio, &check)?;
        drop(ops);
        let mut virtual_bytes = 1024usize;
        let mut virtual_work = 0usize;
        for k in 0..relu_op {
            check()?;
            if let (TwinOp::Conv(c), Some(fin), Some(fout)) =
                (&net.ops[k], geo.contrib[k], geo.frames[k + 1])
            {
                if !fout.full {
                    let bytes = super::support_program::virtual_conv_allocation_bytes(
                        c,
                        (c.kernel.1, fin.ph, fin.pw),
                        (c.kernel.0, fout.ph, fout.pw),
                    )?;
                    virtual_bytes = virtual_bytes
                        .checked_add(bytes)
                        .ok_or_else(|| product_bad("virtual plan overflow"))?;
                    virtual_work = virtual_work
                        .checked_add(
                            bytes
                                .checked_mul(4)
                                .ok_or_else(|| product_bad("virtual work overflow"))?,
                        )
                        .ok_or_else(|| product_bad("virtual work overflow"))?;
                }
            }
        }
        let vrequest = WorkRequest {
            persistent_bytes: virtual_bytes,
            work: u64::try_from(virtual_work).map_err(|_| product_bad("virtual work overflow"))?,
            ..WorkRequest::default()
        };
        prepared_peak(
            source_bytes,
            budget,
            &vrequest,
            geometry_bytes
                .checked_add(scratch)
                .ok_or_else(|| product_bad("construction peak overflow"))?,
            0,
        )?;
        let mut virtual_lease = budget
            .lease_for(LeaseClass::Fallback, vrequest)
            .map_err(patch_refusal)?;
        let deadline = virtual_lease
            .deadline()
            .map_or(deadline, |d| d.min(deadline));
        let check = || {
            root.deadline(Some(deadline))?;
            Ok(())
        };
        let mut vops = patch_reserved(relu_op, Some(&check))?;
        for k in 0..relu_op {
            check()?;
            let vop = match (&net.ops[k], geo.contrib[k], geo.frames[k + 1]) {
                (TwinOp::Conv(c), Some(fin), Some(fout)) if !fout.full => {
                    Some(super::support_program::virtual_conv_from_original(
                        c,
                        (c.kernel.1, fin.ph, fin.pw),
                        (c.kernel.0, fout.ph, fout.pw),
                        budget,
                        Some(deadline),
                    )?)
                }
                _ => None,
            };
            vops.push(vop);
        }
        virtual_lease.mark_materialized().map_err(patch_refusal)?;
        let actual_geometry = size_of::<Self>()
            .checked_add(
                geo.frames
                    .capacity()
                    .checked_add(geo.contrib.capacity())
                    .ok_or_else(|| product_bad("geometry capacity overflow"))?
                    .checked_mul(size_of::<Option<Frame>>())
                    .ok_or_else(|| product_bad("geometry capacity overflow"))?,
            )
            .and_then(|b| {
                vops.capacity()
                    .checked_mul(size_of::<Option<ConvOp>>())
                    .and_then(|v| b.checked_add(v))
            })
            .ok_or_else(|| product_bad("geometry capacity overflow"))?;
        if actual_geometry > geometry_bytes {
            return Err(product_bad("geometry allocator exceeded admission"));
        }
        let mut actual_virtual = 0usize;
        for c in vops.iter().flatten() {
            check()?;
            actual_virtual = actual_virtual
                .checked_add(conv_capacity_bytes(c, &check)?)
                .ok_or_else(|| product_bad("virtual capacity overflow"))?;
        }
        if actual_virtual > virtual_bytes {
            return Err(product_bad("virtual allocator exceeded admission"));
        }
        check()?;
        let geometry_resident = geometry_lease.commit_persistent().map_err(patch_refusal)?;
        let virtual_resident = virtual_lease.commit_persistent().map_err(patch_refusal)?;
        Ok(Self {
            prefix: prefix.clone(),
            geo,
            vops,
            geometry_resident,
            virtual_resident,
            max_ci,
            source_resident: source_bytes,
            deadline,
        })
    }
    pub(super) fn prefix(&self) -> &PreparedPrefix {
        &self.prefix
    }
    pub(super) fn resident_bytes(&self) -> Result<usize> {
        self.geometry_resident
            .bytes()
            .checked_add(self.virtual_resident.bytes())
            .ok_or_else(|| product_bad("plan residency overflow"))
    }
    fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.prefix
            .deadline(Some(
                requested.map_or(self.deadline, |d| d.min(self.deadline)),
            ))?
            .ok_or_else(|| product_bad("missing plan deadline"))
    }
}
fn conv_capacity_bytes(c: &ConvOp, check: &(dyn Fn() -> Result<()> + Sync)) -> Result<usize> {
    let mut b = 0usize;
    for (n, size) in [
        (c.wmat.capacity(), 8),
        (c.wt.capacity(), 8),
        (c.bias.capacity(), 8),
        (c.bias_err.capacity(), 8),
        (c.gather.capacity(), size_of::<usize>()),
        (
            c.back_taps.capacity(),
            size_of::<Vec<(usize, usize, usize)>>(),
        ),
    ] {
        b = b
            .checked_add(
                n.checked_mul(size)
                    .ok_or_else(|| product_bad("virtual capacity overflow"))?,
            )
            .ok_or_else(|| product_bad("virtual capacity overflow"))?;
    }
    for (i, taps) in c.back_taps.iter().enumerate() {
        if i.is_multiple_of(256) {
            check()?;
        }
        b = b
            .checked_add(
                taps.capacity()
                    .checked_mul(size_of::<(usize, usize, usize)>())
                    .ok_or_else(|| product_bad("tap capacity overflow"))?,
            )
            .ok_or_else(|| product_bad("tap capacity overflow"))?;
    }
    Ok(b)
}
fn patch_refusal(error: crate::certify::work::WorkRefusal) -> NyError {
    if error == crate::certify::work::WorkRefusal::Deadline {
        NyError::DeadlineExceeded("margin_row prepared patch inherited deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!("margin_row prepared patch admission: {error}"))
    }
}
fn prepared_peak(
    source: usize,
    budget: &WorkBudget,
    request: &WorkRequest,
    retained_extra: usize,
    future_extra: usize,
) -> Result<()> {
    let live = budget.receipt().map_err(patch_refusal)?;
    let ledger = live
        .live_scratch_bytes
        .checked_add(live.live_persistent_bytes)
        .ok_or_else(|| product_bad("ledger peak overflow"))?;
    let visible = source
        .checked_add(retained_extra)
        .ok_or_else(|| product_bad("source peak overflow"))?;
    let required = ledger
        .max(visible)
        .checked_add(future_extra)
        .and_then(|n| n.checked_add(request.scratch_bytes))
        .and_then(|v| v.checked_add(request.persistent_bytes))
        .ok_or_else(|| product_bad("total live overflow"))?;
    if required > super::root_box::ROOT_SOURCE_BYTES_CAP {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes: super::root_box::ROOT_SOURCE_BYTES_CAP,
            site: "margin_row prepared patch source total live",
        });
    }
    Ok(())
}

/// Fused complete CPU Outward row pair, authenticated by the original immutable
/// prefix. Retained rows own that prefix and their reservation after plan drop.
pub(super) fn run_prepared_prefix_patch_input_rows(
    plan: &PreparedPatchPlan,
    targets: &[usize],
    requested: Option<Instant>,
) -> Result<InputRowPair<'static>> {
    let deadline = plan.deadline(requested)?;
    ny_core::require_f64_interval_proof_environment()?;
    if targets.is_empty() || targets.len() > 2048 {
        return Err(product_bad("prepared patch bounded tile width"));
    }
    let prefix = plan.prefix();
    let root = prefix.root();
    let budget = root.budget();
    let net = root.model().net();
    prefix.for_relu(plan.geo.relu_op)?;
    // Match the target owner's exact preallocation request. This immutable
    // source census was paid during plan construction; neither prefix nor its
    // retained model/domain/gates can change. Include the retained plan and all
    // current same-ledger owners BEFORE the target Vec is admitted or copied.
    let target_scope_bytes = targets
        .len()
        .checked_mul(size_of::<usize>())
        .and_then(|n| n.checked_add(size_of::<InputRowScope<'static>>()))
        .and_then(|n| n.checked_add(256))
        .ok_or_else(|| product_bad("target scope admission overflow"))?;
    prepared_peak(
        plan.source_resident,
        budget,
        &WorkRequest {
            persistent_bytes: target_scope_bytes,
            ..WorkRequest::default()
        },
        plan.resident_bytes()?,
        0,
    )?;
    let scope = InputRowScope::from_prepared(
        prefix,
        OriginalRowDemand {
            relu_op: plan.geo.relu_op,
            targets,
        },
        deadline,
    )?;
    let deadline = deadline.min(scope.deadline());
    let p = targets
        .len()
        .checked_mul(2)
        .ok_or_else(|| product_bad("fused column overflow"))?;
    let mut request = prepared_patch_request(plan, p)?;
    if prefix.layer_count() != 0 {
        let input_rows = plan
            .geo
            .frames
            .first()
            .copied()
            .flatten()
            .ok_or_else(|| product_bad("missing alternate input frame"))?
            .rows();
        let alternative_bytes = input_rows
            .checked_mul(targets.len())
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| targets.len().checked_mul(32).and_then(|b| n.checked_add(b)))
            .and_then(|n| n.checked_add(size_of::<PatchWalk>() + 1024))
            .ok_or_else(|| product_bad("alternate patch payload overflow"))?;
        request.scratch_bytes = request
            .scratch_bytes
            .checked_add(alternative_bytes)
            .ok_or_else(|| product_bad("alternate patch overlap overflow"))?;
        request.work = request
            .work
            .checked_mul(2)
            .and_then(|n| {
                input_rows
                    .checked_mul(targets.len())
                    .and_then(|n| n.checked_mul(1024))
                    .and_then(|n| u64::try_from(n).ok())
                    .and_then(|extra| n.checked_add(extra))
            })
            .ok_or_else(|| product_bad("alternate patch work overflow"))?;
    }
    let frame_bytes = prefix
        .layer_count()
        .checked_mul(size_of::<&super::root::LayerGates>())
        .and_then(|n| n.checked_add(size_of::<PreparedPrefixFrame<'_>>()))
        .ok_or_else(|| product_bad("frame pointer overflow"))?;
    prepared_peak(
        scope.resident_floor(deadline)?,
        budget,
        &request,
        plan.resident_bytes()?,
        frame_bytes,
    )?;
    let mut lease = budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(patch_refusal)?;
    let deadline = lease.deadline().map_or(deadline, |d| d.min(deadline));
    let frame = prefix.frame(Some(deadline))?;
    for result in rayon::broadcast(|_| {
        frame.checkpoint()?;
        ny_core::require_f64_interval_proof_environment()
    }) {
        result?;
    }
    let check = || frame.checkpoint();
    let mut columns = patch_reserved(p, Some(&check))?;
    let mut lowers = patch_reserved(p, Some(&check))?;
    for upper in [false, true] {
        for &target in scope.targets() {
            check()?;
            columns.push(target);
            lowers.push(!upper);
        }
    }
    lease.mark_materialized().map_err(patch_refusal)?;
    let kernel = PatchKernel {
        source_net: net,
        geo: &plan.geo,
        vops: &plan.vops,
    };
    let mut walk = kernel
        .walk_inner(
            net,
            &frame,
            &columns,
            &lowers,
            None,
            &GpuConvRoute::Off,
            Some(deadline),
        )
        .ok_or_else(|| product_bad("prepared patch structural refusal"))??;
    check()?;
    super::errlane::note_pass(super::errlane::PassLane::PerCol, targets.len(), None);
    super::errlane::note_pass(super::errlane::PassLane::PerCol, targets.len(), None);
    if prefix.layer_count() != 0 {
        let zero_frame = ZeroCrossingLowerFrame(&frame);
        // Reuse the already-paid first half of the fused identity seed map.
        // Only an extra LOWER walk is needed; the original upper planes and
        // source geometry remain unchanged.
        let mut alternative = kernel
            .walk_inner(
                net,
                &zero_frame,
                &columns[..targets.len()],
                &lowers[..targets.len()],
                None,
                &GpuConvRoute::Off,
                Some(deadline),
            )
            .ok_or_else(|| product_bad("alternate patch structural refusal"))??;
        check()?;
        super::errlane::note_pass(super::errlane::PassLane::PerCol, targets.len(), None);
        select_prepared_patch_lower_columns(&frame, &mut walk, &mut alternative, deadline)?;
        drop(alternative);
        check()?;
    }
    drop(columns);
    drop(lowers);
    drop(frame);
    root.deadline(Some(deadline))?;
    let resident = lease.commit_persistent().map_err(patch_refusal)?;
    let seed =
        plan.geo.frames[plan.geo.seed_tensor].ok_or_else(|| product_bad("missing seed frame"))?;
    InputRowPair::from_prepared_patch(
        scope,
        walk,
        seed.shape(),
        PatchInputRowsAuthority(()),
        resident,
        deadline,
    )
}

fn select_prepared_patch_lower_columns(
    frame: &dyn FrozenGateFrame,
    baseline: &mut PatchWalk,
    alternative: &mut PatchWalk,
    deadline: Instant,
) -> Result<()> {
    let columns = alternative.a.ncols();
    let local_rows = baseline.a.nrows();
    if columns.checked_mul(2) != Some(baseline.a.ncols())
        || alternative.a.nrows() != local_rows
        || baseline.frame != alternative.frame
        || baseline.pos.len() != baseline.a.ncols()
        || alternative.pos.len() != columns
        || baseline.pos[..columns] != alternative.pos[..]
        || baseline.b.len() != baseline.a.ncols()
        || baseline.eb.len() != baseline.a.ncols()
        || alternative.b.len() != columns
        || alternative.eb.len() != columns
        || baseline
            .e
            .as_ref()
            .is_none_or(|e| e.dim() != baseline.a.dim())
        || alternative
            .e
            .as_ref()
            .is_none_or(|e| e.dim() != alternative.a.dim())
    {
        return Err(product_bad("alternate patch mapping/planes changed"));
    }
    let check = || {
        frame.checkpoint()?;
        check_optional_deadline(Some(deadline))
    };
    for column in 0..columns {
        check()?;
        let geometry = col_geo(&baseline.frame, baseline.pos[column]);
        let bound = |walk: &PatchWalk| {
            prepared_lower_bound_values(
                frame,
                walk.b[column],
                walk.eb[column],
                (0..local_rows).map(|row| {
                    let (ch, iy, ix) = split_row(&walk.frame, row);
                    abs_index(&walk.frame, &geometry, ch, iy, ix)
                        .map(|id| {
                            (
                                id,
                                walk.a[[row, column]],
                                walk.e.as_ref().expect("checked")[[row, column]],
                            )
                        })
                        // Virtual padding is an exact zero extension. Yield its
                        // zero term so the common evaluator polls EVERY bounded
                        // local slab, including long stretches of padding.
                        .unwrap_or((0, 0.0, 0.0))
                }),
                deadline,
            )
        };
        if bound(alternative)? > bound(baseline)? {
            let baseline_error = baseline.e.as_mut().expect("checked");
            let alternative_error = alternative.e.as_mut().expect("checked");
            for row in 0..local_rows {
                poll_patch(Some(deadline), Some(&check), row)?;
                std::mem::swap(
                    &mut baseline.a[[row, column]],
                    &mut alternative.a[[row, column]],
                );
                std::mem::swap(
                    &mut baseline_error[[row, column]],
                    &mut alternative_error[[row, column]],
                );
            }
            std::mem::swap(&mut baseline.b[column], &mut alternative.b[column]);
            std::mem::swap(&mut baseline.eb[column], &mut alternative.eb[column]);
        }
    }
    check()
}
fn prepared_patch_request(plan: &PreparedPatchPlan, p: usize) -> Result<WorkRequest> {
    let geo = &plan.geo;
    let f0 = geo
        .frames
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| product_bad("missing input frame"))?;
    let persistent = f0
        .rows()
        .checked_mul(p)
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| p.checked_mul(32).and_then(|m| n.checked_add(m)))
        .and_then(|n| n.checked_add(super::input_rows::INPUT_ROWS_OWNER_BYTES))
        .ok_or_else(|| product_bad("row payload overflow"))?;
    let rows = geo
        .rows_col_patch
        .checked_add(
            geo.virt_rows_max
                .checked_mul(6)
                .ok_or_else(|| product_bad("frontier overflow"))?,
        )
        .ok_or_else(|| product_bad("frontier overflow"))?;
    let matrices = rows
        .checked_mul(p)
        .and_then(|n| n.checked_mul(16))
        .ok_or_else(|| product_bad("frontier overflow"))?;
    let workers = plan
        .max_ci
        .checked_mul(p)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(4096))
        .and_then(|n| n.checked_mul(rayon::current_num_threads()))
        .ok_or_else(|| product_bad("worker scratch overflow"))?;
    let metadata = geo
        .relu_op
        .checked_add(1)
        .and_then(|n| n.checked_mul(2048))
        .and_then(|n| p.checked_mul(512).and_then(|m| n.checked_add(m)))
        .and_then(|n| n.checked_add(16384))
        .ok_or_else(|| product_bad("metadata overflow"))?;
    let scratch = matrices
        .checked_add(workers)
        .and_then(|n| n.checked_add(metadata))
        .ok_or_else(|| product_bad("scratch overflow"))?;
    let work = geo
        .macs_col_patch
        .checked_mul(p as u64)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| {
            (geo.rows_col_patch as u64)
                .checked_mul(p as u64)
                .and_then(|m| m.checked_mul(256))
                .and_then(|m| n.checked_add(m))
        })
        .ok_or_else(|| product_bad("patch work overflow"))?;
    Ok(WorkRequest {
        scratch_bytes: scratch,
        persistent_bytes: persistent,
        work,
        ..WorkRequest::default()
    })
}

#[cfg(test)]
#[path = "patch_prepared_products_tests.rs"]
mod prepared_products_tests;
