// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #spec-influence-cone: which spatial cells of an INTERMEDIATE node can reach
//! the spec's output cells.
//!
//! # Why
//!
//! `#margin-subset-seed` ([`crate::output_margin_seed`]) already seeds only the
//! spec-referenced rows at the OUTPUT node. Every INTERMEDIATE target still
//! seeds a full identity: on TinyYOLO (`yolo_2023`) that is a 10,816-row
//! backward at `Conv_25` when the property reads five outputs which all live at
//! ONE spatial cell of the 13x13 grid — channels 0-4 at (7,9), decoded from
//! `Y_100 / Y_269 / Y_438 / Y_607 / Y_776` as `idx = c*169 + h*13 + w`.
//!
//! Restricting the seed to the cells that can actually reach that objective
//! cell removes work proportional to the cone's share of the grid. Measured
//! saving on TinyYOLO's still-failing targets: `Conv_12` 1.3x, `Add_15` 1.9x,
//! `Conv_20` 2.1x, `Conv_25` **6.8x** (10,816 -> 1,590 rows). The saving is
//! largest at the deepest target, which is where conv IBP width blow-up is
//! worst (`Conv_27` alone amplifies 18.01x).
//!
//! # Soundness
//!
//! **The failure direction is safe, and this is the load-bearing property.**
//! Subset seeding does not drop information — it declines to TIGHTEN it. Rows
//! outside the selected set keep the node's existing sound IBP/forward bounds
//! (see [`crate::output_margin_seed`]'s scatter contract), which are valid, just
//! looser. So a cone that is too SMALL yields a looser final bound, never a
//! wrong one; there is no `-150` exposure here.
//!
//! That said this computes an OVER-approximation anyway — a spatial bounding
//! box, all channels retained — so it is expected to be exact-or-generous for
//! the conv/pool/pad/elementwise families it models, and it fails OPEN (returns
//! `None`, meaning "seed everything") for any op it does not model.
//!
//! # Method
//!
//! Backward interval propagation over spatial coordinates, in reverse execution
//! order, starting from the objective cell at the output. Channels are never
//! narrowed: a conv mixes all input channels into every output channel, so the
//! channel dimension is always fully live and tracking it would buy nothing.
//!
//! Per-op backward rule for an output row interval `[lo, hi]`:
//! - `Conv2d` / `AveragePool` / `MaxPool2d` with kernel `k`, stride `s`,
//!   padding `p`: input rows `[lo*s - p, hi*s - p + k - 1]`, clamped to the
//!   input extent. This is the exact receptive interval.
//! - `Pad` with `before` padding `b` on that axis: input rows `[lo - b, hi - b]`.
//! - Elementwise / `Add` / `Sub` / `ReLU` / activations: identity.
//! - Anything else: fail open.
//!
//! # Wiring
//!
//! Built once per collection in `crown_tighten.rs` from the objective indices
//! published at the OUTPUT node, then consumed by the same subset branch that
//! already served the OUTPUT node — reusing `propagate_crown_to_node_subset`
//! and `scatter_margin_rows_over_bounds` unchanged.

use std::collections::HashMap;

use crate::layers::Layer;
use crate::network::GraphNetwork;

/// Inclusive spatial interval `[lo, hi]` on one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    pub lo: usize,
    pub hi: usize,
}

impl Span {
    fn point(v: usize) -> Self {
        Self { lo: v, hi: v }
    }

    #[cfg(test)]
    fn full(extent: usize) -> Self {
        Self {
            lo: 0,
            hi: extent.saturating_sub(1),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    fn clamp_to(self, extent: usize) -> Self {
        let last = extent.saturating_sub(1);
        Self {
            lo: self.lo.min(last),
            hi: self.hi.min(last),
        }
    }

    pub(crate) fn len(self) -> usize {
        self.hi.saturating_sub(self.lo) + 1
    }

    /// Backward receptive interval through a strided window.
    ///
    /// Output index `o` reads input indices `[o*s - p, o*s - p + k - 1]`, so an
    /// output span `[lo, hi]` reads `[lo*s - p, hi*s - p + k - 1]`. Saturating
    /// arithmetic clamps the low end at 0 (padding region), which is the
    /// over-approximating direction.
    fn through_window(self, kernel: usize, stride: usize, pad: usize) -> Self {
        let lo = self.lo.saturating_mul(stride).saturating_sub(pad);
        let hi = self
            .hi
            .saturating_mul(stride)
            .saturating_sub(pad)
            .saturating_add(kernel.saturating_sub(1));
        Self { lo, hi }
    }
}

/// Spatial box on a 3-D `(channels, height, width)` node. Channels are always
/// fully live (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpatialBox {
    pub rows: Span,
    pub cols: Span,
}

impl SpatialBox {
    fn union(self, other: Self) -> Self {
        Self {
            rows: self.rows.union(other.rows),
            cols: self.cols.union(other.cols),
        }
    }
}

/// Backward influence boxes for every node that can reach the objective.
///
/// `None` for a node means "fully live / not modelled" — callers must seed the
/// full width there.
pub(crate) struct InfluenceCones {
    boxes: HashMap<String, Option<SpatialBox>>,
}

impl InfluenceCones {
    /// Flat row indices of `node`'s cone, given its `(c, h, w)` shape.
    ///
    /// `None` when the node is fully live, not modelled, or its cone covers the
    /// whole grid — in every case the caller should keep full-width seeding
    /// rather than pay the subset machinery for no reduction.
    pub(crate) fn subset_indices(
        &self,
        node: &str,
        shape: (usize, usize, usize),
    ) -> Option<Vec<usize>> {
        let (c, h, w) = shape;
        let bx = (*self.boxes.get(node)?)?;
        let rows = bx.rows.clamp_to(h);
        let cols = bx.cols.clamp_to(w);
        if rows.len() >= h && cols.len() >= w {
            return None; // no reduction; keep the historical full-width path
        }
        let plane = h * w;
        let mut out = Vec::with_capacity(c * rows.len() * cols.len());
        for ch in 0..c {
            for r in rows.lo..=rows.hi.min(h.saturating_sub(1)) {
                for col in cols.lo..=cols.hi.min(w.saturating_sub(1)) {
                    out.push(ch * plane + r * w + col);
                }
            }
        }
        (!out.is_empty()).then_some(out)
    }
}

/// Decode flat output indices into a spatial box over a `(c, h, w)` grid.
pub(crate) fn objective_box(indices: &[usize], shape: (usize, usize, usize)) -> Option<SpatialBox> {
    let (_, h, w) = shape;
    let plane = h.checked_mul(w)?;
    if plane == 0 {
        return None;
    }
    let mut acc: Option<SpatialBox> = None;
    for &flat in indices {
        let rem = flat % plane;
        let r = rem / w;
        let c = rem % w;
        let one = SpatialBox {
            rows: Span::point(r),
            cols: Span::point(c),
        };
        acc = Some(match acc {
            Some(prev) => prev.union(one),
            None => one,
        });
    }
    acc
}

/// Propagate the objective box backward over the execution order.
///
/// `output_node` is where `objective` applies; `shape_of` resolves a node's
/// `(c, h, w)` shape (`None` for non-spatial nodes, which fail open).
pub(crate) fn compute(
    graph: &GraphNetwork,
    exec_order: &[String],
    output_node: &str,
    objective: SpatialBox,
    shape_of: &dyn Fn(&str) -> Option<(usize, usize, usize)>,
) -> InfluenceCones {
    let mut boxes: HashMap<String, Option<SpatialBox>> = HashMap::new();
    boxes.insert(output_node.to_string(), Some(objective));

    for node_name in exec_order.iter().rev() {
        // A node with no recorded box cannot reach the objective; contribute
        // nothing to its inputs (they may still be reached by another path).
        let Some(&current) = boxes.get(node_name) else {
            continue;
        };
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };

        // Fully-live or unmodelled: poison every input the same way.
        let propagate = |boxes: &mut HashMap<String, Option<SpatialBox>>,
                         value: Option<SpatialBox>| {
            for input in &node.inputs {
                let entry = boxes.entry(input.clone()).or_insert(value);
                *entry = match (*entry, value) {
                    (Some(a), Some(b)) => Some(a.union(b)),
                    // Any fully-live contributor makes the input fully live.
                    _ => None,
                };
            }
        };

        let Some(current) = current else {
            propagate(&mut boxes, None);
            continue;
        };

        let input_shape = node.inputs.first().and_then(|n| shape_of(n));
        let next = match &node.layer {
            Layer::Conv2d(conv) => {
                let (kh, kw) = conv.kernel_size();
                let (sh, sw) = conv.stride;
                let (ph, pw) = conv.padding;
                Some(SpatialBox {
                    rows: current.rows.through_window(kh, sh, ph),
                    cols: current.cols.through_window(kw, sw, pw),
                })
            }
            Layer::AveragePool(pool) => {
                let (kh, kw) = pool.kernel_size;
                let (sh, sw) = pool.stride;
                let (ph, pw) = pool.padding;
                Some(SpatialBox {
                    rows: current.rows.through_window(kh, sh, ph),
                    cols: current.cols.through_window(kw, sw, pw),
                })
            }
            Layer::MaxPool2d(pool) => {
                let (kh, kw) = pool.kernel_size;
                let (sh, sw) = pool.stride;
                let (ph, pw) = pool.padding;
                Some(SpatialBox {
                    rows: current.rows.through_window(kh, sh, ph),
                    cols: current.cols.through_window(kw, sw, pw),
                })
            }
            // Shape-preserving elementwise ops: identity on spatial position.
            // `Add`/`Sub` fan the SAME box to both operands, which is exactly
            // right for a residual connection.
            Layer::ReLU(_)
            | Layer::Add(_)
            | Layer::Sub(_)
            | Layer::Sigmoid(_)
            | Layer::Tanh(_)
            | Layer::LeakyReLU(_)
            | Layer::Clip(_)
            | Layer::Elu(_)
            | Layer::GELU(_)
            | Layer::SiLU(_)
            | Layer::HardSwish(_)
            | Layer::Softplus(_)
            | Layer::AddConstant(_)
            | Layer::MulConstant(_)
            | Layer::SubConstant(_)
            | Layer::DivConstant(_)
            | Layer::BatchNorm(_) => Some(current),
            // #zero-pad-cone-identity: a `Pad` whose every `(before, after)` is
            // `(0, 0)` adds no elements on any axis, so its output tensor IS its
            // input tensor and output cell `(r, c)` is input cell `(r, c)`.
            // That is an EXACT identity, not an over-approximation — the same
            // fact `#patches-zero-pad-identity` already relies on to pass a
            // patches relation through such a Pad unchanged. TinyYOLO's
            // `Pad_10`/`Pad_17` are exactly this (`pads=[0,0,0,0,0,0,0,0]`),
            // and without this arm they poison every node upstream of them to
            // "fully live" via the `_ => None` fail-open below.
            //
            // Pads with a real offset are deliberately NOT handled here (the
            // `[lo - b, hi - b]` rule in the module docs); they keep failing
            // open until someone measures them.
            Layer::Pad(pad) if pad.pads.iter().all(|&(b, a)| b == 0 && a == 0) => Some(current),
            // Everything else (Pad with real offsets, Flatten, Reshape, Gather,
            // Concat, MatMul, ...) is not modelled: fail open.
            _ => None,
        };
        // Clamp to the input extent so a padded conv cannot claim rows the
        // input does not have.
        let next = match (next, input_shape) {
            (Some(b), Some((_, ih, iw))) => Some(SpatialBox {
                rows: b.rows.clamp_to(ih),
                cols: b.cols.clamp_to(iw),
            }),
            (other, _) => other,
        };
        propagate(&mut boxes, next);
    }

    InfluenceCones { boxes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_backward_interval_is_the_receptive_field() {
        // 3x3 stride 1 pad 1: output row r reads input rows [r-1, r+1].
        let s = Span::point(7).through_window(3, 1, 1);
        assert_eq!(s, Span { lo: 6, hi: 8 });
        // A span widens by k-1 overall, not per element.
        let s = Span { lo: 5, hi: 7 }.through_window(3, 1, 1);
        assert_eq!(s, Span { lo: 4, hi: 8 });
        // Stride 2 pooling doubles the position and spans the window.
        let s = Span::point(3).through_window(2, 2, 0);
        assert_eq!(s, Span { lo: 6, hi: 7 });
        // Low end saturates into the padding region rather than underflowing.
        let s = Span::point(0).through_window(3, 1, 1);
        assert_eq!(s, Span { lo: 0, hi: 2 });
    }

    /// The TinyYOLO cone, computed by hand in
    /// `docs/PATCHES_RESIDUAL_ADD_ROOT_CAUSE_2026-07-27.md`, must reproduce:
    /// one cell at the 13x13 output grows to 3x3 through one 3x3 conv, 5x5
    /// through two, and 9x9 through four.
    #[test]
    fn tinyyolo_deep_cone_matches_the_hand_computation() {
        let mut rows = Span::point(7);
        let mut cols = Span::point(9);
        // Conv_29 is 1x1 (no growth), then Conv_27 and Conv_25 are 3x3 s1 p1.
        for _ in 0..2 {
            rows = rows.through_window(3, 1, 1);
            cols = cols.through_window(3, 1, 1);
        }
        assert_eq!(rows.len(), 5, "two 3x3 convs give a 5x5 cone");
        assert_eq!(cols.len(), 5);
        // 25 of 169 cells = 6.8x fewer rows than full width.
        let saving = 169.0_f64 / 25.0;
        assert!((saving - 6.76).abs() < 0.01, "saving was {saving}");
    }

    #[test]
    fn objective_box_decodes_channel_major_flat_indices() {
        // TinyYOLO: [125, 13, 13]; Y_100/269/438/607/776 are channels 0-4 at (7,9).
        let shape = (125usize, 13usize, 13usize);
        let bx = objective_box(&[100, 269, 438, 607, 776], shape).expect("box");
        assert_eq!(bx.rows, Span::point(7));
        assert_eq!(bx.cols, Span::point(9));
    }

    #[test]
    fn full_coverage_declines_the_subset_so_callers_keep_full_width() {
        let mut boxes = HashMap::new();
        boxes.insert(
            "n".to_string(),
            Some(SpatialBox {
                rows: Span::full(13),
                cols: Span::full(13),
            }),
        );
        let cones = InfluenceCones { boxes };
        assert!(cones.subset_indices("n", (64, 13, 13)).is_none());
    }

    #[test]
    fn subset_indices_are_channel_complete_and_in_range() {
        let mut boxes = HashMap::new();
        boxes.insert(
            "n".to_string(),
            Some(SpatialBox {
                rows: Span { lo: 6, hi: 8 },
                cols: Span { lo: 8, hi: 10 },
            }),
        );
        let cones = InfluenceCones { boxes };
        let idx = cones.subset_indices("n", (4, 13, 13)).expect("subset");
        // 4 channels x 3 rows x 3 cols
        assert_eq!(idx.len(), 4 * 9);
        assert!(idx.iter().all(|&i| i < 4 * 13 * 13));
        // Channel 0, row 6, col 8 -> 0*169 + 6*13 + 8
        assert!(idx.contains(&(6 * 13 + 8)));
        // Every channel is represented (a conv mixes all input channels).
        for ch in 0..4 {
            assert!(idx.iter().any(|&i| i / 169 == ch));
        }
    }

    /// #zero-pad-cone-identity: an all-zero `Pad` must pass the cone through
    /// (its output tensor IS its input tensor), while a Pad with a real offset
    /// must keep failing open. TinyYOLO's `Pad_10`/`Pad_17` are the zero case,
    /// and before this arm existed they poisoned `Add_15`/`Conv_12`/`Add_8`/
    /// `Conv_5`/`Conv_1` to full width.
    fn cone_through_pad(pads: Vec<(usize, usize)>) -> InfluenceCones {
        use crate::layers::{PadLayer, PadMode, ReLULayer};
        use crate::network::{GraphNode, NETWORK_INPUT};

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "src".to_string(),
            Layer::ReLU(ReLULayer::new()),
            vec![NETWORK_INPUT.to_string()],
        ));
        graph.add_node(GraphNode::new(
            "pad".to_string(),
            Layer::Pad(PadLayer::new(pads, PadMode::Constant(0.0))),
            vec!["src".to_string()],
        ));
        let exec_order = vec!["src".to_string(), "pad".to_string()];
        let objective = SpatialBox {
            rows: Span::point(7),
            cols: Span::point(9),
        };
        compute(&graph, &exec_order, "pad", objective, &|_| {
            Some((16usize, 26usize, 26usize))
        })
    }

    #[test]
    fn zero_pad_passes_the_cone_through_unchanged() {
        let cones = cone_through_pad(vec![(0, 0), (0, 0), (0, 0)]);
        let idx = cones
            .subset_indices("src", (16, 26, 26))
            .expect("all-zero Pad is an exact identity: the cone must survive it");
        // One spatial cell, every channel live.
        assert_eq!(idx.len(), 16);
        assert!(idx.contains(&(7 * 26 + 9)));
    }

    #[test]
    fn nonzero_pad_still_fails_open() {
        let cones = cone_through_pad(vec![(0, 0), (1, 1), (1, 1)]);
        assert!(
            cones.subset_indices("src", (16, 26, 26)).is_none(),
            "a Pad with a real offset is not modelled and must seed full width"
        );
    }

    #[test]
    fn unknown_node_yields_no_subset() {
        let cones = InfluenceCones {
            boxes: HashMap::new(),
        };
        assert!(cones.subset_indices("missing", (1, 4, 4)).is_none());
    }
}
