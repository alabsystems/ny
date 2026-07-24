// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact support certification for a stored 2-ReLU facet normal.
//!
//! Facet enumeration is deliberately outside this module: an enumerated normal
//! is only a proposal.  For a proposed, finite `f32` normal `a`, this checker
//! independently maximizes
//!
//! `a_x1 x1 + a_x2 x2 + a_y1 ReLU(x1) + a_y2 ReLU(x2)`
//!
//! over the octahedron `P`.  It cuts `P` into the four closed ReLU orthants.  On
//! each cell ReLU is affine, so the maximum is attained at a cell vertex.  Every
//! constraint, pairwise intersection, feasibility comparison, and objective dot
//! product is evaluated as an exact [`BigRational`].  In particular, this path
//! does not consume the tolerance-filtered or tolerance-deduplicated vertices
//! used to *propose* normals.
//!
//! The exact support is converted to `f32` only once.  The candidate conversion
//! is verified by converting it back to an exact rational; if it is inward, the
//! checker steps one ULP toward `+inf` and verifies again.  Any non-finite input,
//! empty/inconsistent `P`, unrepresentable finite upper bound, or conversion
//! pathology fails closed by returning `None` (the caller must drop the facet).

use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use ny_tensor::next_up_f32;

use super::{Facet, Octahedron2};

#[derive(Clone, Debug)]
struct ExactRow {
    // Coefficients are from {-1, 0, 1}; keeping them integral avoids introducing
    // any arithmetic approximation while enumerating the 2D cell vertices.
    a: [i8; 2],
    b: BigRational,
}

#[derive(Clone, Debug)]
struct ExactLiftedVertex {
    w: [BigRational; 4],
}

/// Reusable exact support checker for one sound octahedron `P`.
///
/// Construction enumerates all vertices of `P` intersected with each of the
/// four closed ReLU orthants using exact dyadic rational arithmetic.  Duplicate
/// boundary vertices are intentionally retained: correctness is independent of
/// tolerance-based deduplication, and duplicates merely repeat an exact dot
/// product.  The object has no connection to any authority gate.
#[derive(Clone, Debug)]
pub struct ExactRelu2Support {
    vertices: Vec<ExactLiftedVertex>,
    orthant_vertex_counts: [usize; 4],
}

impl ExactRelu2Support {
    /// Build an exact checker for `P`, failing closed on non-finite or
    /// inconsistent bounds.
    pub fn new(p: &Octahedron2) -> Option<Self> {
        let l1 = rat_f64(p.l1)?;
        let u1 = rat_f64(p.u1)?;
        let l2 = rat_f64(p.l2)?;
        let u2 = rat_f64(p.u2)?;
        let s_lo = rat_f64(p.s_lo)?;
        let s_hi = rat_f64(p.s_hi)?;
        let d_lo = rat_f64(p.d_lo)?;
        let d_hi = rat_f64(p.d_hi)?;

        let base = [
            ExactRow { a: [1, 0], b: u1 },
            ExactRow { a: [-1, 0], b: -l1 },
            ExactRow { a: [0, 1], b: u2 },
            ExactRow { a: [0, -1], b: -l2 },
            ExactRow { a: [1, 1], b: s_hi },
            ExactRow {
                a: [-1, -1],
                b: -s_lo,
            },
            ExactRow {
                a: [1, -1],
                b: d_hi,
            },
            ExactRow {
                a: [-1, 1],
                b: -d_lo,
            },
        ];

        let mut vertices = Vec::new();
        let mut orthant_vertex_counts = [0usize; 4];
        for (orthant, &(s1, s2)) in [(1i8, 1i8), (1, -1), (-1, 1), (-1, -1)].iter().enumerate() {
            let mut rows = base.to_vec();
            // s_i*x_i >= 0  <=>  -s_i*x_i <= 0.
            rows.push(ExactRow {
                a: [-s1, 0],
                b: BigRational::zero(),
            });
            rows.push(ExactRow {
                a: [0, -s2],
                b: BigRational::zero(),
            });

            let before = vertices.len();
            for i in 0..rows.len() {
                for j in (i + 1)..rows.len() {
                    let det = i16::from(rows[i].a[0]) * i16::from(rows[j].a[1])
                        - i16::from(rows[i].a[1]) * i16::from(rows[j].a[0]);
                    if det == 0 {
                        continue;
                    }

                    // Cramer's rule over exact rationals.
                    let x1 = (scale(&rows[i].b, rows[j].a[1]) - scale(&rows[j].b, rows[i].a[1]))
                        / rat_i16(det);
                    let x2 = (scale(&rows[j].b, rows[i].a[0]) - scale(&rows[i].b, rows[j].a[0]))
                        / rat_i16(det);

                    if rows.iter().all(|row| row_holds(row, &x1, &x2)) {
                        let zero = BigRational::zero();
                        vertices.push(ExactLiftedVertex {
                            w: [
                                x1.clone(),
                                x2.clone(),
                                if s1 > 0 { x1 } else { zero.clone() },
                                if s2 > 0 { x2 } else { zero },
                            ],
                        });
                    }
                }
            }
            orthant_vertex_counts[orthant] = vertices.len() - before;
        }

        // A finite axis-bounded nonempty 2D polytope (including a lower-
        // dimensional one encoded by opposite inequalities) has a vertex.  No
        // exact cell vertex therefore means P is empty/inconsistent.
        if vertices.is_empty() {
            return None;
        }
        Some(Self {
            vertices,
            orthant_vertex_counts,
        })
    }

    /// Certify one stored `f32` normal and return its exact-support half-space.
    ///
    /// The returned `Facet` satisfies `a*w <= b` for every
    /// `w=(x,ReLU(x))`, `x in P`.  `None` means "drop this proposal", never
    /// "accept without a certificate".
    pub fn certify_normal(&self, a: [f32; 4]) -> Option<Facet> {
        let a_exact = [
            rat_f32(a[0])?,
            rat_f32(a[1])?,
            rat_f32(a[2])?,
            rat_f32(a[3])?,
        ];
        let mut support: Option<BigRational> = None;
        for vertex in &self.vertices {
            let mut value = BigRational::zero();
            for (coefficient, coordinate) in a_exact.iter().zip(&vertex.w) {
                value += coefficient * coordinate;
            }
            if support.as_ref().is_none_or(|best| value > *best) {
                support = Some(value);
            }
        }
        let b = rational_to_f32_up(&support?)?;
        Some(Facet { a, b })
    }

    /// Number of exact pair-intersection vertices retained from each orthant,
    /// in `(++), (+-), (-+), (--)` order.  This is diagnostic metadata only;
    /// duplicated boundary vertices are deliberately included.
    pub fn orthant_vertex_counts(&self) -> [usize; 4] {
        self.orthant_vertex_counts
    }
}

fn rat_f64(value: f64) -> Option<BigRational> {
    BigRational::from_float(value)
}

fn rat_f32(value: f32) -> Option<BigRational> {
    value
        .is_finite()
        .then(|| rat_f64(f64::from(value)))
        .flatten()
}

fn rat_i16(value: i16) -> BigRational {
    // Every i16 is exactly representable as f64.
    rat_f64(f64::from(value)).expect("finite exact integer")
}

fn scale(value: &BigRational, coefficient: i8) -> BigRational {
    match coefficient {
        -1 => -value,
        0 => BigRational::zero(),
        1 => value.clone(),
        _ => value * rat_f64(f64::from(coefficient)).expect("finite exact coefficient"),
    }
}

fn row_holds(row: &ExactRow, x1: &BigRational, x2: &BigRational) -> bool {
    scale(x1, row.a[0]) + scale(x2, row.a[1]) <= row.b
}

/// Directed conversion verified in exact arithmetic.
fn rational_to_f32_up(value: &BigRational) -> Option<f32> {
    let max = rat_f32(f32::MAX).expect("f32::MAX is finite");
    if value > &max {
        return None;
    }
    if value <= &-max {
        // The least finite f32 is already an outward upper bound.
        return Some(-f32::MAX);
    }

    let mut candidate = value.to_f32()?;
    if !candidate.is_finite() {
        return None;
    }
    // `to_f32` is merely a proposal.  Only the exact round-trip comparison
    // authorizes the returned value.  Correct nearest rounding needs at most one
    // successor; the small cap turns any library pathology into a dropped facet.
    for _ in 0..=2 {
        if rat_f32(candidate).is_some_and(|encoded| encoded >= *value) {
            return Some(candidate);
        }
        candidate = next_up_f32(candidate);
        if !candidate.is_finite() {
            return None;
        }
    }
    None
}
