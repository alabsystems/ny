// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Checked primitive slopes, intercepts, and exact McCormick constants.

use super::{refusal, sqrt_interval, square, I};
use ny_core::Result;

#[derive(Clone, Copy)]
pub(super) struct Line {
    pub(super) slope: f64,
    pub(super) intercept: f64,
}
#[derive(Clone, Copy)]
pub(super) struct Envelope {
    pub(super) lower: Line,
    pub(super) upper: Line,
    pub(super) range: I,
}
#[derive(Clone, Copy)]
pub(super) enum Primitive {
    Square,
    Sqrt,
    Reciprocal,
}

fn evaluate(kind: Primitive, x: I) -> Option<I> {
    match kind {
        Primitive::Square => Some(square(x)),
        Primitive::Sqrt => sqrt_interval(x),
        Primitive::Reciprocal => Some(I::point(1.0).div_positive(x)),
    }
}

fn endpoint_line(kind: Primitive, domain: I, slope: f64, lower: bool) -> Result<Line> {
    if !slope.is_finite() {
        return Err(refusal("primitive slope nonfinite"));
    }
    let at = |x: I| -> Result<I> {
        Ok(evaluate(kind, x)
            .ok_or_else(|| refusal("primitive endpoint check refused"))?
            .sub(I::point(slope).mul(x)))
    };
    let mut candidates = vec![at(I::point(domain.lo))?, at(I::point(domain.hi))?];
    // Interior extrema are mandatory for square/reciprocal lower and sqrt
    // upper lines. Including an outside critical point is conservative too.
    let critical = match kind {
        Primitive::Square if lower => Some(I::point(slope).mul(I::point(0.5))),
        Primitive::Sqrt if !lower && slope > 0.0 => Some(
            I::point(1.0).div_positive(I::point(4.0).mul(I::point(slope)).mul(I::point(slope))),
        ),
        Primitive::Reciprocal if lower && slope < 0.0 => {
            sqrt_interval(I::point(1.0).div_positive(I::point(-slope)))
        }
        _ => None,
    };
    if let Some(x) = critical {
        if x.finite() && x.hi >= domain.lo && x.lo <= domain.hi {
            candidates.push(at(I::new(x.lo.max(domain.lo), x.hi.min(domain.hi)))?);
        }
    }
    let intercept = if lower {
        candidates
            .iter()
            .map(|x| x.lo)
            .fold(f64::INFINITY, f64::min)
    } else {
        candidates
            .iter()
            .map(|x| x.hi)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    if !intercept.is_finite() {
        return Err(refusal("primitive intercept nonfinite"));
    }
    Ok(Line { slope, intercept })
}

pub(super) fn envelope(kind: Primitive, domain: I) -> Result<Envelope> {
    if !domain.finite() {
        return Err(refusal("primitive finite domain required"));
    }
    let center = domain.midpoint();
    let range = evaluate(kind, domain).ok_or_else(|| refusal("primitive range refused"))?;
    let (lower_slope, upper_slope) = match kind {
        Primitive::Square => {
            let s = domain.lo + domain.hi;
            (
                if domain.lo <= 0.0 && domain.hi >= 0.0 {
                    0.0
                } else {
                    s
                },
                s,
            )
        }
        Primitive::Sqrt => {
            if domain.lo <= 0.0 {
                return Err(refusal("sqrt domain must be positive"));
            }
            let l = domain.lo.sqrt();
            let u = domain.hi.sqrt();
            (1.0 / (l + u), 1.0 / (l + u))
        }
        Primitive::Reciprocal => {
            if domain.lo <= 0.0 {
                return Err(refusal("reciprocal domain must be positive"));
            }
            (-1.0 / (center * center), -1.0 / (domain.lo * domain.hi))
        }
    };
    Ok(Envelope {
        lower: endpoint_line(kind, domain, lower_slope, true)?,
        upper: endpoint_line(kind, domain, upper_slope, false)?,
        range,
    })
}

pub(super) fn clip(domain: I, limit: f64) -> Result<Envelope> {
    let f = |x: f64| x.clamp(-limit, limit);
    let lower_slope = if domain.hi <= -limit || domain.lo >= limit {
        0.0
    } else if domain.lo >= -limit && domain.hi <= limit {
        1.0
    } else if domain.lo < -limit && domain.hi <= limit {
        (f(domain.hi) + limit) / (domain.hi + limit)
    } else {
        (limit - f(domain.lo)) / (limit - domain.lo)
    };
    let upper_slope = if domain.hi <= -limit || domain.lo >= limit {
        0.0
    } else if domain.lo >= -limit && domain.hi <= limit {
        1.0
    } else if domain.lo >= -limit && domain.hi > limit {
        (limit - f(domain.lo)) / (limit - domain.lo)
    } else {
        (f(domain.hi) + limit) / (domain.hi + limit)
    };
    let make = |slope: f64, lower: bool| -> Result<Line> {
        if !slope.is_finite() {
            return Err(refusal("clip slope refused"));
        }
        let mut values = Vec::with_capacity(4);
        for x in [domain.lo, domain.hi, -limit, limit] {
            if x >= domain.lo && x <= domain.hi {
                values.push(I::point(f(x)).sub(I::point(slope).mul(I::point(x))));
            }
        }
        let intercept = if lower {
            values.iter().map(|v| v.lo).fold(f64::INFINITY, f64::min)
        } else {
            values
                .iter()
                .map(|v| v.hi)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        Ok(Line { slope, intercept })
    };
    Ok(Envelope {
        lower: make(lower_slope, true)?,
        upper: make(upper_slope, false)?,
        range: I::new(f(domain.lo), f(domain.hi)),
    })
}

pub(super) fn compose(coefficient: I, relax: Envelope, lower: bool) -> Result<(I, I)> {
    if !coefficient.finite() || !relax.range.finite() {
        return Err(refusal("composition nonfinite"));
    }
    let proposal = coefficient.midpoint();
    let error = I::point(coefficient.sub(I::point(proposal)).abs_max())
        .mul(I::point(relax.range.abs_max()));
    let line = if (proposal >= 0.0) == lower {
        relax.lower
    } else {
        relax.upper
    };
    let c = I::point(proposal).mul(I::point(line.slope));
    let b = I::point(proposal).mul(I::point(line.intercept));
    Ok((c, if lower { b.sub(error) } else { b.add(error) }))
}

pub(super) fn mccormick(weight: I, x: I, inv: I, lower: bool) -> Result<(I, I, I)> {
    if !weight.finite() || !x.finite() || !inv.finite() {
        return Err(refusal("McCormick finite inputs required"));
    }
    let w = weight.midpoint();
    let want_lower = (w >= 0.0) == lower;
    let alternatives = if want_lower {
        [(inv.lo, x.lo), (inv.hi, x.hi)]
    } else {
        [(inv.hi, x.lo), (inv.lo, x.hi)]
    };
    let point = (x.midpoint(), inv.midpoint());
    let estimate = |&(p, a): &(f64, f64)| p * point.0 + a * point.1 - p * a;
    let first = estimate(&alternatives[0]);
    let second = estimate(&alternatives[1]);
    let (p, a) = if (first >= second) == want_lower {
        alternatives[0]
    } else {
        alternatives[1]
    };
    let weight_error =
        I::point(weight.sub(I::point(w)).abs_max()).mul(I::point(x.mul(inv).abs_max()));
    let constant = I::point(w).mul(I::point(p).mul(I::point(a)).neg());
    Ok((
        I::point(w).mul(I::point(p)),
        I::point(w).mul(I::point(a)),
        if lower {
            constant.sub(weight_error)
        } else {
            constant.add(weight_error)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    fn q(x: f64) -> BigRational {
        BigRational::from_float(x).unwrap()
    }

    #[test]
    fn signed_mccormick_constants_and_products_enclose_exact_rational_vertices() {
        for w in [-3.125, 0.75, 1e20] {
            let x = I::new(-3.25, -0.125);
            let inv = I::new(0.375, 3.5);
            for lower in [true, false] {
                let (a, c, b) = mccormick(I::point(w), x, inv, lower).unwrap();
                for xv in [x.lo, x.hi] {
                    for iv in [inv.lo, inv.hi] {
                        let truth = q(w) * q(xv) * q(iv);
                        let bound = a.mul(I::point(xv)).add(c.mul(I::point(iv))).add(b);
                        if lower {
                            assert!(q(bound.lo) <= truth);
                        } else {
                            assert!(truth <= q(bound.hi));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reciprocal_lower_checks_interior_not_only_endpoints() {
        let e = envelope(Primitive::Reciprocal, I::new(0.125, 5.0)).unwrap();
        let critical = (-1.0 / e.lower.slope).sqrt();
        assert!(critical > 0.125 && critical < 5.0);
        for x in [0.125, critical, 0.5, 1.0, 2.0, 5.0] {
            let true_value = q(1.0) / q(x);
            assert!(q(e.lower.slope) * q(x) + q(e.lower.intercept) <= true_value);
            assert!(true_value <= q(e.upper.slope) * q(x) + q(e.upper.intercept));
        }
    }

    #[test]
    fn square_and_clip_lines_are_checked_on_negative_and_crossing_domains() {
        for domain in [I::new(-3.0, -0.25), I::new(-3.0, 2.0), I::new(0.5, 2.0)] {
            let square_e = envelope(Primitive::Square, domain).unwrap();
            let clip_e = clip(domain, 1.0).unwrap();
            for step in 0..65 {
                let x = domain.lo + (domain.hi - domain.lo) * step as f64 / 64.0;
                for (e, truth) in [(square_e, q(x) * q(x)), (clip_e, q(x.clamp(-1.0, 1.0)))] {
                    assert!(q(e.lower.slope) * q(x) + q(e.lower.intercept) <= truth);
                    assert!(truth <= q(e.upper.slope) * q(x) + q(e.upper.intercept));
                }
            }
        }
    }
}
