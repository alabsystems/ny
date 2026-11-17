// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sign-safe residual Add in the original input frame.
//! Requires the separately qualified raw-zero Add rule. The caller owns
//! inherited error, support masks, FoldMag and root tensor lifetimes.

use super::*;

struct AddUnionPlan {
    coordinates: Array2<usize>,
    left: Array2<usize>,
    right: Array2<usize>,
}

fn bytes(elements: usize, width: usize) -> Option<usize> {
    elements
        .checked_mul(width)
        .filter(|&n| n <= isize::MAX as usize)
}

impl<T: TabElem> PackedRows<T> {
    fn plan_add(&self, rhs: &Self, deadline: Option<Instant>) -> Result<Option<AddUnionPlan>> {
        check_deadline(deadline)?;
        let unsupported = || Ok::<_, NyError>(None);
        if !T::F32_STORE
            || legacy_rounding_forced()
            || self.p == 0
            || self.n_in == 0
            || self.n_in != rhs.n_in
            || self.p != rhs.p
            || self.m.nrows() != rhs.m.nrows()
            || self.m.nrows() == 0
            || self.m.nrows() % self.p != 0
        {
            return unsupported();
        }
        let Some(naug) = self
            .n_in
            .checked_add(1)
            .filter(|n| n.checked_add(10).is_some())
        else {
            return unsupported();
        };
        for source in [self, rhs] {
            if source.m.dim() != source.d.dim()
                || source.coordinates.nrows() != self.p
                || source.bias_column().checked_add(1) != Some(source.m.ncols())
            {
                return unsupported();
            }
            for sp in 0..self.p {
                check_deadline(deadline)?;
                let (mut previous, mut padded) = (None, false);
                for &i in source.coordinates.row(sp).iter() {
                    if i == usize::MAX {
                        padded = true;
                    } else {
                        if padded || i >= self.n_in || previous.is_some_and(|prev| prev >= i) {
                            return unsupported();
                        }
                        previous = Some(i);
                    }
                }
            }
        }
        let Some(dense_bytes) = self
            .m
            .nrows()
            .checked_mul(naug)
            .and_then(|n| bytes(n, 2 * size_of::<T>()))
        else {
            return unsupported();
        };
        let mut unions = Vec::with_capacity(self.p);
        let mut slots = 0;
        for sp in 0..self.p {
            check_deadline(deadline)?;
            let (mut l, mut r) = (0, 0);
            let mut union = Vec::new();
            loop {
                let li = self.coordinates.get((sp, l)).copied().unwrap_or(usize::MAX);
                let ri = rhs.coordinates.get((sp, r)).copied().unwrap_or(usize::MAX);
                let i = li.min(ri);
                if i == usize::MAX {
                    break;
                }
                let ls = if li == i {
                    let old = l;
                    l += 1;
                    old
                } else {
                    usize::MAX
                };
                let rs = if ri == i {
                    let old = r;
                    r += 1;
                    old
                } else {
                    usize::MAX
                };
                union.push((i, ls, rs));
            }
            slots = slots.max(union.len());
            if slots >= self.n_in {
                return unsupported();
            }
            unions.push(union);
        }
        let Some(map_bytes) = self
            .p
            .checked_mul(slots)
            .and_then(|n| bytes(n, 3 * size_of::<usize>()))
        else {
            return unsupported();
        };
        let Some(payload_bytes) = slots
            .checked_add(1)
            .and_then(|n| self.m.nrows().checked_mul(n))
            .and_then(|n| bytes(n, 2 * size_of::<T>()))
        else {
            return unsupported();
        };
        if payload_bytes
            .checked_add(map_bytes)
            .is_none_or(|n| n >= dense_bytes)
        {
            return unsupported();
        }
        // This compares result + maps with the dense pair. It is not a peak
        // memory cap: the temporary per-spatial unions also occupy storage.
        let mut plan = AddUnionPlan {
            coordinates: Array2::from_elem((self.p, slots), usize::MAX),
            left: Array2::from_elem((self.p, slots), usize::MAX),
            right: Array2::from_elem((self.p, slots), usize::MAX),
        };
        for (sp, union) in unions.iter().enumerate() {
            check_deadline(deadline)?;
            for (slot, &(i, l, r)) in union.iter().enumerate() {
                plan.coordinates[[sp, slot]] = i;
                plan.left[[sp, slot]] = l;
                plan.right[[sp, slot]] = r;
            }
        }
        Ok(Some(plan))
    }

    /// Both operands are borrowed, including Add(x,x). No source mutation or
    /// lifetime decrement is performed here. One-sided entries run the full
    /// arithmetic and stores; they cannot be copied through without charges.
    pub(in crate::margin_row::root) fn add_packed(
        &self,
        rhs: &Self,
        xabs_ext: &[f64],
        deadline: Option<Instant>,
    ) -> Result<Option<(Self, Vec<f64>)>> {
        let Some(plan) = self.plan_add(rhs, deadline)? else {
            return Ok(None);
        };
        let naug = self.n_in + 1;
        if xabs_ext.len() != naug || !xabs_ext.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(NyError::NumericalInstability(
                "margin_row invalid packed Add frame".into(),
            ));
        }
        // Validate absent sites too. Omitting a nonzero padding coefficient
        // would violate the original-coordinate representation invariant.
        for source in [self, rhs] {
            for j in 0..source.m.nrows() {
                check_deadline(deadline)?;
                for slot in 0..source.m.ncols() {
                    let (m, d) = (source.m[[j, slot]], source.d[[j, slot]]);
                    if !m.wide().is_finite() || !d.wide().is_finite() {
                        return Err(NyError::NumericalInstability(
                            "margin_row nonfinite packed Add source".into(),
                        ));
                    }
                    if slot != source.bias_column()
                        && source.coordinates[[j % self.p, slot]] == usize::MAX
                        && (!m.is_positive_zero() || !d.is_positive_zero())
                    {
                        return Ok(None);
                    }
                }
            }
        }
        let bias = plan.coordinates.ncols();
        let stride = bias + 1;
        let mut m = Array2::from_elem((self.m.nrows(), stride), T::ZERO);
        let mut d = Array2::from_elem(m.raw_dim(), T::ZERO);
        let mut gstore = vec![0.0; self.m.nrows()];
        let acc_widen = 1.0 + 2.0 * gamma_n(naug + 8);
        let rwiden = root_round_widen(naug);
        let floor = root_round_floor(xabs_ext);
        m.as_slice_mut()
            .unwrap()
            .par_chunks_mut(stride)
            .zip(d.as_slice_mut().unwrap().par_chunks_mut(stride))
            .zip(gstore.par_iter_mut())
            .enumerate()
            .try_for_each(|(j, ((mr, dr), gs))| {
                check_deadline(deadline)?;
                let sp = j % self.p;
                let (mut racc, mut dust) = (0.0, 0.0);
                for slot in 0..bias {
                    let i = plan.coordinates[[sp, slot]];
                    if i == usize::MAX {
                        break;
                    }
                    let l = plan.left[[sp, slot]];
                    let r = plan.right[[sp, slot]];
                    let (ma, da) = if l == usize::MAX {
                        (T::ZERO, T::ZERO)
                    } else {
                        (self.m[[j, l]], self.d[[j, l]])
                    };
                    let (mb, db) = if r == usize::MAX {
                        (T::ZERO, T::ZERO)
                    } else {
                        (rhs.m[[j, r]], rhs.d[[j, r]])
                    };
                    if ma.is_positive_zero()
                        && da.is_positive_zero()
                        && mb.is_positive_zero()
                        && db.is_positive_zero()
                    {
                        continue;
                    }
                    let sum = ma.wide() + mb.wide();
                    let dsum = da.wide() + db.wide();
                    racc += (sum.abs() + dsum.abs()) * xabs_ext[i];
                    let (stored, charge) = T::store_center(sum);
                    mr[slot] = stored;
                    dr[slot] = store_radius_charged::<T>(dsum, charge);
                    dust += ((stored.wide() - sum).abs() + (dr[slot].wide() - dsum).abs())
                        * xabs_ext[i];
                }
                // Physical bias is last; every bound still uses ORIGINAL width
                // and the complete original frame, including absent columns.
                let sum =
                    self.m[[j, self.bias_column()]].wide() + rhs.m[[j, rhs.bias_column()]].wide();
                let dsum =
                    self.d[[j, self.bias_column()]].wide() + rhs.d[[j, rhs.bias_column()]].wide();
                racc += (sum.abs() + dsum.abs()) * xabs_ext[self.n_in];
                let d64 = next_up(dsum + root_round_charge(ADD_ROUND_REL, racc, rwiden, floor));
                let (stored, charge) = T::store_center(sum);
                mr[bias] = stored;
                dr[bias] = store_radius_charged::<T>(d64, charge);
                dust += ((stored.wide() - sum).abs() + (dr[bias].wide() - d64).abs())
                    * xabs_ext[self.n_in];
                *gs = next_up(dust * acc_widen);
                if !gs.is_finite() || mr.iter().chain(dr.iter()).any(|v| !v.wide().is_finite()) {
                    return Err(NyError::NumericalInstability(
                        "margin_row nonfinite packed Add result".into(),
                    ));
                }
                Ok::<(), NyError>(())
            })?;
        Ok(Some((
            Self {
                n_in: self.n_in,
                p: self.p,
                coordinates: plan.coordinates,
                m,
                d,
            },
            gstore,
        )))
    }
}

#[cfg(test)]
mod tests;
