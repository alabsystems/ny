// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compact root tableaux in original coordinates, through Conv/ReLU/residual Add.
//!
//! The coordinate frame is always the ORIGINAL input box. Missing entries are
//! raw M=D=+0, never dropped small numbers. Error, fold, bias, width and depth
//! charges retain their original meanings. In particular, a smaller physical
//! row must not reduce a rounding envelope. This module is admitted only on the
//! f32 storage lane after the identity-scatter environment qualification.

use super::*;

#[derive(Clone)]
pub(super) struct PackedRows<T> {
    n_in: usize,
    p: usize,
    // [spatial position, slot], strictly increasing original coordinates;
    // absent padding is a suffix of usize::MAX. Shared across output channels.
    coordinates: Array2<usize>,
    pub(super) m: Array2<T>,
    pub(super) d: Array2<T>,
}

#[derive(Clone)]
pub(super) struct State<T> {
    pub(super) rows: ReluRows<T>,
    pub(super) gerr: Vec<f64>,
    pub(super) support: Vec<u64>,
}

#[derive(Clone)]
pub(super) enum ReluRows<T> {
    Dense(Array2<T>, Array2<T>),
    Packed(PackedRows<T>),
}

/// Topology and numerical admission runs before publishing any packed state.
/// The first preactivation has a sole ReLU consumer. Its output may fan out to
/// any number of Conv consumers; they borrow the SAME carrier in graph order.
pub(super) fn admit<T: TabElem>(net: &TwinNet, consumers: &[usize]) -> bool {
    if !T::F32_STORE || clip_rows_enabled() || borrowed_gates_forced() {
        return false;
    }
    let Some(TwinOp::Conv(first)) = net.ops.first() else {
        return false;
    };
    if !matches!(net.ops.get(1), Some(TwinOp::Relu { input: 1, .. }))
        || consumers[1] != 1
        || consumers[2] == 0
        || first.k_fwd >= net.n_in
    {
        return false;
    }
    let last = *net.trunk_relus.last().expect("validated trunk");
    let mut seen = 0;
    for op in &net.ops[2..=last] {
        match op {
            TwinOp::Conv(c) if c.input == 2 => {
                // Packed consumers use relative conversion bounds, so their
                // captured weights must be zero or normal. The first identity
                // Conv charges cast-to-zero loss separately.
                if c.wmat
                    .iter()
                    .any(|&w| !w.is_finite() || (w != 0.0 && !(w as f32).is_normal()))
                {
                    return false;
                }
                seen += 1;
            }
            TwinOp::Relu { input: 2, .. }
            | TwinOp::Flatten { input: 2 }
            | TwinOp::ChannelAffine { input: 2, .. }
            | TwinOp::Add { lhs: 2, .. }
            | TwinOp::Add { rhs: 2, .. } => return false,
            _ => {}
        }
    }
    seen == consumers[2]
}

#[inline]
fn check_deadline(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|d| Instant::now() > d) {
        return Err(NyError::DeadlineExceeded(
            "margin_row sparse stem deadline".into(),
        ));
    }
    Ok::<(), NyError>(())
}

impl<T: TabElem> PackedRows<T> {
    /// Called only by identity_conv with its already qualified captured weights.
    pub(super) fn from_identity(
        c: &ConvOp,
        n_in: usize,
        p: usize,
        k: usize,
        weights: &[f32],
        deadline: Option<Instant>,
    ) -> Result<Self> {
        if !T::F32_STORE {
            return Err(NyError::InvalidSpec(
                "packed stem requires f32 storage".into(),
            ));
        }
        let mut coordinates = Array2::from_elem((p, k), usize::MAX);
        let mut taps = vec![(usize::MAX, usize::MAX); k];
        let mut m = Array2::from_elem((c.oshape.0 * p, k + 1), T::ZERO);
        for sp in 0..p {
            check_deadline(deadline)?;
            for t in 0..k {
                taps[t] = (c.gather[t * p + sp], t);
            }
            taps.sort_unstable();
            for (slot, &(gi, t)) in taps.iter().enumerate() {
                if gi == usize::MAX {
                    break;
                }
                coordinates[[sp, slot]] = gi;
                for oc in 0..c.oshape.0 {
                    let w = weights[oc * k + t];
                    if w.to_bits() & 0x7fff_ffff != 0 {
                        m[[oc * p + sp, slot]] = T::from_acc(w);
                    }
                }
            }
        }
        let d = Array2::from_elem(m.raw_dim(), T::ZERO);
        Ok(Self {
            n_in,
            p,
            coordinates,
            m,
            d,
        })
    }

    pub(super) fn bias_column(&self) -> usize {
        self.coordinates.ncols()
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        (self.m.len() + self.d.len()) * size_of::<T>() + self.coordinates.len() * size_of::<usize>()
    }

    /// Materialize one ranker row only, with the original width and bias slot.
    pub(super) fn row(&self, j: usize) -> (Vec<T>, Vec<T>) {
        let mut m = vec![T::ZERO; self.n_in + 1];
        let mut d = m.clone();
        for (slot, &i) in self.coordinates.row(j % self.p).iter().enumerate() {
            if i == usize::MAX {
                break;
            }
            m[i] = self.m[[j, slot]];
            d[i] = self.d[[j, slot]];
        }
        m[self.n_in] = self.m[[j, self.bias_column()]];
        d[self.n_in] = self.d[[j, self.bias_column()]];
        (m, d)
    }

    pub(super) fn materialize_checked(
        &self,
        deadline: Option<Instant>,
    ) -> Result<(Array2<T>, Array2<T>)> {
        check_deadline(deadline)?;
        let shape = (self.m.nrows(), self.n_in + 1);
        let mut m = Array2::from_elem(shape, T::ZERO);
        let mut d = Array2::from_elem(shape, T::ZERO);
        for j in 0..shape.0 {
            check_deadline(deadline)?;
            for (slot, &i) in self.coordinates.row(j % self.p).iter().enumerate() {
                if i == usize::MAX {
                    break;
                }
                m[[j, i]] = self.m[[j, slot]];
                d[[j, i]] = self.d[[j, slot]];
            }
            m[[j, self.n_in]] = self.m[[j, self.bias_column()]];
            d[[j, self.n_in]] = self.d[[j, self.bias_column()]];
        }
        Ok((m, d))
    }

    #[cfg(test)]
    pub(super) fn materialize(&self) -> (Array2<T>, Array2<T>) {
        self.materialize_checked(None)
            .expect("test materialization")
    }

    pub(super) fn concretize(
        &self,
        mid: &[f64],
        rad: &[f64],
        xabs: &[f64],
        relu_depth: usize,
        gerr: &[f64],
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let gam = gamma_n(self.n_in + 16 + 8 * relu_depth);
        let mut l = vec![0.0; self.m.nrows()];
        let mut u = l.clone();
        l.par_iter_mut()
            .zip(u.par_iter_mut())
            .enumerate()
            .try_for_each(|(j, (lj, uj))| {
                check_deadline(deadline)?;
                let (mut vl, mut rl, mut vu, mut ru, mut tabs) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for (slot, &i) in self.coordinates.row(j % self.p).iter().enumerate() {
                    if i == usize::MAX {
                        break;
                    }
                    let lo_c = self.m[[j, slot]].wide() - self.d[[j, slot]].wide();
                    let up_c = self.m[[j, slot]].wide() + self.d[[j, slot]].wide();
                    vl += lo_c * mid[i];
                    rl += lo_c.abs() * rad[i];
                    vu += up_c * mid[i];
                    ru += up_c.abs() * rad[i];
                    tabs += (lo_c.abs() + up_c.abs()) * xabs[i];
                }
                let b = self.bias_column();
                let bl = self.m[[j, b]].wide() - self.d[[j, b]].wide();
                let bu = self.m[[j, b]].wide() + self.d[[j, b]].wide();
                let low = vl + bl - rl;
                let upp = vu + bu + ru;
                let base = next_up(gam * (tabs + bl.abs() + bu.abs()));
                let slack = next_up(base + gerr[j]);
                *lj = next_down(next_down(low - slack));
                *uj = next_up(next_up(upp + slack));
                Ok::<(), NyError>(())
            })?;
        Ok((l, u))
    }

    /// Same sign-safe gate formulas as the dense lane, in original column order.
    /// Only raw +0 coefficient sites can be omitted. Bias is always processed.
    pub(super) fn apply_gates(
        &mut self,
        alpha: &[f64],
        slopes: &[f64],
        intercepts: &[f64],
        xabs_ext: &[f64],
        deadline: Option<Instant>,
    ) -> Result<Vec<f64>> {
        if !alpha
            .iter()
            .chain(slopes)
            .chain(intercepts)
            .all(|v| v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "margin_row nonfinite sparse gate".into(),
            ));
        }
        if xabs_ext.len() != self.n_in + 1 || !xabs_ext.iter().all(|v| v.is_finite() && *v >= 0.0) {
            return Err(NyError::NumericalInstability(
                "margin_row invalid sparse frame".into(),
            ));
        }
        let naug = self.n_in + 1;
        let acc_widen = 1.0 + 2.0 * gamma_n(naug + 8);
        let rwiden = root_round_widen(naug);
        let floor = root_round_floor(xabs_ext);
        let bias = self.bias_column();
        let stride = bias + 1;
        let mut gstore = vec![0.0; self.m.nrows()];
        let coords = &self.coordinates;
        let p = self.p;
        self.m
            .as_slice_mut()
            .unwrap()
            .par_chunks_mut(stride)
            .zip(self.d.as_slice_mut().unwrap().par_chunks_mut(stride))
            .zip(gstore.par_iter_mut())
            .enumerate()
            .try_for_each(|(j, ((mr, dr), gs))| {
                check_deadline(deadline)?;
                let (a, s) = (alpha[j], slopes[j]);
                let (mut racc, mut dust) = (0.0, 0.0);
                for (slot, &i) in coords.row(j % p).iter().enumerate() {
                    if i == usize::MAX {
                        break;
                    }
                    if mr[slot].is_positive_zero() && dr[slot].is_positive_zero() {
                        continue;
                    }
                    let (mi, di) = (mr[slot].wide(), dr[slot].wide());
                    let lo_c = (mi - di) * a;
                    let up_c = (mi + di) * s;
                    #[allow(clippy::manual_midpoint)]
                    let mm = (lo_c + up_c) * 0.5;
                    let dd = (up_c - lo_c) * 0.5;
                    racc += (lo_c.abs() + up_c.abs()) * xabs_ext[i];
                    let (stored, charge) = T::store_center(mm);
                    mr[slot] = stored;
                    dr[slot] = store_radius_charged::<T>(dd, charge);
                    dust +=
                        ((stored.wide() - mm).abs() + (dr[slot].wide() - dd).abs()) * xabs_ext[i];
                }
                let (mi, di) = (mr[bias].wide(), dr[bias].wide());
                let lo_c = (mi - di) * a;
                let up0 = (mi + di) * s;
                let up_c = up0 + intercepts[j];
                #[allow(clippy::manual_midpoint)]
                let mm = (lo_c + up_c) * 0.5;
                let dd = (up_c - lo_c) * 0.5;
                racc += (lo_c.abs() + up_c.abs() + up0.abs()) * xabs_ext[naug - 1];
                let d64 = next_up(dd + root_round_charge(GATE_ROUND_REL, racc, rwiden, floor));
                let (stored, charge) = T::store_center(mm);
                mr[bias] = stored;
                dr[bias] = store_radius_charged::<T>(d64, charge);
                dust += ((stored.wide() - mm).abs() + (dr[bias].wide() - d64).abs())
                    * xabs_ext[naug - 1];
                *gs = next_up(dust * acc_widen);
                Ok::<(), NyError>(())
            })?;
        Ok(gstore)
    }

    pub(super) fn conv_error(
        &self,
        c: &ConvOp,
        gerr: &[f64],
        xabs: &[f64],
        s_xabs: f64,
        deadline: Option<Instant>,
    ) -> Result<Vec<f64>> {
        let gf32 = next_up(gamma_n_f32(c.k_fwd + 8));
        let bwiden = gamma_n(self.n_in + 3);
        let mut ginp = Array2::<f64>::zeros((self.m.nrows(), 1));
        ginp.as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(j, gt)| {
                check_deadline(deadline)?;
                let (mut braw, mut bd) = (0.0, 0.0);
                for (slot, &i) in self.coordinates.row(j % self.p).iter().enumerate() {
                    if i == usize::MAX {
                        break;
                    }
                    let d = self.d[[j, slot]].wide().abs();
                    braw += (self.m[[j, slot]].wide().abs() + d) * xabs[i];
                    bd += d * xabs[i];
                }
                let bias = self.bias_column();
                braw += self.m[[j, bias]].wide().abs() + self.d[[j, bias]].wide().abs();
                bd += self.d[[j, bias]].wide().abs();
                *gt = conv_f32_ginp_value(gerr[j], braw, bd, gf32, bwiden, s_xabs, true);
                Ok::<(), NyError>(())
            })?;
        Ok(conv_f32_gerr_finish(c, &ginp, s_xabs, true))
    }

    /// Accumulate in the ordinary ascending consumer-tap order. Each gathered
    /// row contributes once to each represented original coordinate. D stays
    /// signed: only the consumer weight is made absolute. Missing entries are
    /// exact +0. Conv consumers borrow this carrier without cloning its payload.
    #[cfg(test)]
    pub(super) fn contract(
        &self,
        c: &ConvOp,
        m: &mut Array2<T>,
        d: &mut Array2<T>,
        deadline: Option<Instant>,
    ) -> Result<()> {
        if self.m.iter().chain(&self.d).any(|v| !v.wide().is_finite()) {
            return Err(NyError::NumericalInstability(
                "margin_row nonfinite sparse source".into(),
            ));
        }
        let weights: Vec<f32> = c.wmat.iter().map(|&w| w as f32).collect();
        let p_out = c.oshape.1 * c.oshape.2;
        let naug = self.n_in + 1;
        let bias = self.bias_column();
        let stride = bias + 1;
        let ms = self.m.as_slice().expect("packed standard layout");
        let ds = self.d.as_slice().expect("packed standard layout");
        let coordinates = self.coordinates.as_slice().expect("packed coordinates");
        m.as_slice_mut()
            .unwrap()
            .par_chunks_mut(naug)
            .zip(d.as_slice_mut().unwrap().par_chunks_mut(naug))
            .enumerate()
            .try_for_each(|(j, (mr, dr))| {
                check_deadline(deadline)?;
                let (oc, sp) = (j / p_out, j % p_out);
                for t in 0..c.k_fwd {
                    if t % 64 == 0 {
                        check_deadline(deadline)?;
                    }
                    let gi = c.gather[t * p_out + sp];
                    if gi == usize::MAX {
                        continue;
                    }
                    let w = weights[oc * c.k_fwd + t];
                    if w == 0.0 {
                        continue;
                    }
                    let source = gi * stride;
                    let sm = &ms[source..source + stride];
                    let sd = &ds[source..source + stride];
                    let coord = (gi % self.p) * bias;
                    let indices = &coordinates[coord..coord + bias];
                    let abs_w = w.abs();
                    // Slice the source once per tap. The inner loop visits
                    // contiguous packed values without repeated 2-D indexing.
                    for (&i, (&mi, &di)) in indices.iter().zip(sm.iter().zip(sd.iter())) {
                        if i == usize::MAX {
                            break;
                        }
                        mr[i] = T::from_acc(mr[i].wide() as f32 + w * mi.wide() as f32);
                        dr[i] = T::from_acc(dr[i].wide() as f32 + abs_w * di.wide() as f32);
                    }
                    mr[self.n_in] =
                        T::from_acc(mr[self.n_in].wide() as f32 + w * sm[bias].wide() as f32);
                    dr[self.n_in] =
                        T::from_acc(dr[self.n_in].wide() as f32 + abs_w * sd[bias].wide() as f32);
                }
                Ok::<(), NyError>(())
            })
    }
}

#[cfg(test)]
thread_local! {
    static TRACE: std::cell::RefCell<Option<(bool, Vec<usize>, Vec<usize>)>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) fn materialize_immediately() -> bool {
    TRACE.with(|slot| slot.borrow().as_ref().is_some_and(|s| s.0))
}

#[cfg(test)]
pub(super) fn record(op: usize) {
    TRACE.with(|slot| {
        if let Some(s) = slot.borrow_mut().as_mut() {
            s.1.push(op);
        }
    });
}

#[cfg(test)]
fn trace_detail<R>(materialize: bool, f: impl FnOnce() -> R) -> (R, Vec<usize>, Vec<usize>) {
    struct Restore(Option<(bool, Vec<usize>, Vec<usize>)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TRACE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let restore =
        Restore(TRACE.with(|slot| slot.replace(Some((materialize, Vec::new(), Vec::new())))));
    let result = f();
    let (_, convs, adds) = TRACE.with(|slot| slot.borrow_mut().take().unwrap());
    drop(restore);
    (result, convs, adds)
}

#[cfg(test)]
fn trace<R>(materialize: bool, f: impl FnOnce() -> R) -> (R, Vec<usize>) {
    let (result, convs, _) = trace_detail(materialize, f);
    (result, convs)
}

#[cfg(test)]
pub(super) fn record_add(op: usize) {
    TRACE.with(|slot| {
        if let Some(s) = slot.borrow_mut().as_mut() {
            s.2.push(op);
        }
    });
}

#[cfg(test)]
mod tests;

mod add_union;
mod conv_union;
mod state;
pub(super) use state::{borrow, release, take_dense, take_rows};

#[cfg(test)]
mod residual_tests;
