// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Move six private kernel arrays through the shared numeric firewall while
//! retaining their original six capacity claims. No new payload is allocated.
//! This is output custody, not a complete CPU kernel or source-capture proof.

use super::super::{relation_sealed, BoundedRelation};
use super::*;
use crate::amortise::{CooperativeWork, POLL_CHUNK_UNITS};
use crate::LinearBounds;
use ndarray::Axis;

/// Field order is deliberate: bound storage drops before all capacity claims.
/// Malformed error planes may be discarded by the shared firewall; retaining
/// their charges until this wrapper drops is conservative over-accounting.
pub(in super::super) struct ClaimedKernelOutput {
    bounds: LinearBounds,
    _claims: [Charge; 6],
    deadline: Option<Instant>,
}

// Keep all claims in one guard while fallible publication owns or drops the
// moved arrays. Any arrays not moved into publication also drop before claims.
struct Transaction {
    arrays: [Option<Array2<f32>>; 6],
    claims: [Charge; 6],
}

struct PublicationWork<'a, P> {
    owner: &'a Arc<SoaStorageBudget>,
    poll: Poll,
    observe: P,
    boundary: usize,
}

impl<P: FnMut(usize) -> Result<()>> CooperativeWork for PublicationWork<'_, P> {
    fn checkpoint(&mut self) -> Result<()> {
        self.boundary += 1;
        (self.observe)(self.boundary)?;
        self.owner.check_live(&mut self.poll)
    }

    fn before(&mut self, units: usize) -> Result<()> {
        if units > POLL_CHUNK_UNITS {
            return Err(invalid(
                "kernel publication batch exceeds original work quantum",
            ));
        }
        self.checkpoint()?;
        self.poll.admit_elements(units, 1, COMMIT)
    }
}

impl ClaimedKernelOutput {
    /// Parts are lower-A, lower-b, upper-A, upper-b, lower-error, upper-error.
    /// Biases use owned (rows,1) arrays; removing the singleton axis preserves
    /// their allocation, capacity, offset and row stride without a reshape.
    /// All inputs must belong to ONE original joined query/account.
    pub(in super::super) fn from_parts(
        parts: [BoundedArray<f32>; 6],
        requested: Option<Instant>,
    ) -> Result<Self> {
        Self::observed(parts, requested, |_| Ok(()))
    }

    fn observed(
        parts: [BoundedArray<f32>; 6],
        requested: Option<Instant>,
        observe: impl FnMut(usize) -> Result<()>,
    ) -> Result<Self> {
        let deadline = parts.iter().fold(requested, |d, p| narrower(d, p.deadline));
        let mut poll = Poll::for_owner(&parts[0]._charge.owner, deadline);
        parts[0]._charge.owner.check_live(&mut poll)?;
        poll.admit_elements(6, 64, ENTRY)?;
        let owner = Arc::clone(&parts[0]._charge.owner);
        let mut work = PublicationWork {
            owner: &owner,
            poll,
            observe,
            boundary: 0,
        };
        work.checkpoint()?;
        if owner.query.is_none() {
            return Err(invalid(
                "kernel publication requires the original joined query",
            ));
        }
        for part in &parts {
            if !Arc::ptr_eq(&owner, &part._charge.owner) {
                return Err(invalid(
                    "kernel output planes belong to different live accounts",
                ));
            }
        }
        if parts[1].storage.ncols() != 1 || parts[3].storage.ncols() != 1 {
            return Err(invalid("kernel output bias must retain one owned column"));
        }
        let [la, lb, ua, ub, le, ue] = parts;
        // All following field operations are infallible ownership moves. The
        // transaction then keeps claims alive across EVERY fallible operation.
        let mut transaction = Transaction {
            arrays: [
                Some(la.storage),
                Some(lb.storage),
                Some(ua.storage),
                Some(ub.storage),
                Some(le.storage),
                Some(ue.storage),
            ],
            claims: [
                la._charge, lb._charge, ua._charge, ub._charge, le._charge, ue._charge,
            ],
        };
        let bounds = LinearBounds::new_or_conservative_with_err_polled_in_place(
            transaction.arrays[0].take().expect("private lower A"),
            transaction.arrays[1]
                .take()
                .expect("private lower b")
                .index_axis_move(Axis(1), 0),
            transaction.arrays[2].take().expect("private upper A"),
            transaction.arrays[3]
                .take()
                .expect("private upper b")
                .index_axis_move(Axis(1), 0),
            transaction.arrays[4].take().expect("private lower error"),
            transaction.arrays[5].take().expect("private upper error"),
            &mut work,
        )?;
        work.checkpoint()?;
        Ok(Self {
            bounds,
            _claims: transaction.claims,
            deadline,
        })
    }
}

impl relation_sealed::Sealed for ClaimedKernelOutput {}
impl BoundedRelation for ClaimedKernelOutput {
    fn bounds(&self) -> &LinearBounds {
        &self.bounds
    }
    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    fn owner(&self) -> &Arc<SoaStorageBudget> {
        &self._claims[0].owner
    }
}

#[cfg(test)]
#[path = "claimed_output_tests.rs"]
mod tests;
