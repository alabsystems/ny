// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Source-owned input-relative preactivation sandwiches. This first interface
//! retains either the completed original RootGates borrow or an authenticated
//! immutable PreparedPrefix and paid ordered original seed IDs. Owned rows never
//! borrow mutable root construction or relabel supplied gates as a proof owner.
//! Constructors are private to the margin-row producers. No default capability,
//! clipping verdict, whole-property coverage or independent replay is added here.

use std::{mem::size_of, time::Instant};

use ndarray::Array2;

mod contiguous_matrix;
use contiguous_matrix::{ContiguousMatrix, MatrixRefusal};
use ny_core::dd::two_sum;
use ny_core::f32_to_f64_exact;
use ny_core::{NyError, Result};

use super::engine::PassOut;
use super::net::{check_deadline, underflow_product_up, TwinNet, TwinOp};
use super::patch::{Frame, PatchWalk, Shape};
use super::root::RootGates;
use super::rounding::next_up;
use super::support_program::PreparedPrefix;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};

const POLL_ITEMS: usize = 4096;
/// Pair, private reservation/Arc metadata, and storage-transfer receipt headers.
/// Scalar array and target allocation capacities are counted separately.
pub(super) const INPUT_ROWS_OWNER_BYTES: usize = 1024;

enum Premises<'a> {
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    Borrowed {
        net: &'a TwinNet,
        root: &'a RootGates,
        targets: &'a [usize],
    },
    Prepared {
        prefix: PreparedPrefix,
        targets: Vec<usize>,
        // Exact immutable linked-owner census, paid and polled once before
        // target allocation. None of these retained allocation claims mutate.
        source_bytes: usize,
        _resident: PersistentReservation,
    },
}

/// Exact original preactivation selection. Ordered IDs are retained by the
/// producer; no compacted position can silently become an original neuron ID.
pub(super) struct OriginalRowDemand<'a> {
    pub(super) relu_op: usize,
    pub(super) targets: &'a [usize],
}

/// Immutable proof premises; owner identity cannot silently change while a row
/// view lives. The declared original endpoints, not rounded `root.xabs`, define
/// the domain used by the pointwise error charge.
pub(super) struct InputRowScope<'a> {
    premises: Premises<'a>,
    relu_op: usize,
    deadline: Instant,
}

impl<'a> InputRowScope<'a> {
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn new(
        net: &'a TwinNet,
        root: &'a RootGates,
        relu_op: usize,
        targets: &'a [usize],
        deadline: Instant,
    ) -> Result<Self> {
        check_deadline(deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        let bad = |message: &str| NyError::InvalidSpec(format!("input rows: {message}"));
        if !root.mode.outward() || targets.is_empty() || relu_op >= net.i_gemm1 {
            return Err(bad("requires an Outward trunk preactivation request"));
        }
        let source = root
            .head_group_source
            .as_ref()
            .ok_or_else(|| bad("missing authenticated original unsplit builder authority"))?;
        let deadline = source.authenticate_input_rows(net, root, deadline)?;
        let seed_tensor = match net.ops.get(relu_op) {
            Some(TwinOp::Relu { input, .. }) => *input,
            _ => return Err(bad("seed is not an original trunk ReLU")),
        };
        let seed_rows = *net
            .tsize
            .get(seed_tensor)
            .ok_or_else(|| bad("missing original seed shape"))?;
        if net.n_in == 0
            || root.lo.len() != net.n_in
            || root.hi.len() != net.n_in
            || root.mid.len() != net.n_in
            || root.rad.len() != net.n_in
            || root.xabs.len() != net.n_in
        {
            return Err(bad("original input shape mismatch"));
        }
        for (i, (&lo, &hi)) in root.lo.iter().zip(&root.hi).enumerate() {
            poll(deadline, i)?;
            if !(lo.is_finite()
                && hi.is_finite()
                && lo <= hi
                && root.mid[i].is_finite()
                && root.rad[i].is_finite()
                && root.rad[i] >= 0.0
                && root.xabs[i].is_finite()
                && root.xabs[i] >= 0.0)
            {
                return Err(bad("invalid original input box"));
            }
        }
        for (i, &target) in targets.iter().enumerate() {
            poll(deadline, i)?;
            let mut duplicate = false;
            for chunk in targets[..i].chunks(POLL_ITEMS) {
                check_deadline(deadline)?;
                duplicate |= chunk.contains(&target);
            }
            if target >= seed_rows || duplicate {
                return Err(bad("out-of-range or duplicate original target"));
            }
        }
        // The completed original generation was authenticated above. These
        // checks additionally reject malformed arithmetic in the actual gates
        // the producer reads; numerical shapes never mint source authority.
        for (op, original) in net.ops[..relu_op].iter().enumerate() {
            check_deadline(deadline)?;
            let TwinOp::Relu { input, layer } = original else {
                continue;
            };
            let gate = root
                .layers
                .get(*layer)
                .ok_or_else(|| bad("missing original prefix gate"))?;
            let n = *net
                .tsize
                .get(*input)
                .ok_or_else(|| bad("missing prefix tensor"))?;
            if gate.op != op
                || gate.n != n
                || [
                    gate.l.len(),
                    gate.u.len(),
                    gate.alpha.len(),
                    gate.s.len(),
                    gate.c.len(),
                    gate.ms.len(),
                ]
                .iter()
                .any(|&len| len != n)
            {
                return Err(bad("original prefix gate shape/order mismatch"));
            }
            for j in 0..n {
                poll(deadline, j)?;
                let (l, u, a, s, c, ms) = (
                    gate.l[j],
                    gate.u[j],
                    gate.alpha[j],
                    gate.s[j],
                    gate.c[j],
                    gate.ms[j],
                );
                if !(l.is_finite()
                    && u.is_finite()
                    && l <= u
                    && (0.0..=1.0).contains(&a)
                    && (0.0..=1.0).contains(&s)
                    && c.is_finite()
                    && c >= 0.0
                    && ms.is_finite()
                    && ms >= a.max(s))
                {
                    return Err(bad("invalid prefix gate arithmetic"));
                }
                let need_negative = if l < 0.0 { mul_up(s, -l) } else { 0.0 };
                let one_minus_s = if s == 1.0 { 0.0 } else { next_up(1.0 - s) };
                let need_positive = if u > 0.0 { mul_up(one_minus_s, u) } else { 0.0 };
                if c < need_negative.max(need_positive) {
                    return Err(bad("prefix upper line lacks original-box authority"));
                }
            }
        }
        check_deadline(deadline)?;
        Ok(Self {
            premises: Premises::Borrowed { net, root, targets },
            relu_op,
            deadline,
        })
    }

    /// Copy only the ordered target IDs, after admission from this exact
    /// original prefix's budget. No supplied RootGates or gate arrays are read.
    pub(super) fn from_prepared(
        prefix: &PreparedPrefix,
        OriginalRowDemand { relu_op, targets }: OriginalRowDemand<'_>,
        requested: Instant,
    ) -> Result<InputRowScope<'static>> {
        let deadline = prefix
            .deadline(Some(requested))?
            .ok_or_else(|| invalid("missing finite prefix deadline"))?;
        let root = prefix.root();
        let budget = root.budget();
        checkpoint_budget(budget, deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        let (_, seed_rows) = prefix.for_relu(relu_op)?;
        if targets.is_empty() {
            return Err(invalid("empty original target request"));
        }
        let source_bytes = prefix.resident_bytes(Some(deadline))?;
        let target_bytes = targets
            .len()
            .checked_mul(size_of::<usize>())
            .ok_or_else(|| invalid("target capacity overflow"))?;
        let bytes = target_bytes
            .checked_add(size_of::<InputRowScope<'static>>())
            .and_then(|v| v.checked_add(256))
            .ok_or_else(|| invalid("target owner overflow"))?;
        let live = source_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("target total-live overflow"))?;
        if live > super::root_box::ROOT_SOURCE_BYTES_CAP {
            return Err(storage_memory(
                live,
                super::root_box::ROOT_SOURCE_BYTES_CAP,
                "input rows prepared target owner",
            ));
        }
        // The duplicate check is quadratic without allocating a second sorted
        // target slab; its entire scan is charged before either copy or scan.
        let work = targets
            .len()
            .checked_mul(targets.len())
            .and_then(|v| v.checked_add(targets.len().checked_mul(16)?))
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| invalid("target validation work overflow"))?;
        let mut lease = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: bytes,
                    work,
                    ..WorkRequest::default()
                },
            )
            .map_err(work_refusal)?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(targets.len())
            .map_err(|_| storage_memory(bytes, bytes, "input rows target allocation"))?;
        checkpoint_budget(budget, deadline)?;
        if owned.capacity() != targets.len() {
            let actual = owned
                .capacity()
                .checked_mul(size_of::<usize>())
                .and_then(|v| v.checked_add(size_of::<InputRowScope<'static>>()))
                .and_then(|v| v.checked_add(256))
                .ok_or_else(|| invalid("target actual capacity overflow"))?;
            return Err(storage_memory(
                actual,
                bytes,
                "input rows original target allocator capacity",
            ));
        }
        lease.mark_materialized().map_err(work_refusal)?;
        for (i, &target) in targets.iter().enumerate() {
            if i.is_multiple_of(POLL_ITEMS) {
                checkpoint_budget(budget, deadline)?;
            }
            if target >= seed_rows {
                return Err(invalid("out-of-range original target"));
            }
            for previous in targets[..i].chunks(POLL_ITEMS) {
                checkpoint_budget(budget, deadline)?;
                if previous.contains(&target) {
                    return Err(invalid("duplicate original target"));
                }
            }
            owned.push(target);
        }
        checkpoint_budget(budget, deadline)?;
        let resident = lease.commit_persistent().map_err(work_refusal)?;
        Ok(InputRowScope {
            premises: Premises::Prepared {
                prefix: prefix.clone(),
                targets: owned,
                source_bytes,
                _resident: resident,
            },
            relu_op,
            deadline,
        })
    }

    pub(super) fn net(&self) -> &TwinNet {
        match &self.premises {
            Premises::Borrowed { net, .. } => net,
            Premises::Prepared { prefix, .. } => prefix.root().model().net(),
        }
    }
    pub(super) fn targets(&self) -> &[usize] {
        match &self.premises {
            Premises::Borrowed { targets, .. } => targets,
            Premises::Prepared { targets, .. } => targets,
        }
    }
    pub(super) fn prepared_prefix(&self) -> Option<&PreparedPrefix> {
        match &self.premises {
            Premises::Prepared { prefix, .. } => Some(prefix),
            Premises::Borrowed { .. } => None,
        }
    }
    pub(super) fn borrowed_root(&self) -> Option<&RootGates> {
        match &self.premises {
            Premises::Borrowed { root, .. } => Some(root),
            Premises::Prepared { .. } => None,
        }
    }
    pub(super) fn budget(&self) -> Option<&WorkBudget> {
        self.prepared_prefix().map(|prefix| prefix.root().budget())
    }
    pub(super) fn lo(&self) -> &[f64] {
        match &self.premises {
            Premises::Borrowed { root, .. } => &root.lo,
            Premises::Prepared { prefix, .. } => prefix.root().lo(),
        }
    }
    pub(super) fn hi(&self) -> &[f64] {
        match &self.premises {
            Premises::Borrowed { root, .. } => &root.hi,
            Premises::Prepared { prefix, .. } => prefix.root().hi(),
        }
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn mid(&self) -> &[f64] {
        match &self.premises {
            Premises::Borrowed { root, .. } => &root.mid,
            Premises::Prepared { prefix, .. } => prefix.root().mid(),
        }
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn rad(&self) -> &[f64] {
        match &self.premises {
            Premises::Borrowed { root, .. } => &root.rad,
            Premises::Prepared { prefix, .. } => prefix.root().rad(),
        }
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn xabs(&self) -> &[f64] {
        match &self.premises {
            Premises::Borrowed { root, .. } => &root.xabs,
            Premises::Prepared { prefix, .. } => prefix.root().xabs(),
        }
    }
    pub(super) fn checkpoint(&self, requested: Instant) -> Result<()> {
        let deadline = requested.min(self.deadline);
        if let Some(budget) = self.budget() {
            checkpoint_budget(budget, deadline)
        } else {
            check_deadline(deadline)
        }
    }
    pub(super) fn n_in(&self) -> usize {
        self.net().n_in
    }
    pub(super) fn columns(&self) -> usize {
        self.targets().len()
    }
    pub(super) fn deadline(&self) -> Instant {
        self.deadline
    }
    pub(super) fn resident_floor(&self, requested: Instant) -> Result<usize> {
        self.checkpoint(requested)?;
        match &self.premises {
            Premises::Prepared {
                source_bytes,
                _resident,
                ..
            } => source_bytes
                .checked_add(_resident.bytes())
                .ok_or_else(|| invalid("prepared source capacity overflow")),
            Premises::Borrowed { net, root, targets } => {
                super::engine::input_product_resident_floor_with_deadline(
                    net,
                    root,
                    requested.min(self.deadline),
                )?
                .checked_add(size_of::<Self>())
                .and_then(|v| v.checked_add(targets.len().checked_mul(size_of::<usize>())?))
                .ok_or_else(|| invalid("borrowed source capacity overflow"))
            }
        }
    }
}

/// Actual center/bias backing, also part of any future immutable recipe key.
/// The f32 variant owns binary32 allocations; read views widen their bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum InputRowsBacking {
    F64,
    F32,
}

enum Coefficients {
    F64(ContiguousMatrix<f64>),
    F32(ContiguousMatrix<f32>),
}
enum Biases {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

impl Coefficients {
    fn dim(&self) -> (usize, usize) {
        match self {
            Self::F64(a) => a.dim(),
            Self::F32(a) => a.dim(),
        }
    }
    fn nrows(&self) -> usize {
        self.dim().0
    }
    fn len(&self) -> usize {
        self.dim().0 * self.dim().1
    }
    fn is_standard_layout(&self) -> bool {
        match self {
            Self::F64(a) => a.is_standard_layout(),
            Self::F32(a) => a.is_standard_layout(),
        }
    }
    fn value(&self, row: usize, column: usize) -> f64 {
        match self {
            Self::F64(a) => a[[row, column]],
            Self::F32(a) => f32_to_f64_exact(a[[row, column]]),
        }
    }
    fn backing(&self) -> InputRowsBacking {
        match self {
            Self::F64(_) => InputRowsBacking::F64,
            Self::F32(_) => InputRowsBacking::F32,
        }
    }
}

impl Biases {
    fn len(&self) -> usize {
        match self {
            Self::F64(b) => b.len(),
            Self::F32(b) => b.len(),
        }
    }
    fn bytes(&self) -> Option<usize> {
        match self {
            Self::F64(b) => b.capacity().checked_mul(size_of::<f64>()),
            Self::F32(b) => b.capacity().checked_mul(size_of::<f32>()),
        }
    }
    fn value(&self, column: usize) -> f64 {
        match self {
            Self::F64(b) => b[column],
            Self::F32(b) => f32_to_f64_exact(b[column]),
        }
    }
    fn backing(&self) -> InputRowsBacking {
        match self {
            Self::F64(_) => InputRowsBacking::F64,
            Self::F32(_) => InputRowsBacking::F32,
        }
    }
}

struct Rows {
    a: Coefficients,
    e: ContiguousMatrix<f64>,
    b: Biases,
    eb: Vec<f64>,
    // Actual owned scalar allocation capacities captured while sealing. No
    // matrix copy, scatter, or second coefficient pass is performed.
    matrix_bytes: usize,
}

enum Storage {
    Dense {
        lower: Rows,
        upper: Rows,
    },
    Fused {
        rows: Rows,
        frame: Frame,
        positions: Vec<(usize, usize)>,
    },
}

/// Explicit lower/upper coefficients, errors and constants. Unlike legacy
/// M/D rows, independent constants are never averaged or made symmetric.
pub(super) struct InputRowPair<'a> {
    scope: InputRowScope<'a>,
    storage: Storage,
    // Prepared products transfer the producer's preadmitted actual output
    // reservation. Borrowed legacy products retain their existing cap contract.
    resident: Option<PersistentReservation>,
}

/// Telemetry about actual paid storage delivery, not new proof authority or
/// a claim that f32 arithmetic kernels, default routing or replay ran.
pub(super) struct InputRowsF32Receipt {
    pub(super) coefficient_cells_stored: usize,
    pub(super) bias_cells_stored: usize,
    pub(super) changed_cells: usize,
    pub(super) old_payload_bytes: usize,
    pub(super) new_payload_bytes: usize,
    pub(super) new_nominal_allocation_bytes: usize,
    pub(super) peak_admitted_bytes: usize,
    pub(super) deadline: Instant,
}

pub(super) struct InputRowsF32Transfer<'a> {
    pub(super) rows: InputRowPair<'a>,
    pub(super) receipt: InputRowsF32Receipt,
}

impl<'a> InputRowPair<'a> {
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn from_dense(
        scope: InputRowScope<'a>,
        lower: PassOut,
        upper: PassOut,
        authority: super::engine::DenseInputRowsAuthority,
        deadline: Instant,
    ) -> Result<Self> {
        if scope.prepared_prefix().is_some() {
            return Err(invalid(
                "prepared rows require original persistent reservation",
            ));
        }
        Self::seal_dense(scope, lower, upper, authority, None, deadline)
    }

    pub(super) fn from_prepared_dense(
        scope: InputRowScope<'a>,
        lower: PassOut,
        upper: PassOut,
        authority: super::engine::DenseInputRowsAuthority,
        resident: PersistentReservation,
        deadline: Instant,
    ) -> Result<Self> {
        if scope.prepared_prefix().is_none() {
            return Err(invalid(
                "prepared producer requires immutable original prefix",
            ));
        }
        Self::seal_dense(scope, lower, upper, authority, Some(resident), deadline)
    }

    fn seal_dense(
        mut scope: InputRowScope<'a>,
        lower: PassOut,
        upper: PassOut,
        _authority: super::engine::DenseInputRowsAuthority,
        resident: Option<PersistentReservation>,
        deadline: Instant,
    ) -> Result<Self> {
        let deadline = deadline.min(scope.deadline);
        scope.deadline = deadline;
        scope.checkpoint(deadline)?;
        let convert = |out: PassOut| -> Result<Rows> {
            if out.coll.is_some() || out.coll_rows.is_some() || out.unst_rows.is_some() {
                return Err(invalid("unexpected collection ownership"));
            }
            let e = out
                .e
                .ok_or_else(|| invalid("missing Outward coefficient errors"))?;
            seal_rows(out.a, e, out.b, out.eb, deadline)
        };
        let lower = convert(lower)?;
        let upper = convert(upper)?;
        validate_rows_checked(
            &lower,
            scope.n_in(),
            scope.columns(),
            deadline,
            scope.budget(),
        )?;
        validate_rows_checked(
            &upper,
            scope.n_in(),
            scope.columns(),
            deadline,
            scope.budget(),
        )?;
        let pair = Self {
            scope,
            storage: Storage::Dense { lower, upper },
            resident,
        };
        pair.validate_owner(deadline)?;
        Ok(pair)
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn from_patch(
        scope: InputRowScope<'a>,
        walk: PatchWalk,
        seed_shape: Shape,
        authority: super::patch::PatchInputRowsAuthority,
        deadline: Instant,
    ) -> Result<Self> {
        if scope.prepared_prefix().is_some() {
            return Err(invalid(
                "prepared patch requires original persistent reservation",
            ));
        }
        Self::seal_patch(scope, walk, seed_shape, authority, None, deadline)
    }

    pub(super) fn from_prepared_patch(
        scope: InputRowScope<'a>,
        walk: PatchWalk,
        seed_shape: Shape,
        authority: super::patch::PatchInputRowsAuthority,
        resident: PersistentReservation,
        deadline: Instant,
    ) -> Result<Self> {
        if scope.prepared_prefix().is_none() {
            return Err(invalid("prepared patch requires immutable original prefix"));
        }
        Self::seal_patch(scope, walk, seed_shape, authority, Some(resident), deadline)
    }

    fn seal_patch(
        mut scope: InputRowScope<'a>,
        walk: PatchWalk,
        seed_shape: Shape,
        _authority: super::patch::PatchInputRowsAuthority,
        resident: Option<PersistentReservation>,
        deadline: Instant,
    ) -> Result<Self> {
        let deadline = deadline.min(scope.deadline);
        scope.deadline = deadline;
        scope.checkpoint(deadline)?;
        let columns = scope
            .columns()
            .checked_mul(2)
            .ok_or_else(|| invalid("column overflow"))?;
        let frame = walk.frame;
        let frame_rows = frame
            .c
            .checked_mul(frame.ph)
            .and_then(|n| n.checked_mul(frame.pw))
            .ok_or_else(|| invalid("frame overflow"))?;
        let input_rows = frame
            .c
            .checked_mul(frame.h)
            .and_then(|n| n.checked_mul(frame.w))
            .ok_or_else(|| invalid("original input frame overflow"))?;
        let seed_hw = seed_shape
            .1
            .checked_mul(seed_shape.2)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("original seed frame overflow"))?;
        let seed_tensor = match scope.net().ops.get(scope.relu_op) {
            Some(TwinOp::Relu { input, .. }) => *input,
            _ => return Err(invalid("changed seed operation")),
        };
        if input_rows != scope.n_in()
            || seed_shape.0.checked_mul(seed_hw) != scope.net().tsize.get(seed_tensor).copied()
            || frame.ph == 0
            || frame.pw == 0
            || [
                frame.h,
                frame.w,
                frame.ph,
                frame.pw,
                seed_shape.1,
                seed_shape.2,
            ]
            .iter()
            .any(|&dimension| isize::try_from(dimension).is_err())
            || walk.pos.len() != columns
            || (frame.full
                && (frame.ph != frame.h
                    || frame.pw != frame.w
                    || frame.sy != 0
                    || frame.sx != 0
                    || frame.oy != 0
                    || frame.ox != 0))
        {
            return Err(invalid("original patch frame mismatch"));
        }
        let e = walk
            .e
            .ok_or_else(|| invalid("missing patch Outward coefficient errors"))?;
        let rows = seal_rows(walk.a, e, walk.b, walk.eb, deadline)?;
        validate_rows_checked(&rows, frame_rows, columns, deadline, scope.budget())?;
        for col in 0..columns {
            if col.is_multiple_of(POLL_ITEMS) {
                scope.checkpoint(deadline)?;
            }
            let target = scope.targets()[col % scope.columns()];
            let spatial = target % seed_hw;
            if walk.pos[col] != (spatial / seed_shape.2, spatial % seed_shape.2) {
                return Err(invalid("patch column uses another original target anchor"));
            }
            let mut previous = None;
            for row in 0..frame_rows {
                if row.is_multiple_of(POLL_ITEMS) {
                    scope.checkpoint(deadline)?;
                }
                match input_index(&frame, walk.pos[col], row)? {
                    Some(index) => {
                        if index >= scope.n_in() || previous.is_some_and(|p| index <= p) {
                            return Err(invalid("invalid absolute original-input order"));
                        }
                        previous = Some(index);
                    }
                    None if rows.a.value(row, col) == 0.0 && rows.e[[row, col]] == 0.0 => {}
                    None => return Err(invalid("virtual padding carries a coefficient/error")),
                }
            }
        }
        scope.checkpoint(deadline)?;
        let pair = Self {
            scope,
            storage: Storage::Fused {
                rows,
                frame,
                positions: walk.pos,
            },
            resident,
        };
        pair.validate_owner(deadline)?;
        Ok(pair)
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn validate_scope(
        &self,
        net: &TwinNet,
        root: &RootGates,
        relu_op: usize,
        targets: &[usize],
    ) -> Result<()> {
        self.scope.checkpoint(self.scope.deadline)?;
        if self.scope.prepared_prefix().is_some()
            || !std::ptr::eq(self.scope.net(), net)
            || !self
                .scope
                .borrowed_root()
                .is_some_and(|original| std::ptr::eq(original, root))
            || self.scope.relu_op != relu_op
            || self.scope.targets() != targets
        {
            return Err(invalid("stale model/root/prefix/ordered target scope"));
        }
        Ok(())
    }

    pub(super) fn validate_prepared_prefix(
        &self,
        prefix: &PreparedPrefix,
        relu_op: usize,
        targets: &[usize],
    ) -> Result<()> {
        self.validate_prepared_prefix_with_deadline(prefix, relu_op, targets, self.deadline())
    }

    /// Constant-time original owner/ledger, prefix, operation and count binding.
    /// Ordered target contents are checked separately after consumer admission.
    pub(super) fn validate_prepared_prefix_binding(
        &self,
        prefix: &PreparedPrefix,
        relu_op: usize,
        target_count: usize,
        requested: Instant,
    ) -> Result<()> {
        let deadline = requested.min(self.deadline());
        self.validate_owner(deadline)?;
        let original = self
            .prepared_prefix()
            .ok_or_else(|| invalid("missing owned original prefix"))?;
        if !original.same_prefix(prefix)
            || self.scope.relu_op != relu_op
            || target_count != self.columns()
        {
            return Err(invalid("stale original prefix/operation/ordered targets"));
        }
        original.for_relu(relu_op)?;
        Ok(())
    }

    /// Validate the full ordered producer slice under the earlier caller clock.
    /// The consuming operation must admit this bounded scan before calling it.
    pub(super) fn validate_prepared_prefix_with_deadline(
        &self,
        prefix: &PreparedPrefix,
        relu_op: usize,
        targets: &[usize],
        requested: Instant,
    ) -> Result<()> {
        let deadline = requested.min(self.deadline());
        self.validate_prepared_prefix_binding(prefix, relu_op, targets.len(), deadline)?;
        for (a, b) in self
            .targets()
            .chunks(POLL_ITEMS)
            .zip(targets.chunks(POLL_ITEMS))
        {
            self.checkpoint(deadline)?;
            if a != b {
                return Err(invalid("changed original target order"));
            }
        }
        self.checkpoint(deadline)
    }

    fn validate_owner(&self, requested: Instant) -> Result<()> {
        self.scope.checkpoint(requested)?;
        match (&self.scope.premises, &self.resident) {
            (
                Premises::Prepared {
                    prefix,
                    targets,
                    _resident,
                    ..
                },
                Some(resident),
            ) => {
                ny_core::require_f64_interval_proof_environment()?;
                let budget = prefix.root().budget();
                let required = self
                    .owned_payload_bytes()?
                    .checked_add(INPUT_ROWS_OWNER_BYTES)
                    .ok_or_else(|| invalid("row owner capacity overflow"))?;
                let target_required = targets
                    .capacity()
                    .checked_mul(size_of::<usize>())
                    .and_then(|v| v.checked_add(size_of::<InputRowScope<'static>>()))
                    .and_then(|v| v.checked_add(256))
                    .ok_or_else(|| invalid("target owner capacity overflow"))?;
                if !resident.shares_ledger(budget) || !_resident.shares_ledger(budget) {
                    return Err(invalid("missing/foreign original row reservation"));
                }
                if required > resident.bytes() {
                    return Err(storage_memory(
                        required,
                        resident.bytes(),
                        "input rows actual original row capacity",
                    ));
                }
                if target_required > _resident.bytes() {
                    return Err(storage_memory(
                        target_required,
                        _resident.bytes(),
                        "input rows actual original target capacity",
                    ));
                }
                prefix.for_relu(self.scope.relu_op)?;
                let live = self
                    .scope
                    .resident_floor(requested)?
                    .checked_add(resident.bytes())
                    .ok_or_else(|| invalid("prepared product total-live overflow"))?;
                if live > super::root_box::ROOT_SOURCE_BYTES_CAP {
                    return Err(storage_memory(
                        live,
                        super::root_box::ROOT_SOURCE_BYTES_CAP,
                        "input rows prepared total live",
                    ));
                }
            }
            (Premises::Borrowed { .. }, None) => {}
            _ => return Err(invalid("row storage and source reservation disagree")),
        }
        self.scope.checkpoint(requested)
    }

    pub(super) fn columns(&self) -> usize {
        self.scope.columns()
    }
    pub(super) fn n_in(&self) -> usize {
        self.scope.n_in()
    }
    pub(super) fn deadline(&self) -> Instant {
        self.scope.deadline
    }
    pub(super) fn root(&self) -> &RootGates {
        self.scope
            .borrowed_root()
            .expect("legacy borrowed input-row consumer")
    }
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn borrowed_root(&self) -> Option<&RootGates> {
        self.scope.borrowed_root()
    }
    pub(super) fn prepared_prefix(&self) -> Option<&PreparedPrefix> {
        self.scope.prepared_prefix()
    }
    pub(super) fn targets(&self) -> &[usize] {
        self.scope.targets()
    }
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn net(&self) -> &TwinNet {
        self.scope.net()
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn lo(&self) -> &[f64] {
        self.scope.lo()
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn hi(&self) -> &[f64] {
        self.scope.hi()
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn mid(&self) -> &[f64] {
        self.scope.mid()
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn rad(&self) -> &[f64] {
        self.scope.rad()
    }
    #[expect(dead_code, reason = "bounded input-row integration")]
    pub(super) fn xabs(&self) -> &[f64] {
        self.scope.xabs()
    }
    pub(super) fn checkpoint(&self, requested: Instant) -> Result<()> {
        self.scope.checkpoint(requested)
    }
    #[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
    pub(super) fn resident_bytes(&self) -> Option<usize> {
        self.resident.as_ref().map(PersistentReservation::bytes)
    }
    /// Actual directly visible source and owned-product residency. The caller
    /// still owns opaque caches, other tiles/workers and global admission.
    pub(super) fn resident_floor(&self, requested: Instant) -> Result<usize> {
        let deadline = requested.min(self.scope.deadline);
        self.validate_owner(deadline)?;
        if self.prepared_prefix().is_some() {
            let bytes = self
                .scope
                .resident_floor(deadline)?
                .checked_add(
                    self.resident
                        .as_ref()
                        .ok_or_else(|| invalid("missing original row reservation"))?
                        .bytes(),
                )
                .ok_or_else(|| invalid("prepared source/product capacity overflow"))?;
            self.checkpoint(deadline)?;
            return Ok(bytes);
        }
        let source = super::engine::input_product_resident_floor_with_deadline(
            self.scope.net(),
            self.root(),
            deadline,
        )?;
        let payload = self.owned_payload_bytes()?;
        let bytes = source
            .checked_add(payload)
            .and_then(|bytes| bytes.checked_add(size_of::<Self>()))
            .and_then(|bytes| {
                bytes.checked_add(self.scope.targets().len().checked_mul(size_of::<usize>())?)
            })
            .ok_or_else(|| invalid("source/product capacity overflow"))?;
        self.checkpoint(deadline)?;
        Ok(bytes)
    }

    /// Direct owned payload capacity only; not a source/ledger credit.
    pub(super) fn owned_payload_bytes(&self) -> Result<usize> {
        let row_bytes = |rows: &Rows| {
            rows.matrix_bytes
                .checked_add(rows.b.bytes()?)?
                .checked_add(rows.eb.capacity().checked_mul(size_of::<f64>())?)
        };
        match &self.storage {
            Storage::Dense { lower, upper } => {
                row_bytes(lower).and_then(|bytes| bytes.checked_add(row_bytes(upper)?))
            }
            Storage::Fused {
                rows, positions, ..
            } => row_bytes(rows).and_then(|bytes| {
                bytes.checked_add(
                    positions
                        .capacity()
                        .checked_mul(size_of::<(usize, usize)>())?,
                )
            }),
        }
        .ok_or_else(|| invalid("owned row capacity overflow"))
    }

    pub(super) fn backing(&self) -> InputRowsBacking {
        match &self.storage {
            Storage::Dense { lower, .. } => lower.a.backing(),
            Storage::Fused { rows, .. } => rows.a.backing(),
        }
    }

    fn nominal_cells(&self) -> Result<(usize, usize)> {
        let cells = |rows: &Rows| (rows.a.len(), rows.b.len());
        match &self.storage {
            Storage::Dense { lower, upper } => {
                let (la, lb) = cells(lower);
                let (ua, ub) = cells(upper);
                Ok((
                    la.checked_add(ua)
                        .ok_or_else(|| invalid("coefficient cell overflow"))?,
                    lb.checked_add(ub)
                        .ok_or_else(|| invalid("bias cell overflow"))?,
                ))
            }
            Storage::Fused { rows, .. } => Ok(cells(rows)),
        }
    }

    /// Conservative old-plus-ALL-new nominal overlap. Carried E/eb allocations
    /// move into the result and are already included in the resident floor.
    pub(super) fn f32_storage_required_bytes(
        &self,
        resident: usize,
        requested: Instant,
    ) -> Result<usize> {
        let deadline = requested.min(self.scope.deadline);
        self.checkpoint(deadline)?;
        if self.backing() != InputRowsBacking::F64 {
            return Err(invalid("f32 storage already delivered"));
        }
        let (a, b) = self.nominal_cells()?;
        let extra = a
            .checked_add(b)
            .and_then(|cells| cells.checked_mul(size_of::<f32>()))
            .ok_or_else(|| invalid("f32 storage allocation overflow"))?;
        resident
            .max(self.resident_floor(deadline)?)
            .checked_add(extra)
            .and_then(|bytes| bytes.checked_add(F32_TRANSFER_FIXED_BYTES))
            .ok_or_else(|| invalid("f32 storage total-live overflow"))
    }

    /// Consuming delivery to REAL f32 A/b buffers. Every stored-center gap is
    /// absorbed into moved f64 errors before publication, keeping the original
    /// authenticated model/root/generation/target scope and earliest deadline.
    pub(super) fn into_f32(
        mut self,
        resident: usize,
        budget: usize,
        requested: Instant,
    ) -> Result<InputRowsF32Transfer<'a>> {
        let deadline = requested.min(self.scope.deadline);
        self.checkpoint(deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        self.scope.deadline = deadline;
        self.validate_owner(deadline)?;
        let required = self.f32_storage_required_bytes(resident, deadline)?;
        let budget = budget.min(super::root_box::ROOT_SOURCE_BYTES_CAP);
        if required > budget {
            return Err(storage_memory(
                required,
                budget,
                "input rows f32 storage total live",
            ));
        }
        let (coefficient_cells_stored, bias_cells_stored) = self.nominal_cells()?;
        if coefficient_cells_stored == 0 || bias_cells_stored == 0 {
            return Err(invalid("no eligible cells for f32 storage"));
        }
        let old_payload_bytes = self.owned_payload_bytes()?;
        let new_bytes = coefficient_cells_stored
            .checked_add(bias_cells_stored)
            .and_then(|cells| cells.checked_mul(size_of::<f32>()))
            .ok_or_else(|| invalid("f32 cell capacity overflow"))?;
        // A prepared product cannot substitute the integer cap for admission
        // from its original shared ledger. Retain the old persistent claim
        // (conservatively larger than the delivered payload); all new A/b
        // overlap is separately leased until the old buffers are gone.
        let original_budget = self.scope.budget().cloned();
        let mut overlap_lease: Option<WorkLease> = if let Some(original) = &original_budget {
            let work = coefficient_cells_stored
                .checked_add(bias_cells_stored)
                .and_then(|n| n.checked_mul(64))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| invalid("f32 storage work overflow"))?;
            Some(
                original
                    .lease_for(
                        LeaseClass::Fallback,
                        WorkRequest {
                            scratch_bytes: new_bytes
                                .checked_add(F32_TRANSFER_FIXED_BYTES)
                                .ok_or_else(|| invalid("f32 overlap overflow"))?,
                            work,
                            ..WorkRequest::default()
                        },
                    )
                    .map_err(work_refusal)?,
            )
        } else {
            None
        };
        if let Some(lease) = &mut overlap_lease {
            lease.mark_materialized().map_err(work_refusal)?;
        }
        self.checkpoint(deadline)?;
        let mut admission = StorageAdmission {
            peak: required,
            remaining: new_bytes,
            allocated: 0,
            budget,
            changed: 0,
            original_budget,
        };
        let storage = match self.storage {
            Storage::Dense { lower, upper } => Storage::Dense {
                lower: store_rows_f32(lower, &mut admission, deadline)?,
                upper: store_rows_f32(upper, &mut admission, deadline)?,
            },
            Storage::Fused {
                rows,
                frame,
                positions,
            } => Storage::Fused {
                rows: store_rows_f32(rows, &mut admission, deadline)?,
                frame,
                positions,
            },
        };
        let rows = Self {
            scope: self.scope,
            storage,
            resident: self.resident,
        };
        let new_payload_bytes = rows.owned_payload_bytes()?;
        rows.validate_owner(deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        if admission.remaining != 0 || rows.backing() != InputRowsBacking::F32 {
            return Err(invalid("incomplete f32 storage delivery"));
        }
        if let Some(lease) = &mut overlap_lease {
            lease.complete().map_err(work_refusal)?;
        }
        drop(overlap_lease);
        Ok(InputRowsF32Transfer {
            rows,
            receipt: InputRowsF32Receipt {
                coefficient_cells_stored,
                bias_cells_stored,
                changed_cells: admission.changed,
                old_payload_bytes,
                new_payload_bytes,
                new_nominal_allocation_bytes: admission.allocated,
                peak_admitted_bytes: admission.peak,
                deadline,
            },
        })
    }
    pub(super) fn target(&self, column: usize) -> Option<usize> {
        self.scope.targets().get(column).copied()
    }
    pub(super) fn lower(&self, column: usize) -> Option<InputRowView<'_>> {
        self.view(column, true)
    }
    pub(super) fn upper(&self, column: usize) -> Option<InputRowView<'_>> {
        self.view(column, false)
    }

    fn view(&self, column: usize, lower: bool) -> Option<InputRowView<'_>> {
        if column >= self.columns() {
            return None;
        }
        let (rows, col, mapping) = match &self.storage {
            Storage::Dense {
                lower: lo,
                upper: up,
            } => (if lower { lo } else { up }, column, None),
            Storage::Fused {
                rows,
                frame,
                positions,
            } => {
                let col = column + if lower { 0 } else { self.columns() };
                (rows, col, Some((frame, positions[col])))
            }
        };
        Some(InputRowView {
            rows,
            col,
            mapping,
            lo: self.scope.lo(),
            hi: self.scope.hi(),
            budget: self.scope.budget(),
            deadline: self.scope.deadline,
        })
    }
}

/// Borrowed view of one lane; exact zero extension outside a patch is implicit.
pub(super) struct InputRowView<'a> {
    rows: &'a Rows,
    col: usize,
    mapping: Option<(&'a Frame, (usize, usize))>,
    lo: &'a [f64],
    hi: &'a [f64],
    budget: Option<&'a WorkBudget>,
    deadline: Instant,
}

impl InputRowView<'_> {
    pub(super) fn bias(&self) -> f64 {
        self.rows.b.value(self.col)
    }
    pub(super) fn bias_error(&self) -> f64 {
        self.rows.eb[self.col]
    }
    pub(super) fn entries(&self) -> InputRowEntries<'_> {
        InputRowEntries {
            view: self,
            next: 0,
        }
    }

    /// Upper bound on algebraic producer error over the DECLARED input box.
    /// Consumers must separately certify their nominal dot/dual evaluation.
    pub(super) fn pointwise_error(&self, deadline: Instant) -> Result<f64> {
        let deadline = deadline.min(self.deadline);
        checkpoint_maybe(self.budget, deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        // Direct owned-view callers share the same finite work allowance;
        // repeated error scans cannot bypass consumer admission. Virtual patch
        // padding is conservatively included in this visible row-length bound.
        let mut scan = if let Some(budget) = self.budget {
            let work = self
                .rows
                .a
                .nrows()
                .checked_mul(32)
                .and_then(|n| n.checked_add(256))
                .and_then(|n| u64::try_from(n).ok())
                .ok_or_else(|| invalid("pointwise error work overflow"))?;
            Some(
                budget
                    .lease_for(
                        LeaseClass::Fallback,
                        WorkRequest {
                            work,
                            ..WorkRequest::default()
                        },
                    )
                    .map_err(work_refusal)?,
            )
        } else {
            None
        };
        let mut error = self.bias_error();
        for (count, (i, _, e)) in self.entries().enumerate() {
            if count.is_multiple_of(POLL_ITEMS) {
                checkpoint_maybe(self.budget, deadline)?;
            }
            let magnitude = self.lo[i].abs().max(self.hi[i].abs());
            let term = mul_up(e, magnitude);
            if term != 0.0 {
                error = next_up(error + term);
            }
        }
        checkpoint_maybe(self.budget, deadline)?;
        if !error.is_finite() {
            return Err(invalid("non-finite pointwise error"));
        }
        if let Some(lease) = &mut scan {
            lease.complete().map_err(work_refusal)?;
        }
        Ok(error)
    }
}

pub(super) struct InputRowEntries<'a> {
    view: &'a InputRowView<'a>,
    next: usize,
}

impl Iterator for InputRowEntries<'_> {
    type Item = (usize, f64, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((frame, pos)) = self.view.mapping {
            // Jump over entire virtual-padding rectangles, rather than walking
            // an unbounded zero tail between deadline-polled real entries.
            self.next = next_real_row(frame, pos, self.next)?;
        }
        while self.next < self.view.rows.a.nrows() {
            let row = self.next;
            self.next += 1;
            let index = match self.view.mapping {
                None => Some(row),
                Some((frame, pos)) => {
                    input_index(frame, pos, row).expect("sealed original-input mapping")
                }
            };
            if let Some(i) = index {
                return Some((
                    i,
                    self.view.rows.a.value(row, self.view.col),
                    self.view.rows.e[[row, self.view.col]],
                ));
            }
        }
        None
    }
}

fn next_real_row(frame: &Frame, pos: (usize, usize), start: usize) -> Option<usize> {
    let plane = frame.ph * frame.pw;
    let ay = (pos.0 as isize)
        .checked_mul(frame.sy)?
        .checked_add(frame.oy)?;
    let ax = (pos.1 as isize)
        .checked_mul(frame.sx)?
        .checked_add(frame.ox)?;
    let bound = |value: isize, max: usize| value.max(0).min(max as isize) as usize;
    let y0 = bound(ay.checked_neg()?, frame.ph);
    let x0 = bound(ax.checked_neg()?, frame.pw);
    let y1 = bound((frame.h as isize).checked_sub(ay)?, frame.ph);
    let x1 = bound((frame.w as isize).checked_sub(ax)?, frame.pw);
    if y0 >= y1 || x0 >= x1 {
        return None;
    }
    let (mut ch, mut y, mut x) = (start / plane, (start % plane) / frame.pw, start % frame.pw);
    if y < y0 {
        y = y0;
        x = x0;
    }
    if x < x0 {
        x = x0;
    }
    if x >= x1 {
        y += 1;
        x = x0;
    }
    if y >= y1 {
        ch += 1;
        y = y0;
        x = x0;
    }
    (ch < frame.c).then(|| ch * plane + y * frame.pw + x)
}

fn input_index(frame: &Frame, pos: (usize, usize), row: usize) -> Result<Option<usize>> {
    let plane = frame
        .ph
        .checked_mul(frame.pw)
        .filter(|&n| n > 0)
        .ok_or_else(|| invalid("patch plane overflow"))?;
    let ch = row / plane;
    let iy = (row % plane) / frame.pw;
    let ix = row % frame.pw;
    let anchor = |position: usize, stride: isize, offset: isize| -> Result<isize> {
        isize::try_from(position)
            .ok()
            .and_then(|p| p.checked_mul(stride))
            .and_then(|p| p.checked_add(offset))
            .ok_or_else(|| invalid("patch anchor overflow"))
    };
    let y = anchor(pos.0, frame.sy, frame.oy)?
        .checked_add(isize::try_from(iy).map_err(|_| invalid("patch y overflow"))?)
        .ok_or_else(|| invalid("patch y overflow"))?;
    let x = anchor(pos.1, frame.sx, frame.ox)?
        .checked_add(isize::try_from(ix).map_err(|_| invalid("patch x overflow"))?)
        .ok_or_else(|| invalid("patch x overflow"))?;
    if ch >= frame.c || y < 0 || x < 0 || y as usize >= frame.h || x as usize >= frame.w {
        return Ok(None);
    }
    ch.checked_mul(frame.h)
        .and_then(|v| v.checked_add(y as usize))
        .and_then(|v| v.checked_mul(frame.w))
        .and_then(|v| v.checked_add(x as usize))
        .map(Some)
        .ok_or_else(|| invalid("absolute input index overflow"))
}

#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
fn validate_rows(rows: &Rows, n: usize, columns: usize, deadline: Instant) -> Result<()> {
    validate_rows_checked(rows, n, columns, deadline, None)
}

fn validate_rows_checked(
    rows: &Rows,
    n: usize,
    columns: usize,
    deadline: Instant,
    budget: Option<&WorkBudget>,
) -> Result<()> {
    checkpoint_maybe(budget, deadline)?;
    if rows.a.dim() != (n, columns)
        || rows.e.dim() != rows.a.dim()
        || !rows.a.is_standard_layout()
        || !rows.e.is_standard_layout()
        || rows.b.len() != columns
        || rows.eb.len() != columns
    {
        return Err(invalid("coefficient/error/bias shape/layout mismatch"));
    }
    if rows.a.backing() != rows.b.backing() {
        return Err(invalid("mixed nominal storage delivery"));
    }
    for (i, &e) in rows.e.iter().enumerate() {
        if i.is_multiple_of(POLL_ITEMS) {
            checkpoint_maybe(budget, deadline)?;
        }
        let a = rows.a.value(i / columns, i % columns);
        if !a.is_finite() || !e.is_finite() || e < 0.0 {
            return Err(invalid("invalid coefficient/error lane"));
        }
    }
    for (i, &e) in rows.eb.iter().enumerate() {
        if i.is_multiple_of(POLL_ITEMS) {
            checkpoint_maybe(budget, deadline)?;
        }
        let b = rows.b.value(i);
        if !b.is_finite() || !e.is_finite() || e < 0.0 {
            return Err(invalid("invalid bias/error lane"));
        }
    }
    checkpoint_maybe(budget, deadline)
}

fn seal_matrix(matrix: Array2<f64>, deadline: Instant) -> Result<(ContiguousMatrix<f64>, usize)> {
    check_deadline(deadline)?;
    let matrix = ContiguousMatrix::from_array(matrix).map_err(|refusal| {
        invalid(match refusal {
            MatrixRefusal::Layout => "nonstandard owned matrix origin",
            MatrixRefusal::Origin => "nonzero or oversized owned matrix origin",
            MatrixRefusal::Shape => "owned matrix reconstruction shape",
        })
    })?;
    let cells = matrix.len();
    let bytes = matrix
        .capacity()
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| invalid("owned matrix capacity overflow"))?;
    // Keep the original exact-capacity admission. The original zero-offset
    // vector moves into the compact carrier without reconstruction or a copy.
    if matrix.capacity() != cells {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: bytes,
            budget_bytes: cells * size_of::<f64>(),
            site: "input rows owned matrix allocator capacity",
        });
    }
    check_deadline(deadline)?;
    Ok((matrix, bytes))
}

fn seal_rows(
    a: Array2<f64>,
    e: Array2<f64>,
    b: Vec<f64>,
    eb: Vec<f64>,
    deadline: Instant,
) -> Result<Rows> {
    let (a, a_bytes) = seal_matrix(a, deadline)?;
    let (e, e_bytes) = seal_matrix(e, deadline)?;
    let matrix_bytes = a_bytes
        .checked_add(e_bytes)
        .ok_or_else(|| invalid("owned matrix pair capacity overflow"))?;
    Ok(Rows {
        a: Coefficients::F64(a),
        e,
        b: Biases::F64(b),
        eb,
        matrix_bytes,
    })
}

// Fixed stack/header overlap while a producer's old payload and all new
// nominal allocations remain resident. No second f64 nominal array is made.
const F32_TRANSFER_FIXED_BYTES: usize =
    size_of::<InputRowsF32Transfer<'static>>() + 4 * size_of::<Vec<f32>>() + 256;
const _: () = assert!(INPUT_ROWS_OWNER_BYTES >= size_of::<InputRowsF32Transfer<'static>>() + 256);

struct StorageAdmission {
    peak: usize,
    remaining: usize,
    allocated: usize,
    budget: usize,
    changed: usize,
    original_budget: Option<WorkBudget>,
}

fn storage_memory(required: usize, budget: usize, site: &'static str) -> NyError {
    NyError::CpuMemoryExceeded {
        required_bytes: required,
        budget_bytes: budget,
        site,
    }
}

impl StorageAdmission {
    fn checkpoint(&self, deadline: Instant) -> Result<()> {
        checkpoint_maybe(self.original_budget.as_ref(), deadline)
    }
    fn allocate(&mut self, cells: usize, deadline: Instant) -> Result<Vec<f32>> {
        self.checkpoint(deadline)?;
        let requested = cells
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid("f32 reserve overflow"))?;
        if requested > self.remaining {
            return Err(invalid("unpaid f32 reserve"));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(cells)
            .map_err(|_| storage_memory(self.peak, self.budget, "input rows f32 allocation"))?;
        self.checkpoint(deadline)?;
        let actual = values
            .capacity()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid("f32 allocator capacity overflow"))?;
        let overage = actual
            .checked_sub(requested)
            .ok_or_else(|| invalid("f32 allocator short reserve"))?;
        if self.original_budget.is_some() && overage != 0 {
            return Err(storage_memory(
                actual,
                requested,
                "input rows f32 original ledger capacity",
            ));
        }
        self.peak = self
            .peak
            .checked_add(overage)
            .ok_or_else(|| invalid("f32 overlap overflow"))?;
        if self.peak > self.budget {
            return Err(storage_memory(
                self.peak,
                self.budget,
                "input rows f32 actual allocator overlap",
            ));
        }
        self.remaining -= requested;
        self.allocated = self
            .allocated
            .checked_add(actual)
            .ok_or_else(|| invalid("f32 actual allocation overflow"))?;
        Ok(values)
    }
}

/// The complete signed source-minus-stored difference is head+residual under
/// qualified binary64 TwoSum semantics. An outward magnitude sum pays both
/// pieces, including gaps below an ulp of a nonzero subtraction head.
fn f32_storage_gap(source: f64, stored: f32) -> Result<f64> {
    let wide = f32_to_f64_exact(stored);
    if !source.is_finite() || !wide.is_finite() {
        return Err(invalid("non-finite f32 storage center"));
    }
    let (head, residual) = two_sum(source, -wide);
    let gap = add_error_up(head.abs(), residual.abs())?;
    Ok(gap)
}

fn add_error_up(carried: f64, extra: f64) -> Result<f64> {
    if !carried.is_finite() || carried < 0.0 || !extra.is_finite() || extra < 0.0 {
        return Err(invalid("invalid f32 conversion error"));
    }
    let sum = if extra == 0.0 {
        carried
    } else if carried == 0.0 {
        extra
    } else {
        next_up(carried + extra)
    };
    if !sum.is_finite() {
        return Err(invalid("f32 conversion error overflow"));
    }
    Ok(sum)
}

#[allow(clippy::cast_possible_truncation)]
fn store_cell(source: f64, error: &mut f64, admission: &mut StorageAdmission) -> Result<f32> {
    let stored = source as f32;
    let gap = f32_storage_gap(source, stored)?;
    *error = add_error_up(*error, gap)?;
    if source.to_bits() != f32_to_f64_exact(stored).to_bits() {
        admission.changed = admission
            .changed
            .checked_add(1)
            .ok_or_else(|| invalid("f32 changed-cell overflow"))?;
    }
    Ok(stored)
}

fn store_rows_f32(rows: Rows, admission: &mut StorageAdmission, deadline: Instant) -> Result<Rows> {
    admission.checkpoint(deadline)?;
    let Rows {
        a,
        mut e,
        b,
        mut eb,
        matrix_bytes,
    } = rows;
    let (Coefficients::F64(a), Biases::F64(b)) = (a, b) else {
        return Err(invalid("f32 storage already delivered"));
    };
    let shape = a.dim();
    let a_cells = a.len();
    let old_a_bytes = a_cells
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| invalid("old nominal capacity overflow"))?;
    let error_bytes = matrix_bytes
        .checked_sub(old_a_bytes)
        .ok_or_else(|| invalid("missing sealed error capacity"))?;
    if !a.is_standard_layout()
        || !e.is_standard_layout()
        || e.dim() != shape
        || b.len() != eb.len()
        || shape.1 != b.len()
    {
        return Err(invalid("f32 conversion shape/layout mismatch"));
    }
    // Both nominal destinations are prepaid before filling. The existing
    // f64 E/eb buffers remain owned and mutate only inside this transaction.
    let mut stored_a = admission.allocate(a_cells, deadline)?;
    let mut stored_b = admission.allocate(b.len(), deadline)?;
    let new_a_bytes = stored_a
        .capacity()
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid("new nominal capacity overflow"))?;
    for (i, (&source, error)) in a.iter().zip(e.iter_mut()).enumerate() {
        if i.is_multiple_of(POLL_ITEMS) {
            admission.checkpoint(deadline)?;
        }
        stored_a.push(store_cell(source, error, admission)?);
    }
    for (i, (&source, error)) in b.iter().zip(eb.iter_mut()).enumerate() {
        if i.is_multiple_of(POLL_ITEMS) {
            admission.checkpoint(deadline)?;
        }
        stored_b.push(store_cell(source, error, admission)?);
    }
    let a = ContiguousMatrix::from_shape_vec(shape, stored_a)
        .map_err(|_| invalid("f32 storage matrix shape"))?;
    let rows = Rows {
        a: Coefficients::F32(a),
        e,
        b: Biases::F32(stored_b),
        eb,
        matrix_bytes: error_bytes
            .checked_add(new_a_bytes)
            .ok_or_else(|| invalid("f32 matrix capacity overflow"))?,
    };
    validate_rows_checked(
        &rows,
        shape.0,
        shape.1,
        deadline,
        admission.original_budget.as_ref(),
    )?;
    admission.checkpoint(deadline)?;
    Ok(rows)
}

fn invalid(message: &str) -> NyError {
    NyError::InvalidSpec(format!("input rows: {message}"))
}
fn work_refusal(refusal: WorkRefusal) -> NyError {
    if refusal == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("input rows original budget".into())
    } else {
        invalid(&format!("original budget refused: {refusal:?}"))
    }
}
fn checkpoint_budget(budget: &WorkBudget, deadline: Instant) -> Result<()> {
    budget
        .checkpoint_for(LeaseClass::Fallback)
        .map_err(work_refusal)?;
    check_deadline(deadline)
}
fn checkpoint_maybe(budget: Option<&WorkBudget>, deadline: Instant) -> Result<()> {
    if let Some(budget) = budget {
        checkpoint_budget(budget, deadline)
    } else {
        check_deadline(deadline)
    }
}
#[cfg_attr(not(test), expect(dead_code, reason = "bounded input-row integration"))]
fn poll(deadline: Instant, index: usize) -> Result<()> {
    if index.is_multiple_of(POLL_ITEMS) {
        check_deadline(deadline)?;
    }
    Ok(())
}
fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        next_up(underflow_product_up(a, b))
    }
}

#[cfg(test)]
#[path = "input_rows_tests.rs"]
mod tests;
