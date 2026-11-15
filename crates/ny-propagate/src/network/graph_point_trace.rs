// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Optional completed diagnostic execution trace. This records software arms,
//! not a numerical certificate or a deployed hardware/kernel qualification.

use super::core::GraphNetwork;
use crate::layers::linear::LinearPointReceipt;
use crate::layers::point_linear::PointLinearReceipt;
use ny_core::{GemmEngine, NyError, Result};
use serde::Serialize;
use std::time::Instant;

const MAX_TRACE_BYTES: usize = 32 * 1024 * 1024;
const MAX_TRACE_STEPS: usize = 100_000;
const MAX_NAME_BYTES: usize = 64 * 1024;
const MAX_BACKEND_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "arm", rename_all = "snake_case")]
pub enum GraphPointArm {
    Linear {
        receipt: LinearPointReceipt,
    },
    PointLinear {
        receipt: PointLinearReceipt,
    },
    CausalSoftmaxF64Libm,
    /// The generic dispatch may internally choose its own implementation.
    /// A configured engine is recorded as configuration, not as an observed
    /// successful backend. The interval midpoint remains diagnostic only.
    IntervalCenter {
        configured_backend: Option<&'static str>,
    },
    ResidentGpuDag {
        /// This names a successful GPU-plan API arm. Its software provenance
        /// may remain unidentified; the API name does not establish hardware.
        backend: &'static str,
    },
}

impl GraphPointArm {
    /// Stable explicit fields for a bounded streaming identity, with no JSON
    /// serialization or allocation proportional to the tensor payload.
    pub fn identity_fields(self) -> (&'static str, &'static str, Option<&'static str>) {
        match self {
            Self::Linear { receipt } => (
                receipt.arm.identity_name(),
                receipt.backend,
                receipt.declined_backend,
            ),
            Self::PointLinear { receipt } => (receipt.arm.identity_name(), receipt.backend, None),
            Self::CausalSoftmaxF64Libm => (
                "causal-softmax-stable-f64-libm-one-f32-store/v1",
                "platform-f64-libm-unqualified",
                None,
            ),
            Self::IntervalCenter { configured_backend } => (
                "generic-interval-center-diagnostic/v1",
                "generic-dispatch-backend-unobserved",
                configured_backend,
            ),
            Self::ResidentGpuDag { backend } => {
                ("successful-resident-dag-api-center/v1", backend, None)
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct GraphPointStep {
    pub node: String,
    pub operator: &'static str,
    pub arm: GraphPointArm,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphPointTraceAdmission {
    pub persistent_bytes: usize,
    pub scratch_bytes: usize,
    pub work: u64,
}

#[derive(Debug, Serialize)]
pub struct GraphPointTrace {
    pub schema_version: u32,
    pub qualification: &'static str,
    steps: Vec<GraphPointStep>,
}

fn checkpoint(deadline: Option<Instant>) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded("concrete graph trace".into()))
    } else {
        Ok(())
    }
}

impl GraphPointTrace {
    pub fn steps(&self) -> &[GraphPointStep] {
        &self.steps
    }

    pub fn retained_bytes(&self) -> Result<usize> {
        self.retained_bytes_with_deadline(ny_core::instance_budget::deadline())
    }

    pub fn retained_bytes_with_deadline(&self, deadline: Option<Instant>) -> Result<usize> {
        self.steps.iter().enumerate().try_fold(
            self.steps
                .capacity()
                .checked_mul(size_of::<GraphPointStep>())
                .and_then(|n| n.checked_add(size_of::<Self>()))
                .ok_or_else(|| NyError::InvalidSpec("graph trace retained overflow".into()))?,
            |n, (index, step)| {
                if index % 64 == 0 {
                    checkpoint(deadline)?;
                }
                n.checked_add(step.node.capacity())
                    .ok_or_else(|| NyError::InvalidSpec("graph trace retained overflow".into()))
            },
        )
    }

    pub(crate) fn new(network: &GraphNetwork, deadline: Option<Instant>) -> Result<Self> {
        checkpoint(deadline)?;
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(network.node_names().len().max(1))
            .map_err(|e| NyError::InvalidSpec(format!("graph trace allocation: {e}")))?;
        checkpoint(deadline)?;
        Ok(Self {
            schema_version: 1,
            qualification: "noncertifying-diagnostic/software-provenance-only/hardware-driver-libm-unqualified/v1",
            steps,
        })
    }

    pub(crate) fn push(
        &mut self,
        node: &str,
        operator: &'static str,
        arm: GraphPointArm,
        deadline: Option<Instant>,
    ) -> Result<()> {
        checkpoint(deadline)?;
        let (_, backend, other) = arm.identity_fields();
        if self.steps.len() == self.steps.capacity()
            || node.len() > MAX_NAME_BYTES
            || backend.len() > MAX_BACKEND_BYTES
            || other.is_some_and(|backend| backend.len() > MAX_BACKEND_BYTES)
        {
            return Err(NyError::UnsupportedOp(
                "graph trace exceeded admission".into(),
            ));
        }
        let mut name = String::new();
        name.try_reserve_exact(node.len())
            .map_err(|e| NyError::InvalidSpec(format!("graph trace name allocation: {e}")))?;
        let mut start = 0;
        while start < node.len() {
            checkpoint(deadline)?;
            let mut end = (start + 4096).min(node.len());
            while !node.is_char_boundary(end) {
                end -= 1;
            }
            name.push_str(&node[start..end]);
            start = end;
        }
        self.steps.push(GraphPointStep {
            node: name,
            operator,
            arm,
        });
        checkpoint(deadline)
    }
}

impl GraphNetwork {
    /// Preflight the optional trace before its first allocation. Its returned
    /// bytes/work must join the caller's original point-execution lease.
    pub fn point_trace_admission(
        &self,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<GraphPointTraceAdmission> {
        checkpoint(deadline)?;
        let count = self.node_names().len();
        let backend = engine.map(|e| e.backend_provenance());
        if count > MAX_TRACE_STEPS || backend.is_some_and(|b| b.len() > MAX_BACKEND_BYTES) {
            return Err(NyError::UnsupportedOp(
                "graph trace metadata profile".into(),
            ));
        }
        let mut names = 0usize;
        for (index, name) in self.node_names().iter().enumerate() {
            if index % 64 == 0 {
                checkpoint(deadline)?;
            }
            if name.len() > MAX_NAME_BYTES {
                return Err(NyError::UnsupportedOp(
                    "graph trace node-name profile".into(),
                ));
            }
            names = names
                .checked_add(name.len())
                .ok_or_else(|| NyError::InvalidSpec("graph trace size overflow".into()))?;
        }
        let persistent_bytes = count
            .max(1)
            .checked_mul(size_of::<GraphPointStep>())
            .and_then(|n| n.checked_add(names))
            .and_then(|n| n.checked_add(16 * 1024))
            .ok_or_else(|| NyError::InvalidSpec("graph trace storage overflow".into()))?;
        if persistent_bytes > MAX_TRACE_BYTES {
            return Err(NyError::UnsupportedOp(
                "graph trace memory admission".into(),
            ));
        }
        // Copy/check/hash every named step and its possible actual/declined
        // backend. Static provenance strings are borrowed, not heap copies.
        let metadata = count
            .max(1)
            .checked_mul(512 + 2 * engine.map_or(0, |_| MAX_BACKEND_BYTES))
            .and_then(|n| n.checked_add(names))
            .ok_or_else(|| NyError::InvalidSpec("graph trace work overflow".into()))?;
        let work = (metadata as u64)
            .checked_mul(8)
            .ok_or_else(|| NyError::InvalidSpec("graph trace work overflow".into()))?;
        checkpoint(deadline)?;
        Ok(GraphPointTraceAdmission {
            persistent_bytes,
            scratch_bytes: 16 * 1024,
            work,
        })
    }
}
