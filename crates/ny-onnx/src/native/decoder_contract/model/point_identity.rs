// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bounded diagnostic identities. No tensor-sized JSON serialization, no
//! certificate authority, and no hardware/kernel qualification is inferred.

use super::super::{DecoderContractRefusal as Refusal, Result};
use ndarray::{ArrayD, IxDyn};
use ny_propagate::{certify::work::WorkLease, network::GraphPointTrace};
use sha2::{Digest, Sha256};
use std::time::Instant;

pub(in super::super) fn poll(lease: &WorkLease, deadline: Option<Instant>) -> Result<()> {
    lease.checkpoint().map_err(Refusal::Budget)?;
    if deadline.is_some_and(|d| Instant::now() >= d) {
        Err(Refusal::Deadline)
    } else {
        Ok(())
    }
}

fn field(
    hash: &mut Sha256,
    bytes: &[u8],
    lease: &WorkLease,
    deadline: Option<Instant>,
) -> Result<()> {
    poll(lease, deadline)?;
    hash.update((bytes.len() as u64).to_le_bytes());
    for chunk in bytes.chunks(4096) {
        poll(lease, deadline)?;
        hash.update(chunk);
    }
    Ok(())
}

pub(in super::super) fn array_digest(
    array: &ArrayD<f32>,
    lease: &WorkLease,
    deadline: Option<Instant>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    field(&mut hash, b"ny-decoder-point-f32-bits/v1", lease, deadline)?;
    hash.update((array.ndim() as u64).to_le_bytes());
    for &dimension in array.shape() {
        poll(lease, deadline)?;
        hash.update((dimension as u64).to_le_bytes());
    }
    let mut buffer = [0u8; 4096];
    let mut used = 0;
    for (index, &value) in array.iter().enumerate() {
        if index % 1024 == 0 {
            poll(lease, deadline)?;
        }
        if !value.is_finite() {
            return Err(Refusal::Tensor("nonfinite diagnostic point".into()));
        }
        buffer[used..used + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        used += 4;
        if used == buffer.len() {
            hash.update(buffer);
            used = 0;
        }
    }
    if used > 0 {
        hash.update(&buffer[..used]);
    }
    poll(lease, deadline)?;
    Ok(hash.finalize().into())
}

pub(in super::super) fn copy_array(
    array: &ArrayD<f32>,
    lease: &WorkLease,
    deadline: Option<Instant>,
) -> Result<ArrayD<f32>> {
    poll(lease, deadline)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(array.len())
        .map_err(|e| Refusal::Build(format!("point copy allocation: {e}")))?;
    for (index, &value) in array.iter().enumerate() {
        if index % 1024 == 0 {
            poll(lease, deadline)?;
        }
        values.push(value);
    }
    let result = ArrayD::from_shape_vec(IxDyn(array.shape()), values)
        .map_err(|e| Refusal::Build(e.to_string()))?;
    poll(lease, deadline)?;
    Ok(result)
}

pub(in super::super) fn copy_name(
    name: &str,
    lease: &WorkLease,
    deadline: Option<Instant>,
) -> Result<String> {
    poll(lease, deadline)?;
    let mut result = String::new();
    result
        .try_reserve_exact(name.len())
        .map_err(|e| Refusal::Build(format!("point name allocation: {e}")))?;
    let mut start = 0;
    while start < name.len() {
        poll(lease, deadline)?;
        let mut end = (start + 4096).min(name.len());
        while !name.is_char_boundary(end) {
            end -= 1;
        }
        result.push_str(&name[start..end]);
        start = end;
    }
    poll(lease, deadline)?;
    Ok(result)
}

pub(in super::super) fn observation_digest(
    lowering: &[u8; 32],
    selected_output: &str,
    input: &[u8; 32],
    output: &[u8; 32],
    trace: &GraphPointTrace,
    lease: &WorkLease,
    deadline: Option<Instant>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    field(
        &mut hash,
        b"ny-decoder-completed-point-observation/v1",
        lease,
        deadline,
    )?;
    field(&mut hash, lowering, lease, deadline)?;
    field(&mut hash, selected_output.as_bytes(), lease, deadline)?;
    field(&mut hash, input, lease, deadline)?;
    field(&mut hash, output, lease, deadline)?;
    hash.update(trace.schema_version.to_le_bytes());
    field(&mut hash, trace.qualification.as_bytes(), lease, deadline)?;
    hash.update((trace.steps().len() as u64).to_le_bytes());
    for step in trace.steps() {
        poll(lease, deadline)?;
        field(&mut hash, step.node.as_bytes(), lease, deadline)?;
        field(&mut hash, step.operator.as_bytes(), lease, deadline)?;
        let (arm, backend, other) = step.arm.identity_fields();
        field(&mut hash, arm.as_bytes(), lease, deadline)?;
        field(&mut hash, backend.as_bytes(), lease, deadline)?;
        hash.update([u8::from(other.is_some())]);
        if let Some(other) = other {
            field(&mut hash, other.as_bytes(), lease, deadline)?;
        }
    }
    poll(lease, deadline)?;
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::certify::work::{WorkBudget, WorkRequest};
    use std::time::Duration;

    #[test]
    fn exact_point_bit_identity_preserves_signed_zero_and_subnormal_payload() {
        let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1000, 1, Duration::ZERO).unwrap();
        let lease = budget
            .lease(WorkRequest {
                work: 1000,
                proposals: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        let values = ArrayD::from_shape_vec(
            IxDyn(&[2]),
            vec![f32::from_bits(0x8000_0000), f32::from_bits(1)],
        )
        .unwrap();
        let actual = array_digest(&values, &lease, None).unwrap();
        let mut oracle = Sha256::new();
        let domain = b"ny-decoder-point-f32-bits/v1";
        oracle.update((domain.len() as u64).to_le_bytes());
        oracle.update(domain);
        oracle.update(1u64.to_le_bytes());
        oracle.update(2u64.to_le_bytes());
        oracle.update(0x8000_0000u32.to_le_bytes());
        oracle.update(1u32.to_le_bytes());
        let expected: [u8; 32] = oracle.finalize().into();
        assert_eq!(actual, expected);
        let copied = copy_array(&values, &lease, None).unwrap();
        assert_eq!(copied[[0]].to_bits(), 0x8000_0000);
        assert_eq!(copied[[1]].to_bits(), 1);
        let changed = ArrayD::zeros(IxDyn(&[2]));
        assert_ne!(array_digest(&changed, &lease, None).unwrap(), actual);
    }

    #[test]
    fn point_bit_hash_and_copy_refuse_the_existing_deadline_before_publication() {
        let budget = WorkBudget::new(None, 1 << 20, 1 << 20, 1000, 1, Duration::ZERO).unwrap();
        let lease = budget.lease(WorkRequest::default()).unwrap();
        let values = ArrayD::zeros(IxDyn(&[1024]));
        let expired = Some(Instant::now());
        assert!(matches!(
            array_digest(&values, &lease, expired),
            Err(Refusal::Deadline)
        ));
        assert!(matches!(
            copy_array(&values, &lease, expired),
            Err(Refusal::Deadline)
        ));
        assert_eq!(budget.receipt().unwrap().completed, 0);
    }
}
