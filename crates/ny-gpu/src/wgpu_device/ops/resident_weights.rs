// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident constant-weight buffer cache (weight residency: upload each
//! constant weight form once per model and reuse it across BaB domains).
//!
//! During BaB the resnet weights are provably constant across domains (all
//! domains share the SAME weight `Arc<[f32]>`; only `Activation` relaxation
//! state mutates per domain — `write_back_alpha`, batched.rs). Yet the sound
//! resident CROWN backward re-uploaded every layer's weight (and a CPU-computed
//! `|W|`, and the joint adjoint a CPU-computed `Wᵀ`) to the GPU **per domain per
//! call**, leaving the GPU ~60 % idle. This cache uploads each constant weight
//! form ONCE per model and keeps it GPU-resident across calls.
//!
//! ## Keying + the keep-alive soundness guard (mandatory)
//!
//! Entries are keyed by **`Arc` pointer identity + length + derived form**
//! (mirroring `crown_plan_key::hash_arc_identity` / `ConvTransposePlanKey`).
//! Distinct live allocations have distinct data addresses, so a pointer hit is
//! a true content hit — **provided the keyed allocation is still alive**. Each
//! entry therefore retains a clone of the weight `Arc` (the keep-alive guard,
//! mirroring `PreparedCrownPlan::static_weight_arcs`): a cached `Arc` cannot be
//! freed, so its address can never be recycled into a *different* weight and
//! silently serve stale bytes (stale hit → wrong weights → wrong bounds →
//! false-VERIFIED). Do not remove it.
//!
//! ## Bit-identical / soundness
//!
//! This is a **pure residency** change: it alters *where the constant bytes
//! live*, never *what is computed*. The derived forms are pure deterministic
//! functions of the immutable `Arc` bytes — `Abs` is the elementwise `|w|`
//! (exact in IEEE-754), `Transposed` is a permutation — so serving a cached
//! upload is byte-identical to recomputing + re-uploading per call. Shader
//! interfaces are untouched: the same read-only storage bindings receive the
//! same bytes. Per-domain data (seeds, boxes, slopes/intercepts, β) is NOT
//! cached — it stays a per-call upload.
//!
//! ## Cache-clear invariant (mirrors `clear_conv_transpose_plan_cache`)
//!
//! The cache is cleared between models via
//! [`clear_resident_weight_buffers`](WgpuDevice::clear_resident_weight_buffers),
//! invoked from `clear_crown_working_set` alongside the CROWN and
//! conv-transpose plan caches, so all pointer-keyed entries (and their
//! keep-alive `Arc`s) drop together before any new model's weights are built.
//!
//! ## A failed upload is never cached
//!
//! wgpu reports a failed `create_buffer` or `write_buffer` (out of memory, or
//! a size past a device limit) through the error scopes and still hands back a
//! buffer handle, so a cached entry could hold an invalid buffer that fails
//! every later call with the same weight until `clear_crown_working_set`. A
//! miss therefore uploads inside its own error scope and caches the buffer
//! only when that scope is clean. Inside a checked transaction (the
//! deadline-bounded intermediate sweeps) a nested scope is not allowed: its
//! pop has no deadline. There the transaction's own scopes cover the upload,
//! the new entry stays UNCOMMITTED, a clean `finish` commits it, and a
//! transaction that ends any other way evicts it
//! (`discard_transaction_resident_weights`).
//!
//! ## Gate
//!
//! `NY_RESIDENT_WEIGHTS=0` opts out (per-call upload, no caching) — the A/B
//! lever for the differential oracle. Both modes bind identical bytes, so they
//! are value-identical by construction.

use std::sync::Arc;

use ny_core::{NyError, Result};

use super::super::error_scope::run_gpu_checked_on_device;
use super::super::WgpuDevice;

/// Which derived form of the constant weight bytes a resident buffer holds.
///
/// All forms are pure deterministic functions of the immutable `Arc<[f32]>`
/// contents, so caching them is bit-identical to per-call recomputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WeightForm {
    /// The weight bytes as-is.
    Raw,
    /// Elementwise `|w|` (IEEE-754 `abs` is exact — a sign-bit clear).
    Abs,
    /// The `(din × dof)` transpose of a `(dof × din)` row-major weight:
    /// `wt[j*dof + i] = w[i*din + j]` — the exact layout the joint adjoint
    /// builds for `Ā @ Wᵀ`. The dims participate in the key so one `Arc` used
    /// under two different shapes could never alias (defense in depth; today
    /// each weight `Arc` has exactly one shape).
    Transposed { dof: usize, din: usize },
}

impl WeightForm {
    fn label(self) -> &'static str {
        match self {
            WeightForm::Raw => "resident_w_raw",
            WeightForm::Abs => "resident_w_abs",
            WeightForm::Transposed { .. } => "resident_w_t",
        }
    }
}

/// Cache key: weight `Arc` identity (data address + length) + derived form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ResidentWeightKey {
    /// `Arc::as_ptr(...).cast::<f32>() as usize` — the allocation's data
    /// address. Distinct **live** allocations have distinct addresses; the
    /// entry's keep-alive `Arc` guarantees liveness for the key's lifetime.
    ptr: usize,
    len: usize,
    form: WeightForm,
}

fn resident_weight_key(weight: &Arc<[f32]>, form: WeightForm) -> ResidentWeightKey {
    ResidentWeightKey {
        ptr: Arc::as_ptr(weight).cast::<f32>() as usize,
        len: weight.len(),
        form,
    }
}

/// A resident weight buffer + the mandatory keep-alive guard.
pub(crate) struct ResidentWeightEntry {
    /// KEEP-ALIVE (soundness-mandatory): retaining the weight `Arc` makes
    /// address recycling impossible while the pointer-keyed entry is live, so
    /// a pointer-identity hit is always a true content hit. Mirrors
    /// `PreparedCrownPlan::static_weight_arcs`. Held for lifetime semantics
    /// only — never read back.
    keep_alive: Arc<[f32]>,
    /// The GPU-resident buffer holding the derived form's bytes, uploaded once.
    buffer: Arc<wgpu::Buffer>,
    /// Uploaded inside a checked transaction whose error scopes have not yet
    /// popped clean, so the upload is not yet known to have succeeded. The
    /// transaction's clean `finish` clears it; a transaction that ends any
    /// other way evicts every entry still carrying it.
    uncommitted: bool,
}

/// Whether the resident-weight cache is enabled (default ON;
/// `NY_RESIDENT_WEIGHTS=0` opts out for A/B differential runs).
fn resident_weights_enabled() -> bool {
    std::env::var("NY_RESIDENT_WEIGHTS").ok().as_deref() != Some("0")
}

impl WgpuDevice {
    /// Checked bytes retained by the resident-weight cache on this exact
    /// device. Use the actual logical buffer size so API-required padding is
    /// charged as well as the retained tensor payload.
    pub(crate) fn resident_weight_cache_bytes(&self) -> Result<usize> {
        let cache = self.resident_weight_buffers.lock().map_err(|err| {
            NyError::InternalError(format!("resident weight cache lock poisoned: {err}"))
        })?;
        cache.values().try_fold(0usize, |total, entry| {
            let bytes = usize::try_from(entry.buffer.size()).map_err(|_| {
                NyError::InternalError("resident weight buffer size does not fit in usize".into())
            })?;
            total.checked_add(bytes).ok_or_else(|| {
                NyError::InternalError("resident weight cache byte count overflow".into())
            })
        })
    }

    /// Look up (or upload once) the GPU-resident buffer for `weight` in `form`.
    ///
    /// On a hit the buffer is returned without touching the queue (no upload,
    /// no CPU abs/transpose). On a miss the derived bytes are computed and
    /// uploaded ONCE, and the entry retains a keep-alive clone of `weight`.
    /// With `NY_RESIDENT_WEIGHTS=0` this degrades to the pre-cache behavior:
    /// a fresh per-call upload (identical bytes, nothing cached).
    ///
    /// The returned buffer is only ever bound READ-ONLY by the CROWN folds
    /// (GEMM operand B / `create_simple_pipeline` read-only slots), so cached
    /// contents can never be mutated by a dispatch.
    pub(crate) fn resident_weight_buf(
        &self,
        weight: &Arc<[f32]>,
        form: WeightForm,
    ) -> Result<Arc<wgpu::Buffer>> {
        self.resident_weight_buf_with(weight, form, || self.upload_weight_form(weight, form))
    }

    /// [`Self::resident_weight_buf`] with the upload passed in. Production
    /// passes [`Self::upload_weight_form`]; the gpu-tests pins pass an upload
    /// that hits a wgpu error, standing in for an allocation failure they
    /// cannot provoke on demand.
    ///
    /// A miss uploads inside its OWN wgpu error scope and caches the buffer
    /// only when that scope is clean, except inside a checked transaction,
    /// where the entry is cached UNCOMMITTED and the transaction commits or
    /// evicts it (module doc, "A failed upload is never cached").
    fn resident_weight_buf_with(
        &self,
        weight: &Arc<[f32]>,
        form: WeightForm,
        upload: impl FnOnce() -> Result<wgpu::Buffer>,
    ) -> Result<Arc<wgpu::Buffer>> {
        if !resident_weights_enabled() {
            return Ok(Arc::new(upload()?));
        }

        let key = resident_weight_key(weight, form);
        {
            let cache = self.resident_weight_buffers.lock().map_err(|err| {
                NyError::InternalError(format!("resident weight cache lock poisoned: {err}"))
            })?;
            if let Some(entry) = cache.get(&key) {
                // The keep-alive guard makes a pointer hit a true content hit:
                // the cached Arc pins the allocation, so `weight` (same live
                // address) must be the same allocation.
                debug_assert!(
                    Arc::ptr_eq(&entry.keep_alive, weight),
                    "resident weight pointer hit must be the same live Arc"
                );
                return Ok(Arc::clone(&entry.buffer));
            }
        }

        // Outside a transaction the upload gets its own scope, so a failed
        // upload returns `Err` before anything is cached. Inside one, a nested
        // scope is not allowed (its pop has no deadline); the transaction's
        // scopes cover the upload and decide the entry's fate.
        let in_transaction = self.checked_transaction_active();
        let buffer = Arc::new(if in_transaction {
            upload()?
        } else {
            run_gpu_checked_on_device(&self.device, form.label(), upload)?
        });
        let mut cache = self.resident_weight_buffers.lock().map_err(|err| {
            NyError::InternalError(format!("resident weight cache lock poisoned: {err}"))
        })?;
        // A concurrent miss may have inserted first; keep the canonical entry
        // (drop our duplicate upload) — mirrors the conv_transpose plan cache.
        let entry = cache.entry(key).or_insert_with(|| ResidentWeightEntry {
            keep_alive: Arc::clone(weight),
            buffer,
            uncommitted: in_transaction,
        });
        Ok(Arc::clone(&entry.buffer))
    }

    /// Trust the uploads a checked transaction made: its error scopes popped
    /// clean, so every upload under them succeeded. Called by the
    /// transaction's successful `finish`/`finish_bounded`.
    ///
    /// Recovers a poisoned lock: a panic elsewhere leaves the map itself
    /// intact, and this runs on the transaction's finishing path.
    pub(in crate::wgpu_device) fn commit_transaction_resident_weights(&self) {
        let mut cache = self
            .resident_weight_buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for entry in cache.values_mut() {
            entry.uncommitted = false;
        }
    }

    /// Evict every upload that no clean transaction finish has committed.
    /// Called when a checked transaction ends (an error, an expired deadline,
    /// or an early `?` exit drops it before a clean `finish`, and after a
    /// clean finish any later upload is still uncommitted), and when one
    /// begins, so uncommitted entries only ever belong to the transaction in
    /// progress. Such an upload may hold an invalid buffer; keeping it would
    /// fail every later call with the same weight until
    /// `clear_crown_working_set`.
    ///
    /// Recovers a poisoned lock and never panics: it runs from the
    /// transaction's `Drop`.
    pub(in crate::wgpu_device) fn discard_transaction_resident_weights(&self) {
        let mut cache = self
            .resident_weight_buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.retain(|_, entry| !entry.uncommitted);
    }

    /// Allocate + upload the derived bytes for `form` (the once-per-key path).
    fn upload_weight_form(&self, weight: &Arc<[f32]>, form: WeightForm) -> Result<wgpu::Buffer> {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(form.label()),
            size: (weight.len().max(1) * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        match form {
            WeightForm::Raw => {
                self.queue
                    .write_buffer(&buf, 0, bytemuck::cast_slice(&weight[..]));
            }
            WeightForm::Abs => {
                // Elementwise |w| — exact (sign-bit clear), bit-identical to the
                // per-call `weight.iter().map(|v| v.abs())` it replaces.
                let absw: Vec<f32> = weight.iter().map(|v| v.abs()).collect();
                self.queue
                    .write_buffer(&buf, 0, bytemuck::cast_slice(&absw));
            }
            WeightForm::Transposed { dof, din } => {
                if dof * din != weight.len() {
                    return Err(NyError::shape_mismatch(vec![dof, din], vec![weight.len()]));
                }
                // Exact replica of the joint adjoint's transpose layout:
                // wt[j*dof + i] = weight[i*din + j] (a pure permutation).
                let mut wt = vec![0.0f32; weight.len()];
                for i in 0..dof {
                    let wr = i * din;
                    for j in 0..din {
                        wt[j * dof + i] = weight[wr + j];
                    }
                }
                self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&wt));
            }
        }
        super::intermediate_sweep::note_host_to_device(
            weight.len().saturating_mul(size_of::<f32>()),
        );
        self.resident_weight_uploads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(buf)
    }

    /// Clear the resident weight cache, dropping the GPU buffers AND their
    /// keep-alive `Arc`s together.
    ///
    /// Mirrors [`clear_conv_transpose_plan_cache`](Self::clear_conv_transpose_plan_cache);
    /// called from `clear_crown_working_set` between models so no freed weight
    /// `Arc` address can alias a stale resident buffer.
    pub(crate) fn clear_resident_weight_buffers(&self) -> Result<()> {
        let mut cache = self.resident_weight_buffers.lock().map_err(|err| {
            NyError::InternalError(format!("resident weight cache lock poisoned: {err}"))
        })?;
        cache.clear();
        Ok(())
    }

    /// Number of cached resident weight buffers (test-only introspection).
    // Introspection hooks for the resident-weight cache's pending regression
    // tests; dead until those land, but part of the cache's public contract.
    #[allow(dead_code)]
    #[cfg(any(test, feature = "gpu-tests"))]
    pub(crate) fn resident_weight_cache_len(&self) -> usize {
        self.resident_weight_buffers
            .lock()
            .map(|c| c.len())
            .unwrap_or(0)
    }

    /// Total resident-weight uploads performed (misses; test-only).
    #[allow(dead_code)]
    #[cfg(any(test, feature = "gpu-tests"))]
    pub(crate) fn resident_weight_upload_count(&self) -> usize {
        self.resident_weight_uploads
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Strong count of the keep-alive `Arc` for `(weight identity, form)`
    /// (test-only): proves the cache pins the allocation after the caller
    /// drops its own `Arc`.
    #[allow(dead_code)]
    #[cfg(any(test, feature = "gpu-tests"))]
    pub(crate) fn resident_weight_keepalive_strong_count(
        &self,
        ptr: usize,
        len: usize,
        form: WeightForm,
    ) -> Option<usize> {
        let cache = self.resident_weight_buffers.lock().ok()?;
        cache
            .get(&ResidentWeightKey { ptr, len, form })
            .map(|e| Arc::strong_count(&e.keep_alive))
    }
}

#[cfg(test)]
mod key_tests {
    use super::*;

    /// Same weight `Arc` + same form ⇒ same key (true hit).
    #[test]
    fn same_arc_same_key() {
        let w: Arc<[f32]> = Arc::from(vec![0.5f32; 12]);
        let a = resident_weight_key(&w, WeightForm::Raw);
        let b = resident_weight_key(&Arc::clone(&w), WeightForm::Raw);
        assert_eq!(a, b, "same weight Arc must key to the same entry");
    }

    /// Distinct `Arc` allocations with IDENTICAL contents ⇒ distinct keys:
    /// identity keying, no content-based false hit.
    #[test]
    fn different_arc_same_contents_different_key() {
        let contents = vec![0.5f32; 12];
        let wa: Arc<[f32]> = Arc::from(contents.clone());
        let wb: Arc<[f32]> = Arc::from(contents);
        assert_ne!(
            resident_weight_key(&wa, WeightForm::Raw),
            resident_weight_key(&wb, WeightForm::Raw),
            "distinct allocations must not collide even with identical contents"
        );
    }

    /// The derived form participates in the cache key — Raw/Abs/Transposed of the
    /// same Arc are distinct entries (different bytes!).
    #[test]
    fn form_affects_key() {
        let w: Arc<[f32]> = Arc::from(vec![-1.0f32; 12]);
        let raw = resident_weight_key(&w, WeightForm::Raw);
        let abs = resident_weight_key(&w, WeightForm::Abs);
        let t34 = resident_weight_key(&w, WeightForm::Transposed { dof: 3, din: 4 });
        let t43 = resident_weight_key(&w, WeightForm::Transposed { dof: 4, din: 3 });
        assert_ne!(raw, abs);
        assert_ne!(raw, t34);
        assert_ne!(abs, t34);
        assert_ne!(t34, t43, "transpose dims participate in the key");
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    fn read_back(device: &WgpuDevice, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident_w_test_staging"),
            size: (n.max(1) * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident_w_test_dl"),
            });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * size_of::<f32>()) as u64);
        device.queue.submit(Some(enc.finish()));
        WgpuDevice::read_buffer(&device.device, &staging, n).expect("readback")
    }

    /// (a) Two lookups of the same Arc return the SAME buffer with NO re-upload
    /// (upload counter unchanged on the second call), for every form.
    /// (b) A DIFFERENT Arc with identical contents gets a DIFFERENT entry
    /// (identity keying). (c) After `clear_crown_working_set` the cache is
    /// empty. Also proves the derived bytes are the exact CPU derivations.
    #[test]
    fn resident_weight_cache_hit_identity_and_clear() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        device.clear_crown_working_set().expect("clear");
        assert_eq!(device.resident_weight_cache_len(), 0);

        let w: Arc<[f32]> = Arc::from(vec![1.5f32, -2.0, 0.25, -0.5, 3.0, -4.0]);
        let base_uploads = device.resident_weight_upload_count();

        // Raw: miss then hit.
        let raw0 = device
            .resident_weight_buf(&w, WeightForm::Raw)
            .expect("raw");
        assert_eq!(device.resident_weight_upload_count(), base_uploads + 1);
        let raw1 = device
            .resident_weight_buf(&w, WeightForm::Raw)
            .expect("raw");
        assert!(
            Arc::ptr_eq(&raw0, &raw1),
            "same Arc must return the same buffer"
        );
        assert_eq!(
            device.resident_weight_upload_count(),
            base_uploads + 1,
            "a cache hit must not re-upload"
        );
        assert_eq!(read_back(&device, &raw0, w.len()), w[..].to_vec());

        // Abs: separate entry; bytes are the exact elementwise |w|.
        let abs0 = device
            .resident_weight_buf(&w, WeightForm::Abs)
            .expect("abs");
        assert!(!Arc::ptr_eq(&raw0, &abs0));
        assert_eq!(device.resident_weight_upload_count(), base_uploads + 2);
        let abs1 = device
            .resident_weight_buf(&w, WeightForm::Abs)
            .expect("abs");
        assert!(Arc::ptr_eq(&abs0, &abs1));
        assert_eq!(device.resident_weight_upload_count(), base_uploads + 2);
        let expect_abs: Vec<f32> = w.iter().map(|v| v.abs()).collect();
        assert_eq!(read_back(&device, &abs0, w.len()), expect_abs);

        // Transposed (2×3 → 3×2): exact adjoint layout wt[j*dof+i] = w[i*din+j].
        let form_t = WeightForm::Transposed { dof: 2, din: 3 };
        let t0 = device.resident_weight_buf(&w, form_t).expect("t");
        assert_eq!(device.resident_weight_upload_count(), base_uploads + 3);
        let t1 = device.resident_weight_buf(&w, form_t).expect("t");
        assert!(Arc::ptr_eq(&t0, &t1));
        assert_eq!(device.resident_weight_upload_count(), base_uploads + 3);
        let mut expect_t = vec![0.0f32; 6];
        for i in 0..2 {
            for j in 0..3 {
                expect_t[j * 2 + i] = w[i * 3 + j];
            }
        }
        assert_eq!(read_back(&device, &t0, w.len()), expect_t);

        // (b) Identity keying: identical CONTENTS in a distinct allocation is a
        // distinct entry (a content-keyed false hit here would be unsound for
        // mutable-content reuse patterns; identity keying never hits).
        let w2: Arc<[f32]> = Arc::from(w[..].to_vec());
        assert!(!Arc::ptr_eq(&w, &w2));
        let raw2 = device
            .resident_weight_buf(&w2, WeightForm::Raw)
            .expect("raw2");
        assert!(
            !Arc::ptr_eq(&raw0, &raw2),
            "distinct Arc with identical contents must get its own entry"
        );
        assert_eq!(device.resident_weight_upload_count(), base_uploads + 4);
        assert_eq!(device.resident_weight_cache_len(), 4);

        // (c) The model-transition clear empties the cache.
        device.clear_crown_working_set().expect("clear");
        assert_eq!(device.resident_weight_cache_len(), 0);
    }

    /// (d) The keep-alive guard: after the CALLER drops its Arc, the cache's
    /// retained clone (strong count 1) keeps the allocation alive, so a new
    /// allocation can never reuse the cached address → no false hit is
    /// possible. (Two live allocations cannot share an address.)
    #[test]
    fn resident_weight_keep_alive_prevents_address_recycling() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        device.clear_crown_working_set().expect("clear");

        let w: Arc<[f32]> = Arc::from(vec![7.0f32; 64]);
        let ptr = Arc::as_ptr(&w).cast::<f32>() as usize;
        let len = w.len();
        let _buf = device
            .resident_weight_buf(&w, WeightForm::Raw)
            .expect("raw");
        assert_eq!(
            device.resident_weight_keepalive_strong_count(ptr, len, WeightForm::Raw),
            Some(2),
            "caller + cache hold the Arc"
        );

        // Drop the caller's Arc: the CACHE keep-alive must still pin it.
        drop(w);
        assert_eq!(
            device.resident_weight_keepalive_strong_count(ptr, len, WeightForm::Raw),
            Some(1),
            "keep-alive must hold the allocation after the caller drops"
        );

        // Any new allocation is a DIFFERENT address while the entry lives:
        // the allocator cannot hand out memory that is still allocated.
        for _ in 0..32 {
            let fresh: Arc<[f32]> = Arc::from(vec![-9.0f32; 64]);
            assert_ne!(
                Arc::as_ptr(&fresh).cast::<f32>() as usize,
                ptr,
                "a live keep-alive address can never be recycled"
            );
            // And therefore a lookup for `fresh` can never hit the old entry.
            let fresh_buf = device
                .resident_weight_buf(&fresh, WeightForm::Raw)
                .expect("fresh");
            assert_eq!(read_back(&device, &fresh_buf, 64), vec![-9.0f32; 64]);
        }

        device.clear_crown_working_set().expect("clear");
        assert_eq!(device.resident_weight_cache_len(), 0);
    }

    /// An upload that wgpu rejects, standing in for a `create_buffer` that
    /// fails (out of memory, or a size past a device limit): wgpu reports the
    /// error through the error scopes and still hands back a handle.
    fn invalid_upload(device: &WgpuDevice) -> Result<wgpu::Buffer> {
        let past_limit = device
            .device
            .limits()
            .max_buffer_size
            .checked_add(4)
            .expect("max_buffer_size + 4 fits u64");
        Ok(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident weight fault pin: past max_buffer_size"),
            size: past_limit,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// (e) An upload that hit a wgpu error never enters the cache. Before the
    /// fix the buffer was cached inside the caller's scope: that call failed,
    /// and every later call with the same weight got the invalid buffer until
    /// `clear_crown_working_set`.
    #[test]
    fn resident_weight_upload_with_gpu_error_is_not_cached() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        device.clear_crown_working_set().expect("clear");

        let w: Arc<[f32]> = Arc::from(vec![1.5f32, -2.0, 0.25, -0.5]);
        let failed = device.run_gpu_checked("resident weight upload fault pin", || {
            device.resident_weight_buf_with(&w, WeightForm::Raw, || invalid_upload(&device))
        });
        let cached_after_failure = device.resident_weight_cache_len();
        match &failed {
            Err(NyError::InternalError(msg)) => assert!(
                msg.contains("wgpu validation error"),
                "the upload's wgpu error must fail the call: {msg}"
            ),
            Err(other) => panic!("expected the upload's wgpu validation error, got {other}"),
            Ok(_) => panic!("an upload that hit a wgpu error must not return a buffer"),
        }
        assert_eq!(
            cached_after_failure, 0,
            "an upload that hit a wgpu error must not enter the resident weight cache"
        );

        let retry = device
            .resident_weight_buf(&w, WeightForm::Raw)
            .expect("the same weight uploads clean after a failed upload");
        assert_eq!(device.resident_weight_cache_len(), 1);
        assert_eq!(read_back(&device, &retry, w.len()), w[..].to_vec());

        device.clear_crown_working_set().expect("clear");
        assert_eq!(device.resident_weight_cache_len(), 0);
    }

    /// (f) Inside a checked transaction an upload cannot push its own error
    /// scope (its pop has no deadline), so the transaction decides: a clean
    /// `finish` keeps the upload, a transaction whose scopes caught the
    /// upload's error evicts it, and so does one dropped before finishing (an
    /// early `?` exit).
    #[test]
    fn resident_weight_upload_in_a_transaction_is_kept_only_after_a_clean_finish() {
        use std::time::{Duration, Instant};

        let _g = gpu_test_serial_guard();
        let device = require_device();
        device.clear_crown_working_set().expect("clear");
        let deadline = || Instant::now() + Duration::from_secs(60);
        let w: Arc<[f32]> = Arc::from(vec![3.0f32, -1.0, 0.5, -0.25]);

        // A clean transaction commits its upload: the entry survives the
        // transaction and serves the next call without a re-upload.
        let committed = {
            let mut transaction = device
                .begin_gpu_checked_transaction("resident weight commit pin", deadline())
                .expect("begin");
            let buf = device
                .resident_weight_buf(&w, WeightForm::Raw)
                .expect("upload inside the transaction");
            transaction
                .finish("resident weight commit pin")
                .expect("a clean transaction finishes");
            buf
        };
        assert_eq!(
            device.resident_weight_cache_len(),
            1,
            "a clean transaction keeps its upload"
        );
        let uploads = device.resident_weight_upload_count();
        let hit = device
            .resident_weight_buf(&w, WeightForm::Raw)
            .expect("hit after the commit");
        assert!(
            Arc::ptr_eq(&committed, &hit),
            "the committed entry serves the hit"
        );
        assert_eq!(
            device.resident_weight_upload_count(),
            uploads,
            "no re-upload"
        );
        assert_eq!(read_back(&device, &hit, w.len()), w[..].to_vec());
        device.clear_crown_working_set().expect("clear");

        // A transaction whose scopes catch the upload's wgpu error evicts it.
        let error = {
            let mut transaction = device
                .begin_gpu_checked_transaction("resident weight fault pin", deadline())
                .expect("begin");
            let buf = device
                .resident_weight_buf_with(&w, WeightForm::Raw, || invalid_upload(&device))
                .expect("inside a transaction the upload leaves its error to the transaction");
            drop(buf);
            assert_eq!(
                device.resident_weight_cache_len(),
                1,
                "the upload is cached, uncommitted, while the transaction runs"
            );
            transaction
                .finish("resident weight fault pin")
                .expect_err("the transaction's scopes catch the upload's validation error")
        };
        assert!(
            error.to_string().contains("validation"),
            "unexpected transaction error: {error}"
        );
        assert_eq!(
            device.resident_weight_cache_len(),
            0,
            "a transaction that did not finish clean must evict the upload it made"
        );

        // A transaction dropped before finishing (an early `?` exit) evicts
        // its uploads too: nothing proved them good.
        {
            let _transaction = device
                .begin_gpu_checked_transaction("resident weight early exit pin", deadline())
                .expect("begin");
            device
                .resident_weight_buf(&w, WeightForm::Raw)
                .expect("upload inside the transaction");
        }
        assert_eq!(
            device.resident_weight_cache_len(),
            0,
            "a transaction dropped before finishing must evict its uploads"
        );

        // Outside any transaction the same weight uploads clean.
        let retry = device
            .resident_weight_buf(&w, WeightForm::Raw)
            .expect("clean upload after the failed transaction");
        assert_eq!(device.resident_weight_cache_len(), 1);
        assert_eq!(read_back(&device, &retry, w.len()), w[..].to_vec());

        device.clear_crown_working_set().expect("clear");
        assert_eq!(device.resident_weight_cache_len(), 0);
    }
}
