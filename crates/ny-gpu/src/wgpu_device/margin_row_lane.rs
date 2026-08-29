// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `MarginRowLaneDevice` — the ny-gpu implementation of
//! [`ny_core::MarginRowLaneGpu`], the injected device executor for the
//! margin-row certified GPU backward lane (#margin-row-gpu-eft,
//! `docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md` §6).
//!
//! # Non-repeat contract (the reason this is NOT the sweep's device)
//!
//! The predecessor seam (`NY_MARGIN_ROW_GPU=1`) dispatched on the
//! process-global backend and installed the caller's deadline there, which —
//! combined with the intermediate sweep's (since-classified) post-submit
//! deadline checks — poisoned the shared device ledger for the whole process.
//! This executor therefore:
//!
//! * owns a DEDICATED `WgpuDevice` (its own `wgpu::Device` + `Queue` on the
//!   same adapter), so it never contends on the sweep device's serialization
//!   mutex and never touches its memory-ledger reservations          [R3];
//! * accepts NO deadline: the lane checks its deadline HOST-SIDE, strictly
//!   before submission and after the drain (design §6)               [R3];
//! * implements the drain-safe transaction: every pass in ONE encoder, ONE
//!   `queue.submit`, then an UNCONDITIONAL drain — no fallible operation, no
//!   deadline check, no early return between submit and the completed drain;
//!   a failed drain still polls the queue to idle before `Err`      [R2].
//!
//! Shader modules are compiled through THE shader-loading seam
//! (`shader_loading::create_compute_module`), so the lane's kernels get the
//! device's resolved DenormPreserve loading path — the same path the lane's
//! admission check attests via [`MarginRowLaneDevice::denorm_preserve_enabled`]
//! (reuse, not fork: a privately-created module would silently bypass the
//! passthrough channel and void the subnormal contract of
//! `ds_primitives.wgsl`). Pipeline/bind-group construction mirrors
//! `create_simple_pipeline` with one extension: a configurable entry point
//! (the lane's kernels are named, e.g. `gate_transform`, not `main`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ny_core::{
    LaneArenaBuffer, LaneBinding, LaneProgramSpec, LaneProgramTiming, LaneResidentId,
    LaneStepBinding, LaneStepSpec, LaneTransactionSpec, MarginRowLaneGpu, NyError, Result,
};

use super::shader_loading;
use super::WgpuDevice;

/// Cap on bytes parked in the reuse pool (B1 arena pool). The pool trades
/// buffer create/destroy churn for resident memory on the lane's DEDICATED
/// device; past this cap, buffers are dropped instead of pooled — the
/// pre-B1 behavior, so the cap is a safety valve, not a correctness knob
/// (GB10 unified-memory OOM discipline: bounded, never unbounded, growth).
const MAX_POOLED_BYTES: u64 = 256 << 20;

/// Pool usage classes (part of the pool key: a uniform buffer must never be
/// handed back as a storage buffer).
const POOL_STORAGE: u8 = 0;
const POOL_STAGING: u8 = 1;
const POOL_UNIFORM: u8 = 2;

/// Size-class buffer reuse pool (B1: intermediates stop being created and
/// destroyed per pass). Keys are (usage class, EXACT byte size): lane
/// programs re-issue identical buffer sizes pass after pass (same net, same
/// shapes), so exact-size keying hits from the second pass on while making
/// stale-tail exposure structurally impossible (a reused buffer is either
/// fully rewritten, or cleared in-encoder before any step reads it).
#[derive(Default)]
struct BufferPool {
    free: HashMap<(u8, u64), Vec<wgpu::Buffer>>,
    pooled_bytes: u64,
}

/// One live entry of the resident-plane registry (B1 residency).
///
/// FIX D3 (hardening session, impl-v9): the registry is a slot table, not an
/// append-only list. `upload_resident` fills a fresh slot, `release_resident`
/// EMPTIES it (dropping the `wgpu::Buffer`, which frees the device
/// allocation), and a released slot is never re-issued — a stale
/// `LaneResidentId` therefore reads as UNKNOWN and fails the whole program
/// closed, instead of silently aliasing another plane's bytes.
struct ResidentEntry {
    buf: wgpu::Buffer,
    len: u64,
}

/// One resolved arena slot of a running program.
enum ArenaEntry {
    /// Created (or pooled) for this program; `reusable` marks it for return
    /// to the pool on success, `needs_clear` marks a pooled Zeroed buffer
    /// that must be `clear_buffer`-ed in the encoder before any step runs.
    Owned {
        buf: wgpu::Buffer,
        kind: u8,
        size: u64,
        reusable: bool,
        needs_clear: bool,
    },
    /// A resident buffer from the registry (never pooled, never cleared —
    /// its contents ARE the pass-invariant payload).
    Resident(wgpu::Buffer),
}

impl ArenaEntry {
    fn buffer(&self) -> &wgpu::Buffer {
        match self {
            ArenaEntry::Owned { buf, .. } => buf,
            ArenaEntry::Resident(buf) => buf,
        }
    }
}

/// Dedicated-device executor for the margin-row certified GPU backward lane.
///
/// Construction is EXPENSIVE (a full `WgpuDevice::new()`, compiling the shared
/// pipeline set once) and should happen at most once per process, behind the
/// lane's armed gate — the registered factory in ny-cli is a closure, so a
/// dark lane never constructs anything.
pub struct MarginRowLaneDevice {
    dev: WgpuDevice,
    /// Compiled pipelines keyed by `LaneTransactionSpec::label` (the spec
    /// contract requires byte-identical shader/entry per label).
    pipelines: Mutex<HashMap<String, Arc<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>>>,
    /// Device-resident buffers minted by [`MarginRowLaneGpu::upload_resident`]
    /// (B1 residency: ids are SLOT indices). FIX D3: a slot is emptied by
    /// [`MarginRowLaneGpu::release_resident`] when its owning plan static is
    /// dropped, and never re-issued — so the live device bytes track the LIVE
    /// plan statics, while the (24-byte) slot vector is the only thing that
    /// grows monotonically.
    resident: Mutex<Vec<Option<ResidentEntry>>>,
    /// B1 size-class reuse pool for per-pass buffers.
    pool: Mutex<BufferPool>,
}

impl MarginRowLaneDevice {
    /// Create the lane's dedicated device. Any adapter/driver failure is a
    /// plain `Err` — the caller fails closed to the CPU lane.
    pub fn new() -> Result<Self> {
        Ok(Self {
            dev: WgpuDevice::new()?,
            pipelines: Mutex::new(HashMap::new()),
            resident: Mutex::new(Vec::new()),
            pool: Mutex::new(BufferPool::default()),
        })
    }

    /// Take a pooled buffer of this exact (class, size), if any.
    fn pool_take(&self, kind: u8, size: u64) -> Option<wgpu::Buffer> {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let buf = pool.free.get_mut(&(kind, size)).and_then(Vec::pop);
        if buf.is_some() {
            pool.pooled_bytes = pool.pooled_bytes.saturating_sub(size);
        }
        buf
    }

    /// Return a buffer to the pool (dropped instead once past the byte cap).
    fn pool_put(&self, kind: u8, size: u64, buf: wgpu::Buffer) {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if pool.pooled_bytes.saturating_add(size) > MAX_POOLED_BYTES {
            return; // drop: bounded pool, pre-B1 behavior past the cap
        }
        pool.pooled_bytes += size;
        pool.free.entry((kind, size)).or_default().push(buf);
    }

    /// Build (or fetch the cached) compute pipeline for one dispatch shape.
    /// Mirrors `create_simple_pipeline` (binding 0 = uniform, 1.. = storage
    /// with per-binding read/write), extended with a configurable entry point;
    /// the module itself goes through the shader-loading seam so the resolved
    /// DenormPreserve path applies. `read_only[i]` is the access mode of
    /// storage binding `i + 1` — part of the label contract (two dispatches
    /// sharing a label must carry byte-identical shader/entry/binding shape).
    fn pipeline_for(
        &self,
        label: &str,
        entry: &str,
        shader_wgsl: &str,
        read_only: &[bool],
    ) -> Arc<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let mut cache = self
            .pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(hit) = cache.get(label) {
            return Arc::clone(hit);
        }
        let shader = shader_loading::create_compute_module(
            self.dev.device(),
            self.dev.denorm_preserve_enabled(),
            label,
            shader_wgsl,
        );
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, ro) in read_only.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (i + 1) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: *ro },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = self
            .dev
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        let pl = self
            .dev
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline =
            self.dev
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
        let built = Arc::new((pipeline, layout));
        cache.insert(label.to_string(), Arc::clone(&built));
        built
    }

    fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.dev.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            // wgpu requires COPY_BUFFER_ALIGNMENT (4); all lane payloads are
            // f32/u32 streams, but round up defensively. Buffers are
            // zero-initialized by wgpu, which is the Output contract.
            size: size.max(4).next_multiple_of(4),
            usage,
            mapped_at_creation: false,
        })
    }

    /// The program body run inside the checked error scopes: prepare
    /// (fallible) -> ONE submit -> UNCONDITIONAL drain -> read back.
    ///
    /// Every step is one compute pass inside ONE command encoder — wgpu's
    /// usage tracking inserts the barriers between passes, so each step's
    /// storage writes are visible to the next — and every readback copy is
    /// encoded after the last step. There is exactly one `queue.submit`
    /// (production shape). In PROFILE mode ([`LaneProgramSpec::profile`],
    /// caller-guaranteed non-verdict-bearing) each step becomes its own
    /// submit + unconditional drain so per-step wall times are observable —
    /// the R2 shape (no fallible operation between a submit and its
    /// completed drain) holds for EVERY submit in both modes.
    fn program_inner(
        &self,
        spec: &LaneProgramSpec<'_>,
        timing: &mut LaneProgramTiming,
    ) -> Result<Vec<Vec<u8>>> {
        // ---- prepare (fallible; nothing submitted yet) ----
        let prepare_started = std::time::Instant::now();
        if spec.steps.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "margin-row lane '{}': empty program",
                spec.label
            )));
        }
        let mut arena: Vec<ArenaEntry> = Vec::with_capacity(spec.buffers.len());
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        for init in spec.buffers {
            match init {
                LaneArenaBuffer::Data(bytes) => {
                    let len = bytes.len() as u64;
                    let padded = len.max(4).next_multiple_of(4);
                    // Pool-reuse only when the write covers the WHOLE buffer
                    // (padded == len): a partially-covered reused buffer could
                    // expose stale tail bytes, and clearing first is
                    // impossible here (queue.write_buffer lands BEFORE this
                    // program's encoder commands at submit).
                    let (buf, reusable) = if padded == len {
                        match self.pool_take(POOL_STORAGE, padded) {
                            Some(buf) => (buf, true),
                            None => (self.create_buffer("mr_lane_data", len, usage), true),
                        }
                    } else {
                        (self.create_buffer("mr_lane_data", len, usage), false)
                    };
                    self.dev.queue().write_buffer(&buf, 0, bytes);
                    arena.push(ArenaEntry::Owned {
                        buf,
                        kind: POOL_STORAGE,
                        size: padded,
                        reusable,
                        needs_clear: false,
                    });
                }
                // wgpu zero-initializes FRESH buffers, which is the Zeroed
                // contract; a POOLED buffer instead gets an in-encoder
                // `clear_buffer` before any step runs (encoder commands
                // execute in order, so every step sees zeros).
                LaneArenaBuffer::Zeroed(len) => {
                    let padded = (*len).max(4).next_multiple_of(4);
                    let (buf, needs_clear) = match self.pool_take(POOL_STORAGE, padded) {
                        Some(buf) => (buf, true),
                        None => (self.create_buffer("mr_lane_zeroed", *len, usage), false),
                    };
                    arena.push(ArenaEntry::Owned {
                        buf,
                        kind: POOL_STORAGE,
                        size: padded,
                        reusable: true,
                        needs_clear,
                    });
                }
                // B1 residency: the bytes are already on-device; fail closed
                // on an unknown id (never substitute).
                // Fail-closed on an unknown OR RELEASED id (never substitute):
                // a released slot reads `None` and takes the same error path.
                LaneArenaBuffer::Resident(id) => {
                    let registry = self
                        .resident
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let slot = usize::try_from(id.raw()).map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "margin-row lane '{}': resident id overflow",
                            spec.label
                        ))
                    })?;
                    let Some(Some(entry)) = registry.get(slot) else {
                        return Err(NyError::InvalidSpec(format!(
                            "margin-row lane '{}': unknown or released resident buffer {}",
                            spec.label,
                            id.raw()
                        )));
                    };
                    arena.push(ArenaEntry::Resident(entry.buf.clone()));
                }
            }
        }

        // Per step: pipeline + uniform buffer + bind group (all fallible-safe:
        // still nothing submitted).
        let mut prepared = Vec::with_capacity(spec.steps.len());
        for step in spec.steps {
            if step.uniforms.is_empty() || !step.uniforms.len().is_multiple_of(16) {
                return Err(NyError::InvalidSpec(format!(
                    "margin-row lane '{}' step '{}': uniforms must be a non-empty 16-byte \
                     multiple, got {}",
                    spec.label,
                    step.label,
                    step.uniforms.len()
                )));
            }
            let read_only: Vec<bool> = step.bindings.iter().map(|b| b.read_only).collect();
            let pipe = self.pipeline_for(step.label, step.entry, step.shader_wgsl, &read_only);
            // Uniform payloads are 16-byte multiples (checked above), so a
            // pooled buffer is always fully rewritten by this write.
            let params_len = step.uniforms.len() as u64;
            let params_buf = self.pool_take(POOL_UNIFORM, params_len).unwrap_or_else(|| {
                self.create_buffer(
                    "mr_lane_params",
                    params_len,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                )
            });
            self.dev.queue().write_buffer(&params_buf, 0, step.uniforms);
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }];
            for (i, binding) in step.bindings.iter().enumerate() {
                let entry = arena.get(binding.buffer).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "margin-row lane '{}' step '{}': binding {} references missing arena \
                         buffer {}",
                        spec.label,
                        step.label,
                        i + 1,
                        binding.buffer
                    ))
                })?;
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: entry.buffer().as_entire_binding(),
                });
            }
            let bind_group = self
                .dev
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(step.label),
                    layout: &pipe.1,
                    entries: &entries,
                });
            prepared.push((pipe, params_buf, bind_group, step.workgroups, step.label));
        }

        // Staging buffers for the readbacks (still fallible-safe). Pooled by
        // exact size; fully overwritten by the encoder copy before mapping.
        let mut outputs: Vec<(usize, wgpu::Buffer, u64)> = Vec::with_capacity(spec.readbacks.len());
        for &idx in spec.readbacks {
            let src = arena.get(idx).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "margin-row lane '{}': readback references missing arena buffer {idx}",
                    spec.label
                ))
            })?;
            let len = src.buffer().size();
            let staging = self.pool_take(POOL_STAGING, len).unwrap_or_else(|| {
                self.create_buffer(
                    "mr_lane_staging",
                    len,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )
            });
            outputs.push((idx, staging, len));
        }
        timing.upload_ms = prepare_started.elapsed().as_secs_f64() * 1e3;

        // Pooled Zeroed buffers must read as zeros: clear them in-encoder
        // BEFORE any step (encoder commands execute in order).
        let clears: Vec<&wgpu::Buffer> = arena
            .iter()
            .filter_map(|entry| match entry {
                ArenaEntry::Owned {
                    buf,
                    needs_clear: true,
                    ..
                } => Some(buf),
                _ => None,
            })
            .collect();

        let dispatch_started = std::time::Instant::now();
        if spec.profile {
            // ---- PROFILE mode (caller-guaranteed non-verdict-bearing): one
            // step per submit, each drained to idle (R2 per submit), timed.
            let mut per_step = Vec::with_capacity(prepared.len());
            for (k, (pipe, _params, bind_group, workgroups, label)) in prepared.iter().enumerate() {
                let step_started = std::time::Instant::now();
                let mut encoder =
                    self.dev
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(*label),
                        });
                if k == 0 {
                    for buf in &clears {
                        encoder.clear_buffer(buf, 0, None);
                    }
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(*label),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipe.0);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
                }
                self.dev.queue().submit(std::iter::once(encoder.finish()));
                let poll_result = self.dev.device().poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                if let Err(error) = poll_result {
                    return Err(NyError::InternalError(format!(
                        "margin-row lane '{}': device poll failed during profiled step \
                         '{label}': {error}",
                        spec.label
                    )));
                }
                per_step.push(step_started.elapsed().as_secs_f64() * 1e3);
            }
            timing.per_step_ms = Some(per_step);
            // Readback copies as a final submit of their own.
            let mut encoder =
                self.dev
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(spec.label),
                    });
            for (i, staging, len) in &outputs {
                encoder.copy_buffer_to_buffer(arena[*i].buffer(), 0, staging, 0, *len);
            }
            self.dev.queue().submit(std::iter::once(encoder.finish()));
        } else {
            // ---- ONE encoder: clears, every step, then every staging copy ----
            let mut encoder =
                self.dev
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(spec.label),
                    });
            for buf in &clears {
                encoder.clear_buffer(buf, 0, None);
            }
            for (pipe, _params, bind_group, workgroups, label) in &prepared {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(*label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipe.0);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
            }
            for (i, staging, len) in &outputs {
                encoder.copy_buffer_to_buffer(arena[*i].buffer(), 0, staging, 0, *len);
            }

            // ---- submit EXACTLY once ----
            self.dev.queue().submit(std::iter::once(encoder.finish()));
        }

        // ---- UNCONDITIONAL drain (R2): map every staging buffer and poll ----
        // No `?`, no deadline check between the submit above and the completed
        // poll below. Failures are collected and reported only after the queue
        // has been driven to idle.
        let mut receivers = Vec::with_capacity(outputs.len());
        for (_, staging, _) in &outputs {
            let (sender, receiver) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            receivers.push(receiver);
        }
        let poll_result = self.dev.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Drain complete. Only now is failure allowed to surface.
        if let Err(error) = poll_result {
            // Unmap nothing: mapping either completed (unmapped by Buffer drop)
            // or never will; the buffers are transaction-local and dropped here.
            return Err(NyError::InternalError(format!(
                "margin-row lane '{}': device poll failed during drain: {error}",
                spec.label
            )));
        }
        timing.dispatch_ms = dispatch_started.elapsed().as_secs_f64() * 1e3;
        let mut results = Vec::with_capacity(outputs.len());
        for ((_, staging, len), receiver) in outputs.iter().zip(&receivers) {
            match receiver.try_recv() {
                Ok(Ok(())) => {
                    let data = staging.slice(..).get_mapped_range();
                    results.push(data[..*len as usize].to_vec());
                    drop(data);
                    staging.unmap();
                }
                Ok(Err(map_error)) => {
                    return Err(NyError::InternalError(format!(
                        "margin-row lane '{}': staging map failed after drain: {map_error:?}",
                        spec.label
                    )));
                }
                Err(_) => {
                    return Err(NyError::InternalError(format!(
                        "margin-row lane '{}': drain completed without the map callback",
                        spec.label
                    )));
                }
            }
        }

        // Success: park per-pass buffers for reuse (B1 arena pool). Only the
        // success path pools — after any device error the buffers' states are
        // not worth reasoning about, so they drop (pre-B1 behavior).
        for entry in arena {
            if let ArenaEntry::Owned {
                buf,
                kind,
                size,
                reusable: true,
                ..
            } = entry
            {
                self.pool_put(kind, size, buf);
            }
        }
        for (pipe_bundle, params_buf, bind_group, _, _) in prepared {
            drop(bind_group);
            drop(pipe_bundle);
            self.pool_put(POOL_UNIFORM, params_buf.size(), params_buf);
        }
        for (_, staging, len) in outputs {
            self.pool_put(POOL_STAGING, len, staging);
        }
        Ok(results)
    }
}

impl MarginRowLaneGpu for MarginRowLaneDevice {
    fn denorm_preserve_enabled(&self) -> bool {
        // The RESOLVED per-device loading profile AND the process-wide
        // passthrough-fallback poison: a module that silently fell back to
        // plain WGSL would void the subnormal contract the lane's admission
        // attests, so report the channel OFF from that moment (fail-closed).
        self.dev.denorm_preserve_enabled() && self.dev.denorm_preserve_contract_intact()
    }

    fn adapter_summary(&self) -> String {
        self.dev.info()
    }

    fn run_lane_transaction(
        &self,
        spec: &LaneTransactionSpec<'_>,
    ) -> std::result::Result<Vec<Vec<u8>>, String> {
        // A single-dispatch transaction is a one-step program; expressing it
        // that way keeps exactly ONE submit/drain implementation in this file.
        let buffers: Vec<LaneArenaBuffer<'_>> = spec
            .bindings
            .iter()
            .map(|b| match b {
                LaneBinding::Read(bytes) => LaneArenaBuffer::Data(bytes),
                LaneBinding::Output(len) => LaneArenaBuffer::Zeroed(*len),
            })
            .collect();
        let step_bindings: Vec<LaneStepBinding> = spec
            .bindings
            .iter()
            .enumerate()
            .map(|(i, b)| LaneStepBinding {
                buffer: i,
                read_only: matches!(b, LaneBinding::Read(_)),
            })
            .collect();
        let mut readbacks = Vec::new();
        let mut want_lens = Vec::new();
        for (i, b) in spec.bindings.iter().enumerate() {
            if let LaneBinding::Output(len) = b {
                readbacks.push(i);
                want_lens.push(*len as usize);
            }
        }
        let step = LaneStepSpec {
            label: spec.label,
            entry: spec.entry,
            shader_wgsl: spec.shader_wgsl,
            uniforms: spec.uniforms,
            bindings: &step_bindings,
            workgroups: spec.workgroups,
        };
        let program = LaneProgramSpec {
            label: spec.label,
            buffers: &buffers,
            steps: std::slice::from_ref(&step),
            readbacks: &readbacks,
            profile: false,
        };
        let mut out = self.run_lane_program(&program)?;
        // Buffer sizes are rounded up to COPY_BUFFER_ALIGNMENT on creation;
        // return exactly the requested Output lengths (the old contract).
        for (bytes, want) in out.iter_mut().zip(&want_lens) {
            bytes.truncate(*want);
        }
        Ok(out)
    }

    fn run_lane_program(
        &self,
        spec: &LaneProgramSpec<'_>,
    ) -> std::result::Result<Vec<Vec<u8>>, String> {
        let mut timing = LaneProgramTiming::default();
        self.run_lane_program_timed(spec, &mut timing)
    }

    fn run_lane_program_timed(
        &self,
        spec: &LaneProgramSpec<'_>,
        timing: &mut LaneProgramTiming,
    ) -> std::result::Result<Vec<Vec<u8>>, String> {
        // `run_gpu_checked` serializes on THIS dedicated device's own lock
        // (never the sweep device's) and wraps the program in wgpu
        // validation/internal/OOM error scopes plus the uncaptured-error
        // generation fence. The closure keeps every fallible step BEFORE the
        // submit; from submit to completed drain there is no early return
        // (see `program_inner`), so a captured validation error surfaces
        // here only after the queue was driven to idle.               [R2,R3]
        self.dev
            .run_gpu_checked(spec.label, || self.program_inner(spec, timing))
            .map_err(|error| error.to_string())
    }

    fn upload_resident(&self, bytes: &[u8]) -> std::result::Result<LaneResidentId, String> {
        // B1 residency: upload once, live until the OWNER releases it (FIX D3
        // — `release_resident`, called when the owning plan static drops) or
        // the executor drops. The write itself rides the same checked path as
        // any upload; an error scope trip surfaces as `Err` and the caller
        // falls back to per-pass `Data` uploads.
        self.dev
            .run_gpu_checked("mr_lane_resident_upload", || {
                let usage = wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC;
                let buf = self.create_buffer("mr_lane_resident", bytes.len() as u64, usage);
                self.dev.queue().write_buffer(&buf, 0, bytes);
                let mut registry = self
                    .resident
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                // APPEND ONLY (never reuse a released slot): a stale id must
                // stay unknown rather than alias a different plane's bytes.
                let id = LaneResidentId::from_raw(registry.len() as u64);
                registry.push(Some(ResidentEntry {
                    buf,
                    len: bytes.len() as u64,
                }));
                Ok(id)
            })
            .map_err(|error| error.to_string())
    }

    fn release_resident(&self, id: LaneResidentId) {
        // FIX D3: empty the slot — dropping the `wgpu::Buffer` frees the
        // device allocation. Total and idempotent by construction: an unknown
        // or already-empty slot is a no-op (the owner may outlive nothing, and
        // a double release must never panic on a Drop path).
        let Ok(slot) = usize::try_from(id.raw()) else {
            return;
        };
        let mut registry = self
            .resident
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entry) = registry.get_mut(slot) {
            *entry = None;
        }
    }

    fn resident_bytes(&self) -> u64 {
        let registry = self
            .resident
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry
            .iter()
            .filter_map(|slot| slot.as_ref().map(|entry| entry.len))
            .sum()
    }
}
