Ordered packed-root CUDA arithmetic
==================================

This kernel accelerates an already admitted compact convolution plan. It does
not strengthen the relaxation by itself. Its purpose is to make more root work
fit within the verifier's time and memory budgets. Competition value requires
an end-to-end benchmark; known-answer tests do not establish that value.

Arithmetic argument
-------------------
Assume CUDA __fmul_rn and __fadd_rn implement binary32 round-to-nearest, ties-even
with gradual underflow, as requested by the committed native compilation flags.
For each output spatial site s, channel c and stored column k, one CUDA thread
owns both M and signed D. The checked owner range partitions all output tuples
exactly once. Scheduling these tuples spatial-first changes no coefficient's
arithmetic order; storage remains channel-major.

The host's admitted runs are sorted, disjoint destination intervals. The source
slot is the sum of prior run lengths plus the offset within the selected run.
Expanding that relation into the inverse map preserves the CPU plan's mapping.
Input channels share the validated spatial map. The last physical column maps
directly to the source bias and does not read the inverse map. A bias-only plan
therefore needs no inverse allocation and passes a null inverse pointer.

For one owner, both implementations initialize M=D=+0 and visit taps in ascending
order. They skip padding, missing coordinates, and either signed zero weight.
Every remaining tap performs M <- RN(M + RN(w*x)) and
D <- RN(D + RN(abs(w)*y)); y retains its sign. Induction on the tap index gives
the same stored bits after every update under the stated arithmetic contract.
There is no reassociation, cross-thread reduction, fused multiply-add, or atomic
update. The kernel does not recalculate the scalar bias, error envelope, support
mask, retained rows, or FoldMag: their existing CPU code consumes the result.
Thus substituting this arithmetic preserves the existing root certificate under
the same assumptions. Nonfinite output is rejected before publication.

This is a source-level equivalence argument, not a machine-checked compiler or
GPU proof. Qualification checks the actual module and transport against literal
expected bits from the archived independent rational oracle. Nine fixtures run
both as complete calls and as fragmented owner ranges, checking future-output
poison, zero-owner behavior, unchanged inputs and allocation guards. This finite
suite is a regression and implementation-admission check, not an exhaustive
proof of floating-point hardware.

Fragmentation includes a launch starting in the second spatial site. A separate
262145-word exact-bit roundtrip crosses the host-copy chunk boundary in both
directions and checks its allocation guards; it uses a host-defined integer
pattern and performs no arithmetic kernel operation.

Ownership and failure
---------------------
Each engine owns a separate non-primary CUDA context and serializes its calls.
The checked core request bounds every index, product and payload before copies.
Each submitted copy or launch has a drain guard, including error and unwind
paths. Buffers and borrowed host storage remain alive through drainage. Every
raw free is attempted at most once, and all explicit frees precede successful
publication. A device error permanently denies the capability. Unproven stream
quiescence or failed caller-context restoration terminates the process rather
than releasing storage that may still be in use or returning a result.

Only a pre-dispatch refusal can select CPU fallback, under its existing storage
limits. A near-dense plan admitted only by compressed GPU storage returns to the
dense CPU route on refusal. A deadline or failure after acceptance is propagated.
Finite-deadline calls never construct or wait for a cold engine; the VNN-COMP
CLI prewarms before creating command deadlines. The factory is enabled by
default with the cuda feature and respects NY_NO_CUDA.
The CPU-only guard is checked before consulting an engine. Unsupported native
architectures retain CPU contraction. There is no runtime NVRTC or PTX fallback.

Dependency limitation: cudarc 0.19.9 creates a context before its final fallible
memory-pool attribute query. If that query fails, its constructor does not free
the raw context. No engine is published on this path, and the one-shot factory
prevents repeated initialization attempts, but that allocation can remain until
process exit. The adapter does not attempt to destroy an unidentified context.

Rebuilding native artifacts
--------------------------
Use a pinned NVRTC 13 library and its matching builtins. Its directory must be
on LD_LIBRARY_PATH before Python starts, for example:

  LD_LIBRARY_PATH=/path/to/cuda/lib build-slot -- bash scripts/ny_box.sh build -- \
    python3 scripts/build_packed_root_kernels.py /path/to/cuda/lib/libnvrtc.so.13

The builder verifies the archived fixture/oracle hashes and records the source,
builder, compiler libraries, flags and native image hashes. It does not derive
expected results from the GPU. Native images exist for SM86, SM120 and SM121;
each device must independently pass the same startup qualification. The build
record is not evidence that a device with each architecture has been tested.

The gpu-tests feature selects packed-root device conformance; absent hardware,
unsupported architecture or failed qualification fails the selected lane.
The benchmarks feature additionally selects the explicit throughput experiment.
NY_PHASE_TELEMETRY=1 emits NY_PACKED_ROOT_CUDA_V1 after the command, recording
completed production calls, dispatch attempts and completed-call wall time.
It also emits NY_PACKED_ROOT_CALL_V1 for individual execute calls, including
refusals, deadlines and failures. Host dispatch time includes mandatory drainage;
it is not device kernel time. Startup qualification is excluded from both.
