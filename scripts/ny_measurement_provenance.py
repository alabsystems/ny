#!/usr/bin/env python3
"""Create immutable provenance records for a NY VNN-COMP measurement run.

The start record is deliberately content-oriented: it binds the solver binary,
the NY worktree (including every non-ignored untracked file), the dependency
pins, benchmark checkout, host, and effective sweep configuration without
copying source patches or arbitrary environment variables into the artifact.
The completion record is separate so the start evidence can never be rewritten.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_GIT_EXECUTABLE_OVERRIDE: str | None = None
V1_INPUT_ASSIGNMENT = re.compile(
    r"\(\s*X_\d+\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*\)"
)
V2_ASSIGNMENT_HEADER = re.compile(r"^(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]$")
STANDARD_SOLVER_VERDICTS = frozenset({"sat", "unsat", "unknown", "timeout", "error"})
PROCESS_SNAPSHOT_LIMIT = 128
PROCESS_ARGS_LIMIT = 384
SNAPSHOT_OUTPUT_LIMIT = 64 * 1024
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
PROC_SELF_MOUNTINFO = Path("/proc/self/mountinfo")
CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
# Memory containment profiles. The competition lane is `gb10-80g` and stays
# the default; `wsl24-20g` is a research lane sized for a 24 GiB WSL2 VM on a
# 31 GiB desktop, preserving the same high:max ratio.
#
# A smaller profile does not make a run equivalent to a larger one — fewer
# instances fit, so more time out. Results under a smaller profile are a LOWER
# BOUND on the same code under a larger one. The selected profile is recorded
# in the start manifest, so evidence never has to be guessed at.
#
# Like NY_MEASURE_EXPECTED_CPUS, the selector is an exact allowlist and the
# kernel's actual cgroup values are still attested against it, so the variable
# chooses which policy to demand and cannot fake compliance with it.
DEFAULT_CONTAINMENT_PROFILE = "gb10-80g"
# One finite per-process virtual-address ceiling is shared by every physical
# profile. CUDA/ONNX Runtime reserve about 53.5 GiB before useful work, and a
# sealed CIFAR100 idx7641 run reached 79.67 GiB of VA with only 24.36 GiB charged
# to its cgroup. 160 GiB leaves that run 80.33 GiB of VA headroom, so the gb10
# cgroup remainder is the narrower allocation envelope, while a runaway mmap
# loop still terminates at a deterministic limit on supported 64-bit Linux.
EXPECTED_RLIMIT_AS_BYTES = 160 * 1024**3
CONTAINMENT_PROFILES = {
    "gb10-80g": {
        "memory_high_bytes": 64 * 1024**3,
        "memory_max_bytes": 80 * 1024**3,
        "rlimit_as_bytes": EXPECTED_RLIMIT_AS_BYTES,
    },
    "wsl24-20g": {
        "memory_high_bytes": 16 * 1024**3,
        "memory_max_bytes": 20 * 1024**3,
        # NOT memory_max. RLIMIT_AS caps VIRTUAL address space, and the CUDA
        # driver plus ONNX Runtime reserve tens of GiB of VA regardless of how
        # much physical memory the host has. Sizing it down with memory.max
        # starves those reservations: measured on this box, a cersyve instance
        # that returns `sat` in 1s under an 80 GiB RLIMIT_AS instead spends 106s
        # and records `timeout` under a 20 GiB one. Physical containment is the
        # cgroup's job; the common finite address-space ceiling stays above both
        # runtime reservations and useful working mappings for every profile.
        "rlimit_as_bytes": EXPECTED_RLIMIT_AS_BYTES,
    },
}
ALLOWED_CONTAINMENT_PROFILES = frozenset(CONTAINMENT_PROFILES)
# Stable aliases for the competition-default profile. Internal policy checks
# use the profile-aware accessors below; callers and tests that reason about
# the default lane can continue to use these exact constants.
EXPECTED_MEMORY_HIGH_BYTES = CONTAINMENT_PROFILES[DEFAULT_CONTAINMENT_PROFILE][
    "memory_high_bytes"
]
EXPECTED_MEMORY_MAX_BYTES = CONTAINMENT_PROFILES[DEFAULT_CONTAINMENT_PROFILE][
    "memory_max_bytes"
]
EXPECTED_MEMORY_SWAP_MAX_BYTES = 8 * 1024**3
EXPECTED_PIDS_MAX = 4096
EXPECTED_CPU_COUNT = 10
CUDA_RUNTIME_INFO_SCHEMA = "ny_cuda_runtime_info_v3"
MEASUREMENT_CUDA_RUNTIME_SCHEMA = "ny_measurement_cuda_runtime_v1"
SEALED_CUDA_RUNTIME_SCHEMA = "ny_measurement_sealed_cuda_runtime_v1"
CUDA_RUNTIME_REQUIRED_ROLES = frozenset({"driver", "cublas", "cublas_lt"})
CUDA_RUNTIME_OPTIONAL_ROLES = frozenset({"nvrtc", "nvrtc_builtins"})
SYSTEM_LD_SO_PRELOAD = Path("/etc/ld.so.preload")
CUDA_RUNTIME_SAFE_LIBRARY_NAME = re.compile(
    r"^lib(?:(?:cuda|nvcuda|cublas|cublasLt|nvrtc)"
    r"(?:(?:32|64)(?:_[0-9]+(?:_[0-9]+)?)?)?"
    r"|nvrtc-builtins|nvblas)[.]so(?:[.][0-9]+)*$"
)
UNSAFE_DYNAMIC_LOADER_ENV = frozenset(
    {
        "DYLD_FORCE_FLAT_NAMESPACE",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
    }
)

# Runtime libraries outside NY also consume process-global environment
# controls. Keep their namespaces fail-closed for score evidence: reviewed
# controls are recorded through ENV_ALLOWLIST below, while any other matching
# key must be explicitly reviewed before it can reach the solver.
UNREVIEWED_SOLVER_RUNTIME_ENV_PREFIXES = (
    "__EGL_",
    "__GL_",
    "__GLX_",
    "__NV_",
    "__VK_",
    "ACCELERATE_",
    "ACO_",
    "AMD_",
    "ANV_",
    "BLIS_",
    "CANDLE_",
    "CARGO_",
    "CUBLAS_",
    "CUBLASLT_",
    "CUDA_",
    "CUDNN_",
    "DISABLE_LAYER_",
    "DRI_",
    "EGL_",
    "ENABLE_LAYER_",
    "GALLIUM_",
    "GBM_",
    "GOMP_",
    "GOTO_",
    "INTEL_",
    "KMP_",
    "LC_",
    "LIBGL_",
    "LP_",
    "LVP_",
    "MALLOC_",
    "MATMUL_",
    "MESA_",
    "MKL_",
    "Malloc",
    "NOUVEAU_",
    "NVBLAS_",
    "NVIDIA_",
    "NVPRESENT_",
    "NVRTC_",
    "NVVM_",
    "OMP_",
    "OPENBLAS_",
    "ORT_",
    "PYTHON",
    "RADV_",
    "RAYON_",
    "RUST_",
    "RUSTUP_",
    "VECLIB_",
    "VK_",
    "VULKAN_",
    "WGPU_",
    "XDG_",
    "ZINK_",
)
UNREVIEWED_SOLVER_RUNTIME_ENV_EXACT = frozenset(
    {
        "DISPLAY",
        "DISABLE_LAYER_NV_OPTIMUS_1",
        "DRI_PRIME",
        "GCONV_PATH",
        "GLIBC_TUNABLES",
        "LANGUAGE",
        "LOCPATH",
        "MALLOC_CONF",
        "LIBGL_ALWAYS_SOFTWARE",
        "LIBGL_DRIVERS_PATH",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "MESA_VK_DEVICE_SELECT",
        "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE",
        "NODEVICE_SELECT",
        "TEMP",
        "TEMPDIR",
        "TMP",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
    }
)
ALLOWED_EXPECTED_CPU_COUNTS = frozenset({"10", "20"})


def _containment_profile_name() -> str:
    raw = os.environ.get("NY_MEASURE_CONTAINMENT_PROFILE")
    if raw is None:
        return DEFAULT_CONTAINMENT_PROFILE
    if raw not in ALLOWED_CONTAINMENT_PROFILES:
        raise ProvenanceError(
            "NY_MEASURE_CONTAINMENT_PROFILE must be exactly "
            + " or ".join(sorted(ALLOWED_CONTAINMENT_PROFILES))
        )
    return raw


def _expected_memory_high_bytes() -> int:
    return CONTAINMENT_PROFILES[_containment_profile_name()]["memory_high_bytes"]


def _expected_memory_max_bytes() -> int:
    return CONTAINMENT_PROFILES[_containment_profile_name()]["memory_max_bytes"]


def _expected_rlimit_as_bytes() -> int:
    return CONTAINMENT_PROFILES[_containment_profile_name()]["rlimit_as_bytes"]

_SENSITIVE_ARGUMENT = (
    r"(?:api[-_]?key|token|password|passwd|secret|credential|authorization|"
    r"auth|cookie|private[-_]?key|access[-_]?key)"
)

# This is intentionally an explicit allowlist, not an NY_* or process-environment
# sweep. Unknown NY_* settings, unreviewed externally inherited AY_* settings,
# and mimalloc runtime controls make capture fail so a solver-affecting knob can
# neither disappear from the record nor accidentally leak a new secret. The
# reviewed AY gates below expose revision-pinned measurement controls; every
# future AY_* knob remains fail-closed until it receives the same provenance
# review. Competition wrappers sanitize AY experiment overrides and mimalloc; a
# local allocator A/B must explicitly remove or provenance-review those knobs.
ENV_ALLOWLIST = frozenset(
    {
        "AY_LRA_WARM_SIMPLEX_STATE",
        "AY_MILP_NODE_PROP",
        "CARGO_BUILD_JOBS",
        "CARGO_TARGET_DIR",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "DBUS_SESSION_BUS_ADDRESS",
        "DYLD_LIBRARY_PATH",
        "GPU_AVAILABLE",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "MKL_NUM_THREADS",
        "NY_ACASXU_PROF",
        "NY_ALLOW_NONCUDA_MEASURE",
        "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
        "NY_ALPHA_FINAL_BOUND_ONLY",
        "NY_ALPHA_REFRESH_FRACTION",
        "NY_ATTR_BRANCH",
        "NY_ATTR_BRANCH_DIAG",
        "NY_AY",
        "NY_AY_BRANCH_HINTS",
        "NY_AY_MARGIN_REFRAME",
        "NY_AY_MILP_TALL_FLIP_CAP",
        "NY_AY_NODE_WARM_CAP_MS",
        "NY_AY_OBJECTIVE_FIRST_SAT",
        "NY_ATTACK_EXTEND",
        "NY_ATTACK_EXTEND_FRAC",
        "NY_ATTACK_EXTEND_MARGIN",
        "NY_ATTACK_POINT_FAST_KERNELS",
        "NY_BAB_CLAUSE_LEARN",
        "NY_BAB_CLAUSE_REPLAY",
        "NY_BAB_QUEUE_MEM_MB",
        "NY_BAB_RESNET_PARALLEL",
        "NY_BAB_RESNET_REFOLD_GUARD",
        "NY_BICCOS_BCP_SHADOW",
        "NY_BICCOS_Q_STAGE0",
        "NY_BICCOS_Q_STAGE1_REPLAY",
        "NY_BN_FOLD_EXT",
        "NY_BRANCH_KFSB_CHILDSIM",
        "NY_BRANCH_RESCORE",
        "NY_BRANCH_STEM",
        "NY_BRANCH_STEM_K",
        "NY_BRANCH_STEM_NODES",
        "NY_BRANCH_STEM_PROBE",
        "NY_BRANCH_TRACE",
        "NY_BROOT",
        "NY_BUILD_DISK_PATH",
        "NY_BUILD_FEATURES",
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CLIP_INTERM_RESNET",
        "NY_COMPACT_TAIL_K16",
        "NY_CONE_REFRESH",
        "NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE",
        "NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE",
        "NY_CONV_SKIP_DEAD_F32",
        "NY_CONVTRANSPOSE_SOUND_F64_GPU",
        "NY_CUT_CROWN_M2_PROJECTED",
        "NY_CUT_CROWN_RESIDENT_SHADOW",
        "NY_CROWN_CUT_SEGMENT",
        "NY_CROWN_CHUNK_AWARE_BUDGET",
        "NY_CROWN_IBP_COLLECTOR_CAP_SECS",
        "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
        "NY_CROWN_IBP_SPARSE_RELU_ROWS",
        "NY_CUDA_CROWN",
        "NY_CUDA_DISCRETE_MODE",
        "NY_CUDA_DGEMM_TRIPLET",
        "NY_CUDA_GEMM_TRANSPORT",
        "NY_CUDA_RESIDENT_PATCHES_ROOT",
        "NY_CUDA_WIDE",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_CROWN_MEM_CAP_MB",
        "NY_CROWN_OBJ_CHUNK",
        "NY_CROWN_SERVE_TRUNCATED_CACHE",
        "NY_DENSE_BUDGET_MB",
        "NY_DISABLE_CROWN_COLLECTION_CACHE",
        "NY_EFT_ERR",
        "NY_ENDGAME_GRACE_SECS",
        "NY_F64_BLAS",
        "NY_F64_LEAF",
        "NY_F64_LINEAGE_RECOVER",
        "NY_F64_TAIL",
        "NY_FACET_BANK",
        "NY_FACET_BANK_MAX_BYTES",
        "NY_FACET_BANK_PLANES",
        "NY_FCHEAD_GRACE_SECS",
        "NY_FCHEAD_TIGHTEN",
        "NY_GPU_MEMORY_BUDGET_MB",
        "NY_GPU_DENORM_PRESERVE",
        "NY_GPU_LOCK_PATH",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_WGPU_CROWN",
        "NY_GAP_ATTRIBUTION",
        "NY_GAP_ATTRIBUTION_BUDGET_SECS",
        "NY_GAP_ATTRIBUTION_ROWS",
        "NY_GRAPH_MIP",
        "NY_GRAPH_MIP_LEAF",
        "NY_GRAPH_MIP_LEAF_MAX_BINARIES",
        "NY_GRAPH_MIP_LEAF_MAX_NNZ",
        "NY_GRAPH_MIP_LEAF_MIN_DEPTH",
        "NY_GRAPH_MIP_LEAF_PIN",
        "NY_GRAPH_MIP_LEAF_SLICE_S",
        "NY_GRAPH_MIP_LEAF_TOTAL_FRAC",
        "NY_GRAPH_MIP_MAX_BINARIES",
        "NY_GRAPH_MIP_MAX_NNZ",
        "NY_GRAPH_MIP_MIN_SLICE_S",
        "NY_GRAPH_MIP_SERIAL",
        "NY_HYDRA_CROWN",
        # Certificate-backed Input-Manifold Bound canary controls. Keep this
        # list explicit: any newly introduced NY_IMB_* knob still fails closed
        # until it receives the same provenance review.
        "NY_IMB",
        "NY_IMB_AY_REGION_PROOF",
        "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_BUDGET_S",
        "NY_IMB_EARLY",
        "NY_IMB_LEAF_MODE",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_IMB_SELECTOR_K2_LIFT",
        "NY_IMB_SELECTOR_K4_LIFT",
        "NY_IMB_SELECTOR_RANGE_CRASH",
        "NY_IMB_SELECTOR_SOLVE_PROFILE",
        "NY_IMB_TAIL_ALPHA",
        "NY_IMB_TAIL_CERT_AY",
        "NY_IMB_WIRE",
        "NY_MIP_CERTIFIED_SHARED_TREE",
        "NY_MIP_SAFENLP_DIRECT_FIRST",
        "NY_MIP_SAFENLP_SHARED_PREFIX",
        "NY_MIP_SAFENLP_TARGET_FSB_PREFIX",
        "NY_MIP_SERIAL",
        "NY_INTERM_REFINE",
        "NY_INTERM_REFINE_ALPHA",
        "NY_INTERM_REFINE_ALPHA_ITERS",
        "NY_INTERM_REFINE_ALPHA_LR",
        "NY_INTERM_REFINE_ALPHA_MAX_ROWS",
        "NY_INTERM_REFINE_ALPHA_REOPT",
        "NY_INTERM_REFINE_LAYERS",
        "NY_INTERM_REFINE_MAX_DIM",
        "NY_INTERM_REFINE_MIN_DEPTH",
        "NY_INTERM_REFINE_PROBE",
        "NY_INTERM_REFINE_PRUNE",
        "NY_INTERM_REFINE_PRUNE_TOL",
        "NY_INTERM_REFINE_ROWS",
        "NY_INTERM_REFINE_SEEDS",
        "NY_INTERM_REFINE_SELECTIVE_TOPK",
        "NY_INPUT_SPLIT_WARM_PARALLEL",
        "NY_INVPROP",
        "NY_INVPROP_LR",
        "NY_INVPROP_OPTIMIZE",
        "NY_INVPROP_SPLIT_LIFT",
        "NY_KFSB_LAYER_QUOTA",
        "NY_MARGIN_ROW_ADAPTIVE_RESERVE",
        "NY_MARGIN_ROW_DOMAIN_STACK",
        "NY_MARGIN_ROW_LRU",
        "NY_MARGIN_ROW_CLASSWISE",
        "NY_MARGIN_ROW_CONV_BWD_BLOCKED",
        "NY_MARGIN_ROW_PARALLEL",
        "NY_MARGIN_ROW_PROFILE",
        "NY_MARGIN_ROW_RESERVE_MAX_FRAC",
        "NY_MARGIN_ROW_RESERVE_SECS",
        "NY_MEASURE_ARTIFACTS",
        "NY_MEASURE_BIN",
        "NY_MEASURE_CAP",
        "NY_MEASURE_CATS",
        "NY_MEASURE_CONFIGS_DIR",
        "NY_MEASURE_GIT_BIN",
        "NY_MEASURE_CONTAINMENT_PROFILE",
        "NY_MEASURE_EXPECTED_CPUS",
        "NY_MEASURE_INSTANCE_INDEX",
        "NY_MEASURE_MAX_ROWS_PER_CATEGORY",
        "NY_MEASURE_OUTPUT_DIR",
        "NY_MEASURE_RUN_ID",
        "NY_MEASURE_RUSTUP_BIN",
        "NY_MEASURE_VNNLIB_VERSION",
        # Capture-only bit-exact solver IR. This one reviewed path does not
        # weaken the fail-closed treatment of any other NY_* diagnostic.
        "NY_MIP_DUMP",
        "NY_MIP_STABILITY_HINTS",
        "NY_MIP_WINDOW_TIMEOUT_SECS",
        "NY_MO_ADAPTIVE_DEPTH_COMMIT",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_MO_BAB_TRACE",
        "NY_MO_BETA_BASELINE_ONLY",
        "NY_MO_BETA_BASELINE_FIRST",
        "NY_MO_CUDA_BETA_SPSA",
        "NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR",
        "NY_MO_CUDA_FACTORY_ENGINE_HANDOFF",
        "NY_MO_GPU_CHUNK",
        "NY_MO_KFSB",
        "NY_MO_KFSB_CACHED_LA",
        "NY_MO_KFSB_CERT_REUSE",
        "NY_MO_KFSB_CHUNK",
        "NY_MO_KFSB_F64_SHADOW",
        "NY_MO_KFSB_K",
        "NY_MO_KFSB_PROBE",
        "NY_MO_KFSB_REDUCE",
        "NY_MO_KFSB_WINNER_PROBE",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS",
        "NY_MO_STALL_OBBT_CANARY",
        "NY_MOAT_SECS",
        "NY_MULTIOBJ_JOINT_ALPHA",
        "NY_MULTIOBJ_JOINT_ALPHA_GPU",
        "NY_NO_ALPHA_BRIDGE",
        "NY_NO_CNF_ROUTE",
        "NY_NO_CUDA",
        "NY_NO_CUDA_F32",
        "NY_NO_FRAC_HEAD",
        "NY_NO_PGD_TIME_CAP",
        "NY_NN4SYS_1D_PHASE_EVENTS",
        "NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS",
        "NY_NN4SYS_MVF_CLIP_DIAG",
        "NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS",
        "NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES",
        "NY_ORACLE_FRONTIER",
        "NY_ORT_ATTACK",
        "NY_ORT_ACTIVE_SET_REPAIR",
        "NY_ORT_REFINE_GRAD",
        "NY_ORT_SESSION_CACHE",
        "NY_PACKED_GRAPH_ALPHA_QUEUE",
        "NY_PATCHES_BUDGET_SECS",
        "NY_PATCHES_DEADLINE_FLAT_BIAS",
        "NY_PATCHES_DEADLINE_PARALLEL_SCATTER",
        "NY_PATCHES_DEADLINE_RELU",
        "NY_PATCHES_EAGER_ERR",
        "NY_PATCHES_EAGER_ERR_7D",
        "NY_PATCHES_GPU",
        "NY_PHASE_TELEMETRY",
        "NY_PGD_DIAG",
        "NY_PGD_EXACT_BATCHED",
        "NY_PGD_GAMA",
        "NY_PGD_GAMA_LAMBDA",
        "NY_PGD_GAMA_LIN_FRAC",
        "NY_PGD_VJP_BATCH",
        "NY_POSTBAB_ATTACK",
        "NY_POSTBAB_BAB_SEEDS",
        "NY_POSTBAB_BAB_SEEDS_K",
        "NY_POSTBAB_FRONTIER_FASTLANE",
        "NY_POSTBAB_FRONTIER_FASTLANE_SECS",
        "NY_POSTBAB_RESERVE_SECS",
        "NY_REL_BAB_DEADLINE_MULT",
        "NY_REL_DIFF_COUPLING",
        "NY_REL_JOINT_RELU_CUTS",
        "NY_REL_JOINT_RELU_CUTS_SUM",
        "NY_RELATIONAL_BAB",
        "NY_RELATIONAL_UNSAT",
        "NY_REL_EDGE_ALPHA",
        "NY_REL_EDGE_ALPHA_ITERS",
        "NY_REL_EDGE_ALPHA_TOP",
        "NY_REL_EDGE_MILP",
        "NY_REL_EDGE_MILP_DEPTH",
        "NY_REL_EDGE_MILP_GAP",
        "NY_REL_WHOLE_MIP",
        "NY_REL_WHOLE_MIP_OBBT",
        "NY_REL_WHOLE_MIP_OBBT_CHUNK",
        "NY_REL_WHOLE_MIP_OBBT_COND",
        "NY_REL_WHOLE_MIP_OBBT_COND_FRAC",
        "NY_REL_WHOLE_MIP_OBBT_MAXN",
        "NY_REL_WHOLE_MIP_OBBT_OUTER",
        "NY_REL_WHOLE_MIP_OBBT_ROUNDS",
        "NY_REL_WHOLE_MIP_OBBT_S",
        "NY_REL_WHOLE_MIP_OBBT_WIDTH",
        "NY_REL_WHOLE_MIP_SLICE_S",
        "NY_RESNET_ERR_MERGE",
        "NY_RESNET_GPU",
        "NY_RESNET_GPU_MAX_OBJECTIVES",
        "NY_RESNET_GPU_MAX_SEED",
        "NY_RESNET_GPU_TIME_BUDGET_MS",
        "NY_RESNET_WARMUP_GPU",
        "NY_RNG_RESTARTS",
        "NY_RNG_SEED",
        "NY_ROOT",
        "NY_ROOT_ALPHA_CUDA_MARGIN_LR_BRACKET",
        "NY_ROOT_ALPHA_CUDA_MARGIN_MW",
        "NY_ROOT_ALPHA_CUDA_MARGIN_STEP",
        "NY_ROOT_ALPHA_CUDA_MARGIN_TOPK",
        "NY_ROOT_ALPHA_CUDA_ROWS",
        "NY_ROOT_ALPHA_GPU",
        "NY_ROOT_ALPHA_ITERS",
        "NY_ROOT_ALPHA_MARGIN",
        "NY_ROOT_ALPHA_MARGIN_GRADIENT",
        "NY_ROOT_ALPHA_PHASE_CHECKPOINT",
        "NY_ROOT_ALPHA_TRUE",
        "NY_ROOT_ALPHA_TRUE_MAXROWS",
        "NY_ROOT_BLAS",
        "NY_ROOT_BLAS_TILE",
        "NY_ROOT_CRITICAL_GPU_ALPHA",
        "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET",
        "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET_CASCADE",
        "NY_ROOT_CRITICAL_GPU_ALPHA_LR_BRACKET",
        "NY_ROOT_CRITICAL_GPU_SPEC",
        "NY_SELECTIVE_ROOT_ALPHA",
        "NY_ROOT_GEMM",
        "NY_ROOT_CROWN_INTERM",
        "NY_ROOT_CROWN_INTERM_LAYERS",
        "NY_ROOT_CROWN_INTERM_MAXDIM",
        "NY_ROOT_CROWN_INTERM_OPTALPHA",
        "NY_ROOT_CROWN_INTERM_SECS",
        "NY_ROOT_INTERM_ALPHA",
        "NY_ROOT_INTERM_ALPHA_SECS",
        "NY_ROOT_INTERM_CUDA_FACTORY",
        "NY_ROOT_JOINT_INTERM_ALPHA",
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
        "NY_ROOT_JOINT_INTERM_ALPHA_ITERS",
        "NY_ROOT_JOINT_INTERM_ALPHA_LR",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_DIM",
        "NY_ROOT_JOINT_INTERM_ALPHA_MAX_SEL",
        "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
        "NY_ROOT_JOINT_INTERM_ALPHA_SECS",
        "NY_ROOT_JOINT_MIN_REMAINING_SECS",
        "NY_ROOT_OUTPUT_CONDITIONED_HEAD",
        "NY_ROOT_POST_C_SURVIVOR",
        "NY_ROOT_SKIP_ADAPTIVE_SPEC",
        "NY_ROOT_SPARSE_INTERM_CROWN",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS",
        "NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS",
        "NY_ROOT_SPARSE_INTERM_CROWN_SECS",
        "NY_ROOT_SPEC_PRUNE",
        "NY_RUMP_F64_ENGINE",
        "NY_SAFENLP_SHORT_GRACE",
        "NY_SCRATCH",
        # Schedule-only screen knobs still affect which clauses close before
        # a deadline. Preserve their raw spellings as launch evidence; do not
        # normalize fallback-equivalent values into a stronger identity.
        "NY_SCREEN_CELL_CHUNK",
        "NY_SCREEN_CROWN_MS",
        "NY_SCREEN_MVF_CHUNK",
        "NY_SCREEN_MVF_WAVE_SIZE",
        "NY_SCREEN_WAVE_SIZE",
        "NY_SEG_RESIDENT",
        "NY_SKIP_DISJ_PGD",
        "NY_SOFTMAX_OBJECTIVE_ENVELOPE",
        "NY_SPEC_ALPHA_DIRECT",
        "NY_SPEC_ROOT_ALPHA",
        "NY_SPEC_ROOT_GPU",
        "NY_SPEC_ROOT_MARGIN",
        "NY_STATE_DIR",
        "NY_STRICT_IBP",
        "NY_UPFRONT_ATTACK",
        "NY_UPFRONT_ATTACK_AUTO_CAP",
        "NY_UPFRONT_ATTACK_CAP",
        "NY_UPFRONT_ATTACK_FRAC",
        "NY_UNSTABLE_COUNT",
        "NY_VNNLIB_CACHE",
        "NY_WARMUP_ITERS",
        "NY_WIDE_ACTIVE_COMPACTION_TELEMETRY",
        "NY_WIDE_ALPHA_CPU",
        "NY_WIDE_ALPHA_LOCAL",
        "NY_WIDE_ALPHA_LR",
        "NY_WIDE_ALPHA_NOBIAS",
        "NY_WIDE_ALPHA_TRUE",
        "NY_WIDE_ALPHA_TRUE_DOMS",
        "NY_WIDE_ALPHA_TRUE_EVERY",
        "NY_WIDE_ALPHA_TRUE_STEP",
        "NY_WIDE_ALPHA_UNSHARED",
        "NY_WIDE_MAX_STACKED_ROWS",
        "NY_WITNESS_DEEPEN",
        "NY_WITNESS_DEEPEN_TARGET",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ORT_LOG_SEVERITY_LEVEL",
        "PATH",
        "PWD",
        "RAYON_NUM_THREADS",
        "RUST_BACKTRACE",
        "RUST_TEST_THREADS",
        "RUST_LOG",
        "RUSTFLAGS",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "SHELL",
        "TMPDIR",
        "USER",
        "XDG_RUNTIME_DIR",
    }
)

# These values are useful to the scorecard/provenance helpers and therefore
# remain recorded in environment.values, but they are not runtime inputs to the
# already-built solver.  Keeping them out of the env-i solver launch prevents
# an isolated HOME from accidentally turning build-tool state into solver
# state, and avoids exposing wrapper paths/controls to child processes.
SOLVER_ENVIRONMENT_EXCLUDED_KEYS = frozenset(
    {
        "CARGO_BUILD_JOBS",
        "CARGO_TARGET_DIR",
        "NY_ALLOW_NONCUDA_MEASURE",
        "NY_BROOT",
        "NY_BUILD_FEATURES",
        "NY_MEASURE_ARTIFACTS",
        "NY_MEASURE_BIN",
        "NY_MEASURE_CAP",
        "NY_MEASURE_CATS",
        "NY_MEASURE_CONFIGS_DIR",
        "NY_MEASURE_GIT_BIN",
        "NY_MEASURE_INSTANCE_INDEX",
        "NY_MEASURE_MAX_ROWS_PER_CATEGORY",
        "NY_MEASURE_OUTPUT_DIR",
        "NY_MEASURE_RUN_ID",
        "NY_MEASURE_RUSTUP_BIN",
        "NY_MEASURE_VNNLIB_VERSION",
        "NY_ROOT",
        "NY_SCRATCH",
        "RUSTFLAGS",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "RUST_TEST_THREADS",
    }
)

TYPED_NONNEGATIVE_INTEGER_ENV = frozenset(
    {
        "NY_CROWN_CUT_SEGMENT",
        "NY_FACET_BANK_MAX_BYTES",
        "NY_FACET_BANK_PLANES",
        "NY_IMB_OBJ",
        "NY_IMB_REGION_K",
        "NY_IMB_REPLAY_ONLY_LEAF",
        "NY_MO_KFSB_K",
        "NY_WARMUP_ITERS",
    }
)

TYPED_POSITIVE_INTEGER_ENV = frozenset(
    {
        "NY_BRANCH_STEM_K",
        "NY_BUILD_MIN_FREE_KIB",
        "NY_CUDA_WIDE_MAX_BYTES",
        "NY_GPU_LOCK_WAIT_SECS",
        "NY_GPU_VMEM_LIMIT_KIB",
        "NY_MO_GPU_CHUNK",
        "NY_MO_KFSB_CHUNK",
        "NY_MO_KFSB_WINNER_PROBE_DOMAINS",
    }
)

TYPED_BOUNDED_NONNEGATIVE_INTEGER_ENV = (
    # Root intermediate CROWN wall budget. Rust parses this as u64, while the
    # global verifier deadline remains authoritative over the diagnostic cap.
    ("NY_ROOT_CROWN_INTERM_SECS", 0, 3_600),
)

# Root intermediate-CROWN selector caps use Rust's trimmed integer parser, but
# sealed measurement syntax is deliberately narrower: canonical base-10 only,
# with no signs, surrounding whitespace, or redundant leading zeroes. Every
# zero below is an audited no-work boundary in the runtime. The dense maximum
# preserves the legacy whole-CIFAR 20,000-element experiment ceiling; the sparse
# maxima match the runtime's absolute clamps exactly.
TYPED_BOUNDED_CANONICAL_NONNEGATIVE_INTEGER_ENV = (
    ("NY_ROOT_CROWN_INTERM_MAXDIM", 0, 20_000),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_DIM", 0, 8_192),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_ROWS", 0, 512),
    ("NY_ROOT_SPARSE_INTERM_CROWN_MAX_TARGETS", 0, 4),
    ("NY_ROOT_SPARSE_INTERM_CROWN_SECS", 0, 8),
)

TYPED_BOUNDED_POSITIVE_INTEGER_ENV = (
    # Typed AY per-node warm-attempt ceiling for exact neural Graph-MIP.
    # Rust accepts the same digit-only millisecond syntax and hard maximum.
    ("NY_AY_NODE_WARM_CAP_MS", 1, 60_000),
    # Default-dark alpha-reference CROWN-IBP tightening-only deadline. The
    # mandatory IBP map and outer alpha/root/BaB deadline remain unchanged.
    ("NY_CROWN_IBP_COLLECTOR_CAP_SECS", 1, 3_600),
    # Exact gap-attribution capture is resident-row bounded in Rust. Seal the
    # same limits so a diagnostic receipt cannot hide a clamped typo.
    ("NY_GAP_ATTRIBUTION_BUDGET_SECS", 1, 3_600),
    ("NY_GAP_ATTRIBUTION_ROWS", 1, 3),
    # Dark NN4SYS point-JVP telemetry: the runtime uses the same hard ceiling,
    # preventing a typo-sized diagnostic sweep from becoming score evidence.
    ("NY_NN4SYS_1D_PHASE_EVENTS_MAX_GROUPS", 1, 4096),
    ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_GROUPS", 1, 4096),
    ("NY_NN4SYS_MVF_CLIP_DIAG_MAX_SAMPLES", 1, 16384),
)

# Dark margin-row reserve-allocation A/B. Measurement syntax is deliberately
# canonical and narrower than Rust's f64 parser: `0.<digits>` with no leading
# sign, exponent, redundant leading/trailing zero, or surrounding whitespace.
# Every admitted spelling is then checked against the runtime's finite open
# interval, so the typed record cannot claim an armed ceiling that Rust would
# silently decline.
TYPED_OPEN_UNIT_DECIMAL_FRACTION_ENV = frozenset({"NY_MARGIN_ROW_RESERVE_MAX_FRAC"})

TYPED_BOUNDED_FLOAT_ENV = (
    # The runtime parses f32 and treats zero as disabled. Seal a research-safe
    # ceiling so a malformed or typo-sized grace cannot become score evidence.
    ("NY_ENDGAME_GRACE_SECS", 0.0, 30.0),
    ("NY_REL_BAB_DEADLINE_MULT", 1.0, 10.0),
)

TYPED_ABSOLUTE_PATH_ENV = frozenset({"NY_MIP_DUMP"})

TYPED_ENUM_ENV = (
    (
        "NY_CUDA_GEMM_TRANSPORT",
        frozenset(
            {
                "auto",
                "direct-host-page-tables",
                "unified-memory",
                "explicit-device-copy",
            }
        ),
    ),
    ("NY_GPU_DENORM_PRESERVE", frozenset({"auto", "0", "1"})),
    ("NY_WGPU_CROWN", frozenset({"auto", "0", "1"})),
    (
        "NY_IMB_AY_REGION_PROOF",
        frozenset({"affine", "reachability", "residual", "selector", "shared"}),
    ),
    ("NY_MEASURE_EXPECTED_CPUS", ALLOWED_EXPECTED_CPU_COUNTS),
    ("NY_MEASURE_CONTAINMENT_PROFILE", ALLOWED_CONTAINMENT_PROFILES),
)

TYPED_STRICT_BOOLEAN_ENV = frozenset({"GPU_AVAILABLE", "NY_AY_BRANCH_HINTS"})

# These sealed gates have exact-string runtime semantics. Keep measurement
# syntax equally exact: unlike GPU_AVAILABLE, an explicitly empty value is
# malformed rather than a false spelling, and only reviewed "0"/"1" launch
# values are accepted.
TYPED_EXACT_BOOLEAN_ENV = frozenset(
    {
        "AY_LRA_WARM_SIMPLEX_STATE",
        "AY_MILP_NODE_PROP",
        "NY_ALLOW_NONCUDA_MEASURE",
        "NY_ADAPTIVE_MICROBATCH_CONTROLLER",
        "NY_ALPHA_FINAL_BOUND_ONLY",
        "NY_ATTR_BRANCH",
        "NY_ATTR_BRANCH_DIAG",
        "NY_ATTACK_POINT_FAST_KERNELS",
        "NY_AY_MARGIN_REFRAME",
        "NY_AY_OBJECTIVE_FIRST_SAT",
        "NY_BAB_CLAUSE_LEARN",
        "NY_BAB_CLAUSE_REPLAY",
        "NY_BAB_RESNET_PARALLEL",
        "NY_BICCOS_BCP_SHADOW",
        "NY_BICCOS_Q_STAGE0",
        "NY_BICCOS_Q_STAGE1_REPLAY",
        "NY_BN_FOLD_EXT",
        "NY_BRANCH_STEM",
        "NY_BRANCH_STEM_PROBE",
        "NY_BRANCH_TRACE",
        "NY_COMPACT_TAIL_K16",
        "NY_CONE_REFRESH",
        "NY_CONSTRAINED_PATCHES_ALPHA_RELU_INPLACE",
        "NY_CONSTRAINED_PATCHES_BETA_SPARSE_CAPTURE",
        "NY_CONVTRANSPOSE_SOUND_F64_GPU",
        "NY_CUT_CROWN_M2_PROJECTED",
        "NY_CUT_CROWN_RESIDENT_SHADOW",
        "NY_CROWN_IBP_DOWNSTREAM_RESWEEP",
        "NY_CROWN_CHUNK_AWARE_BUDGET",
        "NY_CROWN_IBP_SPARSE_RELU_ROWS",
        "NY_CROWN_SERVE_TRUNCATED_CACHE",
        "NY_CUDA_DGEMM_TRIPLET",
        "NY_CUDA_CROWN",
        "NY_CUDA_DISCRETE_MODE",
        "NY_CUDA_RESIDENT_PATCHES_ROOT",
        "NY_DISABLE_CROWN_COLLECTION_CACHE",
        "NY_CUDA_WIDE",
        "NY_EFT_ERR",
        "NY_GAP_ATTRIBUTION",
        "NY_MARGIN_ROW_CONV_BWD_BLOCKED",
        "NY_MIP_CERTIFIED_SHARED_TREE",
        "NY_MIP_SAFENLP_DIRECT_FIRST",
        "NY_MIP_SAFENLP_SHARED_PREFIX",
        "NY_MIP_SAFENLP_TARGET_FSB_PREFIX",
        "NY_MIP_SERIAL",
        "NY_IMB_AY_TAIL_ADAPTIVE_FIVE_COMB",
        "NY_IMB_BATCHED_REPLAY",
        "NY_IMB_REPLAY_ONLY",
        "NY_IMB_SELECTOR_K2_LIFT",
        "NY_IMB_SELECTOR_K4_LIFT",
        "NY_IMB_SELECTOR_RANGE_CRASH",
        "NY_IMB_SELECTOR_SOLVE_PROFILE",
        "NY_IMB_TAIL_CERT_AY",
        "NY_MO_ADAPTIVE_DEPTH_COMMIT",
        "NY_MO_ADAPTIVE_DEPTH_SELECT",
        "NY_MO_ADAPTIVE_DEPTH_SHADOW",
        "NY_INTERM_REFINE_PROBE",
        "NY_MO_BAB_TRACE",
        "NY_MO_BETA_BASELINE_ONLY",
        "NY_MO_BETA_BASELINE_FIRST",
        "NY_MO_CUDA_BETA_SPSA",
        "NY_MO_CUDA_BOUNDED_SHARED_EXECUTOR",
        "NY_MO_CUDA_FACTORY_ENGINE_HANDOFF",
        "NY_MO_KFSB_CERT_REUSE",
        "NY_MO_KFSB_F64_SHADOW",
        "NY_MO_STALL_OBBT_CANARY",
        "NY_NN4SYS_1D_PHASE_EVENTS",
        "NY_NN4SYS_MVF_CLIP_DIAG",
        "NY_PACKED_GRAPH_ALPHA_QUEUE",
        "NY_PATCHES_DEADLINE_FLAT_BIAS",
        "NY_PATCHES_DEADLINE_PARALLEL_SCATTER",
        "NY_PATCHES_DEADLINE_RELU",
        "NY_PATCHES_EAGER_ERR",
        "NY_PATCHES_EAGER_ERR_7D",
        "NY_PHASE_TELEMETRY",
        "NY_RESNET_ERR_MERGE",
        "NY_ROOT_ALPHA_CUDA_MARGIN_LR_BRACKET",
        "NY_ROOT_ALPHA_CUDA_MARGIN_MW",
        "NY_ROOT_ALPHA_CUDA_MARGIN_STEP",
        "NY_ROOT_ALPHA_CUDA_MARGIN_TOPK",
        "NY_ROOT_ALPHA_CUDA_ROWS",
        "NY_ROOT_ALPHA_GPU",
        "NY_ROOT_ALPHA_MARGIN",
        "NY_ROOT_ALPHA_MARGIN_GRADIENT",
        "NY_ROOT_ALPHA_PHASE_CHECKPOINT",
        "NY_ROOT_CRITICAL_GPU_ALPHA",
        "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET",
        "NY_ROOT_CRITICAL_GPU_ALPHA_ACTIVE_SET_CASCADE",
        "NY_ROOT_CRITICAL_GPU_ALPHA_LR_BRACKET",
        "NY_ROOT_CRITICAL_GPU_SPEC",
        "NY_ROOT_CROWN_INTERM",
        "NY_ROOT_INTERM_CUDA_FACTORY",
        "NY_ROOT_JOINT_INTERM_ALPHA_DEADLINE_ASCENT",
        "NY_ROOT_JOINT_INTERM_ALPHA_PROBE",
        "NY_ROOT_OUTPUT_CONDITIONED_HEAD",
        "NY_ROOT_POST_C_SURVIVOR",
        "NY_ROOT_SPARSE_INTERM_CROWN",
        "NY_ROOT_SKIP_ADAPTIVE_SPEC",
        "NY_SAFENLP_SHORT_GRACE",
        "NY_SELECTIVE_ROOT_ALPHA",
        "NY_SEG_RESIDENT",
        "NY_SKIP_DISJ_PGD",
        "NY_SOFTMAX_OBJECTIVE_ENVELOPE",
        "NY_UNSTABLE_COUNT",
        "NY_WIDE_ACTIVE_COMPACTION_TELEMETRY",
    }
)


class ProvenanceError(RuntimeError):
    """Evidence could not be captured completely and reproducibly."""


def _expected_measurement_cpu_count() -> int:
    raw = os.environ.get("NY_MEASURE_EXPECTED_CPUS")
    if raw is None:
        return EXPECTED_CPU_COUNT
    if raw not in ALLOWED_EXPECTED_CPU_COUNTS:
        raise ProvenanceError("NY_MEASURE_EXPECTED_CPUS must be exactly 10 or 20")
    return int(raw)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, int]:
    return _stat_fingerprint(path.stat())


def _stat_fingerprint(file_stat: os.stat_result) -> dict[str, int]:
    return {
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "size_bytes": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "ctime_ns": file_stat.st_ctime_ns,
    }


def _stable_file_hash(path: Path) -> tuple[str, dict[str, int]]:
    before = _file_fingerprint(path)
    digest = _sha256_file(path)
    after = _file_fingerprint(path)
    if before != after:
        raise ProvenanceError(f"file changed while provenance was captured: {path}")
    return digest, after


def _stable_sealed_file_hash(
    path: Path, *, executable: bool
) -> tuple[str, dict[str, int]]:
    """Hash one canonical read-only seal without following its final symlink."""
    expected_permissions = 0o555 if executable else 0o444
    try:
        initial_lstat = path.lstat()
    except OSError as error:
        raise ProvenanceError(f"sealed file is unavailable: {path}: {error}") from error
    if (
        stat.S_ISLNK(initial_lstat.st_mode)
        or not stat.S_ISREG(initial_lstat.st_mode)
    ):
        raise ProvenanceError(f"sealed file is not a regular file: {path}")
    initial_permissions = stat.S_IMODE(initial_lstat.st_mode)
    if initial_permissions & 0o222:
        raise ProvenanceError(
            f"sealed file is writable: {path} (mode {initial_permissions:#o})"
        )
    if initial_permissions != expected_permissions:
        raise ProvenanceError(
            f"sealed file has invalid permissions: {path}; "
            f"expected {expected_permissions:#o}, observed {initial_permissions:#o}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProvenanceError(
            f"cannot open sealed file without following symlinks: {path}: {error}"
        ) from error
    digest = hashlib.sha256()
    try:
        before_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before_stat.st_mode)
            or (before_stat.st_dev, before_stat.st_ino)
            != (initial_lstat.st_dev, initial_lstat.st_ino)
        ):
            raise ProvenanceError(f"sealed file changed before hashing: {path}")
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        after_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    before_fingerprint = _stat_fingerprint(before_stat)
    after_fingerprint = _stat_fingerprint(after_stat)
    if (
        before_fingerprint != after_fingerprint
        or before_stat.st_mode != after_stat.st_mode
    ):
        raise ProvenanceError(f"sealed file changed while hashing: {path}")
    try:
        final_lstat = path.lstat()
    except OSError as error:
        raise ProvenanceError(
            f"sealed file changed after hashing: {path}: {error}"
        ) from error
    if (
        stat.S_ISLNK(final_lstat.st_mode)
        or not stat.S_ISREG(final_lstat.st_mode)
        or (final_lstat.st_dev, final_lstat.st_ino)
        != (after_stat.st_dev, after_stat.st_ino)
        or final_lstat.st_mode != after_stat.st_mode
    ):
        raise ProvenanceError(f"sealed file changed after hashing: {path}")
    return digest.hexdigest(), after_fingerprint


def _capture_executable_identity(
    declared_path: str,
    *,
    base_dir: Path,
    label: str,
) -> dict[str, object]:
    path = Path(declared_path)
    candidate = path if path.is_absolute() else base_dir / path
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"{label} executable does not exist: {declared_path}"
        ) from error
    if not resolved.is_file():
        raise ProvenanceError(f"{label} executable is not a file: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise ProvenanceError(f"{label} executable is not executable: {resolved}")
    digest, fingerprint = _stable_file_hash(resolved)
    try:
        resolved_after = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"{label} executable changed during provenance capture: {declared_path}"
        ) from error
    if resolved_after != resolved or _file_fingerprint(resolved) != fingerprint:
        raise ProvenanceError(
            f"{label} executable changed during provenance capture: {declared_path}"
        )
    return {
        "declared_path": declared_path,
        "resolved_path": str(resolved),
        "size_bytes": fingerprint["size_bytes"],
        "sha256": digest,
        "fingerprint": fingerprint,
    }


def _capture_ay_executable(
    repo_root: Path,
    *,
    expected_revision: str,
    declared_path: str | None = None,
) -> dict[str, object] | None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ProvenanceError(
            f"invalid expected AY Git revision: {expected_revision!r}"
        )
    if declared_path is None:
        declared_path = os.environ.get("NY_AY", "")
    if not declared_path:
        return None
    identity = _capture_executable_identity(
        declared_path,
        base_dir=repo_root,
        label="AY",
    )
    executable = str(identity["resolved_path"])
    version = _run(
        [executable, "--version"],
        check=False,
        timeout=15,
        env=_cuda_probe_environment(None),
    )
    stdout = version.stdout.decode("utf-8", "replace").strip()
    stderr = version.stderr.decode("utf-8", "replace").strip()
    if version.returncode != 0:
        raise ProvenanceError(
            f"AY executable --version failed with status {version.returncode}: {stderr}"
        )
    build_commits = re.findall(r"^build\.commit=([^\r\n]+)$", stdout, re.MULTILINE)
    if build_commits != [expected_revision]:
        observed = build_commits or ["<missing>"]
        raise ProvenanceError(
            "AY executable build.commit does not match the exact Cargo.lock pin: "
            f"expected {expected_revision}, observed {observed}"
        )
    # Executing --version is part of the identity capture. Refuse a binary that
    # rewrites or replaces itself while reporting its provenance.
    if (
        _capture_executable_identity(
            declared_path,
            base_dir=repo_root,
            label="AY",
        )
        != identity
    ):
        raise ProvenanceError(
            f"AY executable changed while --version was captured: {declared_path}"
        )
    identity.update(
        {
            "version_command": [executable, "--version"],
            "version_returncode": version.returncode,
            "version_stdout": stdout,
            "version_stderr": stderr,
            "build_commit": expected_revision,
        }
    )
    return identity


def _config_tree_entries(root: Path) -> list[dict[str, object]]:
    """Return a deterministic, content-addressed manifest of a configs tree."""
    evidence: list[dict[str, object]] = []

    def visit(directory: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
        except OSError as error:
            raise ProvenanceError(
                f"could not enumerate measurement configs directory {directory}: {error}"
            ) from error
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            try:
                if child.is_symlink():
                    target = os.readlink(path)
                    item: dict[str, object] = {
                        "kind": "symlink",
                        "path": relative,
                        "target": target,
                        "target_sha256": _sha256(os.fsencode(target)),
                    }
                    if path.is_file():
                        digest, fingerprint = _stable_file_hash(path)
                        item.update(
                            {
                                "resolved_kind": "file",
                                "resolved_size_bytes": fingerprint["size_bytes"],
                                "resolved_sha256": digest,
                            }
                        )
                    elif path.is_dir():
                        raise ProvenanceError(
                            "measurement configs must not contain directory "
                            f"symlinks: {path}"
                        )
                    elif path.exists():
                        raise ProvenanceError(
                            f"measurement configs symlink targets a special file: {path}"
                        )
                    else:
                        item["resolved_kind"] = "missing"
                    evidence.append(item)
                elif child.is_dir(follow_symlinks=False):
                    evidence.append({"kind": "directory", "path": relative})
                    visit(path)
                elif child.is_file(follow_symlinks=False):
                    digest, fingerprint = _stable_file_hash(path)
                    evidence.append(
                        {
                            "kind": "file",
                            "path": relative,
                            "size_bytes": fingerprint["size_bytes"],
                            "sha256": digest,
                        }
                    )
                else:
                    raise ProvenanceError(
                        f"measurement configs contain a special file: {path}"
                    )
            except OSError as error:
                raise ProvenanceError(
                    f"could not capture measurement config input {path}: {error}"
                ) from error

    visit(root)
    return evidence


def _capture_config_inputs(configs_dir: Path) -> dict[str, object]:
    if not configs_dir.is_absolute():
        raise ProvenanceError(
            f"measurement configs directory must be absolute: {configs_dir}"
        )
    declared_path = str(configs_dir)
    try:
        resolved = configs_dir.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"measurement configs directory does not exist: {configs_dir}"
        ) from error
    if not resolved.is_dir():
        raise ProvenanceError(
            f"measurement configs path is not a directory: {configs_dir}"
        )
    first = _config_tree_entries(resolved)
    second = _config_tree_entries(resolved)
    try:
        resolved_after = configs_dir.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"measurement configs directory changed while provenance was captured: {configs_dir}"
        ) from error
    if first != second or resolved_after != resolved:
        raise ProvenanceError(
            f"measurement configs directory changed while provenance was captured: {configs_dir}"
        )
    manifest = {
        "schema": "ny_measurement_config_inputs_v1",
        "entries": first,
    }
    return {
        "schema": "ny_measurement_config_inputs_v1",
        "declared_path": declared_path,
        "resolved_path": str(resolved),
        "entry_count": len(first),
        "manifest_sha256": _sha256(_json_bytes(manifest)),
        "entries": first,
    }


def _seal_file(
    source: Path,
    destination: Path,
    *,
    executable: bool,
    expected_sha256: str | None = None,
    expected_fingerprint: dict[str, int] | None = None,
) -> dict[str, object]:
    """Copy one stable source into a read-only, content-addressed execution file."""
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"cannot resolve file selected for sealing: {source}"
        ) from error
    if not resolved_source.is_file():
        raise ProvenanceError(
            f"file selected for sealing is not regular: {resolved_source}"
        )
    source_digest, source_fingerprint = _stable_file_hash(resolved_source)
    if expected_sha256 is not None and source_digest != expected_sha256:
        raise ProvenanceError(
            f"file changed before it could be sealed: {resolved_source}"
        )
    if expected_fingerprint is not None and source_fingerprint != expected_fingerprint:
        raise ProvenanceError(
            f"file fingerprint changed before it could be sealed: {resolved_source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_lstat = destination.lstat()
    except FileNotFoundError:
        destination_lstat = None
    except OSError as error:
        raise ProvenanceError(
            f"cannot inspect sealed-file destination: {destination}: {error}"
        ) from error
    if destination_lstat is not None:
        if (
            stat.S_ISLNK(destination_lstat.st_mode)
            or not stat.S_ISREG(destination_lstat.st_mode)
        ):
            raise ProvenanceError(f"sealed-file destination is unsafe: {destination}")
        sealed_digest, sealed_fingerprint = _stable_sealed_file_hash(
            destination, executable=executable
        )
        if sealed_digest != source_digest:
            raise ProvenanceError(
                f"existing sealed file has different content: {destination}"
            )
    else:
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as error:
            raise ProvenanceError(
                f"cannot create sealed-file destination: {destination}: {error}"
            ) from error
        copied_digest = hashlib.sha256()
        try:
            with (
                resolved_source.open("rb") as input_file,
                os.fdopen(descriptor, "wb") as output_file,
            ):
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    copied_digest.update(chunk)
                    output_file.write(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
            if (
                copied_digest.hexdigest() != source_digest
                or _file_fingerprint(resolved_source) != source_fingerprint
            ):
                raise ProvenanceError(
                    f"file changed while it was sealed: {resolved_source}"
                )
            destination.chmod(0o555 if executable else 0o444)
            sealed_digest, sealed_fingerprint = _stable_sealed_file_hash(
                destination, executable=executable
            )
            if sealed_digest != source_digest:
                raise ProvenanceError(
                    f"sealed copy does not match its source: {destination}"
                )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    return {
        "schema": "ny_measurement_sealed_file_v1",
        "source_resolved_path": str(resolved_source),
        "source_sha256": source_digest,
        "source_fingerprint": source_fingerprint,
        "path": str(destination.resolve()),
        "size_bytes": sealed_fingerprint["size_bytes"],
        "sha256": sealed_digest,
        "fingerprint": sealed_fingerprint,
        "mode": "executable_read_only" if executable else "read_only",
    }


def _seal_config_inputs(
    original: dict[str, object], run_dir: Path
) -> dict[str, object]:
    """Materialize the captured config tree without live symlink dependencies."""
    source_value = original.get("resolved_path")
    manifest_digest = original.get("manifest_sha256")
    entries = original.get("entries")
    if (
        not isinstance(source_value, str)
        or not isinstance(manifest_digest, str)
        or not isinstance(entries, list)
    ):
        raise ProvenanceError("captured config identity is incomplete")
    source_root = Path(source_value)
    sealed_root = run_dir / "sealed" / "configs" / manifest_digest
    sealed_root.mkdir(parents=True, exist_ok=True)
    directories = [sealed_root]
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ProvenanceError("captured config entry is invalid")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ProvenanceError(f"unsafe captured config path: {relative}")
        source = source_root / relative
        destination = sealed_root / relative
        kind = item.get("kind")
        if kind == "directory":
            destination.mkdir(parents=True, exist_ok=True)
            directories.append(destination)
        elif kind == "file":
            expected_digest = item.get("sha256")
            if not isinstance(expected_digest, str):
                raise ProvenanceError(f"captured config digest is invalid: {source}")
            _seal_file(
                source,
                destination,
                executable=False,
                expected_sha256=expected_digest,
            )
        elif kind == "symlink" and item.get("resolved_kind") == "file":
            expected_digest = item.get("resolved_sha256")
            if not isinstance(expected_digest, str):
                raise ProvenanceError(f"captured config symlink is invalid: {source}")
            _seal_file(
                source,
                destination,
                executable=False,
                expected_sha256=expected_digest,
            )
        else:
            raise ProvenanceError(f"cannot safely seal non-file config entry: {source}")
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        directory.chmod(0o555)
    sealed = _capture_config_inputs(sealed_root.resolve())
    if _capture_config_inputs(Path(str(original["declared_path"]))) != original:
        raise ProvenanceError("measurement configs changed while they were sealed")
    return sealed


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_immutable(path: Path, data: bytes) -> None:
    """Create evidence exactly once; a reused run ID is always an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace immutable evidence: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProvenanceError(f"could not run {command[0]!r}: {error}") from error
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    return result


def _find_executable(name: str) -> str | None:
    """Find a command, including rustup's conventional user-local install."""
    found = shutil.which(name)
    if found is not None:
        return found
    cargo_candidate = Path.home() / ".cargo" / "bin" / name
    if cargo_candidate.is_file() and os.access(cargo_candidate, os.X_OK):
        return str(cargo_candidate)
    return None


def _capture_git_executable(repo: Path) -> dict[str, object]:
    git = (
        _GIT_EXECUTABLE_OVERRIDE
        or os.environ.get("NY_MEASURE_GIT_BIN")
        or _find_executable("git")
    )
    if git is None:
        raise ProvenanceError("Git executable is unavailable")
    identity = _capture_executable_identity(
        git,
        base_dir=repo,
        label="Git",
    )
    executable = str(identity["resolved_path"])
    command = [executable, "--version"]
    result = _run(
        command,
        check=False,
        timeout=15,
        env={
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": os.environ.get("HOME", "/"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(
            f"Git executable --version failed ({result.returncode}): {detail}"
        )
    if (
        _capture_executable_identity(git, base_dir=repo, label="Git")
        != identity
    ):
        raise ProvenanceError("Git executable changed during identity capture")
    # All later repository reads execute the canonical object, not a caller's
    # possibly retargetable symlink spelling.
    identity["declared_path"] = executable
    identity.update(
        {
            "version_command": command,
            "version_returncode": result.returncode,
            "version_stdout": result.stdout.decode(
                "utf-8", "replace"
            ).strip(),
            "version_stderr": result.stderr.decode(
                "utf-8", "replace"
            ).strip(),
        }
    )
    return identity


@contextmanager
def _bound_git_executable(path: str):
    global _GIT_EXECUTABLE_OVERRIDE
    previous = _GIT_EXECUTABLE_OVERRIDE
    _GIT_EXECUTABLE_OVERRIDE = path
    try:
        yield
    finally:
        _GIT_EXECUTABLE_OVERRIDE = previous


def _recapture_bound_git_executable(
    expected: object, repo: Path
) -> dict[str, object]:
    if not isinstance(expected, dict):
        raise ProvenanceError("start manifest Git executable identity is invalid")
    path = expected.get("resolved_path")
    if not isinstance(path, str):
        raise ProvenanceError("start manifest Git executable path is invalid")
    with _bound_git_executable(path):
        return _capture_git_executable(repo)


def _git_evidence_result(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run Git plumbing without optional locks or mutable acceleration caches."""

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    git = (
        _GIT_EXECUTABLE_OVERRIDE
        or os.environ.get("NY_MEASURE_GIT_BIN")
        or _find_executable("git")
    )
    if git is None:
        raise ProvenanceError("Git executable is unavailable")
    return _run(
        [
            git,
            "-C",
            str(repo),
            "-c",
            "core.fileMode=true",
            "-c",
            "core.symlinks=true",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.ignoreStat=false",
            "-c",
            "core.trustctime=true",
            "-c",
            "core.checkStat=default",
            *args,
        ],
        check=check,
        env=environment,
    )


def _git_evidence(repo: Path, *args: str, check: bool = True) -> bytes:
    return _git_evidence_result(repo, *args, check=check).stdout


# Paths whose content decides what the solver binary DOES. A binary older than
# the newest commit touching these cannot have been built from this worktree.
_BUILD_INPUT_PATHS = ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml")
_BEHAVIOUR_INPUT_PATHS = ("configs",)


def _last_commit_epoch(repo: Path, paths: tuple[str, ...]) -> int | None:
    """Committer timestamp of the newest commit touching any of `paths`."""
    raw = _git_evidence(
        repo, "log", "-1", "--format=%ct", "--", *paths, check=False
    ).decode("ascii", "replace")
    stamp = raw.strip()
    return int(stamp) if stamp.isdigit() else None


def _capture_build_coherence(repo_root: Path, binary: Path) -> dict:
    """Refuse a binary that predates the sources or presets it will be run with.

    The manifest already pins the worktree HEAD and the binary's sha256, but
    nothing tied the two together: seal a binary built at commit A, then move
    the worktree to commit B, and the evidence looked complete while the run
    actually measured a mismatched pair.

    That is not hypothetical. A sweep here sealed configs from a commit that
    added the `bab.branching.input_split.sat_escape_branch` preset key while
    executing a binary built before the reader for it existed. The verifier
    fail-closed per instance ("unrecognized key"), so all 194 nn4sys rows
    recorded `unknown` — a full category scored as a legitimate 0 instead of
    stopping the run.

    Compares the binary's mtime against the newest commit touching build inputs
    (crates/, Cargo manifests, toolchain pin) and behaviour inputs (configs/).
    Rebuilding always refreshes mtime, so a correctly-built binary passes; only
    a stale one trips this.
    """
    binary_mtime = int(binary.stat().st_mtime)
    build_epoch = _last_commit_epoch(repo_root, _BUILD_INPUT_PATHS)
    behaviour_epoch = _last_commit_epoch(repo_root, _BEHAVIOUR_INPUT_PATHS)

    stale_against = []
    for label, epoch in (("sources", build_epoch), ("configs", behaviour_epoch)):
        if epoch is not None and binary_mtime < epoch:
            stale_against.append((label, epoch))

    if stale_against:
        detail = "; ".join(
            f"{label} last changed at epoch {epoch}, binary mtime {binary_mtime}"
            f" ({epoch - binary_mtime}s older)"
            for label, epoch in stale_against
        )
        raise ProvenanceError(
            "solver binary predates the worktree it would be measured against "
            f"({detail}). Rebuild before measuring: a binary that cannot read a "
            "newer preset key fails closed per instance and banks a whole "
            "category of `unknown` rows that score as a real measurement."
        )

    return {
        "binary_mtime_epoch": binary_mtime,
        "build_inputs_last_commit_epoch": build_epoch,
        "behaviour_inputs_last_commit_epoch": behaviour_epoch,
        "build_input_paths": list(_BUILD_INPUT_PATHS),
        "behaviour_input_paths": list(_BEHAVIOUR_INPUT_PATHS),
    }


def _decode_nul_records(data: bytes) -> list[str]:
    return [os.fsdecode(item) for item in data.split(b"\0") if item]


def _validate_worktree_path(raw_path: bytes) -> tuple[bytes, ...]:
    """Validate a Git path without decoding or normalizing its byte identity."""

    if not raw_path or b"\0" in raw_path or raw_path.startswith(b"/"):
        raise ProvenanceError(
            f"unsafe tracked worktree path: {os.fsdecode(raw_path)!r}"
        )
    components = tuple(raw_path.split(b"/"))
    if any(component in {b"", b".", b".."} for component in components):
        raise ProvenanceError(
            f"unsafe tracked worktree path: {os.fsdecode(raw_path)!r}"
        )
    return components


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _namespace_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _verify_open_directory_chain(
    repo: Path,
    root_fd: int,
    chain: list[tuple[int, bytes, int]],
) -> None:
    """Prove every opened parent is still linked beneath the captured root."""

    root_open = os.fstat(root_fd)
    try:
        root_named = os.stat(repo, follow_symlinks=False)
    except OSError as error:
        raise ProvenanceError(
            f"repository root changed during provenance capture: {repo}: {error}"
        ) from error
    if _namespace_identity(root_open) != _namespace_identity(root_named):
        raise ProvenanceError(
            f"repository root changed during provenance capture: {repo}"
        )
    for parent_fd, component, child_fd in chain:
        try:
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(child_fd)
        except OSError as error:
            raise ProvenanceError(
                "tracked worktree parent changed during provenance capture: "
                f"{os.fsdecode(component)!r}: {error}"
            ) from error
        if _namespace_identity(named) != _namespace_identity(opened):
            raise ProvenanceError(
                "tracked worktree parent changed during provenance capture: "
                f"{os.fsdecode(component)!r}"
            )


def _tracked_leaf_state(
    parent_fd: int,
    leaf: bytes,
    display_path: str,
) -> dict[str, object]:
    """Hash one final component without following a symlink."""

    try:
        before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return {"path": display_path, "kind": "missing"}
    except OSError as error:
        raise ProvenanceError(
            f"could not inspect tracked worktree path {display_path!r}: {error}"
        ) from error

    if stat.S_ISLNK(before.st_mode):
        try:
            target = os.readlink(leaf, dir_fd=parent_fd)
            after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ProvenanceError(
                f"could not capture tracked worktree symlink {display_path!r}: {error}"
            ) from error
        if not isinstance(target, bytes):
            target = os.fsencode(target)
        if _stat_identity(before) != _stat_identity(after):
            raise ProvenanceError(
                f"tracked worktree symlink changed during capture: {display_path!r}"
            )
        return {
            "path": display_path,
            "kind": "symlink",
            # Git records symlinks as mode 120000 and does not preserve their
            # host permission bits.  Darwin commonly reports 0755 here while
            # Linux reports 0777, even for the same checkout.  Bind the
            # repository-canonical payload rather than nondeterministic host
            # metadata.
            "mode": 0o777,
            "size_bytes": len(target),
            "sha256": _sha256(target),
        }

    if stat.S_ISREG(before.st_mode):
        mode = before.st_mode & 0o7777
        file_flags = (
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(leaf, file_flags, dir_fd=parent_fd)
        except OSError as error:
            raise ProvenanceError(
                f"could not safely open tracked worktree file {display_path!r}: {error}"
            ) from error
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _namespace_identity(
                before
            ) != _namespace_identity(opened):
                raise ProvenanceError(
                    f"tracked worktree file changed before hashing: {display_path!r}"
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after_open = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            after_named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ProvenanceError(
                "tracked worktree file changed during capture: "
                f"{display_path!r}: {error}"
            ) from error
        if _stat_identity(before) != _stat_identity(after_open) or _stat_identity(
            after_open
        ) != _stat_identity(after_named):
            raise ProvenanceError(
                f"tracked worktree file changed during capture: {display_path!r}"
            )
        return {
            "path": display_path,
            "kind": "file",
            "mode": mode,
            "size_bytes": after_open.st_size,
            "sha256": digest.hexdigest(),
        }

    raise ProvenanceError(
        f"cannot hash tracked worktree special entry: {display_path!r}"
    )


def _tracked_path_states(
    repo: Path,
    raw_paths: list[bytes],
) -> list[dict[str, object]]:
    root_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_fd = os.open(repo, root_flags)
    except OSError as error:
        raise ProvenanceError(
            f"could not safely open repository root {repo}: {error}"
        ) from error
    open_components: list[bytes] = []
    open_fds: list[int] = [root_fd]
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )

    def open_chain() -> list[tuple[int, bytes, int]]:
        return [
            (open_fds[index], component, open_fds[index + 1])
            for index, component in enumerate(open_components)
        ]

    def verify_missing_component(
        parent_fd: int,
        component: bytes,
        display_path: str,
    ) -> None:
        try:
            os.stat(
                component,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as error:
            raise ProvenanceError(
                f"tracked worktree missing path changed during capture for "
                f"{display_path!r}: {error}"
            ) from error
        raise ProvenanceError(
            f"tracked worktree path appeared during capture: {display_path!r}"
        )

    try:
        states: list[dict[str, object]] = []
        for raw_path in sorted(set(raw_paths)):
            components = _validate_worktree_path(raw_path)
            display_path = os.fsdecode(raw_path)
            wanted_parents = components[:-1]
            common = 0
            while (
                common < len(open_components)
                and common < len(wanted_parents)
                and open_components[common] == wanted_parents[common]
            ):
                common += 1
            if common < len(open_components):
                _verify_open_directory_chain(repo, root_fd, open_chain())
                while len(open_components) > common:
                    open_components.pop()
                    os.close(open_fds.pop())

            parent_missing = False
            for component in wanted_parents[common:]:
                try:
                    child_fd = os.open(component, directory_flags, dir_fd=open_fds[-1])
                except FileNotFoundError:
                    _verify_open_directory_chain(repo, root_fd, open_chain())
                    verify_missing_component(open_fds[-1], component, display_path)
                    parent_missing = True
                    break
                except OSError as error:
                    raise ProvenanceError(
                        "could not safely open tracked worktree parent for "
                        f"{display_path!r}: {error}"
                    ) from error
                open_components.append(component)
                open_fds.append(child_fd)

            if parent_missing:
                state = {"path": display_path, "kind": "missing"}
            else:
                state = _tracked_leaf_state(open_fds[-1], components[-1], display_path)
                if state["kind"] == "missing":
                    _verify_open_directory_chain(repo, root_fd, open_chain())
                    verify_missing_component(
                        open_fds[-1],
                        components[-1],
                        display_path,
                    )
            states.append(state)
        _verify_open_directory_chain(repo, root_fd, open_chain())
        return states
    finally:
        for descriptor in reversed(open_fds):
            os.close(descriptor)


def _tracked_path_state(repo: Path, relative: str) -> dict[str, object]:
    """Compatibility wrapper used by focused provenance tests."""

    return _tracked_path_states(repo, [os.fsencode(relative)])[0]


def _parse_index_stage(
    data: bytes,
    *,
    object_format: str,
) -> list[bytes]:
    expected_oid_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if expected_oid_length is None:
        raise ProvenanceError(f"unsupported Git object format: {object_format!r}")
    paths: list[bytes] = []
    for record in (item for item in data.split(b"\0") if item):
        header, separator, path = record.partition(b"\t")
        fields = header.split()
        if not separator or len(fields) != 3:
            raise ProvenanceError("could not parse canonical Git index entry")
        mode, oid, stage = fields
        _validate_worktree_path(path)
        if stage != b"0":
            raise ProvenanceError(
                f"unmerged Git index entry is not measurable: {os.fsdecode(path)!r}"
            )
        if mode == b"160000":
            raise ProvenanceError(
                f"Gitlink entry is not measurable: {os.fsdecode(path)!r}"
            )
        if mode not in {b"100644", b"100755", b"120000"}:
            raise ProvenanceError(
                f"unsupported Git index mode {mode!r}: {os.fsdecode(path)!r}"
            )
        if len(oid) != expected_oid_length or any(
            byte not in b"0123456789abcdef" for byte in oid
        ):
            raise ProvenanceError(f"invalid Git index object ID: {os.fsdecode(path)!r}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ProvenanceError("Git index contains duplicate stage-zero paths")
    return paths


def _parse_index_flags(data: bytes, *, label: str) -> list[bytes]:
    paths: list[bytes] = []
    for record in (item for item in data.split(b"\0") if item):
        if len(record) < 3 or record[1:2] != b" ":
            raise ProvenanceError(f"could not parse Git {label} flags")
        tag = record[:1]
        path = record[2:]
        _validate_worktree_path(path)
        if tag != b"H":
            raise ProvenanceError(
                f"unsupported Git {label} flag {tag!r}: {os.fsdecode(path)!r}"
            )
        paths.append(path)
    return paths


def _tracked_index_snapshot(repo: Path) -> dict[str, object]:
    top_level_bytes = _git_evidence(repo, "rev-parse", "--show-toplevel").strip()
    inside = _git_evidence(repo, "rev-parse", "--is-inside-work-tree").strip()
    bare = _git_evidence(repo, "rev-parse", "--is-bare-repository").strip()
    try:
        top_level = Path(os.fsdecode(top_level_bytes)).resolve(strict=True)
        expected_top_level = repo.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"could not resolve Git worktree identity for {repo}: {error}"
        ) from error
    if inside != b"true" or bare != b"false" or top_level != expected_top_level:
        raise ProvenanceError(
            f"Git worktree identity does not match repository root: {repo}"
        )
    head = _git_evidence(repo, "rev-parse", "--verify", "HEAD").strip()
    object_format_bytes = _git_evidence(
        repo, "rev-parse", "--show-object-format"
    ).strip()
    try:
        object_format = object_format_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ProvenanceError("Git object format is not ASCII") from error
    index_stage = _git_evidence(repo, "ls-files", "--stage", "-z", "--full-name", "--")
    index_paths = _parse_index_stage(index_stage, object_format=object_format)
    flags_v = _git_evidence(repo, "ls-files", "-v", "-z", "--full-name", "--")
    flags_f = _git_evidence(repo, "ls-files", "-f", "-z", "--full-name", "--")
    if (
        _parse_index_flags(flags_v, label="assume-unchanged/skip-worktree")
        != index_paths
    ):
        raise ProvenanceError("Git index flag paths differ from canonical index paths")
    if _parse_index_flags(flags_f, label="fsmonitor") != index_paths:
        raise ProvenanceError("Git fsmonitor paths differ from canonical index paths")
    return {
        "top_level": os.fsencode(top_level),
        "head": head,
        "object_format": object_format,
        "index_stage": index_stage,
        "index_paths": index_paths,
        "flags_v": flags_v,
        "flags_f": flags_f,
    }


def _raw_diff_paths(data: bytes) -> list[bytes]:
    records = data.split(b"\0")
    if records[-1:] != [b""]:
        raise ProvenanceError("Git raw diff is not NUL terminated")
    records.pop()
    if len(records) % 2:
        raise ProvenanceError("Git raw diff has an incomplete path record")
    paths: list[bytes] = []
    for offset in range(0, len(records), 2):
        header = records[offset]
        path = records[offset + 1]
        fields = header.split()
        if (
            not header.startswith(b":")
            or len(fields) != 5
            or fields[-1] not in {b"A", b"D", b"M", b"T", b"U"}
        ):
            raise ProvenanceError("could not parse canonical Git raw diff")
        _validate_worktree_path(path)
        paths.append(path)
    return paths


def _tracked_worktree_evidence(
    repo: Path,
) -> tuple[bytes, list[dict[str, object]]]:
    """Return a compact canonical binding of tracked HEAD/index/worktree state."""

    before = _tracked_index_snapshot(repo)
    diff_args = (
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--ignore-submodules=none",
        "--raw",
        "-z",
        "--full-index",
        "--no-abbrev",
    )
    head_to_index = _git_evidence(
        repo,
        "diff",
        "--cached",
        *diff_args,
        "HEAD",
        "--",
    )
    index_to_worktree = _git_evidence(repo, "diff-files", *diff_args, "--")
    changed_paths = _raw_diff_paths(head_to_index) + _raw_diff_paths(index_to_worktree)
    entries = _tracked_path_states(repo, changed_paths)
    after = _tracked_index_snapshot(repo)
    if before != after:
        raise ProvenanceError("Git HEAD or index changed during provenance capture")
    if not head_to_index and not index_to_worktree and not entries:
        return b"", []
    index_stage = before["index_stage"]
    flags_v = before["flags_v"]
    flags_f = before["flags_f"]
    if not all(isinstance(value, bytes) for value in (index_stage, flags_v, flags_f)):
        raise AssertionError("internal Git index snapshot type mismatch")
    payload = {
        "schema": "ny_tracked_worktree_evidence_v2",
        "head_oid": os.fsdecode(before["head"]),
        "object_format": before["object_format"],
        "index_stage": {
            "bytes": len(index_stage),
            "count": len(before["index_paths"]),
            "sha256": _sha256(index_stage),
        },
        "index_flags_v": {
            "bytes": len(flags_v),
            "sha256": _sha256(flags_v),
        },
        "index_flags_f": {
            "bytes": len(flags_f),
            "sha256": _sha256(flags_f),
        },
        "head_to_index_raw": {
            "bytes": len(head_to_index),
            "sha256": _sha256(head_to_index),
        },
        "index_to_worktree_raw": {
            "bytes": len(index_to_worktree),
            "sha256": _sha256(index_to_worktree),
        },
        "paths": entries,
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        entries,
    )


def _untracked_evidence(repo: Path) -> list[dict[str, object]]:
    names = _git_evidence(
        repo, "ls-files", "--others", "--exclude-standard", "-z", "--full-name", "--"
    )
    raw_names = [item for item in names.split(b"\0") if item]
    evidence: list[dict[str, object]] = []
    for state in _tracked_path_states(repo, raw_names):
        if state["kind"] not in {"file", "symlink"}:
            raise ProvenanceError(
                "untracked worktree entry disappeared during capture: "
                f"{state['path']!r}"
            )
        state.pop("mode")
        evidence.append(state)
    return evidence


def _capture_worktree_once(repo: Path) -> dict[str, object]:
    commit = _git_evidence(repo, "rev-parse", "--verify", "HEAD").decode().strip()
    status = _git_evidence(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        "--no-renames",
    )
    tracked_diff, tracked_paths = _tracked_worktree_evidence(repo)
    untracked = _untracked_evidence(repo)
    digest_payload = {
        "commit": commit,
        "status_sha256": _sha256(status),
        "tracked_diff_format": "ny_tracked_worktree_evidence_v2",
        "tracked_diff_sha256": _sha256(tracked_diff),
        "untracked_files": untracked,
    }
    return {
        "commit": commit,
        "clean": not status,
        "status_porcelain_v1_z_entries": _decode_nul_records(status),
        "status_sha256": _sha256(status),
        "tracked_diff_format": "ny_tracked_worktree_evidence_v2",
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": _sha256(tracked_diff),
        "tracked_worktree_paths": tracked_paths,
        "untracked_files": untracked,
        "worktree_evidence_sha256": _sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ),
    }


def _capture_worktree(repo: Path) -> dict[str, object]:
    first = _capture_worktree_once(repo)
    second = _capture_worktree_once(repo)
    if first != second:
        raise ProvenanceError("NY worktree changed while provenance was captured")
    branch_result = _git_evidence_result(
        repo,
        "symbolic-ref",
        "--short",
        "-q",
        "HEAD",
        check=False,
    )
    first["branch"] = (
        branch_result.stdout.decode("utf-8", "replace").strip()
        if branch_result.returncode == 0
        else None
    )
    first["repo_root"] = str(repo)
    return first


def _parse_toolchain(
    repo: Path,
    *,
    declared_tool_path: str | None = None,
    declared_tool_kind: str | None = None,
    declared_rustc_path: str | None = None,
) -> dict[str, object]:
    path = repo / "rust-toolchain.toml"
    if not path.is_file():
        raise ProvenanceError(f"missing pinned toolchain file: {path}")
    data = path.read_bytes()
    text = data.decode("utf-8")
    channel_match = re.search(r'^\s*channel\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if channel_match is None:
        raise ProvenanceError(f"could not parse toolchain channel from {path}")
    components_match = re.search(
        r"^\s*components\s*=\s*\[([^]]*)\]", text, re.MULTILINE
    )
    components = (
        re.findall(r'"([^"]+)"', components_match.group(1))
        if components_match is not None
        else []
    )
    channel = channel_match.group(1)
    if declared_tool_path is None:
        configured_rustup = os.environ.get("NY_MEASURE_RUSTUP_BIN", "")
        if configured_rustup:
            declared_tool_path = configured_rustup
            declared_tool_kind = "rustup"
    if declared_tool_path is not None:
        tool_kind = declared_tool_kind or "rustup"
        if tool_kind not in {"rustup", "rustc"}:
            raise ProvenanceError(
                f"unsupported Rust toolchain probe kind: {tool_kind!r}"
            )
        tool_identity = _capture_executable_identity(
            declared_tool_path,
            base_dir=repo,
            label=f"Rust {tool_kind}",
        )
        tool = str(tool_identity["resolved_path"])
    else:
        rustup = _find_executable("rustup")
        if rustup is not None:
            tool_kind = "rustup"
            tool = rustup
        else:
            rustc = _find_executable("rustc")
            if rustc is None:
                raise ProvenanceError("neither rustup nor rustc is available")
            tool_kind = "rustc"
            tool = rustc
        tool_identity = _capture_executable_identity(
            tool,
            base_dir=repo,
            label=f"Rust {tool_kind}",
        )
        tool = str(tool_identity["resolved_path"])
    helper_environment = {
        "HOME": os.environ.get("HOME", "/"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    rustup_home = os.environ.get("RUSTUP_HOME")
    if rustup_home:
        helper_environment["RUSTUP_HOME"] = rustup_home
    if tool_kind == "rustup":
        selector_command = [tool, "which", "--toolchain", channel, "rustc"]
        if declared_rustc_path is None:
            selector_result = _run(
                selector_command,
                check=False,
                env=helper_environment,
            )
            if selector_result.returncode != 0:
                detail = selector_result.stderr.decode(
                    "utf-8", "replace"
                ).strip()
                raise ProvenanceError(
                    "pinned rustc selection probe failed "
                    f"({selector_result.returncode}): {detail}"
                )
            selected_rustc = selector_result.stdout.decode(
                "utf-8", "strict"
            ).strip()
            if (
                not selected_rustc
                or "\n" in selected_rustc
                or not Path(selected_rustc).is_absolute()
            ):
                raise ProvenanceError(
                    "pinned rustc selection returned an invalid executable path"
                )
        else:
            selected_rustc = declared_rustc_path
        rustc_identity = _capture_executable_identity(
            selected_rustc,
            base_dir=repo,
            label="pinned rustc",
        )
    else:
        selector_command = None
        rustc_identity = tool_identity
    rustc = str(rustc_identity["resolved_path"])
    version_command = [rustc, "--version", "--verbose"]
    version_result = _run(
        version_command,
        check=False,
        env=helper_environment,
    )
    if version_result.returncode != 0:
        detail = version_result.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(
            "pinned rustc version probe failed "
            f"({version_result.returncode}): {detail}"
        )
    if (
        _capture_executable_identity(
            str(tool_identity["declared_path"]),
            base_dir=repo,
            label=f"Rust {tool_kind}",
        )
        != tool_identity
    ):
        raise ProvenanceError(
            f"Rust {tool_kind} executable changed during toolchain capture"
        )
    if (
        _capture_executable_identity(
            str(rustc_identity["declared_path"]),
            base_dir=repo,
            label="pinned rustc",
        )
        != rustc_identity
    ):
        raise ProvenanceError(
            "pinned rustc executable changed during toolchain capture"
        )
    return {
        "path": str(path),
        "sha256": _sha256(data),
        "channel": channel,
        "components": components,
        "probe_tool": {
            "kind": tool_kind,
            **tool_identity,
        },
        "rustc": rustc_identity,
        "selector_command": selector_command,
        "selector_returncode": 0 if selector_command is not None else None,
        "selector_stdout": (
            str(rustc_identity["resolved_path"])
            if selector_command is not None
            else None
        ),
        "selector_stderr": "" if selector_command is not None else None,
        "version_command": version_command,
        "version_returncode": version_result.returncode,
        "version_stdout": version_result.stdout.decode("utf-8", "replace").strip(),
        "version_stderr": version_result.stderr.decode("utf-8", "replace").strip(),
    }


def _parse_ay_pin(repo: Path) -> dict[str, str]:
    lock_path = repo / "Cargo.lock"
    if not lock_path.is_file():
        raise ProvenanceError(f"missing Cargo.lock: {lock_path}")
    data = lock_path.read_bytes()
    matches = re.findall(
        rb"git\+https://github\.com/alabsystems/ay\.git"
        rb"\?rev=([0-9a-f]{40})#([0-9a-f]{40})",
        data,
    )
    pins = {(requested.decode(), resolved.decode()) for requested, resolved in matches}
    if len(pins) != 1:
        raise ProvenanceError(
            f"expected one exact AY revision in Cargo.lock, found {len(pins)}"
        )
    requested, resolved = next(iter(pins))
    if requested != resolved:
        raise ProvenanceError(
            f"AY requested revision {requested} resolved to different commit {resolved}"
        )
    return {
        "git_revision": resolved,
        "requested_revision": requested,
        "cargo_lock_sha256": _sha256(data),
    }


def _sanitize_remote_url(url: str) -> str:
    """Remove credentials, query strings, and fragments from a Git remote."""
    value = url.strip()
    if "://" in value:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    value = value.split("?", 1)[0].split("#", 1)[0]
    scp_match = re.match(r"(?:[^@/:]+@)?([^:]+):(.+)", value)
    if scp_match is not None:
        return f"{scp_match.group(1)}:{scp_match.group(2)}"
    return value


def _capture_benchmark(benchmark_root: Path) -> dict[str, object]:
    top = _git_evidence(benchmark_root, "rev-parse", "--show-toplevel").decode().strip()
    repo = Path(top).resolve()
    worktree = _capture_worktree(repo)
    remotes: list[dict[str, str]] = []
    for name in _git_evidence(repo, "remote").decode("utf-8", "replace").splitlines():
        remote = _git_evidence(repo, "remote", "get-url", name).decode(
            "utf-8", "replace"
        )
        remotes.append({"name": name, "fetch_url": _sanitize_remote_url(remote)})
    return {
        **worktree,
        "benchmark_root": str(benchmark_root),
        "remotes": remotes,
    }


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"ID", "VERSION_ID", "PRETTY_NAME", "BUILD_ID"}:
            values[key] = value.strip().strip('"')
    return values


def _optional_command(command: list[str]) -> dict[str, object] | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    actual = [executable, *command[1:]]
    result = _run(actual, check=False, timeout=15)
    return {
        "command": actual,
        "returncode": result.returncode,
        "stdout": result.stdout.decode("utf-8", "replace").strip(),
        "stderr": result.stderr.decode("utf-8", "replace").strip(),
    }


def _bounded_decode(
    data: bytes, limit: int = SNAPSHOT_OUTPUT_LIMIT
) -> tuple[str, bool]:
    truncated = len(data) > limit
    return data[:limit].decode("utf-8", "replace").strip(), truncated


def _snapshot_command(command: list[str], *, timeout: int = 5) -> dict[str, object]:
    """Run a fixed diagnostic command without making provenance capture fatal."""
    result: dict[str, object] = {
        "command": command,
        "status": "unavailable",
        "returncode": None,
        "stdout": "",
        "stdout_truncated": False,
        "stderr": "",
        "stderr_truncated": False,
    }
    executable = shutil.which(command[0])
    if executable is None:
        return result
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout, stdout_truncated = _bounded_decode(error.stdout or b"")
        stderr, stderr_truncated = _bounded_decode(error.stderr or b"")
        result.update(
            {
                "status": "timeout",
                "stdout": stdout,
                "stdout_truncated": stdout_truncated,
                "stderr": stderr,
                "stderr_truncated": stderr_truncated,
            }
        )
        return result
    except OSError as error:
        result.update({"status": "error", "stderr": str(error)[:1024]})
        return result
    stdout, stdout_truncated = _bounded_decode(completed.stdout)
    stderr, stderr_truncated = _bounded_decode(completed.stderr)
    result.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": stdout,
            "stdout_truncated": stdout_truncated,
            "stderr": stderr,
            "stderr_truncated": stderr_truncated,
        }
    )
    return result


def _redact_process_args(args: str) -> str:
    """Bound process arguments and redact common credential-bearing forms."""
    safe = " ".join(args.replace("\x00", " ").split())
    safe = re.sub(
        r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s]+@",
        r"\1[REDACTED]@",
        safe,
    )
    safe = re.sub(
        rf"(?i)(\b(?=[A-Za-z_])[A-Za-z0-9_]*{_SENSITIVE_ARGUMENT}"
        rf"[A-Za-z0-9_]*\s*=\s*)(?:\"[^\"]*\"|'[^']*'|\S+)",
        r"\1[REDACTED]",
        safe,
    )
    safe = re.sub(
        rf"(?i)((?:^|\s)--?[A-Za-z0-9_-]*{_SENSITIVE_ARGUMENT}[A-Za-z0-9_-]*"
        rf"(?:\s*=\s*|\s+))(?:\"[^\"]*\"|'[^']*'|\S+)",
        r"\1[REDACTED]",
        safe,
    )
    safe = re.sub(r"(?i)(\bBearer\s+)\S+", r"\1[REDACTED]", safe)
    if len(safe) > PROCESS_ARGS_LIMIT:
        return safe[: PROCESS_ARGS_LIMIT - 1] + "…"
    return safe


def _process_snapshot() -> dict[str, object]:
    command = [
        "ps",
        "-eo",
        "pid=,ppid=,stat=,etimes=,pcpu=,pmem=,comm=,args=",
    ]
    captured = _snapshot_command(command)
    entries: list[dict[str, object]] = []
    if captured["status"] == "ok":
        for line in str(captured["stdout"]).splitlines():
            fields = line.strip().split(None, 7)
            if len(fields) < 7:
                continue
            try:
                pid = int(fields[0])
                ppid = int(fields[1])
                elapsed_seconds = int(fields[3])
                cpu_percent = float(fields[4])
                memory_percent = float(fields[5])
            except ValueError:
                continue
            entries.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "state": fields[2][:16],
                    "elapsed_seconds": elapsed_seconds,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "command": fields[6][:64],
                    "args_redacted": _redact_process_args(
                        fields[7] if len(fields) == 8 else fields[6]
                    ),
                }
            )
    entries.sort(
        key=lambda entry: (
            -float(entry["cpu_percent"]),
            -float(entry["memory_percent"]),
            int(entry["pid"]),
        )
    )
    total_observed = len(entries)
    return {
        "source": "ps",
        "status": captured["status"],
        "returncode": captured["returncode"],
        "stderr": captured["stderr"],
        "stderr_truncated": captured["stderr_truncated"],
        "max_entries": PROCESS_SNAPSHOT_LIMIT,
        "max_args_chars": PROCESS_ARGS_LIMIT,
        "sort": ["cpu_percent_desc", "memory_percent_desc", "pid_asc"],
        "total_observed": total_observed,
        "truncated": total_observed > PROCESS_SNAPSHOT_LIMIT
        or bool(captured["stdout_truncated"]),
        "entries": entries[:PROCESS_SNAPSHOT_LIMIT],
    }


def _load_average_snapshot() -> dict[str, object]:
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return {
            "available": False,
            "one_minute": None,
            "five_minutes": None,
            "fifteen_minutes": None,
        }
    return {
        "available": True,
        "one_minute": one,
        "five_minutes": five,
        "fifteen_minutes": fifteen,
    }


def _gpu_activity_snapshot() -> dict[str, object]:
    return {
        "utilization": _snapshot_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute_processes": _snapshot_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def _capture_host_state() -> dict[str, object]:
    return {
        "schema": "ny_measurement_host_state_v1",
        "captured_at_utc": _utc_now(),
        "load_average": _load_average_snapshot(),
        "processes": _process_snapshot(),
        "gpu": _gpu_activity_snapshot(),
    }


def _cpu_identity() -> dict[str, object]:
    identity: dict[str, object] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        identity["allowed_logical_cpus"] = sorted(os.sched_getaffinity(0))
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        selected: dict[str, list[str]] = {}
        accepted = {
            "model name",
            "hardware",
            "processor",
            "cpu implementer",
            "cpu part",
        }
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = (item.strip() for item in line.split(":", 1))
            if key.lower() in accepted and value and value not in selected.get(key, []):
                selected.setdefault(key, []).append(value)
        identity["proc_cpuinfo_identity"] = selected
    identity["lscpu"] = _optional_command(["lscpu", "--json"])
    return identity


def _memory_identity() -> dict[str, object]:
    identity: dict[str, object] = {}
    if hasattr(os, "sysconf"):
        try:
            identity["physical_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
                "SC_PHYS_PAGES"
            )
        except (OSError, ValueError):
            pass
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("MemTotal:", "SwapTotal:")):
                key, value = line.split(":", 1)
                identity[key] = value.strip()
    return identity


def _gpu_identity() -> dict[str, object]:
    query = _optional_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "nvidia_smi_query": query,
        "nvcc_version": _optional_command(["nvcc", "--version"]),
        "display_pci_devices": _optional_command(["lspci", "-nn"]),
        "macos_displays": _optional_command(
            ["system_profiler", "SPDisplaysDataType", "-json"]
        ),
    }


def _resource_limits() -> dict[str, list[int]]:
    names = [
        "RLIMIT_AS",
        "RLIMIT_CORE",
        "RLIMIT_CPU",
        "RLIMIT_DATA",
        "RLIMIT_FSIZE",
        "RLIMIT_MEMLOCK",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
        "RLIMIT_STACK",
    ]
    limits: dict[str, list[int]] = {}
    for name in names:
        if hasattr(resource, name):
            soft, hard = resource.getrlimit(getattr(resource, name))
            limits[name] = [soft, hard]
    return limits


def _capture_measurement_containment(
    *,
    proc_self_cgroup: Path | None = None,
    proc_self_mountinfo: Path | None = None,
    cgroup_root: Path | None = None,
) -> dict[str, object]:
    """Capture and validate the scorecard process's effective containment."""
    expected_cpu_count = _expected_measurement_cpu_count()
    expected_cpu_period_us = 100_000
    expected_cpu_quota_us = expected_cpu_count * expected_cpu_period_us
    proc_self_cgroup = proc_self_cgroup or PROC_SELF_CGROUP
    proc_self_mountinfo = proc_self_mountinfo or PROC_SELF_MOUNTINFO
    cgroup_root = cgroup_root or CGROUP_V2_ROOT
    try:
        cgroup_lines = proc_self_cgroup.read_text(encoding="utf-8").splitlines()
        mount_lines = proc_self_mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProvenanceError(
            f"cannot read kernel cgroup containment identity: {error}"
        ) from error

    unified = [line[3:] for line in cgroup_lines if line.startswith("0::")]
    if len(unified) != 1:
        raise ProvenanceError("expected exactly one unified cgroup-v2 membership")
    membership = unified[0]
    if (
        not membership.startswith("/")
        or "/../" in membership
        or "/./" in membership
        or "//" in membership
        or "\x00" in membership
    ):
        raise ProvenanceError("malformed unified cgroup-v2 membership")

    uid = os.getuid()
    slice_membership = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/ny.slice/ny-build.slice"
    )
    leaf_pattern = re.compile(
        re.escape(slice_membership) + rf"/ny-safe-gpu-{uid}-[0-9]+-[0-9]+[.]service"
    )
    if leaf_pattern.fullmatch(membership) is None:
        raise ProvenanceError(
            "scorecard process is not in the exact ny-safe-gpu service hierarchy"
        )

    cgroup2_mounts: list[tuple[str, str]] = []
    for line in mount_lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 1 >= len(fields) or fields[separator + 1] != "cgroup2":
            continue
        if len(fields) < 5:
            raise ProvenanceError("malformed cgroup-v2 mountinfo entry")
        cgroup2_mounts.append((fields[3], fields[4]))
    if len(cgroup2_mounts) != 1:
        raise ProvenanceError("expected exactly one cgroup-v2 mount")
    mount_root, mount_point = cgroup2_mounts[0]
    if mount_root != "/" or mount_point != str(cgroup_root):
        raise ProvenanceError(
            "cgroup-v2 must be rooted at the reviewed containment mount"
        )

    try:
        root_resolved = cgroup_root.resolve(strict=True)
        current_lexical = root_resolved / membership.lstrip("/")
        slice_lexical = root_resolved / slice_membership.lstrip("/")
        current_resolved = current_lexical.resolve(strict=True)
        slice_resolved = slice_lexical.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(f"cannot resolve scorecard cgroup: {error}") from error
    if (
        not root_resolved.is_dir()
        or not current_resolved.is_dir()
        or not slice_resolved.is_dir()
        or current_resolved != current_lexical
        or slice_resolved != slice_lexical
        or not _is_within(current_resolved, slice_resolved)
    ):
        raise ProvenanceError(
            "resolved scorecard cgroup escapes the reviewed hierarchy"
        )
    # The guard applies the reviewed policy to this exact transient leaf. This
    # keeps measurement containment self-contained while the ancestor scan
    # below still rejects tighter or malformed host policy.
    policy_resolved = current_resolved

    def membership_for(path: Path) -> str:
        relative = path.relative_to(root_resolved).as_posix()
        return f"/{relative}" if relative != "." else "/"

    def read_control(path: Path) -> str:
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except OSError as error:
            raise ProvenanceError(
                f"cannot read cgroup control {path}: {error}"
            ) from error
        if len(lines) != 1 or not lines[0]:
            raise ProvenanceError(f"malformed cgroup control: {path}")
        return lines[0]

    expected_policy_raw = {
        "memory.high": str(_expected_memory_high_bytes()),
        "memory.max": str(_expected_memory_max_bytes()),
        "memory.swap.max": str(EXPECTED_MEMORY_SWAP_MAX_BYTES),
        "pids.max": str(EXPECTED_PIDS_MAX),
        "cpu.max": f"{expected_cpu_quota_us} {expected_cpu_period_us}",
    }
    policy_raw = {
        name: read_control(policy_resolved / name) for name in expected_policy_raw
    }
    if policy_raw != expected_policy_raw:
        raise ProvenanceError(
            "ny-safe-gpu service cgroup controls differ from the reviewed policy"
        )

    def cgroup_chain() -> list[Path]:
        chain: list[Path] = []
        scan = current_resolved
        while True:
            chain.append(scan)
            if scan == root_resolved:
                return chain
            if root_resolved not in scan.parents:
                raise ProvenanceError("cgroup ancestor traversal escaped its mount")
            scan = scan.parent

    chain = cgroup_chain()

    def scalar_effective(
        control_name: str,
        expected: int,
        value_key: str,
    ) -> dict[str, object]:
        levels: list[dict[str, str]] = []
        finite: list[tuple[int, str]] = []
        for scan in chain:
            control_path = scan / control_name
            if not control_path.is_file():
                continue
            raw = read_control(control_path)
            source = membership_for(scan)
            levels.append({"cgroup_path": source, "raw": raw})
            if raw == "max":
                continue
            if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
                raise ProvenanceError(f"malformed {control_name} at cgroup {source}")
            finite.append((int(raw), source))
        if not finite:
            raise ProvenanceError(f"no finite effective {control_name} cgroup limit")
        value, source = min(finite, key=lambda item: item[0])
        if value != expected:
            raise ProvenanceError(
                f"effective {control_name} is {value}, expected {expected}"
            )
        return {
            value_key: value,
            "source_cgroup_path": source,
            "levels": levels,
        }

    def cpu_effective() -> dict[str, object]:
        levels: list[dict[str, str]] = []
        finite: list[tuple[Fraction, int, int, str]] = []
        for scan in chain:
            control_path = scan / "cpu.max"
            if not control_path.is_file():
                continue
            raw = read_control(control_path)
            source = membership_for(scan)
            levels.append({"cgroup_path": source, "raw": raw})
            fields = raw.split()
            if len(fields) != 2 or re.fullmatch(r"[1-9][0-9]*", fields[1]) is None:
                raise ProvenanceError(f"malformed cpu.max at cgroup {source}")
            if fields[0] == "max":
                continue
            if re.fullmatch(r"[1-9][0-9]*", fields[0]) is None:
                raise ProvenanceError(f"malformed cpu.max at cgroup {source}")
            quota = int(fields[0])
            period = int(fields[1])
            finite.append((Fraction(quota, period), quota, period, source))
        if not finite:
            raise ProvenanceError("no finite effective cpu.max cgroup quota")
        ratio, quota, period, source = min(finite, key=lambda item: item[0])
        if ratio != expected_cpu_count:
            raise ProvenanceError(
                f"effective cpu.max is {ratio} CPUs, expected {expected_cpu_count}"
            )
        return {
            "quota_us": quota,
            "period_us": period,
            "equivalent_cpus": expected_cpu_count,
            "source_cgroup_path": source,
            "levels": levels,
        }

    soft_as, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    expected_as = _expected_rlimit_as_bytes()
    if soft_as != expected_as or hard_as != expected_as:
        raise ProvenanceError(
            "soft and hard RLIMIT_AS must both be exactly "
            f"{expected_as} bytes for containment profile "
            f"{_containment_profile_name()}"
        )

    return {
        "schema": "ny_measurement_containment_v1",
        "cgroup_version": 2,
        # Name the profile in the evidence itself: the byte values alone do not
        # say which lane was intended, and results are only comparable within a
        # profile.
        "containment_profile": _containment_profile_name(),
        "membership": membership,
        "mount_point": str(root_resolved),
        "current_cgroup": str(current_resolved),
        "policy_cgroup": str(policy_resolved),
        "policy": {
            "memory.high": {
                "raw": policy_raw["memory.high"],
                "value_bytes": _expected_memory_high_bytes(),
            },
            "memory.max": {
                "raw": policy_raw["memory.max"],
                "value_bytes": _expected_memory_max_bytes(),
            },
            "memory.swap.max": {
                "raw": policy_raw["memory.swap.max"],
                "value_bytes": EXPECTED_MEMORY_SWAP_MAX_BYTES,
            },
            "pids.max": {
                "raw": policy_raw["pids.max"],
                "value": EXPECTED_PIDS_MAX,
            },
            "cpu.max": {
                "raw": policy_raw["cpu.max"],
                "quota_us": expected_cpu_quota_us,
                "period_us": expected_cpu_period_us,
                "equivalent_cpus": expected_cpu_count,
            },
        },
        "effective": {
            "memory.high": scalar_effective(
                "memory.high", _expected_memory_high_bytes(), "value_bytes"
            ),
            "memory.max": scalar_effective(
                "memory.max", _expected_memory_max_bytes(), "value_bytes"
            ),
            "memory.swap.max": scalar_effective(
                "memory.swap.max", EXPECTED_MEMORY_SWAP_MAX_BYTES, "value_bytes"
            ),
            "pids.max": scalar_effective("pids.max", EXPECTED_PIDS_MAX, "value"),
            "cpu.max": cpu_effective(),
        },
        "rlimit_as": {
            "soft_bytes": soft_as,
            "hard_bytes": hard_as,
        },
    }


def _validate_loader_preload_configuration() -> None:
    """Reject process-wide loader injection that an env scrub cannot remove."""
    try:
        if os.path.lexists(SYSTEM_LD_SO_PRELOAD):
            if SYSTEM_LD_SO_PRELOAD.is_symlink():
                raise ProvenanceError(
                    f"system dynamic-loader preload file must not be a symlink: "
                    f"{SYSTEM_LD_SO_PRELOAD}"
                )
            contents = SYSTEM_LD_SO_PRELOAD.read_bytes()
            if contents.strip():
                raise ProvenanceError(
                    "non-empty /etc/ld.so.preload is forbidden for measurement "
                    "provenance"
                )
    except OSError as error:
        raise ProvenanceError(
            "cannot attest the system dynamic-loader preload configuration: "
            f"{error}"
        ) from error


def _validate_cuda_library_directory(component: str) -> None:
    """Allow only dedicated CUDA-library directories in the source search path."""
    declared = Path(component)
    try:
        resolved = declared.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(
            f"LD_LIBRARY_PATH directory does not exist: {declared}"
        ) from error
    if not resolved.is_dir():
        raise ProvenanceError(
            f"LD_LIBRARY_PATH component is not a directory: {declared}"
        )
    try:
        with os.scandir(resolved) as iterator:
            entries = list(iterator)
    except OSError as error:
        raise ProvenanceError(
            f"cannot enumerate LD_LIBRARY_PATH directory {declared}: {error}"
        ) from error
    for entry in entries:
        if CUDA_RUNTIME_SAFE_LIBRARY_NAME.fullmatch(entry.name) is None:
            raise ProvenanceError(
                "LD_LIBRARY_PATH must contain dedicated CUDA directories only; "
                f"unsafe entry {entry.name!r} is present in {declared}"
            )
        try:
            if not entry.is_file(follow_symlinks=True):
                raise ProvenanceError(
                    "LD_LIBRARY_PATH CUDA entries must resolve to regular files: "
                    f"{Path(entry.path)}"
                )
        except OSError as error:
            raise ProvenanceError(
                f"cannot inspect LD_LIBRARY_PATH entry {entry.path}: {error}"
            ) from error


def _capture_environment() -> dict[str, object]:
    unsafe_loader_environment = sorted(
        key for key in UNSAFE_DYNAMIC_LOADER_ENV if key in os.environ
    )
    if unsafe_loader_environment:
        raise ProvenanceError(
            "dynamic-loader injection is forbidden in measurement provenance: "
            + ", ".join(unsafe_loader_environment)
        )
    unknown_darwin_loader_environment = sorted(
        key for key in os.environ if key.startswith("DYLD_")
    )
    if unknown_darwin_loader_environment:
        raise ProvenanceError(
            "dynamic-loader injection is forbidden in measurement provenance: "
            + ", ".join(unknown_darwin_loader_environment)
        )
    unknown_linux_loader_environment = sorted(
        key
        for key in os.environ
        if key.startswith("LD_")
        and key not in {"LD_LIBRARY_PATH"}
        and key not in UNSAFE_DYNAMIC_LOADER_ENV
    )
    if unknown_linux_loader_environment:
        raise ProvenanceError(
            "unreviewed dynamic-loader controls are forbidden in measurement "
            "provenance: " + ", ".join(unknown_linux_loader_environment)
        )
    _validate_loader_preload_configuration()
    if "LD_LIBRARY_PATH" in os.environ:
        components = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
        unsafe_components = [
            component
            for component in components
            if not component or not Path(component).is_absolute()
        ]
        if unsafe_components:
            raise ProvenanceError(
                "LD_LIBRARY_PATH must contain only non-empty absolute directories "
                "for measurement provenance"
            )
        for component in components:
            _validate_cuda_library_directory(component)
    unsafe_shell_environment = sorted(
        key
        for key in os.environ
        if key in {"BASH_ENV", "ENV", "BASHOPTS", "SHELLOPTS"}
        or key.startswith("BASH_FUNC_")
    )
    if unsafe_shell_environment:
        raise ProvenanceError(
            "unsafe shell launch environment is forbidden in measurement "
            "provenance: " + ", ".join(unsafe_shell_environment)
        )
    unreviewed_solver_runtime_environment = sorted(
        key
        for key in os.environ
        if key not in ENV_ALLOWLIST
        and (
            key in UNREVIEWED_SOLVER_RUNTIME_ENV_EXACT
            or key.startswith(UNREVIEWED_SOLVER_RUNTIME_ENV_PREFIXES)
            or key.upper().startswith("MIMALLOC_")
        )
    )
    if unreviewed_solver_runtime_environment:
        raise ProvenanceError(
            "unreviewed solver-runtime environment controls are forbidden in "
            "measurement provenance; add a reviewed non-secret key to the fixed "
            "allowlist or unset it: "
            + ", ".join(unreviewed_solver_runtime_environment)
        )
    unknown_solver = sorted(
        key
        for key in os.environ
        if (
            key.startswith(("NY_", "AY_"))
            or key.upper().startswith("MIMALLOC_")
        )
        and key not in ENV_ALLOWLIST
    )
    if unknown_solver:
        raise ProvenanceError(
            "unrecorded NY_*/AY_*/MIMALLOC_* environment variables are set; "
            "add a reviewed non-secret key to the fixed allowlist or unset it: "
            f"{', '.join(unknown_solver)}"
        )
    values: dict[str, str] = {
        key: os.environ[key] for key in sorted(ENV_ALLOWLIST) if key in os.environ
    }
    typed_values: dict[str, dict[str, object]] = {}
    usize_max = (sys.maxsize * 2) + 1
    for key in sorted(TYPED_NONNEGATIVE_INTEGER_ENV):
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ProvenanceError(
                f"{key} must be a nonnegative integer for measurement provenance"
            )
        value = int(raw)
        if value > usize_max:
            raise ProvenanceError(
                f"{key} is outside the native usize range for measurement provenance"
            )
        typed_values[key] = {
            "type": "nonnegative_integer",
            "value": value,
        }
    for key in sorted(TYPED_POSITIVE_INTEGER_ENV):
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ProvenanceError(
                f"{key} must be a positive integer for measurement provenance"
            )
        value = int(raw)
        if value > usize_max:
            raise ProvenanceError(
                f"{key} is outside the native usize range for measurement provenance"
            )
        if value == 0:
            raise ProvenanceError(
                f"{key} must be a positive integer for measurement provenance"
            )
        typed_values[key] = {
            "type": "positive_integer",
            "value": value,
        }
    for key, minimum, maximum in TYPED_BOUNDED_NONNEGATIVE_INTEGER_ENV:
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ProvenanceError(
                f"{key} must be a bounded nonnegative integer "
                "for measurement provenance"
            )
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ProvenanceError(
                f"{key} must be within [{minimum}, {maximum}] "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "bounded_nonnegative_integer",
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
        }
    canonical_nonnegative_integer_pattern = r"(?:0|[1-9][0-9]*)"
    for (
        key,
        minimum,
        maximum,
    ) in TYPED_BOUNDED_CANONICAL_NONNEGATIVE_INTEGER_ENV:
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(canonical_nonnegative_integer_pattern, raw) is None:
            raise ProvenanceError(
                f"{key} must be a bounded canonical nonnegative integer "
                "for measurement provenance"
            )
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ProvenanceError(
                f"{key} must be within [{minimum}, {maximum}] "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "bounded_canonical_nonnegative_integer",
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
        }
    for key, minimum, maximum in TYPED_BOUNDED_POSITIVE_INTEGER_ENV:
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(r"[0-9]+", raw) is None:
            raise ProvenanceError(
                f"{key} must be a bounded positive integer for measurement provenance"
            )
        value = int(raw)
        if not minimum <= value <= maximum:
            raise ProvenanceError(
                f"{key} must be within [{minimum}, {maximum}] "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "bounded_positive_integer",
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
        }
    canonical_open_unit_decimal_pattern = r"0\.(?:[0-9]*[1-9])"
    for key in sorted(TYPED_OPEN_UNIT_DECIMAL_FRACTION_ENV):
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(canonical_open_unit_decimal_pattern, raw) is None:
            raise ProvenanceError(
                f"{key} must be a canonical decimal fraction in (0, 1) "
                "for measurement provenance"
            )
        value = float(raw)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ProvenanceError(
                f"{key} must parse to a finite f64-compatible value in (0, 1) "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "open_unit_decimal_fraction",
            "value": value,
            "minimum_exclusive": 0.0,
            "maximum_exclusive": 1.0,
        }
    decimal_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for key, minimum, maximum in TYPED_BOUNDED_FLOAT_ENV:
        if key not in values:
            continue
        raw = values[key]
        if re.fullmatch(decimal_pattern, raw) is None:
            raise ProvenanceError(
                f"{key} must be a decimal number for measurement provenance"
            )
        value = float(raw)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ProvenanceError(
                f"{key} must be finite and within [{minimum}, {maximum}] "
                "for measurement provenance"
            )
        typed_values[key] = {
            "type": "bounded_float",
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
        }
    for key in sorted(TYPED_ABSOLUTE_PATH_ENV):
        if key not in values:
            continue
        raw = values[key]
        if not raw or not Path(raw).is_absolute():
            raise ProvenanceError(
                f"{key} must be a non-empty absolute path for measurement provenance"
            )
        typed_values[key] = {
            "type": "absolute_path",
            "value": raw,
        }
    for key, allowed_values in TYPED_ENUM_ENV:
        if key not in values:
            continue
        raw = values[key]
        if raw not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            raise ProvenanceError(
                f"{key} must be unset or one of [{allowed}] for measurement provenance"
            )
        typed_values[key] = {
            "type": "enum",
            "value": raw,
            "allowed_values": sorted(allowed_values),
        }
    for key in sorted(TYPED_STRICT_BOOLEAN_ENV):
        if key not in values:
            continue
        raw = values[key]
        # The runtime gives an explicitly empty value the same false meaning as
        # "0". Measurement syntax deliberately narrows the runtime's broader
        # u32/fallback parser to the reviewed boolean spellings below so a typo
        # cannot silently disable GPU routing in scored evidence.
        if raw not in {"", "0", "1"}:
            raise ProvenanceError(
                f"{key} must be unset, empty, '0', or '1' for measurement provenance"
            )
        typed_values[key] = {
            "type": "boolean",
            "value": raw == "1",
        }
    for key in sorted(TYPED_EXACT_BOOLEAN_ENV):
        if key not in values:
            continue
        raw = values[key]
        if raw not in {"0", "1"}:
            raise ProvenanceError(
                f"{key} must be unset, '0', or '1' for measurement provenance"
            )
        typed_values[key] = {
            "type": "boolean",
            "value": raw == "1",
        }
    return {
        "allowlist_schema": "ny_measurement_environment_v1",
        "values": values,
        "typed_values": typed_values,
    }


def _declared_build_features() -> tuple[str, list[str]]:
    raw = os.environ.get("NY_BUILD_FEATURES", "").strip()
    if not raw:
        raise ProvenanceError(
            "NY_BUILD_FEATURES is required; provenance cannot infer Cargo "
            "features from an already-built binary"
        )
    features = [feature.strip() for feature in raw.split(",")]
    if any(
        not feature or re.fullmatch(r"[A-Za-z0-9_+./?:-]+", feature) is None
        for feature in features
    ):
        raise ProvenanceError(
            "NY_BUILD_FEATURES must be a comma-separated list of Cargo feature names"
        )
    if len(set(features)) != len(features):
        raise ProvenanceError("NY_BUILD_FEATURES contains a duplicate feature")
    return raw, features


def _cuda_runtime_required(
    environment: dict[str, object], build_features: list[str]
) -> bool:
    values = environment.get("values")
    if not isinstance(values, dict):
        raise ProvenanceError("captured measurement environment is malformed")
    allow_noncuda = values.get("NY_ALLOW_NONCUDA_MEASURE") == "1"
    if "cuda" in build_features and not allow_noncuda and "NY_NO_CUDA" in values:
        raise ProvenanceError(
            "NY_NO_CUDA is forbidden for CUDA score measurement; use the explicit "
            "NY_ALLOW_NONCUDA_MEASURE=1 CPU-debug path"
        )
    return "cuda" in build_features and not allow_noncuda


def _not_required_cuda_runtime_identity(
    environment: dict[str, object], build_features: list[str]
) -> dict[str, object]:
    if "cuda" not in build_features:
        reason = "cuda_build_feature_not_declared"
    else:
        values = environment.get("values")
        if not isinstance(values, dict) or values.get("NY_ALLOW_NONCUDA_MEASURE") != "1":
            raise ProvenanceError(
                "CUDA runtime may be omitted only for a non-CUDA build or an "
                "explicit NY_ALLOW_NONCUDA_MEASURE=1 debug capture"
            )
        if values.get("NY_NO_CUDA") != "1":
            raise ProvenanceError(
                "the NY_ALLOW_NONCUDA_MEASURE=1 debug path must bind NY_NO_CUDA=1"
            )
        reason = "noncuda_measurement_explicitly_allowed"
    return {
        "schema": MEASUREMENT_CUDA_RUNTIME_SCHEMA,
        "status": "not_required",
        "reason": reason,
    }


def _cuda_probe_environment(loader_path: str | Path | None) -> dict[str, str]:
    # CUDA identity probes execute the measured binary before the scorecard
    # shell reconstructs its final env-i launch. Give those probes the same
    # reviewed namespace instead of leaking arbitrary ambient controls.
    environment = {
        key: os.environ[key]
        for key in ENV_ALLOWLIST - SOLVER_ENVIRONMENT_EXCLUDED_KEYS
        if key in os.environ
    }
    environment["PATH"] = "/usr/bin:/bin"
    environment["RUST_LOG"] = "error"
    if loader_path is not None:
        environment["LD_LIBRARY_PATH"] = str(loader_path)
        environment.pop("DYLD_LIBRARY_PATH", None)
    else:
        # A source loader directory is qualification input only.  Probes which
        # do not explicitly request it must not inherit it by accident.
        environment.pop("LD_LIBRARY_PATH", None)
    return environment


def _normalized_runtime_fingerprint(
    value: object, *, role: str
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ProvenanceError(
            f"sealed solver CUDA runtime {role} fingerprint is missing"
        )
    normalized: dict[str, int] = {}
    for field in ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns"):
        item = value.get(field)
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or (field == "inode" and item == 0)
        ):
            raise ProvenanceError(
                f"sealed solver CUDA runtime {role} fingerprint is invalid"
            )
        normalized[field] = item
    return normalized


def _cuda_runtime_probe(
    binary: Path, *, loader_path: str | Path | None = None
) -> dict[str, object]:
    result = _run(
        [str(binary), "--cuda-runtime-info"],
        check=False,
        timeout=60,
        env=_cuda_probe_environment(loader_path),
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ProvenanceError(
            "sealed solver failed CUDA runtime identity qualification "
            f"(status {result.returncode}): {detail}"
        )
    try:
        report = json.loads(result.stdout.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(
            "sealed solver returned malformed CUDA runtime identity JSON"
        ) from error
    if not isinstance(report, dict) or report.get("schema") != CUDA_RUNTIME_INFO_SCHEMA:
        raise ProvenanceError("sealed solver returned an unsupported CUDA runtime schema")

    device_name = report.get("device_name")
    pageable_host_ptr = report.get("pageable_host_ptr")
    pageable_memory_access = report.get("pageable_memory_access")
    pageable_access_uses_host_page_tables = report.get(
        "pageable_access_uses_host_page_tables"
    )
    integrated_device = report.get("integrated_device")
    ordinary_gemm_transport = report.get("ordinary_gemm_transport")
    ordinary_gemm_transport_policy = report.get("ordinary_gemm_transport_policy")
    ordinary_gemm_transport_reason = report.get("ordinary_gemm_transport_reason")
    explicit_device_copy = report.get("explicit_device_copy")
    discrete_mode = report.get("discrete_mode")
    deadline_f64_transport = report.get("deadline_f64_transport")
    nvrtc_status = report.get("nvrtc_status")
    if (
        not isinstance(device_name, str)
        or not device_name
        or not isinstance(pageable_host_ptr, bool)
        or not isinstance(pageable_memory_access, bool)
        or not isinstance(pageable_access_uses_host_page_tables, bool)
        or not (integrated_device is None or type(integrated_device) is bool)
        or ordinary_gemm_transport_policy
        not in {
            "auto",
            "override-direct-host-page-tables",
            "override-unified-memory",
            "override-explicit-device-copy",
            "legacy-discrete-mode-override",
        }
        or ordinary_gemm_transport_reason
        not in {
            "pageable-access-uses-host-page-tables",
            "integrated-device",
            "discrete-device",
            "topology-query-failed-explicit-copy",
            "explicit-transport-override",
            "legacy-discrete-mode-override",
        }
        or ordinary_gemm_transport
        not in {
            "direct-host-page-tables",
            "unified-memory",
            "explicit-device-copy",
        }
        or not isinstance(explicit_device_copy, bool)
        or explicit_device_copy
        != (ordinary_gemm_transport == "explicit-device-copy")
        or not isinstance(discrete_mode, bool)
        or discrete_mode
        != (ordinary_gemm_transport == "explicit-device-copy")
        or deadline_f64_transport
        not in {"direct-host-page-tables", "explicit-device-copy"}
        or pageable_host_ptr
        != (
            pageable_memory_access
            and pageable_access_uses_host_page_tables
        )
        or pageable_access_uses_host_page_tables and not pageable_memory_access
        or deadline_f64_transport
        != (
            "direct-host-page-tables"
            if pageable_host_ptr
            else "explicit-device-copy"
        )
        or not isinstance(nvrtc_status, str)
    ):
        raise ProvenanceError("sealed solver CUDA runtime qualification is incomplete")

    expected_profile = {
        "auto": (
            ("direct-host-page-tables", "pageable-access-uses-host-page-tables")
            if pageable_host_ptr
            else ("unified-memory", "integrated-device")
            if integrated_device is True
            else ("explicit-device-copy", "discrete-device")
            if integrated_device is False
            else (
                "explicit-device-copy",
                "topology-query-failed-explicit-copy",
            )
        ),
        "override-direct-host-page-tables": (
            "direct-host-page-tables",
            "explicit-transport-override",
        ),
        "override-unified-memory": (
            "unified-memory",
            "explicit-transport-override",
        ),
        "override-explicit-device-copy": (
            "explicit-device-copy",
            "explicit-transport-override",
        ),
        "legacy-discrete-mode-override": (
            "explicit-device-copy",
            "legacy-discrete-mode-override",
        ),
    }[ordinary_gemm_transport_policy]
    if (ordinary_gemm_transport, ordinary_gemm_transport_reason) != expected_profile:
        raise ProvenanceError("sealed solver CUDA runtime qualification is incomplete")
    if (
        ordinary_gemm_transport_policy == "override-direct-host-page-tables"
        and not pageable_host_ptr
    ):
        raise ProvenanceError("sealed solver CUDA runtime qualification is incomplete")

    raw_candidates = report.get("candidates")
    if not isinstance(raw_candidates, dict):
        raise ProvenanceError("sealed solver CUDA runtime candidates are missing")
    candidates: dict[str, list[str]] = {}
    for role in ("driver", "cublas", "cublas_lt", "nvrtc"):
        values = raw_candidates.get(role)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str)
                or CUDA_RUNTIME_SAFE_LIBRARY_NAME.fullmatch(value) is None
                for value in values
            )
        ):
            raise ProvenanceError(
                f"sealed solver CUDA runtime {role} candidate list is invalid"
            )
        # cudarc may emit the same soname through more than one base-name
        # expansion. Preserve its first-choice order while materializing one
        # hardlink per distinct loader-visible name.
        candidates[role] = list(dict.fromkeys(values))

    raw_objects = report.get("objects")
    if not isinstance(raw_objects, list):
        raise ProvenanceError("sealed solver CUDA runtime object list is missing")
    objects: list[dict[str, object]] = []
    seen_roles: set[str] = set()
    known_roles = CUDA_RUNTIME_REQUIRED_ROLES | CUDA_RUNTIME_OPTIONAL_ROLES
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            raise ProvenanceError("sealed solver CUDA runtime object is malformed")
        role = raw_object.get("role")
        mapped_path = raw_object.get("mapped_path")
        resolved_path = raw_object.get("resolved_path")
        mapped_device_major = raw_object.get("mapped_device_major")
        mapped_device_minor = raw_object.get("mapped_device_minor")
        mapped_inode = raw_object.get("mapped_inode")
        provider_symbol = raw_object.get("provider_symbol")
        size_bytes = raw_object.get("size_bytes")
        sha256 = raw_object.get("sha256")
        if (
            not isinstance(role, str)
            or role not in known_roles
            or role in seen_roles
            or not isinstance(provider_symbol, str)
            or not provider_symbol
            or not isinstance(mapped_path, str)
            or not Path(mapped_path).is_absolute()
            or not isinstance(resolved_path, str)
            or not Path(resolved_path).is_absolute()
            or not isinstance(mapped_device_major, int)
            or isinstance(mapped_device_major, bool)
            or mapped_device_major < 0
            or not isinstance(mapped_device_minor, int)
            or isinstance(mapped_device_minor, bool)
            or mapped_device_minor < 0
            or not isinstance(mapped_inode, int)
            or isinstance(mapped_inode, bool)
            or mapped_inode <= 0
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise ProvenanceError("sealed solver CUDA runtime object identity is invalid")
        fingerprint = _normalized_runtime_fingerprint(
            raw_object.get("fingerprint"), role=role
        )
        if (
            fingerprint["inode"] != mapped_inode
            or fingerprint["size_bytes"] != size_bytes
            or os.major(fingerprint["device"]) != mapped_device_major
            or os.minor(fingerprint["device"]) != mapped_device_minor
        ):
            raise ProvenanceError(
                "sealed solver CUDA runtime mapped identity contradicts its "
                f"in-process file fingerprint for {role}"
            )
        seen_roles.add(role)
        objects.append(
            {
                "role": role,
                "provider_symbol": provider_symbol,
                "mapped_path": mapped_path,
                "resolved_path": resolved_path,
                "mapped_device_major": mapped_device_major,
                "mapped_device_minor": mapped_device_minor,
                "mapped_inode": mapped_inode,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "fingerprint": fingerprint,
            }
        )
    missing_roles = sorted(CUDA_RUNTIME_REQUIRED_ROLES - seen_roles)
    if missing_roles:
        raise ProvenanceError(
            "sealed solver CUDA runtime is missing required mapped objects: "
            + ", ".join(missing_roles)
        )

    has_nvrtc = "nvrtc" in seen_roles
    has_nvrtc_builtins = "nvrtc_builtins" in seen_roles
    expected_nvrtc_status = {
        (False, False): "not_loaded_feature_disabled",
        (True, False): "loaded",
        (True, True): "loaded_with_builtins",
        (False, True): "builtins_loaded_without_nvrtc",
    }[(has_nvrtc, has_nvrtc_builtins)]
    if nvrtc_status != expected_nvrtc_status:
        raise ProvenanceError(
            "sealed solver CUDA runtime NVRTC status contradicts mapped objects"
        )

    objects.sort(key=lambda item: str(item["role"]))
    return {
        "schema": CUDA_RUNTIME_INFO_SCHEMA,
        "device_name": device_name,
        "pageable_host_ptr": pageable_host_ptr,
        "pageable_memory_access": pageable_memory_access,
        "pageable_access_uses_host_page_tables": pageable_access_uses_host_page_tables,
        "integrated_device": integrated_device,
        "ordinary_gemm_transport": ordinary_gemm_transport,
        "ordinary_gemm_transport_policy": ordinary_gemm_transport_policy,
        "ordinary_gemm_transport_reason": ordinary_gemm_transport_reason,
        "explicit_device_copy": explicit_device_copy,
        "discrete_mode": discrete_mode,
        "deadline_f64_transport": deadline_f64_transport,
        "candidates": candidates,
        "objects": objects,
        "nvrtc_status": nvrtc_status,
    }


def _capture_cuda_runtime_identity(
    binary: Path, *, loader_path: str | Path | None = None
) -> dict[str, object]:
    first_probe = _cuda_runtime_probe(binary, loader_path=loader_path)
    captured_objects: list[dict[str, object]] = []
    selected_files: set[tuple[int, int]] = set()
    for probed in first_probe["objects"]:
        if not isinstance(probed, dict):
            raise ProvenanceError("sealed solver CUDA runtime object is malformed")
        resolved_value = probed.get("resolved_path")
        if not isinstance(resolved_value, str):
            raise ProvenanceError("sealed solver CUDA runtime path is missing")
        path = Path(resolved_value)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ProvenanceError(
                f"cannot resolve mapped CUDA runtime object: {path}"
            ) from error
        if resolved != path or not resolved.is_file():
            raise ProvenanceError(
                "mapped CUDA runtime resolved path must name a canonical regular "
                f"file: {path}"
            )
        digest, fingerprint = _stable_file_hash(resolved)
        if (
            fingerprint["inode"] != probed.get("mapped_inode")
            or os.major(fingerprint["device"]) != probed.get("mapped_device_major")
            or os.minor(fingerprint["device"]) != probed.get("mapped_device_minor")
            or fingerprint != probed.get("fingerprint")
            or fingerprint["size_bytes"] != probed.get("size_bytes")
            or digest != probed.get("sha256")
        ):
            raise ProvenanceError(
                "mapped CUDA runtime file identity or in-process hash changed "
                "before it could be independently verified: "
                f"{resolved}"
            )
        file_key = (fingerprint["device"], fingerprint["inode"])
        if file_key in selected_files:
            raise ProvenanceError(
                "distinct CUDA runtime roles resolved to the same file object"
            )
        selected_files.add(file_key)
        captured_objects.append(
            {
                **probed,
                "size_bytes": fingerprint["size_bytes"],
                "sha256": digest,
                "fingerprint": fingerprint,
            }
        )

    second_probe = _cuda_runtime_probe(binary, loader_path=loader_path)
    if second_probe != first_probe:
        raise ProvenanceError(
            "CUDA runtime object selection changed while provenance was captured"
        )
    return {
        "schema": MEASUREMENT_CUDA_RUNTIME_SCHEMA,
        "status": "captured",
        "probe": first_probe,
        "objects": captured_objects,
    }


def _capture_cuda_runtime_dependency(
    binary: Path,
    environment: dict[str, object],
    build_features: list[str],
) -> dict[str, object]:
    if _cuda_runtime_required(environment, build_features):
        values = environment.get("values")
        source_loader = values.get("LD_LIBRARY_PATH") if isinstance(values, dict) else None
        if source_loader is not None and not isinstance(source_loader, str):
            raise ProvenanceError("captured CUDA loader path is invalid")
        return _capture_cuda_runtime_identity(
            binary,
            loader_path=source_loader or None,
        )
    return _not_required_cuda_runtime_identity(environment, build_features)


def _runtime_object_by_role(
    identity: dict[str, object],
) -> dict[str, dict[str, object]]:
    objects = identity.get("objects")
    if not isinstance(objects, list):
        raise ProvenanceError("captured CUDA runtime object list is invalid")
    result: dict[str, dict[str, object]] = {}
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ProvenanceError("captured CUDA runtime object is invalid")
        role = str(item["role"])
        if role in result:
            raise ProvenanceError(f"duplicate captured CUDA runtime role: {role}")
        result[role] = item
    return result


def _seal_cuda_runtime(
    *,
    binary: Path,
    source_identity: dict[str, object],
    run_dir: Path,
) -> dict[str, object]:
    if (
        source_identity.get("schema") != MEASUREMENT_CUDA_RUNTIME_SCHEMA
        or source_identity.get("status") != "captured"
    ):
        raise ProvenanceError("source CUDA runtime identity is not sealable")
    source_objects = _runtime_object_by_role(source_identity)
    source_probe = source_identity.get("probe")
    candidates = (
        source_probe.get("candidates") if isinstance(source_probe, dict) else None
    )
    if not isinstance(candidates, dict):
        raise ProvenanceError("captured CUDA runtime candidates are missing")

    identity_digest = _sha256(_json_bytes(source_identity))
    runtime_parent = run_dir / "sealed" / "cuda-runtime"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = runtime_parent / identity_digest
    try:
        runtime_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ProvenanceError(
            f"refusing to reuse a CUDA runtime seal directory: {runtime_dir}"
        ) from error

    entry_roles: dict[str, str] = {}
    role_primary: dict[str, Path] = {}
    role_digest: dict[str, str] = {}
    for role in sorted(source_objects):
        source_object = source_objects[role]
        source_value = source_object.get("resolved_path")
        expected_digest = source_object.get("sha256")
        expected_fingerprint = source_object.get("fingerprint")
        if (
            not isinstance(source_value, str)
            or not isinstance(expected_digest, str)
            or not isinstance(expected_fingerprint, dict)
        ):
            raise ProvenanceError(
                f"captured CUDA runtime {role} identity is incomplete"
            )
        primary_name = Path(source_value).name
        if CUDA_RUNTIME_SAFE_LIBRARY_NAME.fullmatch(primary_name) is None:
            raise ProvenanceError(
                f"captured CUDA runtime filename is unsafe: {primary_name!r}"
            )
        role_names = [primary_name]
        role_candidates = candidates.get(role, [])
        if role != "nvrtc_builtins" and (
            not isinstance(role_candidates, list) or not role_candidates
        ):
            raise ProvenanceError(
                f"captured CUDA runtime candidate list is missing for {role}"
            )
        role_names.extend(str(value) for value in role_candidates)
        for name in role_names:
            if CUDA_RUNTIME_SAFE_LIBRARY_NAME.fullmatch(name) is None:
                raise ProvenanceError(
                    f"captured CUDA runtime candidate is unsafe: {name!r}"
                )
            owner = entry_roles.get(name)
            if owner is not None and owner != role:
                raise ProvenanceError(
                    f"CUDA runtime candidate {name!r} is shared by roles "
                    f"{owner} and {role}"
                )
            entry_roles[name] = role

        primary = runtime_dir / primary_name
        _seal_file(
            Path(source_value),
            primary,
            executable=False,
            expected_sha256=expected_digest,
            expected_fingerprint=expected_fingerprint,
        )
        role_primary[role] = primary
        role_digest[role] = expected_digest

    for name, role in sorted(entry_roles.items()):
        destination = runtime_dir / name
        primary = role_primary[role]
        if destination == primary:
            continue
        try:
            os.link(primary, destination, follow_symlinks=False)
        except OSError as error:
            raise ProvenanceError(
                f"cannot create sealed CUDA runtime alias {destination}: {error}"
            ) from error

    runtime_dir.chmod(0o555)
    sealed_capture = _capture_cuda_runtime_identity(binary, loader_path=runtime_dir)

    entries: list[dict[str, object]] = []
    for name, role in sorted(entry_roles.items()):
        path = runtime_dir / name
        fingerprint = _file_fingerprint(path)
        entries.append(
            {
                "name": name,
                "path": str(path),
                "role": role,
                "size_bytes": fingerprint["size_bytes"],
                "sha256": role_digest[role],
                "fingerprint": fingerprint,
                "mode": "read_only_hardlink",
            }
        )
    sealed_execution = {
        "schema": SEALED_CUDA_RUNTIME_SCHEMA,
        "path": str(runtime_dir.resolve()),
        "source_identity_sha256": identity_digest,
        "fingerprint": _file_fingerprint(runtime_dir),
        "mode": "directory_read_only",
        "entries": entries,
    }
    sealed_identity = {
        "schema": MEASUREMENT_CUDA_RUNTIME_SCHEMA,
        "status": "captured",
        "source_capture": source_identity,
        "sealed_execution": sealed_execution,
        "probe": sealed_capture["probe"],
        "objects": sealed_capture["objects"],
    }
    _validate_sealed_cuda_runtime_capture(sealed_identity)
    return sealed_identity


def _capture_and_seal_cuda_runtime_dependency(
    binary: Path,
    environment: dict[str, object],
    build_features: list[str],
    run_dir: Path,
) -> dict[str, object]:
    source_identity = _capture_cuda_runtime_dependency(
        binary, environment, build_features
    )
    if source_identity.get("status") == "not_required":
        return source_identity
    return _seal_cuda_runtime(
        binary=binary,
        source_identity=source_identity,
        run_dir=run_dir,
    )


def _sealed_cuda_runtime_entry_map(
    expected: dict[str, object],
) -> tuple[Path, dict[str, dict[str, object]]]:
    sealed_execution = expected.get("sealed_execution")
    if (
        not isinstance(sealed_execution, dict)
        or sealed_execution.get("schema") != SEALED_CUDA_RUNTIME_SCHEMA
    ):
        raise ProvenanceError("sealed CUDA runtime execution identity is invalid")
    path_value = sealed_execution.get("path")
    entries = sealed_execution.get("entries")
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(entries, list)
        or not entries
    ):
        raise ProvenanceError("sealed CUDA runtime directory identity is incomplete")
    runtime_dir = Path(path_value)
    entry_map: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProvenanceError("sealed CUDA runtime entry identity is invalid")
        name = entry.get("name")
        entry_path = entry.get("path")
        role = entry.get("role")
        digest = entry.get("sha256")
        fingerprint = entry.get("fingerprint")
        if (
            not isinstance(name, str)
            or CUDA_RUNTIME_SAFE_LIBRARY_NAME.fullmatch(name) is None
            or name in entry_map
            or entry_path != str(runtime_dir / name)
            or not isinstance(role, str)
            or role
            not in CUDA_RUNTIME_REQUIRED_ROLES | CUDA_RUNTIME_OPTIONAL_ROLES
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(fingerprint, dict)
        ):
            raise ProvenanceError("sealed CUDA runtime entry identity is malformed")
        entry_map[name] = entry
    return runtime_dir, entry_map


def _validate_sealed_cuda_runtime_capture(expected: dict[str, object]) -> None:
    runtime_dir, entry_map = _sealed_cuda_runtime_entry_map(expected)
    probe = expected.get("probe")
    objects = expected.get("objects")
    candidates = probe.get("candidates") if isinstance(probe, dict) else None
    source_capture = expected.get("source_capture")
    source_objects = (
        source_capture.get("objects") if isinstance(source_capture, dict) else None
    )
    if (
        not isinstance(objects, list)
        or not isinstance(candidates, dict)
        or not isinstance(source_objects, list)
    ):
        raise ProvenanceError("sealed CUDA runtime capture is incomplete")
    seen_roles: set[str] = set()
    mapped_names: dict[str, set[str]] = {}
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ProvenanceError("sealed CUDA runtime mapped object is invalid")
        role = str(item["role"])
        if role in seen_roles:
            raise ProvenanceError(
                f"sealed CUDA runtime mapped role is duplicated: {role}"
            )
        seen_roles.add(role)
        mapped_names[role] = set()
        for field in ("mapped_path", "resolved_path"):
            value = item.get(field)
            if (
                not isinstance(value, str)
                or Path(value).parent != runtime_dir
                or Path(value).name not in entry_map
                or entry_map[Path(value).name].get("role") != role
            ):
                raise ProvenanceError(
                    f"sealed CUDA runtime {role} mapping escaped its run directory"
                )
            mapped_names[role].add(Path(value).name)
        resolved_entry = entry_map[Path(str(item["resolved_path"])).name]
        if (
            resolved_entry.get("sha256") != item.get("sha256")
            or resolved_entry.get("fingerprint") != item.get("fingerprint")
        ):
            raise ProvenanceError(
                f"sealed CUDA runtime {role} mapping differs from its sealed file"
            )
    if CUDA_RUNTIME_REQUIRED_ROLES - seen_roles:
        raise ProvenanceError("sealed CUDA runtime capture lost a required role")
    source_names: dict[str, set[str]] = {}
    for item in source_objects:
        if not isinstance(item, dict):
            raise ProvenanceError("source CUDA runtime object identity is invalid")
        role = item.get("role")
        resolved_path = item.get("resolved_path")
        if (
            not isinstance(role, str)
            or role not in seen_roles
            or not isinstance(resolved_path, str)
        ):
            raise ProvenanceError("source CUDA runtime object identity is incomplete")
        source_names.setdefault(role, set()).add(Path(resolved_path).name)
    allowed_names = {
        role: mapped_names.get(role, set()) | source_names.get(role, set())
        for role in seen_roles
    }
    for role, names in candidates.items():
        if role not in CUDA_RUNTIME_REQUIRED_ROLES | {"nvrtc"}:
            raise ProvenanceError(f"sealed CUDA runtime has unknown role {role!r}")
        if not isinstance(names, list):
            raise ProvenanceError(
                f"sealed CUDA runtime candidate list is invalid for {role}"
            )
        if role not in seen_roles:
            continue
        for name in names:
            if (
                not isinstance(name, str)
                or name not in entry_map
                or entry_map[name].get("role") != role
            ):
                raise ProvenanceError(
                    f"sealed CUDA runtime is missing candidate alias {name!r}"
                )
            allowed_names[role].add(name)
    observed_entry_names = {
        role: {
            name for name, entry in entry_map.items() if entry.get("role") == role
        }
        for role in seen_roles
    }
    if observed_entry_names != allowed_names:
        raise ProvenanceError(
            "sealed CUDA runtime aliases differ from the qualified candidate namespace"
        )


def _validate_sealed_cuda_runtime(
    expected: object, *, hash_files: bool
) -> Path | None:
    _validate_loader_preload_configuration()
    if not isinstance(expected, dict):
        raise ProvenanceError("start manifest CUDA runtime identity is invalid")
    if expected.get("schema") != MEASUREMENT_CUDA_RUNTIME_SCHEMA:
        raise ProvenanceError("start manifest CUDA runtime schema is invalid")
    status = expected.get("status")
    if status == "not_required":
        if expected.get("reason") not in {
            "cuda_build_feature_not_declared",
            "noncuda_measurement_explicitly_allowed",
        }:
            raise ProvenanceError("CUDA runtime omission reason is invalid")
        return None
    if status != "captured":
        raise ProvenanceError("start manifest CUDA runtime status is invalid")
    _validate_sealed_cuda_runtime_capture(expected)
    runtime_dir, entry_map = _sealed_cuda_runtime_entry_map(expected)
    sealed_execution = expected["sealed_execution"]
    try:
        resolved = runtime_dir.resolve(strict=True)
        directory_lstat = runtime_dir.lstat()
    except OSError as error:
        raise ProvenanceError(
            f"sealed CUDA runtime directory is unavailable: {runtime_dir}: {error}"
        ) from error
    if (
        resolved != runtime_dir
        or not stat.S_ISDIR(directory_lstat.st_mode)
        or stat.S_ISLNK(directory_lstat.st_mode)
        or directory_lstat.st_mode & 0o222
    ):
        raise ProvenanceError(
            f"sealed CUDA runtime directory is unsafe: {runtime_dir}"
        )
    try:
        actual_names = {
            entry.name
            for entry in os.scandir(runtime_dir)
        }
    except OSError as error:
        raise ProvenanceError(
            f"cannot enumerate sealed CUDA runtime directory: {error}"
        ) from error
    if actual_names != set(entry_map):
        raise ProvenanceError(
            "sealed CUDA runtime directory contents changed after start capture"
        )
    expected_directory_fingerprint = sealed_execution.get("fingerprint")
    if (
        not isinstance(expected_directory_fingerprint, dict)
        or _file_fingerprint(runtime_dir) != expected_directory_fingerprint
    ):
        raise ProvenanceError(
            "sealed CUDA runtime directory fingerprint changed after start capture"
        )

    hashed: dict[tuple[int, int], str] = {}
    role_files: dict[str, tuple[int, int, str]] = {}
    for name, entry in entry_map.items():
        path = runtime_dir / name
        try:
            file_lstat = path.lstat()
            fingerprint = _file_fingerprint(path)
        except OSError as error:
            raise ProvenanceError(
                f"sealed CUDA runtime entry is unavailable: {path}: {error}"
            ) from error
        if (
            not stat.S_ISREG(file_lstat.st_mode)
            or stat.S_ISLNK(file_lstat.st_mode)
            or file_lstat.st_mode & 0o222
            or fingerprint != entry.get("fingerprint")
            or fingerprint["size_bytes"] != entry.get("size_bytes")
        ):
            raise ProvenanceError(
                f"sealed CUDA runtime entry changed after start capture: {path}"
            )
        file_key = (fingerprint["device"], fingerprint["inode"])
        role = str(entry["role"])
        digest = str(entry["sha256"])
        previous = role_files.get(role)
        if previous is None:
            role_files[role] = (file_key[0], file_key[1], digest)
        elif previous != (file_key[0], file_key[1], digest):
            raise ProvenanceError(
                f"sealed CUDA runtime aliases diverged for role {role}"
            )
        if hash_files:
            observed_digest = hashed.get(file_key)
            if observed_digest is None:
                observed_digest, observed_fingerprint = _stable_file_hash(path)
                if observed_fingerprint != fingerprint:
                    raise ProvenanceError(
                        f"sealed CUDA runtime entry changed while hashing: {path}"
                    )
                hashed[file_key] = observed_digest
            if observed_digest != digest:
                raise ProvenanceError(
                    f"sealed CUDA runtime entry hash changed after start: {path}"
                )
    if len({(device, inode) for device, inode, _digest in role_files.values()}) != len(
        role_files
    ):
        raise ProvenanceError(
            "distinct sealed CUDA runtime roles share one file object"
        )
    return runtime_dir


def _cuda_runtime_from_start(start: dict[str, object]) -> object:
    dependencies = start.get("dependencies")
    if not isinstance(dependencies, dict) or "cuda_runtime" not in dependencies:
        raise ProvenanceError("start manifest CUDA runtime dependency is missing")
    return dependencies["cuda_runtime"]


def _recapture_cuda_runtime_from_start(start: dict[str, object]) -> dict[str, object]:
    expected = _cuda_runtime_from_start(start)
    runtime_dir = _validate_sealed_cuda_runtime(expected, hash_files=True)
    if not isinstance(expected, dict):
        raise ProvenanceError("start manifest CUDA runtime identity is invalid")
    if runtime_dir is None:
        return dict(expected)
    solver = start.get("solver_binary")
    sealed = solver.get("sealed_execution") if isinstance(solver, dict) else None
    path_value = sealed.get("path") if isinstance(sealed, dict) else None
    if not isinstance(path_value, str):
        raise ProvenanceError(
            "start manifest cannot reconstruct CUDA runtime capture inputs"
        )
    observed = _capture_cuda_runtime_identity(
        Path(path_value), loader_path=runtime_dir
    )
    recaptured = {
        "schema": MEASUREMENT_CUDA_RUNTIME_SCHEMA,
        "status": "captured",
        "source_capture": expected.get("source_capture"),
        "sealed_execution": expected.get("sealed_execution"),
        "probe": observed["probe"],
        "objects": observed["objects"],
    }
    _validate_sealed_cuda_runtime_capture(recaptured)
    return recaptured


def _resolve_from(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_mutation_root(path: Path, repo_root: Path, label: str) -> None:
    filesystem_root = Path(path.anchor)
    if path == filesystem_root or path == repo_root or path in repo_root.parents:
        raise ProvenanceError(f"unsafe broad {label}: {path}")
    git_dir = (repo_root / ".git").resolve()
    if path == git_dir or _is_within(path, git_dir):
        raise ProvenanceError(f"{label} must not be inside the NY Git metadata: {path}")
    if path.exists() and not path.is_dir():
        raise ProvenanceError(f"{label} exists but is not a directory: {path}")
    if _is_within(path, repo_root):
        relative = path.relative_to(repo_root)
        tracked = _git_evidence(repo_root, "ls-files", "-z", "--", str(relative))
        if tracked:
            raise ProvenanceError(
                f"{label} contains tracked NY paths and would invalidate its own "
                f"measurement: {path}"
            )
        ignored = _git_evidence_result(
            repo_root,
            "check-ignore",
            "-q",
            "--no-index",
            "--",
            str(relative),
            check=False,
        )
        if ignored.returncode != 0:
            raise ProvenanceError(
                f"{label} inside the NY worktree must be Git-ignored: {path}"
            )


def capture_start_manifest(
    *,
    repo_root: Path,
    binary: Path,
    benchmark_root: Path,
    artifact_root: Path,
    run_id: str,
    output_dir: Path,
    scratch_dir: Path,
    result_file: Path,
    solver_log_file: Path,
    categories_raw: str,
    timeout_cap_seconds: int,
    watchdog_grace_seconds: int,
    max_rows_per_category: int,
    instance_index: int,
    vnnlib_version: str,
    sweep_script: Path,
    configs_dir: Path | None = None,
) -> Path:
    """Capture and immutably create one run's start manifest."""
    if SAFE_COMPONENT.fullmatch(run_id) is None:
        raise ValueError(f"unsafe run ID: {run_id!r}")
    repo_root = repo_root.resolve()
    # Bind the exact Git executable before the first repository-dependent
    # safety check.  Every plumbing call below uses the same configured path.
    git_executable = _capture_git_executable(repo_root)
    binary = _resolve_from(repo_root, binary)
    benchmark_root = _resolve_from(repo_root, benchmark_root)
    artifact_root = _resolve_from(repo_root, artifact_root)
    output_dir = _resolve_from(repo_root, output_dir)
    scratch_dir = _resolve_from(repo_root, scratch_dir)
    result_file = _resolve_from(repo_root, result_file)
    solver_log_file = _resolve_from(repo_root, solver_log_file)
    sweep_script = _resolve_from(repo_root, sweep_script)
    if not binary.is_file():
        raise ProvenanceError(f"solver binary does not exist: {binary}")
    if not benchmark_root.is_dir():
        raise ProvenanceError(f"benchmark root does not exist: {benchmark_root}")
    if not sweep_script.is_file():
        raise ProvenanceError(f"sweep script does not exist: {sweep_script}")
    # 0 means UNCAPPED — every instance gets its official per-instance budget from
    # field 3 of instances.csv. That is now the default in
    # scripts/measure_ny_scorecard.sh, because a default cap silently manufactured
    # capability limits (#measure-cap-truncation): at the old 120s default, nn4sys
    # rows budgeted 300-800s and cgan rows budgeted 900-1200s were measured at ~110s
    # and banked as `timeout`. Rejecting 0 here would make every uncapped sealed run
    # fail closed before execution, which is how a parity fix broke sealing once
    # before (repaired in 9ade1779).
    if timeout_cap_seconds < 0 or watchdog_grace_seconds < 0:
        raise ValueError("timeout cap and watchdog grace must be nonnegative")
    if max_rows_per_category < 0:
        raise ValueError("maximum rows per category must be nonnegative")
    if instance_index < 0:
        raise ValueError("instance index must be nonnegative")
    if vnnlib_version not in {"", "1.0", "2.0"}:
        raise ValueError("VNN-LIB version selection must be empty, 1.0, or 2.0")
    git_bound_path = str(git_executable["resolved_path"])
    with _bound_git_executable(git_bound_path):
        _validate_mutation_root(
            output_dir, repo_root, "measurement output directory"
        )
        _validate_mutation_root(
            artifact_root, repo_root, "measurement artifact root"
        )
        _validate_mutation_root(
            scratch_dir, repo_root, "measurement scratch directory"
        )
    if not _is_within(result_file, scratch_dir):
        raise ProvenanceError(
            "result scratch file must be inside the scratch directory"
        )
    if not _is_within(solver_log_file, scratch_dir):
        raise ProvenanceError("solver log file must be inside the scratch directory")
    if result_file == solver_log_file:
        raise ProvenanceError("result and solver-log scratch files must be distinct")

    # Validate containment before creating or sealing any run artifact. The
    # scorecard shell performs an independent Bash-native gate; this recapture
    # binds the kernel-observed identity and effective controls into evidence.
    containment = _capture_measurement_containment()
    start_path = artifact_root / "runs" / run_id / "start.json"
    run_dir = start_path.parent
    config_inputs = (
        _capture_config_inputs(configs_dir) if configs_dir is not None else None
    )
    environment = _capture_environment()
    build_features_raw, build_features = _declared_build_features()
    ay_dependency: dict[str, object] = dict(_parse_ay_pin(repo_root))
    ay_dependency["executable"] = _capture_ay_executable(
        repo_root,
        expected_revision=str(ay_dependency["git_revision"]),
    )
    binary_digest, binary_fingerprint = _stable_file_hash(binary)
    build_coherence = _capture_build_coherence(repo_root, binary)
    sealed_binary = _seal_file(
        binary,
        run_dir / "sealed" / "solver" / binary_digest / binary.name,
        executable=True,
        expected_sha256=binary_digest,
        expected_fingerprint=binary_fingerprint,
    )
    ay_executable = ay_dependency["executable"]
    if isinstance(ay_executable, dict):
        ay_path = ay_executable.get("resolved_path")
        ay_digest = ay_executable.get("sha256")
        ay_fingerprint = ay_executable.get("fingerprint")
        if (
            not isinstance(ay_path, str)
            or not isinstance(ay_digest, str)
            or not isinstance(ay_fingerprint, dict)
        ):
            raise ProvenanceError("captured AY executable identity is incomplete")
        ay_dependency["sealed_executable"] = _seal_file(
            Path(ay_path),
            run_dir / "sealed" / "ay" / ay_digest / Path(ay_path).name,
            executable=True,
            expected_sha256=ay_digest,
            expected_fingerprint=ay_fingerprint,
        )
    else:
        ay_dependency["sealed_executable"] = None
    sealed_config_inputs = (
        _seal_config_inputs(config_inputs, run_dir)
        if config_inputs is not None
        else None
    )
    sealed_binary_path = str(sealed_binary["path"])
    binary_version = _run(
        [sealed_binary_path, "--version"],
        check=False,
        timeout=15,
        env=_cuda_probe_environment(None),
    )
    cuda_runtime_dependency = _capture_and_seal_cuda_runtime_dependency(
        Path(sealed_binary_path),
        environment,
        build_features,
        run_dir,
    )
    if _file_fingerprint(binary) != binary_fingerprint:
        raise ProvenanceError(
            f"solver binary changed during provenance capture: {binary}"
        )
    categories = categories_raw.split()
    if not categories:
        raise ValueError("measurement category list is empty")
    solver_command_template = [
        sealed_binary_path,
        "vnncomp",
        "v1",
        "<category>",
        "<onnx>",
        "<vnnlib>",
        str(result_file),
        "<capped_timeout_seconds>",
    ]
    if sealed_config_inputs is not None:
        solver_command_template.extend(
            ["--configs-dir", str(sealed_config_inputs["resolved_path"])]
        )
    solver_environment_overrides = {
        "PATH": "/usr/bin:/bin",
        "RUST_LOG": "error",
    }
    if cuda_runtime_dependency.get("status") == "captured":
        solver_environment_overrides["LD_LIBRARY_PATH"] = str(
            cuda_runtime_dependency["sealed_execution"]["path"]
        )
    sealed_ay_executable = ay_dependency.get("sealed_executable")
    if isinstance(sealed_ay_executable, dict):
        sealed_ay_path = sealed_ay_executable.get("path")
        if not isinstance(sealed_ay_path, str):
            raise ProvenanceError("sealed AY execution path is invalid")
        solver_environment_overrides["NY_AY"] = sealed_ay_path
    if (
        environment.get("values", {}).get("NY_ALLOW_NONCUDA_MEASURE") == "1"
    ):
        solver_environment_overrides["NY_NO_CUDA"] = "1"
    captured_environment_values = environment.get("values")
    if not isinstance(captured_environment_values, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in captured_environment_values.items()
    ):
        raise ProvenanceError("captured measurement environment is invalid")
    solver_environment_values = {
        key: value
        for key, value in captured_environment_values.items()
        if key not in SOLVER_ENVIRONMENT_EXCLUDED_KEYS
    }
    solver_environment_unsets = sorted(
        key
        for key in captured_environment_values
        if key in SOLVER_ENVIRONMENT_EXCLUDED_KEYS
    )
    if cuda_runtime_dependency.get("status") != "captured":
        if "LD_LIBRARY_PATH" in solver_environment_values:
            solver_environment_values.pop("LD_LIBRARY_PATH")
            solver_environment_unsets.append("LD_LIBRARY_PATH")
    solver_environment_values.update(solver_environment_overrides)
    solver_environment_unsets = sorted(set(solver_environment_unsets))
    with _bound_git_executable(git_bound_path):
        if _capture_git_executable(repo_root) != git_executable:
            raise ProvenanceError("Git executable changed before worktree capture")
        ny_worktree = _capture_worktree(repo_root)
        rust_toolchain = _parse_toolchain(repo_root)
        benchmark_identity = _capture_benchmark(benchmark_root)
        if _capture_git_executable(repo_root) != git_executable:
            raise ProvenanceError("Git executable changed during provenance capture")
    payload = {
        "schema": "ny_measurement_start_v1",
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "ny": ny_worktree,
        "solver_binary": {
            "path": str(binary),
            "size_bytes": binary_fingerprint["size_bytes"],
            "sha256": binary_digest,
            "fingerprint": binary_fingerprint,
            "sealed_execution": sealed_binary,
            "version_returncode": binary_version.returncode,
            "version_stdout": binary_version.stdout.decode("utf-8", "replace").strip(),
            "version_stderr": binary_version.stderr.decode("utf-8", "replace").strip(),
            "declared_build_features": build_features,
            "declared_build_features_raw": build_features_raw,
            "build_coherence": build_coherence,
        },
        "dependencies": {
            "ay": ay_dependency,
            "cuda_runtime": cuda_runtime_dependency,
        },
        "provenance_tools": {
            "git": git_executable,
        },
        "rust_toolchain": rust_toolchain,
        "benchmark": benchmark_identity,
        "measurement": {
            "sweep_invocation": [str(sweep_script)],
            "categories_raw": categories_raw,
            "categories": categories,
            "timeout_cap_seconds": timeout_cap_seconds,
            "watchdog_grace_seconds": watchdog_grace_seconds,
            "max_rows_per_category": (
                max_rows_per_category if max_rows_per_category > 0 else None
            ),
            "instance_index": instance_index if instance_index > 0 else None,
            "vnnlib_version_selection": vnnlib_version or None,
            "config_inputs": config_inputs,
            "sealed_config_inputs": sealed_config_inputs,
            "benchmark_root": str(benchmark_root),
            "output_dir": str(output_dir),
            "artifact_root": str(artifact_root),
            "scratch_dir": str(scratch_dir),
            "result_file": str(result_file),
            "solver_log_file": str(solver_log_file),
            "flight_record_file": f"{result_file}.flight.json",
            "flight_record_capture": (
                "validated-structured-row-metadata-or-explicit-missing-v1"
            ),
            "solver_output_capture": "combined_stdout_stderr_exact_bytes",
            "csv_columns": [
                "category",
                "onnx",
                "vnnlib",
                "prepare_seconds",
                "result",
                "runtime_seconds",
                "run_id",
            ],
            "solver_command_template": solver_command_template,
            "solver_environment": {
                "mode": "env-i-reviewed-record-v1",
                "values": solver_environment_values,
            },
            "solver_environment_overrides": solver_environment_overrides,
            "solver_environment_unsets": solver_environment_unsets,
        },
        "environment": environment,
        "host": {
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "os_release": _read_os_release(),
            "cpu": _cpu_identity(),
            "memory": _memory_identity(),
            "gpu": _gpu_identity(),
            "resource_limits": _resource_limits(),
            "containment": containment,
        },
        "host_state": _capture_host_state(),
    }
    _write_immutable(start_path, _json_bytes(payload))
    return start_path


def _add_integrity_violation(
    violations: list[dict[str, str]],
    code: str,
    detail: str,
) -> None:
    violations.append({"code": code, "detail": detail})


def _identity_sha256(identity: object) -> str:
    return _sha256(_json_bytes(identity))


def _validate_solver_binary(
    start: dict[str, object],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    expected = start.get("solver_binary")
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            "solver_binary_start_identity_invalid",
            "start manifest solver-binary identity is missing or invalid",
        )
        return check
    path_value = expected.get("path")
    expected_digest = expected.get("sha256")
    expected_fingerprint = expected.get("fingerprint")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_digest, str)
        or not isinstance(expected_fingerprint, dict)
    ):
        _add_integrity_violation(
            violations,
            "solver_binary_start_identity_invalid",
            "start manifest solver-binary identity is incomplete",
        )
        return check
    check.update(
        {
            "path": path_value,
            "expected_sha256": expected_digest,
            "expected_fingerprint": expected_fingerprint,
        }
    )
    try:
        path = Path(path_value)
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ProvenanceError(
                f"solver binary is not an executable file: {resolved}"
            )
        observed_digest, observed_fingerprint = _stable_file_hash(resolved)
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            "solver_binary_unavailable",
            str(error),
        )
        return check
    check.update(
        {
            "resolved_path": str(resolved),
            "observed_sha256": observed_digest,
            "observed_fingerprint": observed_fingerprint,
        }
    )
    if str(resolved) != path_value:
        _add_integrity_violation(
            violations,
            "solver_binary_resolved_path_mismatch",
            f"expected {path_value}, observed {resolved}",
        )
    if observed_digest != expected_digest:
        _add_integrity_violation(
            violations,
            "solver_binary_sha256_mismatch",
            f"expected {expected_digest}, observed {observed_digest}",
        )
    if observed_fingerprint != expected_fingerprint:
        _add_integrity_violation(
            violations,
            "solver_binary_fingerprint_mismatch",
            "solver-binary stat fingerprint changed after start capture",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_ay_executable(
    start: dict[str, object],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    dependencies = start.get("dependencies")
    ay = dependencies.get("ay") if isinstance(dependencies, dict) else None
    if not isinstance(ay, dict) or "executable" not in ay:
        _add_integrity_violation(
            violations,
            "ay_executable_start_identity_invalid",
            "start manifest AY executable identity is missing",
        )
        return check
    expected = ay["executable"]
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            "ay_executable_start_identity_invalid",
            "start manifest AY executable identity is invalid",
        )
        return check
    declared_path = expected.get("declared_path")
    ny = start.get("ny")
    repo_root = ny.get("repo_root") if isinstance(ny, dict) else None
    git_revision = ay.get("git_revision")
    if (
        not isinstance(declared_path, str)
        or not isinstance(repo_root, str)
        or not isinstance(git_revision, str)
    ):
        _add_integrity_violation(
            violations,
            "ay_executable_start_identity_invalid",
            "start manifest AY executable path, NY repository root, or revision is invalid",
        )
        return check
    check["expected_identity_sha256"] = _identity_sha256(expected)
    try:
        observed = _capture_ay_executable(
            Path(repo_root),
            expected_revision=git_revision,
            declared_path=declared_path,
        )
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            "ay_executable_unavailable",
            str(error),
        )
        return check
    if observed is None:
        _add_integrity_violation(
            violations,
            "ay_executable_unavailable",
            "start manifest AY executable path unexpectedly resolved to no executable",
        )
        return check
    check["observed_identity_sha256"] = _identity_sha256(observed)
    check["resolved_path"] = observed["resolved_path"]
    if observed != expected:
        _add_integrity_violation(
            violations,
            "ay_executable_identity_mismatch",
            "resolved AY executable identity changed after start capture",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_recaptured_identity(
    *,
    name: str,
    expected: object,
    capture: Callable[[], dict[str, object]],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            f"{name}_start_identity_invalid",
            f"start manifest {name.replace('_', ' ')} identity is missing or invalid",
        )
        return check
    check["expected_identity_sha256"] = _identity_sha256(expected)
    try:
        observed = capture()
    except (OSError, ValueError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            f"{name}_unavailable",
            str(error),
        )
        return check
    check["observed_identity_sha256"] = _identity_sha256(observed)
    if observed != expected:
        _add_integrity_violation(
            violations,
            f"{name}_identity_mismatch",
            f"{name.replace('_', ' ')} identity changed after start capture",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _recapture_rust_toolchain(
    repo: Path, expected: object
) -> dict[str, object]:
    if not isinstance(expected, dict):
        raise ProvenanceError("start manifest Rust toolchain identity is invalid")
    probe_tool = expected.get("probe_tool")
    rustc = expected.get("rustc")
    if not isinstance(probe_tool, dict) or not isinstance(rustc, dict):
        raise ProvenanceError("start manifest Rust probe-tool identity is missing")
    declared_path = probe_tool.get("declared_path")
    declared_rustc_path = rustc.get("declared_path")
    kind = probe_tool.get("kind")
    if (
        not isinstance(declared_path, str)
        or not isinstance(declared_rustc_path, str)
        or kind not in {"rustup", "rustc"}
    ):
        raise ProvenanceError("start manifest Rust probe-tool identity is invalid")
    return _parse_toolchain(
        repo,
        declared_tool_path=declared_path,
        declared_tool_kind=str(kind),
        declared_rustc_path=declared_rustc_path,
    )


def _validate_config_inputs(
    start: dict[str, object],
    violations: list[dict[str, str]],
) -> dict[str, object]:
    measurement = start.get("measurement")
    if not isinstance(measurement, dict) or "config_inputs" not in measurement:
        _add_integrity_violation(
            violations,
            "config_inputs_start_identity_invalid",
            "start manifest config-input identity is missing",
        )
        return {"status": "invalid"}
    expected = measurement["config_inputs"]
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            "config_inputs_start_identity_invalid",
            "start manifest config-input identity is invalid",
        )
        return {"status": "invalid"}
    declared_path = expected.get("declared_path")
    if not isinstance(declared_path, str):
        _add_integrity_violation(
            violations,
            "config_inputs_start_identity_invalid",
            "start manifest config-input path is invalid",
        )
        return {"status": "invalid"}
    return _validate_recaptured_identity(
        name="config_inputs",
        expected=expected,
        capture=lambda: _capture_config_inputs(Path(declared_path)),
        violations=violations,
    )


def _validate_sealed_file(
    *,
    expected: object,
    name: str,
    executable: bool,
    violations: list[dict[str, str]],
) -> dict[str, object]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict):
        _add_integrity_violation(
            violations,
            f"{name}_start_identity_invalid",
            f"start manifest {name.replace('_', ' ')} identity is invalid",
        )
        return check
    path_value = expected.get("path")
    expected_digest = expected.get("sha256")
    expected_fingerprint = expected.get("fingerprint")
    expected_mode = "executable_read_only" if executable else "read_only"
    if (
        expected.get("schema") != "ny_measurement_sealed_file_v1"
        or not isinstance(path_value, str)
        or not isinstance(expected_digest, str)
        or not isinstance(expected_fingerprint, dict)
        or expected.get("mode") != expected_mode
    ):
        _add_integrity_violation(
            violations,
            f"{name}_start_identity_invalid",
            f"start manifest {name.replace('_', ' ')} identity is incomplete",
        )
        return check
    try:
        path = Path(path_value)
        observed_digest, observed_fingerprint = _stable_sealed_file_hash(
            path, executable=executable
        )
        resolved = path.resolve(strict=True)
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(violations, f"{name}_unavailable", str(error))
        return check
    check.update(
        {
            "path": str(resolved),
            "expected_sha256": expected_digest,
            "observed_sha256": observed_digest,
            "expected_fingerprint": expected_fingerprint,
            "observed_fingerprint": observed_fingerprint,
        }
    )
    if str(resolved) != path_value:
        _add_integrity_violation(
            violations,
            f"{name}_resolved_path_mismatch",
            f"expected {path_value}, observed {resolved}",
        )
    if observed_digest != expected_digest:
        _add_integrity_violation(
            violations,
            f"{name}_sha256_mismatch",
            f"expected {expected_digest}, observed {observed_digest}",
        )
    if observed_fingerprint != expected_fingerprint:
        _add_integrity_violation(
            violations,
            f"{name}_fingerprint_mismatch",
            f"{name.replace('_', ' ')} fingerprint changed after sealing",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    return check


def _validate_sealed_config_inputs(
    start: dict[str, object], violations: list[dict[str, str]]
) -> dict[str, object]:
    measurement = start.get("measurement")
    expected = (
        measurement.get("sealed_config_inputs")
        if isinstance(measurement, dict)
        else object()
    )
    if expected is None:
        return {"status": "not_configured"}
    if not isinstance(expected, dict) or not isinstance(
        expected.get("declared_path"), str
    ):
        _add_integrity_violation(
            violations,
            "sealed_config_inputs_start_identity_invalid",
            "start manifest sealed config-input identity is invalid",
        )
        return {"status": "invalid"}
    return _validate_recaptured_identity(
        name="sealed_config_inputs",
        expected=expected,
        capture=lambda: _capture_config_inputs(Path(str(expected["declared_path"]))),
        violations=violations,
    )


def _input_cache_key(path: str, fingerprint: dict[str, int]) -> str:
    identity = json.dumps(
        [path, fingerprint], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(identity)


def _stable_file_bytes(path: Path) -> tuple[bytes, str, dict[str, int]]:
    """Read a regular non-symlink file and reject concurrent replacement."""
    if path.is_symlink() or not path.is_file():
        raise ProvenanceError(
            f"evidence path is not a regular non-symlink file: {path}"
        )
    before = _file_fingerprint(path)
    data = path.read_bytes()
    after = _file_fingerprint(path)
    if before != after:
        raise ProvenanceError(f"file changed while evidence was captured: {path}")
    return data, _sha256(data), after


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _structured_sat_assignment(lines: list[bytes]) -> bool:
    """Recognize the same complete VNN-LIB 1.x/2.0 witness as the archiver."""
    try:
        payload = [line.decode("utf-8").strip() for line in lines if line.strip()]
    except UnicodeDecodeError:
        return False
    if not payload:
        return False
    legacy = "\n".join(payload)
    if legacy.startswith("("):
        return (
            _balanced_parentheses(legacy)
            and V1_INPUT_ASSIGNMENT.search(legacy) is not None
        )

    position = 0
    declarations = 0
    while position < len(payload):
        match = V2_ASSIGNMENT_HEADER.fullmatch(payload[position])
        if match is None:
            return False
        dimensions = match.group(3).strip()
        try:
            shape = (
                []
                if not dimensions
                else [int(value.strip()) for value in dimensions.split(",")]
            )
        except ValueError:
            return False
        if any(dimension <= 0 for dimension in shape):
            return False
        value_count = 1
        for dimension in shape:
            value_count *= dimension
        position += 1
        if position + value_count > len(payload):
            return False
        for value in payload[position : position + value_count]:
            if len(value.split()) != 1:
                return False
            try:
                float(value)
            except ValueError:
                if value.lower() not in {"true", "false"}:
                    return False
        position += value_count
        declarations += 1
    return declarations > 0


def _expected_metadata_config_identity(start: dict[str, object]) -> object:
    measurement = start.get("measurement")
    config_inputs = (
        measurement.get("config_inputs") if isinstance(measurement, dict) else None
    )
    if config_inputs is None:
        return None
    if not isinstance(config_inputs, dict):
        return object()
    required = (
        "schema",
        "declared_path",
        "resolved_path",
        "entry_count",
        "manifest_sha256",
    )
    if any(key not in config_inputs for key in required):
        return object()
    return {key: config_inputs[key] for key in required}


def _discover_run_artifacts(
    *,
    artifact_root: Path,
    run_id: str,
    violations: list[dict[str, str]],
) -> dict[Path, dict[str, Path]]:
    """Discover exact current-run artifact names without trusting metadata paths."""
    discovered: dict[Path, dict[str, Path]] = {}
    expected_names = {
        f"{run_id}.json": "metadata",
        f"{run_id}.results": "result",
        f"{run_id}.solver.log": "solver_log",
        f"{run_id}.preflight.json": "preflight",
    }
    try:
        for directory, directory_names, file_names in os.walk(
            artifact_root, followlinks=False
        ):
            root = Path(directory)
            if root == artifact_root and "runs" in directory_names:
                directory_names.remove("runs")
            for name in list(directory_names):
                child = root / name
                if child.is_symlink():
                    _add_integrity_violation(
                        violations,
                        "run_artifact_symlink_directory",
                        f"artifact tree contains a directory symlink: {child}",
                    )
                    directory_names.remove(name)
            for name in file_names:
                kind = expected_names.get(name)
                if kind is None:
                    continue
                path = root / name
                discovered.setdefault(root, {})[kind] = path
    except OSError as error:
        _add_integrity_violation(
            violations,
            "run_artifact_discovery_failed",
            str(error),
        )
    return discovered


def _run_artifact_namespace(
    discovered: dict[Path, dict[str, Path]], artifact_root: Path
) -> list[tuple[str, str]]:
    return sorted(
        (
            path.relative_to(artifact_root).as_posix(),
            kind,
        )
        for artifacts in discovered.values()
        for kind, path in artifacts.items()
    )


def _revalidate_run_evidence_snapshot(
    *,
    snapshot: dict[str, object],
    run_id: str,
    violations: list[dict[str, str]],
) -> None:
    """Reject artifact/CSV mutations that race the cache postflight."""
    artifact_root = snapshot.get("artifact_root")
    output_dir = snapshot.get("output_dir")
    categories = snapshot.get("categories")
    expected_namespace = snapshot.get("artifact_namespace")
    file_fingerprints = snapshot.get("file_fingerprints")
    expected_csv_paths = snapshot.get("csv_paths")
    if (
        not isinstance(artifact_root, Path)
        or not isinstance(output_dir, Path)
        or not isinstance(categories, list)
        or not isinstance(expected_namespace, list)
        or not isinstance(file_fingerprints, dict)
        or not isinstance(expected_csv_paths, set)
    ):
        _add_integrity_violation(
            violations,
            "run_evidence_snapshot_invalid",
            "internal run-evidence snapshot is missing or malformed",
        )
        return

    discovery_violations: list[dict[str, str]] = []
    observed = _discover_run_artifacts(
        artifact_root=artifact_root,
        run_id=run_id,
        violations=discovery_violations,
    )
    if (
        discovery_violations
        or _run_artifact_namespace(observed, artifact_root) != expected_namespace
    ):
        _add_integrity_violation(
            violations,
            "run_artifact_namespace_changed_during_completion",
            "current-run artifact namespace changed during completion validation",
        )

    candidate_csvs: set[Path] = {
        output_dir / f"{category}.csv"
        for category in categories
        if isinstance(category, str)
    }
    if output_dir.is_dir():
        candidate_csvs.update(output_dir.glob("*.csv"))
    observed_csv_paths = {path for path in candidate_csvs if path.exists()}
    if observed_csv_paths != expected_csv_paths:
        _add_integrity_violation(
            violations,
            "run_csv_namespace_changed_during_completion",
            "measurement CSV namespace changed during completion validation",
        )

    for path, expected_fingerprint in file_fingerprints.items():
        if not isinstance(path, Path) or not isinstance(expected_fingerprint, dict):
            _add_integrity_violation(
                violations,
                "run_evidence_snapshot_invalid",
                "internal evidence fingerprint is malformed",
            )
            continue
        try:
            if path.is_symlink() or not path.is_file():
                raise ProvenanceError(f"evidence path is unavailable: {path}")
            observed_fingerprint = _file_fingerprint(path)
        except (OSError, ProvenanceError) as error:
            _add_integrity_violation(
                violations,
                "run_evidence_file_changed_during_completion",
                str(error),
            )
            continue
        if observed_fingerprint != expected_fingerprint:
            _add_integrity_violation(
                violations,
                "run_evidence_file_changed_during_completion",
                f"evidence file changed during completion validation: {path}",
            )


def _metadata_input_reference(
    *,
    metadata: dict[str, object],
    label: str,
    metadata_path: Path,
    references: dict[str, dict[str, object]],
    violations: list[dict[str, str]],
) -> tuple[str, str, str, int] | None:
    value = metadata.get(label)
    if not isinstance(value, dict):
        _add_integrity_violation(
            violations,
            "run_metadata_input_invalid",
            f"{metadata_path} has invalid {label} evidence",
        )
        return None
    declared_path = value.get("declared_path")
    resolved_path = value.get("resolved_path")
    digest = value.get("sha256")
    size_bytes = value.get("size_bytes")
    cache_key = value.get("hash_cache_key")
    cache_hit = value.get("hash_cache_hit")
    if (
        not isinstance(declared_path, str)
        or not isinstance(resolved_path, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size_bytes) is not int
        or size_bytes < 0
        or not isinstance(cache_key, str)
        or re.fullmatch(r"[0-9a-f]{64}", cache_key) is None
        or type(cache_hit) is not bool
    ):
        _add_integrity_violation(
            violations,
            "run_metadata_input_invalid",
            f"{metadata_path} has incomplete {label} evidence",
        )
        return None
    reference = {
        "path": resolved_path,
        "sha256": digest,
        "size_bytes": size_bytes,
    }
    previous = references.get(cache_key)
    if previous is not None and previous != reference:
        _add_integrity_violation(
            violations,
            "run_metadata_cache_reference_conflict",
            f"cache key {cache_key} has conflicting row evidence",
        )
        return None
    references[cache_key] = reference
    return declared_path, resolved_path, cache_key, size_bytes


def _validate_row_preflight(
    *,
    preflight_path: Path,
    metadata_path: Path,
    metadata: dict[str, object],
    artifact_root: Path,
    start_manifest: Path,
    start_digest: str,
    category: str,
    instance_index: int,
    onnx: str,
    vnnlib: str,
    file_fingerprints: dict[Path, dict[str, int]],
    violations: list[dict[str, str]],
) -> dict[str, object] | None:
    """Validate and rehash a row's before-execution inputs and sealed copies."""
    try:
        data, digest, fingerprint = _stable_file_bytes(preflight_path)
        preflight = json.loads(data)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ProvenanceError,
    ) as error:
        _add_integrity_violation(
            violations,
            "run_preflight_unreadable",
            f"could not validate {preflight_path}: {error}",
        )
        return None
    file_fingerprints[preflight_path] = fingerprint
    try:
        start_artifact = start_manifest.relative_to(artifact_root).as_posix()
        preflight_artifact = preflight_path.relative_to(artifact_root).as_posix()
    except ValueError:
        _add_integrity_violation(
            violations,
            "run_preflight_path_invalid",
            f"preflight is outside the artifact root: {preflight_path}",
        )
        return None
    metadata_link = metadata.get("input_preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "ny_measurement_input_preflight_v1"
        or preflight.get("run_id") != metadata.get("run_id")
        or preflight.get("category") != category
        or preflight.get("instance_index") != instance_index
        or preflight.get("start_manifest") != start_artifact
        or preflight.get("start_manifest_sha256") != start_digest
        or not isinstance(metadata_link, dict)
        or metadata_link.get("schema") != "ny_measurement_input_preflight_v1"
        or metadata_link.get("artifact") != preflight_artifact
        or metadata_link.get("sha256") != digest
    ):
        _add_integrity_violation(
            violations,
            "run_preflight_identity_mismatch",
            f"preflight identity or metadata link does not match: {preflight_path}",
        )
        return None
    inputs = preflight.get("inputs")
    execution_inputs = metadata.get("execution_inputs")
    if not isinstance(inputs, dict) or not isinstance(execution_inputs, dict):
        _add_integrity_violation(
            violations,
            "run_preflight_inputs_invalid",
            f"preflight execution inputs are missing: {preflight_path}",
        )
        return None
    evidence: dict[str, object] = {
        "artifact": preflight_artifact,
        "sha256": digest,
        "size_bytes": fingerprint["size_bytes"],
        "inputs": {},
    }
    evidence_inputs = evidence["inputs"]
    assert isinstance(evidence_inputs, dict)
    for label, declared_name in (("onnx", onnx), ("vnnlib", vnnlib)):
        value = inputs.get(label)
        metadata_original = metadata.get(label)
        if (
            not isinstance(value, dict)
            or value.get("declared_name") != declared_name
            or not isinstance(metadata_original, dict)
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_inputs_invalid",
                f"preflight {label} declaration is invalid: {preflight_path}",
            )
            continue
        original = value.get("original")
        sealed = value.get("sealed")
        if not isinstance(original, dict) or not isinstance(sealed, dict):
            _add_integrity_violation(
                violations,
                "run_preflight_inputs_invalid",
                f"preflight {label} identities are invalid: {preflight_path}",
            )
            continue
        original_path_value = original.get("resolved_path")
        original_digest = original.get("sha256")
        original_size = original.get("size_bytes")
        original_fingerprint = original.get("fingerprint")
        sealed_path_value = sealed.get("resolved_path")
        sealed_digest = sealed.get("sha256")
        sealed_size = sealed.get("size_bytes")
        sealed_fingerprint = sealed.get("fingerprint")
        sealed_artifact = sealed.get("artifact")
        if (
            not isinstance(original_path_value, str)
            or not isinstance(original_digest, str)
            or type(original_size) is not int
            or not isinstance(original_fingerprint, dict)
            or not isinstance(sealed_path_value, str)
            or not isinstance(sealed_digest, str)
            or type(sealed_size) is not int
            or not isinstance(sealed_fingerprint, dict)
            or not isinstance(sealed_artifact, str)
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_inputs_invalid",
                f"preflight {label} identity is incomplete: {preflight_path}",
            )
            continue
        if (
            metadata_original.get("declared_path") != declared_name
            or metadata_original.get("resolved_path") != original_path_value
            or metadata_original.get("sha256") != original_digest
            or metadata_original.get("size_bytes") != original_size
            or execution_inputs.get(label) != sealed
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_metadata_mismatch",
                f"metadata does not bind the {label} preflight: {metadata_path}",
            )
        original_path = Path(original_path_value)
        sealed_path = Path(sealed_path_value)
        expected_sealed = (
            start_manifest.parent
            / "sealed"
            / "inputs"
            / category
            / preflight_path.parent.name
            / label
            / original_digest
            / original_path.name
        )
        try:
            if original_path.is_symlink():
                raise ProvenanceError(
                    f"original input became a symlink: {original_path}"
                )
            observed_original_digest, observed_original_fingerprint = _stable_file_hash(
                original_path
            )
            if sealed_path.is_symlink():
                raise ProvenanceError(f"sealed input became a symlink: {sealed_path}")
            observed_sealed_digest, observed_sealed_fingerprint = _stable_file_hash(
                sealed_path
            )
        except (OSError, ProvenanceError) as error:
            _add_integrity_violation(
                violations,
                "run_preflight_input_unavailable",
                str(error),
            )
            continue
        file_fingerprints[original_path] = observed_original_fingerprint
        file_fingerprints[sealed_path] = observed_sealed_fingerprint
        if (
            observed_original_digest != original_digest
            or observed_original_fingerprint != original_fingerprint
            or observed_sealed_digest != sealed_digest
            or observed_sealed_fingerprint != sealed_fingerprint
            or original_digest != sealed_digest
            or original_size != sealed_size
            or sealed_path.resolve() != expected_sealed.resolve()
            or (artifact_root / sealed_artifact).resolve() != sealed_path.resolve()
        ):
            _add_integrity_violation(
                violations,
                "run_preflight_input_drift",
                f"pre-run {label} binding changed: {preflight_path}",
            )
        evidence_inputs[label] = {
            "original_sha256": observed_original_digest,
            "sealed_sha256": observed_sealed_digest,
            "sealed_artifact": sealed_artifact,
        }
    return evidence


def _validate_run_evidence(
    *,
    start: dict[str, object],
    start_manifest: Path,
    start_digest: str,
    run_id: str,
    violations: list[dict[str, str]],
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    bool,
    dict[str, object],
]:
    """Bind each current-run CSV row to one complete, content-checked trio."""
    measurement = start.get("measurement")
    ny = start.get("ny")
    if not isinstance(measurement, dict) or not isinstance(ny, dict):
        _add_integrity_violation(
            violations,
            "run_evidence_start_identity_invalid",
            "start manifest lacks measurement or NY identity",
        )
        return (
            {
                "schema": "ny_measurement_run_evidence_v1",
                "status": "invalid",
            },
            {},
            False,
            {},
        )
    artifact_root_value = measurement.get("artifact_root")
    output_dir_value = measurement.get("output_dir")
    repo_root_value = ny.get("repo_root")
    categories_value = measurement.get("categories")
    if (
        not isinstance(artifact_root_value, str)
        or not isinstance(output_dir_value, str)
        or not isinstance(repo_root_value, str)
        or not isinstance(categories_value, list)
        or any(
            not isinstance(category, str) or SAFE_COMPONENT.fullmatch(category) is None
            for category in categories_value
        )
    ):
        _add_integrity_violation(
            violations,
            "run_evidence_start_identity_invalid",
            "start manifest has invalid evidence roots or categories",
        )
        return (
            {
                "schema": "ny_measurement_run_evidence_v1",
                "status": "invalid",
            },
            {},
            False,
            {},
        )

    artifact_root = Path(artifact_root_value)
    output_dir = Path(output_dir_value)
    repo_root = Path(repo_root_value)
    categories = list(categories_value)
    category_order = {category: index for index, category in enumerate(categories)}
    initial_violation_count = len(violations)
    discovered = _discover_run_artifacts(
        artifact_root=artifact_root,
        run_id=run_id,
        violations=violations,
    )
    metadata_paths: list[Path] = []
    result_count = 0
    solver_log_count = 0
    preflight_count = 0
    for instance_dir, artifacts in sorted(
        discovered.items(), key=lambda item: str(item[0])
    ):
        missing = {"metadata", "result", "solver_log", "preflight"} - set(artifacts)
        if missing:
            _add_integrity_violation(
                violations,
                "run_artifact_trio_incomplete",
                f"{instance_dir} is missing current-run artifacts: {', '.join(sorted(missing))}",
            )
        metadata_path = artifacts.get("metadata")
        if metadata_path is not None:
            metadata_paths.append(metadata_path)
        result_count += int("result" in artifacts)
        solver_log_count += int("solver_log" in artifacts)
        preflight_count += int("preflight" in artifacts)

    file_fingerprints: dict[Path, dict[str, int]] = {}
    references: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    seen_instances: set[tuple[str, int]] = set()
    expected_config_identity = _expected_metadata_config_identity(start)
    expected_execution_config_identity = (
        measurement.get("sealed_config_inputs")
        if isinstance(measurement, dict)
        else object()
    )
    try:
        expected_start_artifact = start_manifest.relative_to(artifact_root).as_posix()
        expected_cache_artifact = (
            start_manifest.with_name("input_hash_cache.json")
            .relative_to(artifact_root)
            .as_posix()
        )
    except ValueError:
        expected_start_artifact = ""
        expected_cache_artifact = ""
        _add_integrity_violation(
            violations,
            "run_evidence_start_path_invalid",
            "start manifest is outside its declared artifact root",
        )

    for metadata_path in sorted(metadata_paths):
        instance_dir = metadata_path.parent
        artifacts = discovered[instance_dir]
        result_path = artifacts.get("result")
        solver_log_path = artifacts.get("solver_log")
        preflight_path = artifacts.get("preflight")
        if result_path is None or solver_log_path is None or preflight_path is None:
            continue
        try:
            metadata_data, metadata_digest, metadata_fingerprint = _stable_file_bytes(
                metadata_path
            )
            result_data, result_digest, result_fingerprint = _stable_file_bytes(
                result_path
            )
            solver_log_data, solver_log_digest, solver_log_fingerprint = (
                _stable_file_bytes(solver_log_path)
            )
            metadata = json.loads(metadata_data)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ProvenanceError,
        ) as error:
            _add_integrity_violation(
                violations,
                "run_artifact_unreadable",
                f"could not validate {metadata_path}: {error}",
            )
            continue
        file_fingerprints[metadata_path] = metadata_fingerprint
        file_fingerprints[result_path] = result_fingerprint
        file_fingerprints[solver_log_path] = solver_log_fingerprint
        if not isinstance(metadata, dict) or metadata.get("schema") != (
            "ny_measurement_result_v2"
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_schema_invalid",
                f"unsupported result metadata: {metadata_path}",
            )
            continue
        category = metadata.get("category")
        instance_index = metadata.get("instance_index")
        verdict = metadata.get("solver_verdict")
        elapsed_seconds = metadata.get("elapsed_seconds")
        timeout_seconds = metadata.get("timeout_seconds")
        solver_exit_status = metadata.get("solver_exit_status")
        witness_present = metadata.get("witness_present")
        counterexample_validation = metadata.get("counterexample_validation")
        source_csv = metadata.get("source_csv")
        solver_log = metadata.get("solver_log")
        if (
            metadata.get("schema_version") != 2
            or not isinstance(category, str)
            or category not in category_order
            or type(instance_index) is not int
            or instance_index <= 0
            or not isinstance(verdict, str)
            or verdict not in STANDARD_SOLVER_VERDICTS
            or type(elapsed_seconds) is not int
            or elapsed_seconds < 0
            or type(timeout_seconds) is not int
            or timeout_seconds <= 0
            or type(solver_exit_status) is not int
            or not 0 <= solver_exit_status <= 255
            or (verdict in {"sat", "unsat"} and solver_exit_status != 0)
            or type(witness_present) is not bool
            or witness_present != (verdict == "sat")
            or not isinstance(counterexample_validation, dict)
            or counterexample_validation.get("status")
            != ("not_checked" if verdict == "sat" else "not_applicable")
            or not isinstance(source_csv, str)
            or not isinstance(solver_log, dict)
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_fields_invalid",
                f"result metadata fields are invalid: {metadata_path}",
            )
            continue
        identity = (category, instance_index)
        if identity in seen_instances:
            _add_integrity_violation(
                violations,
                "run_metadata_instance_duplicate",
                f"duplicate current-run instance identity: {identity}",
            )
            continue
        seen_instances.add(identity)
        onnx_reference = _metadata_input_reference(
            metadata=metadata,
            label="onnx",
            metadata_path=metadata_path,
            references=references,
            violations=violations,
        )
        vnnlib_reference = _metadata_input_reference(
            metadata=metadata,
            label="vnnlib",
            metadata_path=metadata_path,
            references=references,
            violations=violations,
        )
        if onnx_reference is None or vnnlib_reference is None:
            continue
        onnx, _onnx_resolved, onnx_cache_key, _onnx_size = onnx_reference
        vnnlib, _vnnlib_resolved, vnnlib_cache_key, _vnnlib_size = vnnlib_reference
        instance_identity = json.dumps(
            [category, instance_index, onnx, vnnlib],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_instance_dir = (
            f"{instance_index:05d}-{_sha256(instance_identity)[:16]}"
        )
        if instance_dir.name != expected_instance_dir or instance_dir.parent != (
            artifact_root / category
        ):
            _add_integrity_violation(
                violations,
                "run_artifact_identity_path_mismatch",
                f"metadata is outside its content-addressed instance path: {metadata_path}",
            )
        expected_result_artifact = result_path.relative_to(artifact_root).as_posix()
        expected_log_artifact = solver_log_path.relative_to(artifact_root).as_posix()
        if metadata.get("run_id") != run_id:
            _add_integrity_violation(
                violations,
                "run_metadata_run_id_mismatch",
                f"metadata run ID mismatch: {metadata_path}",
            )
        if (
            metadata.get("start_manifest") != expected_start_artifact
            or metadata.get("start_manifest_sha256") != start_digest
            or metadata.get("input_hash_cache") != expected_cache_artifact
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_provenance_link_mismatch",
                f"metadata provenance link mismatch: {metadata_path}",
            )
        if metadata.get("config_inputs") != expected_config_identity:
            _add_integrity_violation(
                violations,
                "run_metadata_config_identity_mismatch",
                f"metadata config identity mismatch: {metadata_path}",
            )
        if (
            metadata.get("execution_config_inputs")
            != expected_execution_config_identity
        ):
            _add_integrity_violation(
                violations,
                "run_metadata_execution_config_identity_mismatch",
                f"metadata sealed config identity mismatch: {metadata_path}",
            )
        if (
            metadata.get("result_artifact") != expected_result_artifact
            or metadata.get("result_sha256") != result_digest
            or metadata.get("raw_result_sha256") != result_digest
        ):
            _add_integrity_violation(
                violations,
                "run_result_artifact_mismatch",
                f"raw result does not match metadata: {metadata_path}",
            )
        if (
            solver_log.get("artifact") != expected_log_artifact
            or solver_log.get("sha256") != solver_log_digest
            or solver_log.get("size_bytes") != len(solver_log_data)
        ):
            _add_integrity_violation(
                violations,
                "run_solver_log_artifact_mismatch",
                f"solver log does not match metadata: {metadata_path}",
            )
        result_lines = result_data.splitlines()
        first_line = (
            b"".join(result_lines[0].split()).decode("utf-8", "replace").lower()
            if result_lines
            else ""
        )
        verdict_matches = (
            first_line == verdict
            if verdict != "timeout"
            else first_line in {"", "timeout"}
        )
        if not verdict_matches:
            _add_integrity_violation(
                violations,
                "run_result_verdict_mismatch",
                f"raw result verdict does not match metadata: {metadata_path}",
            )
        if verdict == "sat" and not _structured_sat_assignment(result_lines[1:]):
            _add_integrity_violation(
                violations,
                "run_sat_witness_invalid",
                f"SAT result has no structured assignment: {metadata_path}",
            )
        preflight_evidence = _validate_row_preflight(
            preflight_path=preflight_path,
            metadata_path=metadata_path,
            metadata=metadata,
            artifact_root=artifact_root,
            start_manifest=start_manifest,
            start_digest=start_digest,
            category=category,
            instance_index=instance_index,
            onnx=onnx,
            vnnlib=vnnlib,
            file_fingerprints=file_fingerprints,
            violations=violations,
        )
        declared_source = Path(source_csv)
        resolved_source = (
            declared_source
            if declared_source.is_absolute()
            else repo_root / declared_source
        ).resolve()
        expected_csv = (output_dir / f"{category}.csv").resolve()
        if resolved_source != expected_csv:
            _add_integrity_violation(
                violations,
                "run_metadata_source_csv_mismatch",
                f"metadata source CSV does not match the declared output: {metadata_path}",
            )
        records.append(
            {
                "category": category,
                "instance_index": instance_index,
                "onnx": onnx,
                "vnnlib": vnnlib,
                "solver_verdict": verdict,
                "solver_exit_status": solver_exit_status,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
                "input_hash_cache_keys": [onnx_cache_key, vnnlib_cache_key],
                "metadata": {
                    "artifact": metadata_path.relative_to(artifact_root).as_posix(),
                    "sha256": metadata_digest,
                    "size_bytes": metadata_fingerprint["size_bytes"],
                },
                "result": {
                    "artifact": expected_result_artifact,
                    "sha256": result_digest,
                    "size_bytes": result_fingerprint["size_bytes"],
                },
                "solver_log": {
                    "artifact": expected_log_artifact,
                    "sha256": solver_log_digest,
                    "size_bytes": solver_log_fingerprint["size_bytes"],
                },
                "preflight": preflight_evidence,
            }
        )

    records.sort(
        key=lambda record: (
            category_order.get(str(record["category"]), len(category_order)),
            int(record["instance_index"]),
            str(record["metadata"]),
        )
    )
    expected_rows: dict[str, list[list[str]]] = {
        category: [] for category in categories
    }
    for record in records:
        category = str(record["category"])
        expected_rows[category].append(
            [
                category,
                str(record["onnx"]),
                str(record["vnnlib"]),
                "0",
                str(record["solver_verdict"]),
                str(record["elapsed_seconds"]),
                run_id,
            ]
        )

    csv_evidence: list[dict[str, object]] = []
    observed_csv_paths: set[Path] = set()
    actual_rows: dict[str, list[list[str]]] = {category: [] for category in categories}
    candidate_csvs: set[Path] = {
        output_dir / f"{category}.csv" for category in categories
    }
    if output_dir.is_dir():
        candidate_csvs.update(output_dir.glob("*.csv"))
    for csv_path in sorted(candidate_csvs):
        if not csv_path.exists():
            continue
        observed_csv_paths.add(csv_path)
        try:
            data, digest, fingerprint = _stable_file_bytes(csv_path)
            rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
        except (OSError, UnicodeDecodeError, csv.Error, ProvenanceError) as error:
            _add_integrity_violation(
                violations,
                "run_csv_unreadable",
                f"could not validate {csv_path}: {error}",
            )
            continue
        current_rows: list[list[str]] = []
        for row in rows:
            if len(row) > 6 and row[6] == run_id:
                current_rows.append(row)
                if len(row) != 7:
                    _add_integrity_violation(
                        violations,
                        "run_csv_row_invalid",
                        f"current-run row does not have seven columns: {csv_path}",
                    )
            elif run_id in row:
                _add_integrity_violation(
                    violations,
                    "run_csv_row_invalid",
                    f"current-run ID appears outside the provenance column: {csv_path}",
                )
        category = csv_path.stem
        if current_rows and category not in actual_rows:
            _add_integrity_violation(
                violations,
                "run_csv_unexpected_category",
                f"current-run rows appear in unexpected CSV: {csv_path}",
            )
        elif category in actual_rows:
            actual_rows[category] = current_rows
        csv_evidence.append(
            {
                "path": str(csv_path.resolve()),
                "sha256": digest,
                "size_bytes": fingerprint["size_bytes"],
                "current_run_row_count": len(current_rows),
                "current_run_rows_sha256": _identity_sha256(current_rows),
            }
        )
        file_fingerprints[csv_path] = fingerprint
    for category in categories:
        if actual_rows[category] != expected_rows[category]:
            _add_integrity_violation(
                violations,
                "run_csv_artifact_bijection_mismatch",
                f"current-run CSV rows and artifacts differ for category {category}",
            )

    csv_row_count = sum(len(rows) for rows in actual_rows.values())
    evidence_required = bool(
        metadata_paths
        or result_count
        or solver_log_count
        or preflight_count
        or csv_row_count
    )
    check = {
        "schema": "ny_measurement_run_evidence_v1",
        "status": (
            "valid" if len(violations) == initial_violation_count else "invalid"
        ),
        "produced_rows": csv_row_count > 0,
        "metadata_count": len(metadata_paths),
        "result_count": result_count,
        "solver_log_count": solver_log_count,
        "preflight_count": preflight_count,
        "validated_record_count": len(records),
        "csv_row_count": csv_row_count,
        "records_sha256": _identity_sha256(records),
        "csv_evidence_sha256": _identity_sha256(csv_evidence),
        "records": records,
        "csv_evidence": csv_evidence,
    }
    snapshot = {
        "artifact_root": artifact_root,
        "output_dir": output_dir,
        "categories": categories,
        "artifact_namespace": _run_artifact_namespace(discovered, artifact_root),
        "file_fingerprints": file_fingerprints,
        "csv_paths": observed_csv_paths,
    }
    return check, references, evidence_required, snapshot


def _validate_input_hash_cache(
    *,
    cache_path: Path,
    run_id: str,
    start_digest: str,
    referenced_entries: dict[str, dict[str, object]],
    evidence_required: bool,
    violations: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, object]]:
    violation_count = len(violations)
    check: dict[str, object] = {"status": "invalid"}
    if cache_path.is_symlink():
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_file",
            f"input hash cache must not be a symlink: {cache_path}",
        )
        return {"present": True, "artifact": cache_path.name}, check
    if not cache_path.exists():
        if evidence_required or referenced_entries:
            _add_integrity_violation(
                violations,
                "input_hash_cache_missing_for_run_artifacts",
                "current-run rows or artifacts exist without their input hash cache",
            )
        return (
            {
                "present": False,
                "artifact": cache_path.name,
                "entry_count": 0,
            },
            {
                "status": (
                    "invalid" if len(violations) != violation_count else "absent"
                ),
                "referenced_entry_count": len(referenced_entries),
            },
        )
    evidence: dict[str, object] = {
        "present": True,
        "artifact": cache_path.name,
    }
    if not cache_path.is_file():
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_file",
            f"input hash cache must be a regular non-symlink file: {cache_path}",
        )
        return evidence, check
    try:
        cache_data, cache_digest, cache_fingerprint = _stable_file_bytes(cache_path)
    except (OSError, ProvenanceError) as error:
        _add_integrity_violation(
            violations,
            "input_hash_cache_unreadable",
            str(error),
        )
        return evidence, check
    evidence.update(
        {
            "sha256": cache_digest,
            "size_bytes": cache_fingerprint["size_bytes"],
        }
    )
    try:
        cache = json.loads(cache_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_json",
            str(error),
        )
        return evidence, check
    if not isinstance(cache, dict):
        _add_integrity_violation(
            violations,
            "input_hash_cache_invalid_manifest",
            "input hash cache root must be an object",
        )
        return evidence, check
    entries = cache.get("entries")
    evidence["entry_count"] = len(entries) if isinstance(entries, dict) else None
    if cache.get("schema") != "ny_measurement_input_hash_cache_v1":
        _add_integrity_violation(
            violations,
            "input_hash_cache_schema_mismatch",
            "input hash cache schema does not match ny_measurement_input_hash_cache_v1",
        )
    if cache.get("run_id") != run_id:
        _add_integrity_violation(
            violations,
            "input_hash_cache_run_id_mismatch",
            "input hash cache run ID does not match the start manifest",
        )
    if cache.get("start_manifest_sha256") != start_digest:
        _add_integrity_violation(
            violations,
            "input_hash_cache_start_manifest_mismatch",
            "input hash cache start-manifest digest does not match",
        )
    fingerprint_keys = {"device", "inode", "size_bytes", "mtime_ns", "ctime_ns"}
    entries_well_formed = isinstance(entries, dict)
    if isinstance(entries, dict):
        for key, entry in sorted(entries.items()):
            if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
                entries_well_formed = False
                break
            if not isinstance(entry, dict):
                entries_well_formed = False
                break
            path_value = entry.get("path")
            fingerprint = entry.get("fingerprint")
            digest = entry.get("sha256")
            if (
                not isinstance(path_value, str)
                or not isinstance(fingerprint, dict)
                or set(fingerprint) != fingerprint_keys
                or any(type(value) is not int for value in fingerprint.values())
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or _input_cache_key(path_value, fingerprint) != key
            ):
                entries_well_formed = False
                break
    if not entries_well_formed:
        _add_integrity_violation(
            violations,
            "input_hash_cache_entries_invalid",
            "input hash cache entries are malformed or not content-addressed",
        )
    rehashed_entries: list[dict[str, object]] = []
    if entries_well_formed and isinstance(entries, dict):
        for key, entry_object in sorted(entries.items()):
            assert isinstance(entry_object, dict)
            path_value = entry_object["path"]
            fingerprint = entry_object["fingerprint"]
            digest = entry_object["sha256"]
            assert isinstance(path_value, str)
            assert isinstance(fingerprint, dict)
            assert isinstance(digest, str)
            path = Path(path_value)
            try:
                resolved = path.resolve(strict=True)
                if (
                    not path.is_absolute()
                    or resolved != path
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    raise ProvenanceError(
                        f"cached input path is not canonical: {path_value}"
                    )
                observed_digest, observed_fingerprint = _stable_file_hash(path)
            except (OSError, ProvenanceError) as error:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_unavailable",
                    f"could not rehash cache entry {key}: {error}",
                )
                continue
            if observed_digest != digest:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_sha256_mismatch",
                    f"cached input digest changed for {path_value}",
                )
            if observed_fingerprint != fingerprint:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_fingerprint_mismatch",
                    f"cached input stat identity changed for {path_value}",
                )
            reference = referenced_entries.get(key)
            if reference is None:
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_entry_unreferenced",
                    f"cache entry {key} is not referenced by a current-run artifact",
                )
            elif (
                reference.get("path") != path_value
                or reference.get("sha256") != digest
                or reference.get("size_bytes") != fingerprint.get("size_bytes")
            ):
                _add_integrity_violation(
                    violations,
                    "input_hash_cache_metadata_mismatch",
                    f"cache entry {key} differs from row metadata",
                )
            rehashed_entries.append(
                {
                    "key": key,
                    "path": path_value,
                    "sha256": observed_digest,
                    "size_bytes": observed_fingerprint["size_bytes"],
                }
            )
        missing_references = set(referenced_entries) - set(entries)
        if missing_references:
            _add_integrity_violation(
                violations,
                "input_hash_cache_reference_missing",
                "row metadata references cache keys that are absent: "
                + ", ".join(sorted(missing_references)),
            )
    if evidence_required and isinstance(entries, dict) and not entries:
        _add_integrity_violation(
            violations,
            "input_hash_cache_empty_for_run_artifacts",
            "current-run rows or artifacts exist with an empty input hash cache",
        )
    check["status"] = "valid" if len(violations) == violation_count else "invalid"
    check["sha256"] = evidence.get("sha256")
    check["entry_count"] = evidence.get("entry_count")
    check["referenced_entry_count"] = len(referenced_entries)
    check["rehashed_entry_count"] = len(rehashed_entries)
    check["entries_sha256"] = _identity_sha256(rehashed_entries)
    return evidence, check


def create_completion(*, start_manifest: Path, exit_status: int) -> Path:
    """Create the immutable completion sibling for an existing start record."""
    if not 0 <= exit_status <= 255:
        raise ValueError(f"exit status is outside the shell range: {exit_status}")
    if start_manifest.is_symlink():
        raise ProvenanceError(f"start manifest must not be a symlink: {start_manifest}")
    start_manifest = start_manifest.resolve()
    start_data = start_manifest.read_bytes()
    start = json.loads(start_data)
    if start.get("schema") != "ny_measurement_start_v1":
        raise ProvenanceError(f"unsupported start manifest schema: {start_manifest}")
    run_id = start.get("run_id")
    if not isinstance(run_id, str) or SAFE_COMPONENT.fullmatch(run_id) is None:
        raise ProvenanceError(f"invalid run ID in start manifest: {start_manifest}")
    completion_path = start_manifest.with_name("completion.json")
    start_digest = _sha256(start_data)
    violations: list[dict[str, str]] = []
    solver = start.get("solver_binary")
    dependencies = start.get("dependencies")
    ay = dependencies.get("ay") if isinstance(dependencies, dict) else None
    ny = start.get("ny")
    ny_repo_root = ny.get("repo_root") if isinstance(ny, dict) else None
    checks: dict[str, object] = {
        "solver_binary": _validate_solver_binary(start, violations),
        "sealed_solver_binary": _validate_sealed_file(
            expected=(
                solver.get("sealed_execution") if isinstance(solver, dict) else object()
            ),
            name="sealed_solver_binary",
            executable=True,
            violations=violations,
        ),
        "ay_executable": _validate_ay_executable(start, violations),
        "sealed_ay_executable": _validate_sealed_file(
            expected=(
                ay.get("sealed_executable") if isinstance(ay, dict) else object()
            ),
            name="sealed_ay_executable",
            executable=True,
            violations=violations,
        ),
        "config_inputs": _validate_config_inputs(start, violations),
        "sealed_config_inputs": _validate_sealed_config_inputs(start, violations),
        "cuda_runtime": _validate_recaptured_identity(
            name="cuda_runtime",
            expected=(
                dependencies.get("cuda_runtime")
                if isinstance(dependencies, dict)
                else object()
            ),
            capture=lambda: _recapture_cuda_runtime_from_start(start),
            violations=violations,
        ),
    }
    provenance_tools = start.get("provenance_tools")
    expected_git = (
        provenance_tools.get("git")
        if isinstance(provenance_tools, dict)
        else object()
    )
    checks["git_executable"] = _validate_recaptured_identity(
        name="git_executable",
        expected=expected_git,
        capture=lambda: _recapture_bound_git_executable(
            expected_git,
            (
                Path(ny_repo_root)
                if isinstance(ny_repo_root, str)
                else start_manifest.parent
            ),
        ),
        violations=violations,
    )
    expected_rust_toolchain = start.get("rust_toolchain")
    checks["rust_toolchain"] = _validate_recaptured_identity(
        name="rust_toolchain",
        expected=expected_rust_toolchain,
        capture=lambda: _recapture_rust_toolchain(
            (
                Path(ny_repo_root)
                if isinstance(ny_repo_root, str)
                else start_manifest.parent
            ),
            expected_rust_toolchain,
        ),
        violations=violations,
    )
    start_host = start.get("host")
    start_containment = (
        start_host.get("containment") if isinstance(start_host, dict) else object()
    )
    checks["containment"] = _validate_recaptured_identity(
        name="containment",
        expected=start_containment,
        capture=_capture_measurement_containment,
        violations=violations,
    )

    benchmark = start.get("benchmark")
    benchmark_root = (
        benchmark.get("benchmark_root") if isinstance(benchmark, dict) else None
    )
    git_bound_path = (
        expected_git.get("resolved_path")
        if isinstance(expected_git, dict)
        else None
    )
    if (
        checks["git_executable"].get("status") == "valid"
        and isinstance(git_bound_path, str)
    ):
        with _bound_git_executable(git_bound_path):
            if isinstance(ny_repo_root, str):
                checks["ny_worktree"] = _validate_recaptured_identity(
                    name="ny_worktree",
                    expected=ny,
                    capture=lambda: _capture_worktree(Path(ny_repo_root)),
                    violations=violations,
                )
            else:
                _add_integrity_violation(
                    violations,
                    "ny_worktree_start_identity_invalid",
                    "start manifest NY repository root is missing or invalid",
                )
                checks["ny_worktree"] = {"status": "invalid"}
            if isinstance(benchmark_root, str):
                checks["benchmark"] = _validate_recaptured_identity(
                    name="benchmark",
                    expected=benchmark,
                    capture=lambda: _capture_benchmark(Path(benchmark_root)),
                    violations=violations,
                )
            else:
                _add_integrity_violation(
                    violations,
                    "benchmark_start_identity_invalid",
                    "start manifest benchmark root is missing or invalid",
                )
                checks["benchmark"] = {"status": "invalid"}
            checks["git_executable_post"] = _validate_recaptured_identity(
                name="git_executable_post",
                expected=expected_git,
                capture=lambda: _recapture_bound_git_executable(
                    expected_git,
                    (
                        Path(ny_repo_root)
                        if isinstance(ny_repo_root, str)
                        else start_manifest.parent
                    ),
                ),
                violations=violations,
            )
    else:
        checks["ny_worktree"] = {
            "status": "not_performed",
            "reason": "bound_git_identity_invalid",
        }
        checks["benchmark"] = {
            "status": "not_performed",
            "reason": "bound_git_identity_invalid",
        }
        checks["git_executable_post"] = {
            "status": "not_performed",
            "reason": "bound_git_identity_invalid",
        }

    run_evidence_violation_count = len(violations)
    (
        run_evidence,
        referenced_cache_entries,
        evidence_required,
        run_evidence_snapshot,
    ) = _validate_run_evidence(
        start=start,
        start_manifest=start_manifest,
        start_digest=start_digest,
        run_id=run_id,
        violations=violations,
    )
    cache_path = start_manifest.with_name("input_hash_cache.json")
    cache_lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with cache_lock_path.open("a+b") as cache_lock:
        fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
        cache_evidence, checks["input_hash_cache"] = _validate_input_hash_cache(
            cache_path=cache_path,
            run_id=run_id,
            start_digest=start_digest,
            referenced_entries=referenced_cache_entries,
            evidence_required=evidence_required,
            violations=violations,
        )
    _revalidate_run_evidence_snapshot(
        snapshot=run_evidence_snapshot,
        run_id=run_id,
        violations=violations,
    )
    run_evidence["input_hash_cache_entry_count"] = checks["input_hash_cache"].get(
        "entry_count"
    )
    run_evidence["referenced_input_hash_cache_entry_count"] = len(
        referenced_cache_entries
    )
    run_evidence["status"] = (
        "valid" if len(violations) == run_evidence_violation_count else "invalid"
    )
    checks["run_evidence"] = run_evidence
    integrity_status = "valid" if not violations else "invalid"
    payload = {
        "schema": "ny_measurement_completion_v1",
        "run_id": run_id,
        "ended_at_utc": _utc_now(),
        "exit_status": exit_status,
        "completed_successfully": exit_status == 0 and integrity_status == "valid",
        "start_manifest": start_manifest.name,
        "start_manifest_sha256": start_digest,
        "input_hash_cache": cache_evidence,
        "integrity": {
            "schema": "ny_measurement_completion_integrity_v1",
            "status": integrity_status,
            "violations": violations,
            "checks": checks,
        },
        "host_state": _capture_host_state(),
    }
    _write_immutable(completion_path, _json_bytes(payload))
    return completion_path


def verify_start_cuda_runtime(
    *, start_manifest: Path, hash_files: bool
) -> Path | None:
    """Verify the run-local CUDA seal before launching one measured child."""
    if start_manifest.is_symlink():
        raise ProvenanceError(
            f"start manifest must not be a symlink: {start_manifest}"
        )
    try:
        resolved = start_manifest.resolve(strict=True)
        before = _file_fingerprint(resolved)
        data = resolved.read_bytes()
        after = _file_fingerprint(resolved)
    except OSError as error:
        raise ProvenanceError(
            f"cannot read measurement start manifest: {start_manifest}: {error}"
        ) from error
    if before != after:
        raise ProvenanceError(
            "measurement start manifest changed while CUDA runtime was verified"
        )
    try:
        start = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError("measurement start manifest is malformed") from error
    if not isinstance(start, dict) or start.get("schema") != "ny_measurement_start_v1":
        raise ProvenanceError("unsupported measurement start manifest schema")
    return _validate_sealed_cuda_runtime(
        _cuda_runtime_from_start(start), hash_files=hash_files
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="capture immutable start evidence")
    start.add_argument("--repo-root", type=Path, required=True)
    start.add_argument("--binary", type=Path, required=True)
    start.add_argument("--benchmark-root", type=Path, required=True)
    start.add_argument("--artifact-root", type=Path, required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--output-dir", type=Path, required=True)
    start.add_argument("--scratch-dir", type=Path, required=True)
    start.add_argument("--result-file", type=Path, required=True)
    start.add_argument("--solver-log-file", type=Path, required=True)
    start.add_argument("--categories", required=True)
    start.add_argument("--timeout-cap-seconds", type=int, required=True)
    start.add_argument("--watchdog-grace-seconds", type=int, required=True)
    start.add_argument("--max-rows-per-category", type=int, required=True)
    start.add_argument("--instance-index", type=int, required=True)
    start.add_argument("--vnnlib-version", required=True)
    start.add_argument("--configs-dir", type=Path)
    start.add_argument("--sweep-script", type=Path, required=True)
    complete = commands.add_parser("complete", help="record immutable completion")
    complete.add_argument("--start-manifest", type=Path, required=True)
    complete.add_argument("--exit-status", type=int, required=True)
    verify_cuda = commands.add_parser(
        "verify-cuda-runtime",
        help="verify the run-local CUDA runtime seal before child execution",
    )
    verify_cuda.add_argument("--start-manifest", type=Path, required=True)
    verify_cuda.add_argument(
        "--fast",
        action="store_true",
        help="verify namespace and stat fingerprints without rehashing large objects",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    completion_integrity_valid = True
    try:
        if args.command == "start":
            path = capture_start_manifest(
                repo_root=args.repo_root,
                binary=args.binary,
                benchmark_root=args.benchmark_root,
                artifact_root=args.artifact_root,
                run_id=args.run_id,
                output_dir=args.output_dir,
                scratch_dir=args.scratch_dir,
                result_file=args.result_file,
                solver_log_file=args.solver_log_file,
                categories_raw=args.categories,
                timeout_cap_seconds=args.timeout_cap_seconds,
                watchdog_grace_seconds=args.watchdog_grace_seconds,
                max_rows_per_category=args.max_rows_per_category,
                instance_index=args.instance_index,
                vnnlib_version=args.vnnlib_version,
                sweep_script=args.sweep_script,
                configs_dir=args.configs_dir,
            )
        elif args.command == "complete":
            path = create_completion(
                start_manifest=args.start_manifest,
                exit_status=args.exit_status,
            )
            completion = json.loads(path.read_bytes())
            integrity = completion.get("integrity")
            completion_integrity_valid = (
                isinstance(integrity, dict) and integrity.get("status") == "valid"
            )
        else:
            runtime_path = verify_start_cuda_runtime(
                start_manifest=args.start_manifest,
                hash_files=not args.fast,
            )
            path = "" if runtime_path is None else str(runtime_path)
    except (OSError, ValueError, ProvenanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(path)
    if args.command == "complete" and not completion_integrity_valid:
        print(
            "ERROR: measurement completion integrity validation failed", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
