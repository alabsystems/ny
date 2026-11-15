#!/bin/bash
# Resumable NY scorecard measurement: runs the REAL competition entry point
# (`ny vnncomp v1`) over each regular-track benchmark's instances.csv at
# min(the row's official timeout, NY_MEASURE_CAP). THE CAP DEFAULTS TO 0,
# MEANING NO CAP: every row gets the official budget, the only budget the
# competition awards points for. (This header said "defaults to 120 seconds"
# for long after `866e21946` changed it, contradicting the code ~850 lines
# below and outliving the defect by months.) Set NY_MEASURE_CAP=<secs>
# deliberately for a fast LOWER-BOUND sweep, and then read every resulting
# `timeout` as "not measured", never as "cannot solve". The exact cap is
# sealed in the run manifest. Results go to an isolated per-run evidence
# directory (reports/measured-runs/<run-id> by default) in the official
# results.csv format plus a trailing provenance field
# (category,onnx,vnnlib,prep,RESULT,time,run_id). Legacy six-column rows remain
# valid and all scorers ignore the optional trailing field.
# Smallest benchmarks first so each completes (and scores accurately) soonest.
# Resumable: skips instances already recorded. GPU-serial (one instance at a time).
set -u

# Bash imports BASH_ENV content and exported functions before executing this
# file. BASH_ENV can erase itself and define functions named `builtin` or
# `command`. Bash optimizes the redirection-only substitution below without
# executing a command, so /proc/self is this original Bash process and still
# exposes its initial exec environment. `case` is reserved syntax and cannot be
# function-shadowed. Bash strips NUL separators, so search that snapshot only
# for names which Bash itself may already have consumed (`BASH_ENV`, exported
# functions, and shell-option variables). Check `ENV` as an exact live shell
# parameter instead: matching `*ENV=*` in the stripped snapshot also matches
# benign names such as `VIRTUAL_ENV`, while a self-erasing startup payload must
# have arrived through the separately detected `BASH_ENV`. The gate deliberately
# assigns no variables: BASH_ENV cannot defeat it by predeclaring names readonly.
# Group redirection suppresses only Bash's known ignored-NUL diagnostic.
if {
  case "$(</proc/self/environ)" in
    ''|*BASH_ENV=*|*BASH_FUNC_*|*BASHOPTS=*|*SHELLOPTS=*|\
*LD_AUDIT=*|*LD_PRELOAD=*|*DYLD_*=*) ((0)) ;;
    *)
      case "${ENV+x}" in
        x) ((0)) ;;
        *) ((1)) ;;
      esac
      ;;
  esac
} 2>/dev/null; then

# Before invoking the guard or any other external child, accept at most one
# reviewed loader control: a source LD_LIBRARY_PATH made exclusively of
# dedicated CUDA-library directories. The source path is used only to qualify
# objects that are subsequently copied into the run-local seal.
_scorecard_attest_initial_loader() {
  local loader_name
  local component
  local entry
  local basename
  local preload_contents=""
  local -a dyld_names
  local -a loader_names
  local -a loader_components

  dyld_names=("${!DYLD_@}")
  if (( ${#dyld_names[@]} > 0 )); then
    echo "ERROR: unreviewed dynamic-loader control is forbidden: ${dyld_names[*]}" >&2
    return 1
  fi
  loader_names=("${!LD_@}")
  for loader_name in "${loader_names[@]}"; do
    if [[ "$loader_name" != "LD_LIBRARY_PATH" ]]; then
      echo "ERROR: unreviewed dynamic-loader control is forbidden: $loader_name" >&2
      return 1
    fi
  done
  if [[ -L /etc/ld.so.preload ]]; then
    echo "ERROR: /etc/ld.so.preload must not be a symlink" >&2
    return 1
  fi
  if [[ -e /etc/ld.so.preload ]]; then
    preload_contents="$(</etc/ld.so.preload)" || {
      echo "ERROR: cannot attest /etc/ld.so.preload" >&2
      return 1
    }
    if [[ "$preload_contents" == *[![:space:]]* ]]; then
      echo "ERROR: non-empty /etc/ld.so.preload is forbidden" >&2
      return 1
    fi
  fi
  if [[ "${LD_LIBRARY_PATH+x}" != x ]]; then
    return 0
  fi
  case "$LD_LIBRARY_PATH" in
    ''|:*|*:|*::*)
      echo "ERROR: LD_LIBRARY_PATH contains an implicit current directory" >&2
      return 1
      ;;
  esac
  IFS=: builtin read -r -a loader_components <<< "$LD_LIBRARY_PATH"
  if (( ${#loader_components[@]} == 0 )); then
    echo "ERROR: LD_LIBRARY_PATH must contain absolute CUDA directories" >&2
    return 1
  fi
  for component in "${loader_components[@]}"; do
    if [[ -z "$component" || "$component" != /* || ! -d "$component" ]]; then
      echo "ERROR: LD_LIBRARY_PATH must contain existing absolute directories" >&2
      return 1
    fi
    for entry in "$component"/* "$component"/.[!.]* "$component"/..?*; do
      if [[ ! -e "$entry" && ! -L "$entry" ]]; then
        continue
      fi
      basename="${entry##*/}"
      if [[ ! "$basename" =~ ^lib((cuda|nvcuda|cublas|cublasLt|nvrtc)((32|64)(_[0-9]+(_[0-9]+)?)?)?|nvrtc-builtins|nvblas)[.]so([.][0-9]+)*$ \
        || ! -f "$entry" ]]; then
        echo "ERROR: unsafe LD_LIBRARY_PATH entry: $entry" >&2
        return 1
      fi
    done
  done
}

_scorecard_attest_initial_loader || exit 2

# Scorecard evidence must be collected inside the same host containment used by
# long-lived CUDA validation. Re-enter through the installed guard before any
# repository or output side effect, then independently attest the exact cgroup-v2
# policy and both sides of RLIMIT_AS in the child. These literal kernel paths are
# deliberately not configurable: tests patch only their private copied script.
readonly scorecard_proc_cgroup="/proc/self/cgroup"
readonly scorecard_proc_mountinfo="/proc/self/mountinfo"
readonly scorecard_proc_limits="/proc/self/limits"
readonly scorecard_cgroup_root="/sys/fs/cgroup"
# The competition-default memory lane is `gb10-80g` (64/80 GiB). A `wsl24-20g`
# research lane sized for a 24 GiB WSL2 VM is available only through this
# exact, provenance-recorded opt-in; every other value fails before guard
# entry. These are ATTESTED figures — the sweep refuses unless ny-build.slice
# carries them exactly — so a profile must name limits the host can really
# enforce. Install the matching unit from scripts/systemd/.
#
# A smaller profile yields a LOWER BOUND, not an equivalent result: less memory
# means fewer instances fit, so more time out. Do not compare across profiles.
if [[ "${NY_MEASURE_CONTAINMENT_PROFILE+x}" != "x" ]]; then
  NY_MEASURE_CONTAINMENT_PROFILE=gb10-80g
fi
case "$NY_MEASURE_CONTAINMENT_PROFILE" in
  gb10-80g)
    readonly scorecard_memory_high_bytes=68719476736
    readonly scorecard_memory_max_bytes=85899345920
    ;;
  wsl24-20g)
    readonly scorecard_memory_high_bytes=17179869184
    readonly scorecard_memory_max_bytes=21474836480
    ;;
  *)
    echo "ERROR: NY_MEASURE_CONTAINMENT_PROFILE must be exactly gb10-80g or wsl24-20g" >&2
    exit 2
    ;;
esac
# RLIMIT_AS caps VIRTUAL address space and is deliberately NOT sized with the
# profile. CUDA/ONNX Runtime reserve about 53.5 GiB before useful work no matter
# how little physical memory the host has. Shrinking it to a small profile's
# memory.max starves those reservations — measured here, a cersyve instance that
# returns `sat` in 1s under 80 GiB instead spends 106s and records `timeout`.
# An 80 GiB cap is also too close to real BaB: sealed CIFAR100 idx7641 reached
# 79.67 GiB of VA while the cgroup peaked at only 24.36 GiB. The finite 160 GiB
# ceiling leaves 80.33 GiB of VA headroom there, making the gb10 lane's cgroup
# remainder authoritative while still bounding runaway per-process mappings.
# It is the exact attested value for every physical containment profile.
readonly scorecard_rlimit_as_bytes=171798691840
export NY_MEASURE_CONTAINMENT_PROFILE
readonly scorecard_containment_profile="$NY_MEASURE_CONTAINMENT_PROFILE"
readonly scorecard_swap_max_bytes=8589934592
readonly scorecard_pids_max=4096

# The competition-default lane remains exactly 10 CPUs. A 20-CPU research lane
# is available only through this exact, provenance-recorded opt-in; empty,
# numeric aliases, whitespace, and every other value fail before guard entry.
if [[ "${NY_MEASURE_EXPECTED_CPUS+x}" != "x" ]]; then
  NY_MEASURE_EXPECTED_CPUS=10
fi
case "$NY_MEASURE_EXPECTED_CPUS" in
  10|20) ;;
  *)
    echo "ERROR: NY_MEASURE_EXPECTED_CPUS must be exactly 10 or 20" >&2
    exit 2
    ;;
esac
export NY_MEASURE_EXPECTED_CPUS
readonly scorecard_cpu_count="$NY_MEASURE_EXPECTED_CPUS"
readonly scorecard_cpu_period_us=100000
readonly scorecard_cpu_quota_us="$((scorecard_cpu_count * scorecard_cpu_period_us))"

guard_path="$(builtin type -P ny-safe-gpu-run 2>/dev/null)" || guard_path=""
if [[ -z "$guard_path" || ! -x "$guard_path" ]]; then
  echo "ERROR: ny-safe-gpu-run is required for scorecard measurement; refusing an unguarded run." >&2
  exit 2
fi

containment_error=""
_scorecard_read_control() {
  local control_path="$1"
  local control_value
  if [[ ! -f "$control_path" || ! -r "$control_path" ]]; then
    containment_error="missing readable cgroup control: $control_path"
    return 1
  fi
  control_value="$(<"$control_path")"
  if [[ -z "$control_value" || "$control_value" == *$'\n'* ]]; then
    containment_error="malformed cgroup control: $control_path"
    return 1
  fi
  scorecard_control_value="$control_value"
}

_scorecard_effective_scalar() {
  local control_name="$1"
  local scan_dir="$scorecard_current_cgroup_dir"
  local raw_value
  local numeric_value
  scorecard_effective_value=""
  while :; do
    if [[ -f "$scan_dir/$control_name" ]]; then
      _scorecard_read_control "$scan_dir/$control_name" || return 1
      raw_value="$scorecard_control_value"
      if [[ "$raw_value" != "max" ]]; then
        if [[ ! "$raw_value" =~ ^(0|[1-9][0-9]*)$ || ${#raw_value} -gt 18 ]]; then
          containment_error="malformed $control_name at $scan_dir"
          return 1
        fi
        numeric_value=$((10#$raw_value))
        if [[ -z "$scorecard_effective_value" ]] \
          || (( numeric_value < scorecard_effective_value )); then
          scorecard_effective_value="$numeric_value"
        fi
      fi
    fi
    if [[ "$scan_dir" == "$scorecard_cgroup_root_resolved" ]]; then
      break
    fi
    scan_dir="${scan_dir%/*}"
    [[ -n "$scan_dir" ]] || scan_dir="/"
  done
  if [[ -z "$scorecard_effective_value" ]]; then
    containment_error="no finite effective $control_name cgroup policy"
    return 1
  fi
}

_scorecard_effective_cpu() {
  local scan_dir="$scorecard_current_cgroup_dir"
  local quota
  local period
  local extra
  local numeric_quota
  local numeric_period
  local best_quota=""
  local best_period=""
  while :; do
    if [[ -f "$scan_dir/cpu.max" ]]; then
      _scorecard_read_control "$scan_dir/cpu.max" || return 1
      quota=""
      period=""
      extra=""
      IFS=' ' builtin read -r quota period extra <<< "$scorecard_control_value"
      if [[ -n "$extra" || ! "$period" =~ ^[1-9][0-9]*$ \
        || ${#period} -gt 9 ]]; then
        containment_error="malformed cpu.max at $scan_dir"
        return 1
      fi
      if [[ "$quota" != "max" ]]; then
        if [[ ! "$quota" =~ ^[1-9][0-9]*$ || ${#quota} -gt 9 ]]; then
          containment_error="malformed cpu.max at $scan_dir"
          return 1
        fi
        numeric_quota=$((10#$quota))
        numeric_period=$((10#$period))
        if [[ -z "$best_quota" ]] \
          || (( numeric_quota * best_period < best_quota * numeric_period )); then
          best_quota="$numeric_quota"
          best_period="$numeric_period"
        fi
      fi
    fi
    if [[ "$scan_dir" == "$scorecard_cgroup_root_resolved" ]]; then
      break
    fi
    scan_dir="${scan_dir%/*}"
    [[ -n "$scan_dir" ]] || scan_dir="/"
  done
  if [[ -z "$best_quota" ]] \
    || (( best_quota != scorecard_cpu_count * best_period )); then
    containment_error="effective cpu.max is not exactly ${scorecard_cpu_count} CPUs"
    return 1
  fi
}

_scorecard_attest_containment() {
  local cgroup_line
  local candidate
  local unified_count=0
  local mount_line
  local mount_count=0
  local separator
  local index
  local -a mount_fields
  local expected_slice
  local leaf
  local limit_line
  local limit_count=0
  local soft_limit
  local hard_limit
  local units

  if [[ ! -r "$scorecard_proc_cgroup" \
    || ! -r "$scorecard_proc_mountinfo" \
    || ! -r "$scorecard_proc_limits" \
    || ! -d "$scorecard_cgroup_root" ]]; then
    containment_error="required kernel containment interfaces are unavailable"
    return 1
  fi
  while IFS= builtin read -r cgroup_line || [[ -n "$cgroup_line" ]]; do
    case "$cgroup_line" in
      0::*)
        candidate="${cgroup_line#0::}"
        unified_count=$((unified_count + 1))
        scorecard_cgroup_path="$candidate"
        ;;
    esac
  done < "$scorecard_proc_cgroup"
  if (( unified_count != 1 )) \
    || [[ "$scorecard_cgroup_path" != /* \
      || "$scorecard_cgroup_path" == *"/../"* \
      || "$scorecard_cgroup_path" == *"/./"* \
      || "$scorecard_cgroup_path" == *"//"* ]]; then
    containment_error="cgroup-v2 membership is missing or malformed"
    return 1
  fi

  expected_slice="/user.slice/user-${UID}.slice/user@${UID}.service/ny.slice/ny-build.slice"
  if [[ "$scorecard_cgroup_path" != "$expected_slice/"* ]]; then
    containment_error="process is outside the exact ny-build.slice hierarchy"
    return 1
  fi
  leaf="${scorecard_cgroup_path#"$expected_slice/ny-safe-gpu-${UID}-"}"
  if [[ "$leaf" == "$scorecard_cgroup_path" \
    || ! "$leaf" =~ ^[0-9]+-[0-9]+[.]service$ ]]; then
    containment_error="process is not in an immediate ny-safe-gpu service cgroup"
    return 1
  fi

  while IFS= builtin read -r mount_line || [[ -n "$mount_line" ]]; do
    mount_fields=()
    IFS=' ' builtin read -r -a mount_fields <<< "$mount_line"
    separator=-1
    for ((index = 6; index < ${#mount_fields[@]}; index++)); do
      if [[ "${mount_fields[index]}" == "-" ]]; then
        separator="$index"
        break
      fi
    done
    if (( separator >= 0 && separator + 1 < ${#mount_fields[@]} )) \
      && [[ "${mount_fields[3]:-}" == "/" \
        && "${mount_fields[4]:-}" == "$scorecard_cgroup_root" \
        && "${mount_fields[separator + 1]:-}" == "cgroup2" ]]; then
      mount_count=$((mount_count + 1))
    fi
  done < "$scorecard_proc_mountinfo"
  if (( mount_count != 1 )); then
    containment_error="expected one root cgroup-v2 mount at $scorecard_cgroup_root"
    return 1
  fi

  scorecard_cgroup_root_resolved="$(
    CDPATH='' builtin cd -- "$scorecard_cgroup_root" && builtin pwd -P
  )" || {
    containment_error="cannot resolve cgroup-v2 mount"
    return 1
  }
  scorecard_current_cgroup_dir="$(
    CDPATH='' builtin cd -- "$scorecard_cgroup_root$scorecard_cgroup_path" \
      && builtin pwd -P
  )" || {
    containment_error="cannot resolve current process cgroup"
    return 1
  }
  scorecard_slice_cgroup_dir="$(
    CDPATH='' builtin cd -- "$scorecard_cgroup_root$expected_slice" \
      && builtin pwd -P
  )" || {
    containment_error="cannot resolve ny-build.slice cgroup"
    return 1
  }
  case "$scorecard_current_cgroup_dir/" in
    "$scorecard_slice_cgroup_dir"/*) ;;
    *)
      containment_error="resolved process cgroup escapes ny-build.slice"
      return 1
      ;;
  esac
  # The guard installs the reviewed limits on this exact transient service.
  # Keeping policy on the leaf makes every run self-contained instead of
  # depending on persistent host-wide ny-build.slice drop-ins. The effective
  # checks below still reject any tighter or malformed ancestor policy.
  scorecard_policy_cgroup_dir="$scorecard_current_cgroup_dir"

  _scorecard_read_control "$scorecard_policy_cgroup_dir/memory.high" || return 1
  [[ "$scorecard_control_value" == "$scorecard_memory_high_bytes" ]] || {
    containment_error="ny-safe-gpu service memory.high policy mismatch"
    return 1
  }
  _scorecard_read_control "$scorecard_policy_cgroup_dir/memory.max" || return 1
  [[ "$scorecard_control_value" == "$scorecard_memory_max_bytes" ]] || {
    containment_error="ny-safe-gpu service memory.max policy mismatch"
    return 1
  }
  _scorecard_read_control "$scorecard_policy_cgroup_dir/memory.swap.max" || return 1
  [[ "$scorecard_control_value" == "$scorecard_swap_max_bytes" ]] || {
    containment_error="ny-safe-gpu service memory.swap.max policy mismatch"
    return 1
  }
  _scorecard_read_control "$scorecard_policy_cgroup_dir/pids.max" || return 1
  [[ "$scorecard_control_value" == "$scorecard_pids_max" ]] || {
    containment_error="ny-safe-gpu service pids.max policy mismatch"
    return 1
  }
  _scorecard_read_control "$scorecard_policy_cgroup_dir/cpu.max" || return 1
  [[ "$scorecard_control_value" \
    == "${scorecard_cpu_quota_us} ${scorecard_cpu_period_us}" ]] || {
    containment_error="ny-safe-gpu service cpu.max policy mismatch"
    return 1
  }

  _scorecard_effective_scalar "memory.high" || return 1
  [[ "$scorecard_effective_value" == "$scorecard_memory_high_bytes" ]] || {
    containment_error="effective memory.high policy mismatch"
    return 1
  }
  _scorecard_effective_scalar "memory.max" || return 1
  [[ "$scorecard_effective_value" == "$scorecard_memory_max_bytes" ]] || {
    containment_error="effective memory.max policy mismatch"
    return 1
  }
  _scorecard_effective_scalar "memory.swap.max" || return 1
  [[ "$scorecard_effective_value" == "$scorecard_swap_max_bytes" ]] || {
    containment_error="effective memory.swap.max policy mismatch"
    return 1
  }
  _scorecard_effective_scalar "pids.max" || return 1
  [[ "$scorecard_effective_value" == "$scorecard_pids_max" ]] || {
    containment_error="effective pids.max policy mismatch"
    return 1
  }
  _scorecard_effective_cpu || return 1

  while IFS= builtin read -r limit_line || [[ -n "$limit_line" ]]; do
    if [[ "$limit_line" =~ ^Max[[:space:]]+address[[:space:]]+space[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]*$ ]]; then
      limit_count=$((limit_count + 1))
      soft_limit="${BASH_REMATCH[1]}"
      hard_limit="${BASH_REMATCH[2]}"
      units="${BASH_REMATCH[3]}"
    fi
  done < "$scorecard_proc_limits"
  if (( limit_count != 1 )) \
    || [[ "$soft_limit" != "$scorecard_rlimit_as_bytes" \
      || "$hard_limit" != "$scorecard_rlimit_as_bytes" \
      || "$units" != "bytes" ]]; then
    containment_error="soft/hard RLIMIT_AS policy is not exactly \
$scorecard_rlimit_as_bytes bytes"
    return 1
  fi
}

guard_attested=0
if _scorecard_attest_containment; then
  guard_attested=1
fi
if [[ "$guard_attested" != "1" ]]; then
  if [[ "${NY_MEASURE_SAFE_GPU_WRAPPED:-0}" = "1" ]]; then
    echo "ERROR: NY_MEASURE_SAFE_GPU_WRAPPED was set without complete containment attestation: $containment_error" >&2
    exit 2
  fi
  script_dir="$(
    CDPATH='' builtin cd -- "$(/usr/bin/dirname -- "$0")" && builtin pwd -P
  )" || exit 1
  script_path="$script_dir/$(/usr/bin/basename -- "$0")"
  export NY_MEASURE_SAFE_GPU_WRAPPED=1
  # Deliberately does NOT lower NY_GPU_VMEM_LIMIT_KIB with the profile. RLIMIT_AS
  # caps virtual address space, which CUDA/ONNX Runtime reserve independently of
  # charged memory; shrinking it to memory.max converts fast verdicts into
  # timeouts. The guard's finite 160 GiB ceiling is attested for every profile;
  # the profile-specific cgroup remains the physical-memory authority.
  builtin exec /bin/bash "$guard_path" /bin/bash "$script_path" "$@"
fi
# The private recursion marker is not solver launch authority and must not leak
# into the fail-closed NY_* provenance namespace.
builtin unset NY_MEASURE_SAFE_GPU_WRAPPED
# Desktop/session discovery variables are needed only by the outer user-systemd
# guard. They are not solver inputs: discard them after containment admission
# so Vulkan/WGPU display discovery and user-session services cannot perturb a
# score run.
for scorecard_session_variable in "${!XDG_@}"; do
  builtin unset "$scorecard_session_variable"
done
builtin unset DBUS_SESSION_BUS_ADDRESS DISPLAY WAYLAND_DISPLAY XAUTHORITY
# Locale search paths and per-category overrides can load host data or change
# parsing.  Bind one UTF-8 locale for both helpers and solver evidence.
for scorecard_locale_variable in "${!LC_@}"; do
  builtin unset "$scorecard_locale_variable"
done
builtin unset LANGUAGE LOCPATH GCONV_PATH
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
# Every later helper/solver command is either absolute or resolved through this
# host-qualified search path. In particular, caller-prepended `env`, `timeout`,
# Python, or Git shims cannot participate in score evidence. Preserve the
# invoking account's physical home only to locate its toolchain and optional AY
# executable; the scored process receives a separate empty HOME below.
scorecard_user_home_input="${HOME:-}"
case "$scorecard_user_home_input" in
  /*) ;;
  *) echo "ERROR: HOME must name an absolute existing directory" >&2; exit 2 ;;
esac
scorecard_user_home="$(
  CDPATH='' builtin cd -- "$scorecard_user_home_input" && builtin pwd -P
)" || {
  echo "ERROR: HOME must name an absolute existing directory" >&2
  exit 2
}
readonly scorecard_user_home
# Resolve the invoking Git installation once, then use its actual executable
# rather than a user-writable PATH wrapper.  The Python provenance layer hashes
# this exact file before the first repository query and revalidates it later.
if [[ -z "${NY_MEASURE_GIT_BIN:-}" ]]; then
  scorecard_git_selector="$(builtin type -P git 2>/dev/null)" \
    || scorecard_git_selector=""
  [[ -n "$scorecard_git_selector" && -x "$scorecard_git_selector" ]] || {
    echo "ERROR: a Git executable is required for scorecard provenance" >&2
    exit 1
  }
  scorecard_git_exec_path="$(
    /usr/bin/env -i HOME="$scorecard_user_home" PATH=/usr/bin:/bin \
      "$scorecard_git_selector" --exec-path
  )" || {
    echo "ERROR: cannot resolve the selected Git execution path" >&2
    exit 1
  }
  if [[ "$scorecard_git_exec_path" == /* \
    && -f "$scorecard_git_exec_path/git" \
    && -x "$scorecard_git_exec_path/git" ]]; then
    NY_MEASURE_GIT_BIN="$scorecard_git_exec_path/git"
  else
    NY_MEASURE_GIT_BIN="$scorecard_git_selector"
  fi
fi
case "$NY_MEASURE_GIT_BIN" in
  /*) ;;
  *) echo "ERROR: NY_MEASURE_GIT_BIN must name an absolute executable" >&2; exit 2 ;;
esac
[ -f "$NY_MEASURE_GIT_BIN" ] && [ -x "$NY_MEASURE_GIT_BIN" ] || {
  echo "ERROR: selected Git executable is unavailable: $NY_MEASURE_GIT_BIN" >&2
  exit 1
}
export NY_MEASURE_GIT_BIN
readonly scorecard_canonical_path="/usr/bin:/bin"
export PATH="$scorecard_canonical_path"

# Repo root: auto-derive from this script's location; override with NY_ROOT.
cd "${NY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 1
# An explicit binary lets guarded/private CARGO_TARGET_DIR builds enter the
# provenance seal directly, without copying or symlinking them into the shared
# repository target directory. Match the scored run_instance.sh boundary:
# explicit developer overrides are allowed but announced, while automatic
# target/release selection must carry a receipt binding the exact binary bytes
# to the current source identity. This check precedes start-manifest/output
# creation so a stale or receipt-less automatic binary cannot contaminate a
# measurement bank.
scorecard_explicit_bin=0
if [ -n "${NY_MEASURE_BIN:-}" ]; then
  scorecard_explicit_bin=1
  BIN="$NY_MEASURE_BIN"
  if [ ! -f "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "ERROR: explicit NY_MEASURE_BIN is not a regular executable: $BIN" >&2
    exit 1
  fi
  echo "NOTE: using explicit NY_MEASURE_BIN override; automatic NY provenance receipt validation is bypassed: $BIN" >&2
else
  BIN=target/release/ny
fi

if [ "$scorecard_explicit_bin" -eq 0 ]; then
  SCORECARD_RECEIPT_HELPER="$PWD/vnncomp_scripts/submission_binary_receipt.sh"
  if [ -L "$SCORECARD_RECEIPT_HELPER" ] || [ ! -f "$SCORECARD_RECEIPT_HELPER" ]; then
    echo "ERROR: NY receipt validator is missing: $SCORECARD_RECEIPT_HELPER" >&2
    exit 1
  fi
  # The scorecard has already resolved and authenticated its Git executable,
  # while the solver-facing canonical PATH intentionally omits user tools.
  # Expose only that Git executable's directory to the receipt helper so its
  # source-identity check uses the same selected implementation.
  if ! TMPDIR=/tmp \
      PATH="${NY_MEASURE_GIT_BIN%/*}:$scorecard_canonical_path" \
      bash "$SCORECARD_RECEIPT_HELPER" validate "$BIN" "$PWD" >&2; then
    echo "ERROR: refusing stale or unproven automatic NY binary $BIN." >&2
    echo "  Rebuild with './vnncomp_scripts/build_submission_binary.sh'." >&2
    echo "  Developers may set NY_MEASURE_BIN explicitly to opt out for a controlled A/B." >&2
    exit 1
  fi
fi
# Materialize the compile-time root-GEMM default before provenance capture so
# an unset performance knob is still explicit in every immutable measurement.
# Preserve any caller value verbatim: Rust handles explicit faer/ndarray and
# conservatively falls back for unknown or empty values.
if [ "${NY_ROOT_GEMM+x}" != x ]; then
  case "$(uname -s 2>/dev/null):$(uname -m 2>/dev/null)" in
    Linux:aarch64|Linux:arm64) export NY_ROOT_GEMM=faer ;;
    *) export NY_ROOT_GEMM=ndarray ;;
  esac
fi
# The explicit non-CUDA escape hatch is a CPU-debug mode, not permission to
# execute an unsealed CUDA stack. Make that routing decision exact and bind it
# into the start environment before provenance capture.
if [ "${NY_ALLOW_NONCUDA_MEASURE:-0}" = "1" ]; then
  export NY_NO_CUDA=1
fi
# SUBMISSION PARITY (#measure-submission-env-drift). The scored path is
# vnncomp_scripts/run_instance.sh; this script must export the same verifier
# environment or the sealed scorecard measures a DIFFERENT verifier than the
# one submitted. Today that is only OMP_NUM_THREADS=1. The former
# NY_MARGIN_ROW_CONV_BWD_BLOCKED=1 / NY_MARGIN_ROW_PARALLEL=1 exports were
# dropped from every copy of this environment: 1 has been the compiled default
# since 7b004fba (parallel frontier) and 2eaa6b13 (cache-blocked backward
# conv), so exporting =1 was a bit-exact no-op (MEASURED: margin_row/tests.rs
# asserts unset == "1" for both blocked_backward_enabled_from_env and
# margin_row_frontier_from_env). Exported before provenance capture so it is
# sealed into the manifest. The caller's value wins, so an A/B — including the
# =0 margin-row serial kill switches — is still possible.
[ "${OMP_NUM_THREADS+x}" = x ] || export OMP_NUM_THREADS=1
# Bind an external ay executable when available for the legacy ay-proc and
# exact-SMT lanes. The sat_relu CNF route and default MIP lane are linked
# in-process and do not depend on this discovery.
if [ -z "${NY_AY:-}" ]; then
  for c in \
    "$scorecard_user_home/ay/target/release/ay" \
    "$scorecard_user_home/.cargo/bin/ay" \
    "$scorecard_user_home/.local/bin/ay" \
    "$(command -v ay 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && { export NY_AY="$c"; break; }
  done
fi
# Benchmark root: newest local mirror first; override with NY_BROOT.
BROOT="${NY_BROOT:-benchmarks/vnncomp2026/benchmarks}"
[ -d "$BROOT" ] || BROOT=benchmarks/vnncomp2025/benchmarks
# The score ledger is corpus-specific. Resolve the selected benchmark tree
# physically, then prefer exact canonical repository roots and finally the two
# reviewed external-checkout layouts (.../vnncomp20YY/benchmarks). An unknown
# layout may still be useful for a research measurement, but it must never be
# compared with either competition year's checked-in bank by guesswork.
NORMALIZED_BROOT="$(
  CDPATH='' builtin cd -- "$BROOT" && builtin pwd -P
)" || {
  echo "ERROR: cannot resolve benchmark root: $BROOT" >&2
  exit 1
}
known_vnncomp2025_broot=""
known_vnncomp2026_broot=""
if [ -d "$PWD/benchmarks/vnncomp2025/benchmarks" ]; then
  known_vnncomp2025_broot="$(
    CDPATH='' builtin cd -- "$PWD/benchmarks/vnncomp2025/benchmarks" \
      && builtin pwd -P
  )" || exit 1
fi
if [ -d "$PWD/benchmarks/vnncomp2026/benchmarks" ]; then
  known_vnncomp2026_broot="$(
    CDPATH='' builtin cd -- "$PWD/benchmarks/vnncomp2026/benchmarks" \
      && builtin pwd -P
  )" || exit 1
fi
CHECKED_IN_BANK_DIR=""
BENCHMARK_CONFIG_YEAR_DIR=""
if [ -n "$known_vnncomp2026_broot" ] \
  && [ "$NORMALIZED_BROOT" = "$known_vnncomp2026_broot" ]; then
  CHECKED_IN_BANK_DIR="reports/measured-2026"
  BENCHMARK_CONFIG_YEAR_DIR="vnncomp26"
elif [ -n "$known_vnncomp2025_broot" ] \
  && [ "$NORMALIZED_BROOT" = "$known_vnncomp2025_broot" ]; then
  CHECKED_IN_BANK_DIR="reports/measured"
  BENCHMARK_CONFIG_YEAR_DIR="vnncomp25"
elif [ "${NORMALIZED_BROOT##*/}" = benchmarks ]; then
  benchmark_corpus_parent=${NORMALIZED_BROOT%/*}
  case "${benchmark_corpus_parent##*/}" in
    vnncomp2026)
      CHECKED_IN_BANK_DIR="reports/measured-2026"
      BENCHMARK_CONFIG_YEAR_DIR="vnncomp26"
      ;;
    vnncomp2025)
      CHECKED_IN_BANK_DIR="reports/measured"
      BENCHMARK_CONFIG_YEAR_DIR="vnncomp25"
      ;;
  esac
fi
CONFIGS_DIR="${NY_MEASURE_CONFIGS_DIR:-}"
# NY normally auto-discovers the repository configs tree from its executable.
# Make that implicit input explicit and select the exact competition year so a
# 2025 corpus cannot silently load a same-named preset from vnncomp26 merely
# because the newer directory exists. Unknown research corpora must name an
# already-scoped vnncompYY directory; a multi-year parent is ambiguous.
if [ -z "$CONFIGS_DIR" ] && [ -d "$PWD/configs" ]; then
  CONFIGS_DIR="$PWD/configs"
fi
PROVENANCE_CONFIG_ARGS=()
if [ -n "$CONFIGS_DIR" ]; then
  case "$CONFIGS_DIR" in
    /*) ;;
    *) echo "ERROR: NY_MEASURE_CONFIGS_DIR must be an absolute path: $CONFIGS_DIR" >&2; exit 2 ;;
  esac
  [ -d "$CONFIGS_DIR" ] || {
    echo "ERROR: NY_MEASURE_CONFIGS_DIR is not an existing directory: $CONFIGS_DIR" >&2
    exit 2
  }
  CONFIGS_DIR="$(
    CDPATH='' builtin cd -- "$CONFIGS_DIR" && builtin pwd -P
  )" || {
    echo "ERROR: cannot resolve NY_MEASURE_CONFIGS_DIR: $CONFIGS_DIR" >&2
    exit 2
  }
  configs_basename=${CONFIGS_DIR##*/}
  if [ -n "$BENCHMARK_CONFIG_YEAR_DIR" ]; then
    case "$configs_basename" in
      vnncomp[0-9][0-9])
        if [ "$configs_basename" != "$BENCHMARK_CONFIG_YEAR_DIR" ]; then
          echo "ERROR: benchmark corpus requires $BENCHMARK_CONFIG_YEAR_DIR, not $configs_basename" >&2
          exit 2
        fi
        ;;
      *)
        expected_configs_dir="$CONFIGS_DIR/$BENCHMARK_CONFIG_YEAR_DIR"
        [ -d "$expected_configs_dir" ] || {
          echo "ERROR: config parent has no exact $BENCHMARK_CONFIG_YEAR_DIR directory: $CONFIGS_DIR" >&2
          exit 2
        }
        CONFIGS_DIR="$(
          CDPATH='' builtin cd -- "$expected_configs_dir" && builtin pwd -P
        )" || {
          echo "ERROR: cannot resolve exact $BENCHMARK_CONFIG_YEAR_DIR config directory" >&2
          exit 2
        }
        if [ "${CONFIGS_DIR##*/}" != "$BENCHMARK_CONFIG_YEAR_DIR" ]; then
          echo "ERROR: exact $BENCHMARK_CONFIG_YEAR_DIR config directory resolves to ${CONFIGS_DIR##*/}" >&2
          exit 2
        fi
        ;;
    esac
  else
    case "$configs_basename" in
      vnncomp[0-9][0-9]) ;;
      *)
        echo "ERROR: an unknown benchmark corpus requires an explicitly scoped vnncompYY config directory" >&2
        exit 2
        ;;
    esac
  fi
  NY_MEASURE_CONFIGS_DIR="$CONFIGS_DIR"
  export NY_MEASURE_CONFIGS_DIR
  PROVENANCE_CONFIG_ARGS=(--configs-dir "$CONFIGS_DIR")
fi
RUN_ID="${NY_MEASURE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
case "$RUN_ID" in
  *[!A-Za-z0-9_.-]*|'') echo "ERROR: unsafe NY_MEASURE_RUN_ID: $RUN_ID" >&2; exit 2 ;;
esac
OUT="${NY_MEASURE_OUTPUT_DIR:-reports/measured-runs/$RUN_ID}"
[ -n "$OUT" ] || { echo "ERROR: NY_MEASURE_OUTPUT_DIR is empty" >&2; exit 2; }
ARTIFACT_ROOT="${NY_MEASURE_ARTIFACTS:-$OUT/artifacts}"
SCRATCH="${NY_SCRATCH:-${TMPDIR:-/tmp}/ny_measure_scratch/$RUN_ID}"
case "$SCRATCH" in
  /*) ;;
  *) echo "ERROR: NY_SCRATCH must resolve from an absolute path: $SCRATCH" >&2; exit 2 ;;
esac
if [[ -L "$SCRATCH" || ( -e "$SCRATCH" && ! -d "$SCRATCH" ) ]]; then
  echo "ERROR: scratch path must be a real directory, not a symlink or file: $SCRATCH" >&2
  exit 2
fi
mkdir -p "$SCRATCH" || {
  echo "ERROR: cannot create scratch directory: $SCRATCH" >&2
  exit 1
}
SCRATCH="$(
  CDPATH='' builtin cd -- "$SCRATCH" && builtin pwd -P
)" || {
  echo "ERROR: cannot resolve scratch directory: $SCRATCH" >&2
  exit 1
}
SCORECARD_HOME="$SCRATCH/home"
SCORECARD_TMPDIR="$SCRATCH/tmp"
if [[ -e "$SCORECARD_HOME" || -L "$SCORECARD_HOME" \
  || -e "$SCORECARD_TMPDIR" || -L "$SCORECARD_TMPDIR" ]]; then
  echo "ERROR: isolated HOME/TMPDIR must not already exist below scratch" >&2
  exit 2
fi
( umask 077 && mkdir "$SCORECARD_HOME" "$SCORECARD_TMPDIR" ) || {
  echo "ERROR: cannot exclusively create isolated HOME/TMPDIR below scratch" >&2
  exit 1
}
chmod 700 "$SCORECARD_HOME" "$SCORECARD_TMPDIR" || {
  echo "ERROR: cannot secure isolated HOME/TMPDIR" >&2
  exit 1
}
SCORECARD_PYCACHE="$SCORECARD_TMPDIR/pycache"
( umask 077 && mkdir "$SCORECARD_PYCACHE" ) || {
  echo "ERROR: cannot create isolated Python bytecode directory" >&2
  exit 1
}
chmod 700 "$SCORECARD_PYCACHE" || {
  echo "ERROR: cannot secure isolated Python bytecode directory" >&2
  exit 1
}
readonly -a SCORECARD_PYTHON=(
  /usr/bin/python3 -E -s -S -B -X "pycache_prefix=$SCORECARD_PYCACHE"
)
export HOME="$SCORECARD_HOME"
export TMPDIR="$SCORECARD_TMPDIR"
# Toolchain identity capture must resolve the repository's pinned rustup
# channel, not the freshly isolated HOME. This helper-only path is recorded;
# the scored solver still receives a minimal PATH and never invokes rustup.
scorecard_rustup_home_input="${RUSTUP_HOME:-$scorecard_user_home/.rustup}"
case "$scorecard_rustup_home_input" in
  /*) ;;
  *) echo "ERROR: RUSTUP_HOME must name an absolute existing directory" >&2; exit 2 ;;
esac
scorecard_rustup_home="$(
  CDPATH='' builtin cd -- "$scorecard_rustup_home_input" && builtin pwd -P
)" || {
  echo "ERROR: pinned rustup home is unavailable: $scorecard_rustup_home_input" >&2
  exit 1
}
readonly scorecard_rustup_home
export RUSTUP_HOME="$scorecard_rustup_home"
NY_MEASURE_RUSTUP_BIN="${NY_MEASURE_RUSTUP_BIN:-$scorecard_user_home/.cargo/bin/rustup}"
case "$NY_MEASURE_RUSTUP_BIN" in
  /*) ;;
  *) echo "ERROR: NY_MEASURE_RUSTUP_BIN must name an absolute executable" >&2; exit 2 ;;
esac
[ -f "$NY_MEASURE_RUSTUP_BIN" ] && [ -x "$NY_MEASURE_RUSTUP_BIN" ] || {
  echo "ERROR: pinned rustup executable is unavailable: $NY_MEASURE_RUSTUP_BIN" >&2
  exit 1
}
export NY_MEASURE_RUSTUP_BIN
RF="$SCRATCH/ny_vnncomp_result.txt"
LOGF="$SCRATCH/ny_vnncomp_output.log"
FLIGHTF="${RF}.flight.json"

# Likely-fast benchmarks first (small nets NY verifies quickly -> NY's core
# standing across many benchmarks becomes visible soonest), hard conv/GAN nets
# last. Override order with NY_MEASURE_CATS.
CATS="${NY_MEASURE_CATS:-cersyve tllverifybench_2023 collins_rul_cnn_2022 linearizenn_2024 dist_shift_2023 soundnessbench sat_relu acasxu_2023 metaroom_2023 nn4sys malbeware cgan_2023 cora_2024 cifar100_2024 tinyimagenet_2024 safenlp_2024}"
# TIMEOUT CAP — OPT-IN, because a default cap silently manufactures capability
# limits (#measure-cap-truncation). This defaulted to 120s and truncated every
# per-instance budget above it, so nn4sys rows budgeted 300-800s and cgan rows
# budgeted 900-1200s were measured at ~110s and banked as `timeout`. Those rows
# were then read as capability limits for weeks; a corrected per-instance audit
# on 2026-07-29 found 219 such rows corpus-wide and could not explain the uniform
# ~100-110s cap fingerprint until this line was found.
#
# 0 (the default) means NO CAP: every instance gets the official budget from
# field 3 of instances.csv, which is the only budget the competition awards
# points for. Set NY_MEASURE_CAP=<secs> deliberately for a fast LOWER-BOUND
# sweep, and read every resulting `timeout` as "not measured", never as
# "cannot solve" — the emitted CSV and the manifest both record the cap.
CAP="${NY_MEASURE_CAP:-0}"
case "$CAP" in
  *[!0-9]*|'') echo "ERROR: NY_MEASURE_CAP must be a non-negative integer: $CAP" >&2; exit 2 ;;
esac
WATCHDOG_GRACE=30
MAX_ROWS_PER_CATEGORY="${NY_MEASURE_MAX_ROWS_PER_CATEGORY:-0}"
case "$MAX_ROWS_PER_CATEGORY" in
  *[!0-9]*|'') echo "ERROR: NY_MEASURE_MAX_ROWS_PER_CATEGORY must be a positive integer when set" >&2; exit 2 ;;
esac
if [ -n "${NY_MEASURE_MAX_ROWS_PER_CATEGORY:-}" ] && [ "$MAX_ROWS_PER_CATEGORY" -eq 0 ]; then
  echo "ERROR: NY_MEASURE_MAX_ROWS_PER_CATEGORY must be greater than zero" >&2
  exit 2
fi
INSTANCE_INDEX="${NY_MEASURE_INSTANCE_INDEX:-0}"
case "$INSTANCE_INDEX" in
  *[!0-9]*|'') echo "ERROR: NY_MEASURE_INSTANCE_INDEX must be a positive integer when set" >&2; exit 2 ;;
esac
if [ -n "${NY_MEASURE_INSTANCE_INDEX:-}" ] && [ "$INSTANCE_INDEX" -eq 0 ]; then
  echo "ERROR: NY_MEASURE_INSTANCE_INDEX must be greater than zero" >&2
  exit 2
fi
VNNLIB_VERSION="${NY_MEASURE_VNNLIB_VERSION:-}"
case "$VNNLIB_VERSION" in
  ''|1.0|2.0) ;;
  *) echo "ERROR: NY_MEASURE_VNNLIB_VERSION must be 1.0 or 2.0" >&2; exit 2 ;;
esac

# Resolve the categories that can actually produce rows before sealing the
# start manifest. VNN-COMP 2026 has a smaller category set than the historical
# 2025 default above; carrying absent 2025 names into a 2026 start would either
# demand nonexistent presets or, worse, let an unrelated baseline participate
# in one cross-category predicate assignment. Use this same resolver again in
# the execution loop so any benchmark-tree change after start fails closed.
_scorecard_instances_for_category() {
  local selected_cat="$1"
  local search_root
  local search_depth
  if [ -n "$VNNLIB_VERSION" ]; then
    search_root="$BROOT/$selected_cat/$VNNLIB_VERSION"
    search_depth=1
    find "$search_root" -maxdepth "$search_depth" -type f \
      -name instances.csv 2>/dev/null | LC_ALL=C sort
  elif [ -f "$BROOT/$selected_cat/instances.csv" ]; then
    printf '%s\n' "$BROOT/$selected_cat/instances.csv"
  else
    search_root="$BROOT/$selected_cat"
    search_depth=2
    find "$search_root" -maxdepth "$search_depth" -type f \
      -name instances.csv 2>/dev/null | LC_ALL=C sort
  fi
}

ACTIVE_CATS=""
SEALED_CATEGORY_INSTANCE_PATHS="$SCRATCH/category-instance-paths.nul"
: > "$SEALED_CATEGORY_INSTANCE_PATHS" || {
  echo "ERROR: cannot seal category instance paths: $SEALED_CATEGORY_INSTANCE_PATHS" >&2
  exit 1
}
for selected_cat in $CATS; do
  selected_instances=$(_scorecard_instances_for_category "$selected_cat")
  selected_count=$(printf '%s\n' "$selected_instances" | \
    awk 'NF { count++ } END { print count + 0 }')
  if [ "$selected_count" -eq 0 ]; then
    if [ "$INSTANCE_INDEX" -gt 0 ] || [ -n "$VNNLIB_VERSION" ]; then
      echo "ERROR: selected category $selected_cat has no instances.csv" >&2
      exit 1
    fi
    echo "SKIP $selected_cat (no instances.csv)"
    continue
  fi
  if [ "$selected_count" -gt 1 ]; then
    echo "ERROR: $selected_cat has multiple instances.csv files; set NY_MEASURE_VNNLIB_VERSION to 1.0 or 2.0" >&2
    printf '%s\n' "$selected_instances" >&2
    exit 1
  fi
  printf '%s\0' "$selected_instances" >> "$SEALED_CATEGORY_INSTANCE_PATHS" || {
    echo "ERROR: cannot seal instances.csv path for $selected_cat" >&2
    exit 1
  }
  ACTIVE_CATS="${ACTIVE_CATS:+$ACTIVE_CATS }$selected_cat"
done
if [ -z "$ACTIVE_CATS" ]; then
  echo "ERROR: selected measurement has no runnable benchmark categories" >&2
  exit 1
fi
CATS="$ACTIVE_CATS"

# Mandatory current-evidence treatment-engagement contract. Every newly
# produced run names one scalar-v1 predicate or one same-family joint-v2
# predicate. Joint coordinates are comma-separated and must already be sorted:
#   NY_MEASURE_REQUIRE_FLIGHT_EVENT=ny-exec-v1.pgd-fast-gradient-parameter-engaged.NY_PGD_FAST_GRAD.0=ran
#   NY_MEASURE_REQUIRE_FLIGHT_EVENT=ny-exec-v2.exact-vjp-parameter-engaged.NY_PGD_VJP_K.52,NY_PGD_VJP_STEPS.250=ran
# The row archiver validates the predicate against exact lever delivery and a
# strict terminal runtime schema before writing immutable evidence or appending
# the CSV, so a dark/wrong-lane treatment cannot masquerade as a valid A/B arm.
REQUIRED_FLIGHT_EVENT="${NY_MEASURE_REQUIRE_FLIGHT_EVENT:-}"
case "$REQUIRED_FLIGHT_EVENT" in
  '')
    echo "ERROR: current measurement evidence requires NY_MEASURE_REQUIRE_FLIGHT_EVENT=VERSIONED_PREDICATE=ran" >&2
    exit 2
    ;;
  *=*)
    predicate_transport=${REQUIRED_FLIGHT_EVENT%%=*}
    predicate_status=${REQUIRED_FLIGHT_EVENT#*=}
    case "$predicate_transport" in
      ''|*[!A-Za-z0-9_.,-]*|*=*)
        echo "ERROR: NY_MEASURE_REQUIRE_FLIGHT_EVENT must be VERSIONED_PREDICATE=ran" >&2
        exit 2
        ;;
    esac
    case "$predicate_status" in
      ran) ;;
      *)
        echo "ERROR: NY_MEASURE_REQUIRE_FLIGHT_EVENT must be VERSIONED_PREDICATE=ran" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "ERROR: NY_MEASURE_REQUIRE_FLIGHT_EVENT must be VERSIONED_PREDICATE=ran" >&2
    exit 2
    ;;
esac

# Capture the immutable run evidence before any result row can be appended.
# This is deliberately fail-closed: a missing binary, unstable dirty worktree,
# unpinned dependency, unknown NY_* environment knob, or incomplete benchmark
# checkout leaves the measured bank untouched.
# Bash 3.2 treats an empty array expansion as unset under `set -u`. The +
# form below expands to zero arguments when no external config was selected.
START_MANIFEST=$("${SCORECARD_PYTHON[@]}" scripts/ny_measurement_provenance.py start \
  --repo-root . \
  --binary "$BIN" \
  --benchmark-root "$BROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --output-dir "$OUT" \
  --scratch-dir "$SCRATCH" \
  --result-file "$RF" \
  --solver-log-file "$LOGF" \
  --categories "$CATS" \
  --timeout-cap-seconds "$CAP" \
  --watchdog-grace-seconds "$WATCHDOG_GRACE" \
  --max-rows-per-category "$MAX_ROWS_PER_CATEGORY" \
  --instance-index "$INSTANCE_INDEX" \
  --vnnlib-version "$VNNLIB_VERSION" \
  ${PROVENANCE_CONFIG_ARGS[@]+"${PROVENANCE_CONFIG_ARGS[@]}"} \
  --sweep-script "$0") || {
    echo "ERROR: refusing to measure without complete immutable start provenance" >&2
    exit 1
  }
[ -f "$START_MANIFEST" ] || {
  echo "ERROR: provenance helper did not create its reported start manifest" >&2
  exit 1
}

# Recover the sealed loader directory without consulting the caller's library
# path. Once recovered, every subsequent provenance helper and GPU solver child
# runs with this single exact LD_LIBRARY_PATH component.
CUDA_RUNTIME_DIR=$(/usr/bin/env -u LD_LIBRARY_PATH -u LD_AUDIT -u LD_PRELOAD \
  -u DYLD_INSERT_LIBRARIES -u DYLD_FORCE_FLAT_NAMESPACE \
  -u DYLD_LIBRARY_PATH "${SCORECARD_PYTHON[@]}" -c \
  'import json, sys
start=json.load(open(sys.argv[1], encoding="utf-8"))
runtime=start["dependencies"]["cuda_runtime"]
print("" if runtime["status"] == "not_required" else runtime["sealed_execution"]["path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound CUDA runtime path" >&2
    exit 1
  }

PROVENANCE_ENV=(/usr/bin/env -i \
  HOME="$SCORECARD_HOME" \
  PATH="$PATH" \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  TMPDIR="$SCORECARD_TMPDIR")
if [ -n "$CUDA_RUNTIME_DIR" ]; then
  PROVENANCE_ENV+=(LD_LIBRARY_PATH="$CUDA_RUNTIME_DIR")
fi

# Read source-ledger state through the same empty environment as provenance.
# Literal pathspecs and an explicit repository root keep ambient Git variables,
# aliases, global config, and pathspec magic from suppressing a mismatch.
_scorecard_source_git() {
  "${PROVENANCE_ENV[@]}" GIT_CONFIG_NOSYSTEM=1 GIT_NO_REPLACE_OBJECTS=1 \
    "$NY_MEASURE_GIT_BIN" --literal-pathspecs -C "$PWD" \
      -c core.fileMode=true \
      -c core.symlinks=true \
      -c core.fsmonitor=false \
      -c core.untrackedCache=false \
      -c core.ignoreStat=false \
      -c core.trustctime=true \
      -c core.checkStat=default \
      "$@"
}

# Compare bytes, not Git's index stat cache: assume-unchanged and skip-worktree
# can make `git diff` report a modified bank as clean. ls-tree selects the HEAD
# blob with a literal pathspec; hash-object reads the regular worktree file
# directly and does not consult either index flag.
_scorecard_bank_matches_head() {
  local bank="$1" entry mode kind head_oid entry_path worktree_oid
  entry=$(_scorecard_source_git ls-tree HEAD -- "$bank") || return 1
  case "$entry" in
    ''|*$'\n'*) return 1 ;;
  esac
  IFS=$' \t' read -r mode kind head_oid entry_path <<< "$entry"
  case "$mode:$kind:$head_oid" in
    100644:blob:*|100755:blob:*) ;;
    *) return 1 ;;
  esac
  case "$head_oid" in
    ''|*[!0-9a-f]*) return 1 ;;
  esac
  worktree_oid=$(_scorecard_source_git hash-object --no-filters \
    -- "$bank") || return 1
  [ "$worktree_oid" = "$head_oid" ]
}

# Decode one checked-in bank occurrence with the standard library CSV parser.
# Historical banks have five-, six-, and seven-column rows, so the verdict is
# the unique protocol token after the three identity fields, not a fixed
# column. Structural errors, zero tokens, and multiple tokens are explicit
# advisory mismatches rather than opportunities to suppress the warning.
_scorecard_bank_lookup() {
  "${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c '
import csv
import sys

path, category, onnx, vnnlib, occurrence_text = sys.argv[1:]
protocol = {"sat", "unsat", "timeout", "unknown", "error"}
try:
    occurrence = int(occurrence_text)
    if occurrence <= 0:
        raise ValueError("non-positive occurrence")
    seen = 0
    with open(path, "r", encoding="utf-8", newline="") as source:
        for row in csv.reader(source, strict=True):
            if len(row) < 3:
                print("malformed:")
                raise SystemExit(0)
            if row[0] != category or row[1] != onnx or row[2] != vnnlib:
                continue
            seen += 1
            if seen != occurrence:
                continue
            tokens = []
            for field in row[3:]:
                normalized = field.strip().lower()
                if normalized in protocol:
                    tokens.append(normalized)
            if len(tokens) == 1:
                print("found:" + tokens[0])
            elif len(tokens) > 1:
                print("ambiguous:")
            else:
                print("malformed:")
            raise SystemExit(0)
    print("missing:")
except (OSError, UnicodeError, csv.Error, ValueError):
    print("malformed:")
' "$@"
}

# Parse the official list and existing run CSV once into a NUL-delimited stream.
# This preserves quoted commas (used by 2026 relational ONNX identities) while
# keeping Bash 3.2 compatibility and occurrence-aware resume. Newlines/NULs
# cannot be represented safely downstream, so reject them before any row runs.
_scorecard_prepare_instance_rows() {
  "${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c '
import csv
import sys

path, measurement_path, category = sys.argv[1:]
try:
    protocol = {"sat", "unsat", "timeout", "unknown", "error"}
    recorded = {}
    with open(measurement_path, "r", encoding="utf-8", newline="") as source:
        for row in csv.reader(source, strict=True):
            if len(row) not in (6, 7):
                raise ValueError("measurement row must have six or seven fields")
            if row[4].strip().lower() not in protocol:
                raise ValueError("measurement row has a non-protocol verdict")
            if row[0] == category:
                key = (row[1], row[2])
                recorded[key] = recorded.get(key, 0) + 1
    with open(path, "r", encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source, strict=True))
    for index, row in enumerate(rows, 1):
        if len(row) != 3:
            raise ValueError(f"record {index} has {len(row)} fields, expected 3")
        if any(any(control in field for control in ("\0", "\r", "\n")) for field in row):
            raise ValueError(f"record {index} contains an unsupported control byte")
    official_counts = {}
    for onnx, vnnlib, _timeout in rows:
        key = (onnx, vnnlib)
        official_counts[key] = official_counts.get(key, 0) + 1
    unsafe_resume = [
        key
        for key, count in official_counts.items()
        if count > 1 and recorded.get(key, 0) > 0
    ]
    if unsafe_resume:
        onnx, vnnlib = unsafe_resume[0]
        raise ValueError(
            "cannot safely resume duplicated official instance "
            f"({onnx!r}, {vnnlib!r}): existing measurement rows lack "
            "occurrence provenance; use a fresh output/evidence directory"
        )
    output = sys.stdout.buffer
    emit = lambda value: output.write(str(value).encode("utf-8") + b"\0")
    emit(len(rows))
    occurrences = {}
    for index, (onnx, vnnlib, timeout) in enumerate(rows, 1):
        key = (onnx, vnnlib)
        occurrences[key] = occurrences.get(key, 0) + 1
        for value in (
            index,
            onnx,
            vnnlib,
            timeout,
            occurrences[key],
            recorded.get(key, 0),
        ):
            emit(value)
except (OSError, UnicodeError, csv.Error, ValueError) as error:
    print(f"ERROR: malformed instance/measurement CSV: {error}", file=sys.stderr)
    raise SystemExit(1)
' "$@"
}

_scorecard_append_measurement_row() {
  "${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c '
import csv
import sys

path, *row = sys.argv[1:]
try:
    with open(path, "a", encoding="utf-8", newline="") as output:
        csv.writer(output, lineterminator="\n").writerow(row)
except (OSError, UnicodeError, csv.Error) as error:
    print(f"ERROR: cannot append measurement CSV {path}: {error}", file=sys.stderr)
    raise SystemExit(1)
' "$@"
}

# Reconstruct the solver's complete environment from the reviewed values bound
# into start.json, then execute it through `env -i`. This is the closure that a
# prefix denylist alone cannot provide: arbitrary Vulkan implicit-layer knobs,
# future runtime controls, and unrelated ambient variables never reach NY.
SOLVER_ENV_FILE="$SCRATCH/solver_environment.nul"
"${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
  'import json, sys
start=json.load(open(sys.argv[1], encoding="utf-8"))
solver=start["measurement"]["solver_environment"]
if solver.get("mode") != "env-i-reviewed-record-v1":
    raise SystemExit("unsupported solver environment mode")
values=solver.get("values")
if not isinstance(values, dict) or not values:
    raise SystemExit("missing solver environment values")
for key, value in sorted(values.items()):
    if not isinstance(key, str) or not isinstance(value, str) or "=" in key or "\0" in value:
        raise SystemExit("invalid solver environment entry")
    sys.stdout.buffer.write(f"{key}={value}".encode() + b"\0")' \
  "$START_MANIFEST" >"$SOLVER_ENV_FILE" || {
    echo "ERROR: could not reconstruct the provenance-bound solver environment" >&2
    exit 1
  }
SOLVER_ENV=(/usr/bin/env -i)
while IFS= read -r -d '' solver_environment_entry; do
  SOLVER_ENV+=("$solver_environment_entry")
done < "$SOLVER_ENV_FILE"
if (( ${#SOLVER_ENV[@]} <= 1 )); then
  echo "ERROR: provenance-bound solver environment is empty" >&2
  exit 1
fi

_verify_cuda_runtime() {
  local observed_runtime
  observed_runtime=$("${SOLVER_ENV[@]}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${SCORECARD_PYTHON[@]}" scripts/ny_measurement_provenance.py \
    verify-cuda-runtime --start-manifest "$START_MANIFEST" --fast) || {
      echo "ERROR: sealed CUDA runtime verification failed before child execution" >&2
      return 1
    }
  if [ "$observed_runtime" != "$CUDA_RUNTIME_DIR" ]; then
    echo "ERROR: sealed CUDA runtime path changed after start provenance" >&2
    return 1
  fi
}

# State the attested containment in the run log. Rows carry $RUN_ID and the
# start manifest records the profile, but this is what lets a reader tell at a
# glance which lane produced a CSV.
echo "containment profile: $scorecard_containment_profile" \
     "(memory.high=$scorecard_memory_high_bytes" \
     "memory.max=$scorecard_memory_max_bytes" \
     "cpus=$scorecard_cpu_count)"

_record_completion() {
  local rc=$?
  local completion_rc=0
  trap - EXIT
  trap '' HUP INT TERM
  "${SOLVER_ENV[@]}" \
    PYTHONDONTWRITEBYTECODE=1 \
    NY_MEASURE_GIT_BIN="$NY_MEASURE_GIT_BIN" \
    RUSTUP_HOME="$scorecard_rustup_home" \
    "${SCORECARD_PYTHON[@]}" scripts/ny_measurement_provenance.py complete \
    --start-manifest "$START_MANIFEST" --exit-status "$rc" >/dev/null \
    || completion_rc=$?
  if [ "$completion_rc" -ne 0 ]; then
    echo "ERROR: immutable measurement completion or integrity validation failed" >&2
    if [ "$rc" -eq 0 ]; then rc=1; fi
  fi
  exit "$rc"
}

_measurement_signal() {
  local signal="$1"
  local rc=1
  case "$signal" in
    HUP) rc=129 ;;
    INT) rc=130 ;;
    TERM) rc=143 ;;
  esac
  trap - "$signal"
  exit "$rc"
}

trap _record_completion EXIT
trap '_measurement_signal HUP' HUP
trap '_measurement_signal INT' INT
trap '_measurement_signal TERM' TERM

_verify_cuda_runtime || exit 1
# Execute only the run-local copies that were sealed into the start manifest.
# Originals remain bound too and are rehashed by the completion postflight.
BIN=$("${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["solver_binary"]["sealed_execution"]["path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound solver path" >&2
    exit 1
  }
[ -x "$BIN" ] || {
  echo "ERROR: provenance-bound solver is not executable: $BIN" >&2
  exit 1
}
# A plain build (no cuda/mip) silently measures 0/99 on cifar100/tinyimagenet
# (docs/CIFAR100_CUDA_BOOTSTRAP_BREAKTHROUGH_2026-07-15.md), so probe the sealed
# binary's compiled features and refuse to measure without both. Binaries too
# old to know --build-info exit nonzero and fail this gate the same way.
_verify_cuda_runtime || exit 1
BUILD_INFO=$("${SOLVER_ENV[@]}" "$BIN" --build-info 2>/dev/null) || BUILD_INFO=""
if [ "${NY_ALLOW_NONCUDA_MEASURE:-0}" != 1 ]; then
  case "$BUILD_INFO" in
    *"cuda=on"*"mip=on"*) ;;
    *)
      echo "ERROR: solver binary was not built with the cuda+mip competition features" >&2
      echo "  --build-info reported: ${BUILD_INFO:-<unavailable>}" >&2
      echo "  Rebuild with: targo --unverified build --release -p ny-cli --features cuda,mip" >&2
      echo "  (set NY_ALLOW_NONCUDA_MEASURE=1 only for CPU-track debugging)" >&2
      exit 2
      ;;
  esac
  # Compile-time features are insufficient: a missing cuBLAS runtime, broken
  # driver, or refused IEEE/device qualification otherwise falls through to
  # CPU after this gate. Trust only the sealed binary's exit status; self-check
  # wording is deliberately not part of measurement admission.
  _verify_cuda_runtime || exit 1
  if ! "${SOLVER_ENV[@]}" "$BIN" --cuda-selfcheck >/dev/null 2>&1; then
    echo "ERROR: sealed solver failed CUDA runtime/device qualification" >&2
    echo "  --cuda-selfcheck must exit successfully before GPU score measurement" >&2
    echo "  (set NY_ALLOW_NONCUDA_MEASURE=1 only for CPU-track debugging)" >&2
    exit 2
  fi
fi

# #binary-identity. --build-info answers "which FEATURES", which is necessary
# and not sufficient: 330 rows were poisoned on 2026-08-21 by a binary that was
# swapped at a stable path mid-run, and both sides answered `ny 0.3.0`. Capture
# the full canonical identity ONCE, state it in the run log, and re-check the
# solver's exact bytes before EVERY row.
#
# $BIN here is already the run-local sealed copy the start manifest bound, so a
# rebuild of target/release/ny cannot reach it. The per-row check is what
# catches everything else: a shared artifact root, a concurrent harness, a
# helpful cleanup script.
_verify_cuda_runtime || exit 1
SCORECARD_BUILD_ID=$("${SOLVER_ENV[@]}" "$BIN" --build-id 2>/dev/null | head -1) \
  || SCORECARD_BUILD_ID=""
case "$SCORECARD_BUILD_ID" in
  "NY_BUILD_ID_V1 "*) ;;
  *)
    echo "ERROR: sealed solver cannot state its own build identity" >&2
    echo "  $BIN does not support --build-id, so no row it measures could name" >&2
    echo "  the verifier that produced it — the 2026-08-21 failure mode." >&2
    echo "  Rebuild, then re-run:" >&2
    echo "    targo --unverified build --release -p ny-cli --features mip,cuda --bin ny" >&2
    echo "    scripts/ny_binary_identity.sh --expect-scored" >&2
    exit 2
    ;;
esac
echo "$SCORECARD_BUILD_ID"
# macOS ships shasum, not sha256sum; the scorecard already carries per-platform
# branches for `timeout`, so do the same here rather than assume coreutils.
_sha256_of() {
  local path="$1" output
  if command -v sha256sum >/dev/null 2>&1; then
    output=$(sha256sum -- "$path") || return 1
  elif command -v shasum >/dev/null 2>&1; then
    output=$(shasum -a 256 -- "$path") || return 1
  else
    return 1
  fi
  printf '%s\n' "${output%% *}"
}
SCORECARD_BIN_SHA256=$(_sha256_of "$BIN") || SCORECARD_BIN_SHA256=""
[ -n "$SCORECARD_BIN_SHA256" ] || {
  echo "ERROR: could not hash the sealed solver binary: $BIN" >&2
  exit 1
}

# Fail closed, per row, with both ids side by side. A warning would be read as
# noise; a bank of mixed provenance is worse than no bank.
_verify_binary_identity() {
  local observed
  observed=$(_sha256_of "$BIN") || observed=""
  if [ "$observed" != "$SCORECARD_BIN_SHA256" ]; then
    echo "ERROR: the solver binary changed underneath this measurement" >&2
    echo "  path:   $BIN" >&2
    echo "  sealed: $SCORECARD_BIN_SHA256" >&2
    echo "  found:  ${observed:-<unhashable>}" >&2
    echo "  sealed identity: $SCORECARD_BUILD_ID" >&2
    echo "  Rows before and after this point were measured by different" >&2
    echo "  verifiers and are not comparable. Refusing to continue." >&2
    return 1
  fi
}

SEALED_AY=$("${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
  'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8"))["dependencies"]["ay"]["sealed_executable"]; print("" if value is None else value["path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound AY path" >&2
    exit 1
  }
if [ -n "$SEALED_AY" ]; then
  export NY_AY="$SEALED_AY"
fi
CONFIGS_DIR=$("${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
  'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8"))["measurement"]["sealed_config_inputs"]; print("" if value is None else value["resolved_path"])' \
  "$START_MANIFEST") || {
    echo "ERROR: could not recover the provenance-bound config path" >&2
    exit 1
  }

mkdir -p "$OUT" || { echo "ERROR: cannot create output directory: $OUT" >&2; exit 1; }

# Portable per-instance watchdog: macOS ships no GNU `timeout` (the Linux original
# used it), so a bare `timeout ... ny` fails "command not found" and records every
# instance as timeout/0s. Prefer gtimeout/timeout when present; else background the
# run and hard-kill it after the budget. `env` is SIP-safe here (it does not strip
# DYLD when exec'ing our own non-protected binary), so only the watchdog needs help.
_run_to() {
  local secs="$1"; shift
  if [ -x /usr/bin/timeout ]; then
    /usr/bin/timeout "$secs" "$@"
    return $?
  fi
  if [ -x /opt/homebrew/bin/gtimeout ]; then
    /opt/homebrew/bin/gtimeout "$secs" "$@"
    return $?
  fi
  if [ -x /usr/local/bin/gtimeout ]; then
    /usr/local/bin/gtimeout "$secs" "$@"
    return $?
  fi
  "$@" & local pid=$!
  ( sleep "$secs"; kill -9 "$pid" 2>/dev/null ) & local wd=$!
  wait "$pid" 2>/dev/null; local rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  return $rc
}

exec 8< "$SEALED_CATEGORY_INSTANCE_PATHS" || {
  echo "ERROR: cannot reopen sealed category instance paths" >&2
  exit 1
}
for cat in $CATS; do
  IFS= read -r -d '' sealed_instances <&8 || {
    echo "ERROR: sealed instances.csv path is missing for $cat" >&2
    exit 1
  }
  instances=$(_scorecard_instances_for_category "$cat")
  instance_list_count=$(printf '%s\n' "$instances" | awk 'NF { count++ } END { print count + 0 }')
  if [ "$instance_list_count" -ne 1 ] || [ "$instances" != "$sealed_instances" ]; then
    echo "ERROR: $cat instances.csv selection changed after the measurement start was sealed" >&2
    echo "  sealed:   $sealed_instances" >&2
    echo "  observed: ${instances:-<missing>}" >&2
    exit 1
  fi
  inst="$sealed_instances"
  bdir=$(dirname "$inst")
  csv="$OUT/$cat.csv"
  touch "$csv" || { echo "ERROR: cannot create measurement CSV: $csv" >&2; exit 1; }
  # Do not inject the shared test_nano/test_tiny overhead probes. The official
  # result processor explicitly filters those harness checks before scoring.
  instance_rows_file="$SCRATCH/instances.csv.nul"
  if ! _scorecard_prepare_instance_rows \
    "$inst" "$csv" "$cat" > "$instance_rows_file"; then
    echo "ERROR: refusing malformed instance or measurement CSV for $cat" >&2
    exit 1
  fi
  exec 9< "$instance_rows_file" || {
    echo "ERROR: cannot open parsed instance rows: $instance_rows_file" >&2
    exit 1
  }
  IFS= read -r -d '' total <&9 || {
    echo "ERROR: parsed instance row stream has no count: $inst" >&2
    exit 1
  }
  if [ "$INSTANCE_INDEX" -gt "$total" ]; then
    echo "ERROR: selected instance index $INSTANCE_INDEX exceeds $cat row count $total" >&2
    exit 1
  fi
  echo "=== $cat ($total instances) $(date +%H:%M:%S) ==="
  run_rows=0
  while IFS= read -r -d '' n <&9 \
    && IFS= read -r -d '' onnx <&9 \
    && IFS= read -r -d '' vnnlib <&9 \
    && IFS= read -r -d '' timeout <&9 \
    && IFS= read -r -d '' wanted_occurrence <&9 \
    && IFS= read -r -d '' recorded_occurrence <&9; do
    if [ "$INSTANCE_INDEX" -gt 0 ] && [ "$n" -ne "$INSTANCE_INDEX" ]; then
      continue
    fi
    # Resume occurrence-by-occurrence. Some official instance lists contain an
    # identical decoded (onnx,vnnlib) pair more than once. The CSV-safe parser
    # supplied both occurrence counts so quoted commas cannot collapse or
    # invent scored rows.
    if [ "$recorded_occurrence" -ge "$wanted_occurrence" ]; then continue; fi
    if [ "$MAX_ROWS_PER_CATEGORY" -gt 0 ] && [ "$run_rows" -ge "$MAX_ROWS_PER_CATEGORY" ]; then
      break
    fi
    op="$bdir/$onnx"; vp="$bdir/$vnnlib"
    [ -f "$op" ] && [ -f "$vp" ] || {
      echo "ERROR: refusing to record an unarchived row with missing inputs: $cat row $n" >&2
      exit 1
    }
    ROW_BINDING=$("${PROVENANCE_ENV[@]}" PYTHONDONTWRITEBYTECODE=1 \
      "${SCORECARD_PYTHON[@]}" scripts/seal_ny_measurement_inputs.py \
      --artifact-root "$ARTIFACT_ROOT" \
      --run-id "$RUN_ID" \
      --category "$cat" \
      --instance-index "$n" \
      --onnx "$onnx" \
      --vnnlib "$vnnlib" \
      --onnx-file "$op" \
      --vnnlib-file "$vp" \
      --start-manifest "$START_MANIFEST") || {
        echo "ERROR: refusing to execute without immutable pre-run input binding" >&2
        exit 1
      }
    PREFLIGHT_MANIFEST=$("${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
      'import json, sys; print(json.loads(sys.argv[1])["preflight_manifest"])' \
      "$ROW_BINDING") || exit 1
    SEALED_OP=$("${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
      'import json, sys; print(json.loads(sys.argv[1])["onnx_file"])' \
      "$ROW_BINDING") || exit 1
    SEALED_VP=$("${PROVENANCE_ENV[@]}" "${SCORECARD_PYTHON[@]}" -c \
      'import json, sys; print(json.loads(sys.argv[1])["vnnlib_file"])' \
      "$ROW_BINDING") || exit 1
    to=${timeout%%[!0-9]*}; [ -z "$to" ] && to=100
    # Apply the cap ONLY when one was deliberately requested. CAP=0 is the default
    # and means "the official per-instance budget", which is the only budget the
    # competition awards points for. A capped row that comes back `timeout` has NOT
    # been shown to be beyond NY — it has merely not been measured.
    if [ "$CAP" -gt 0 ] && [ "$to" -gt "$CAP" ]; then
      to=$CAP
    fi
    : > "$RF" || { echo "ERROR: cannot clear result scratch file: $RF" >&2; exit 1; }
    : > "$LOGF" || { echo "ERROR: cannot clear solver log: $LOGF" >&2; exit 1; }
    # RF is reused across rows, so its adjacent best-effort flight sidecar must
    # not survive into the next row and be misattributed if that row is killed
    # before NY writes a new one. The archive helper validates category, budget,
    # and terminal verdict before embedding a present record in row metadata.
    rm -f "$FLIGHTF" || {
      echo "ERROR: cannot clear flight-record scratch file: $FLIGHTF" >&2
      exit 1
    }
    # #binary-identity: per ROW, never per manifest. This is the check the
    # 2026-08-21 re-sweep did not have. BEFORE `t0`, deliberately: the sealed
    # binary's hash is harness work, and charging ~0.07 s of it to the recorded
    # elapsed time would corrupt the very measurement it protects.
    _verify_binary_identity || exit 1
    t0=$SECONDS
    _verify_cuda_runtime || exit 1
    if [ -n "$CONFIGS_DIR" ]; then
      _run_to $((to+WATCHDOG_GRACE)) "${SOLVER_ENV[@]}" RUST_LOG=error \
        "$BIN" vnncomp v1 "$cat" "$SEALED_OP" "$SEALED_VP" "$RF" "$to" \
        --configs-dir "$CONFIGS_DIR" >"$LOGF" 2>&1
    else
      _run_to $((to+WATCHDOG_GRACE)) "${SOLVER_ENV[@]}" RUST_LOG=error \
        "$BIN" vnncomp v1 "$cat" "$SEALED_OP" "$SEALED_VP" "$RF" "$to" \
        >"$LOGF" 2>&1
    fi
    solver_rc=$?
    el=$((SECONDS-t0))
    res=$(head -1 "$RF" 2>/dev/null | tr -d '\r\n ' )
    [ -z "$res" ] && res=timeout
    # RF is reused on the next instance. Before any non-missing row is recorded,
    # preserve its complete raw bytes and bind them to the exact ONNX/VNN-LIB
    # hashes and start manifest. SAT remains stricter: the helper rejects it if
    # the raw result has no counterexample assignment.
    "${PROVENANCE_ENV[@]}" PYTHONDONTWRITEBYTECODE=1 \
      "${SCORECARD_PYTHON[@]}" scripts/archive_vnncomp_sat_result.py \
      --result-file "$RF" \
      --solver-log-file "$LOGF" \
      --artifact-root "$ARTIFACT_ROOT" \
      --run-id "$RUN_ID" \
      --category "$cat" \
      --instance-index "$n" \
      --onnx "$onnx" \
      --vnnlib "$vnnlib" \
      --onnx-file "$op" \
      --vnnlib-file "$vp" \
      --solver-verdict "$res" \
      --solver-exit-status "$solver_rc" \
      --timeout-seconds "$to" \
      --elapsed-seconds "$el" \
      --source-csv "$csv" \
      --start-manifest "$START_MANIFEST" \
      --preflight-manifest "$PREFLIGHT_MANIFEST" \
      --flight-file "$FLIGHTF" >/dev/null || {
        if [ "$res" = sat ]; then
          echo "ERROR: refusing to record SAT without its complete witness artifact" >&2
        else
          echo "ERROR: refusing to record result without complete immutable row evidence" >&2
        fi
        exit 1
      }
    _scorecard_append_measurement_row \
      "$csv" "$cat" "$onnx" "$vnnlib" 0 "$res" "$el" "$RUN_ID" || exit 1
    run_rows=$((run_rows+1))
    # A decided sealed run is not a score improvement until the checked-in
    # regular bank reflects it.  This is deliberately an ACTION REQUIRED
    # notice, not an automatic promotion: each corpus's reviewed evidence
    # workflow still owns the replay, official-truth, and validation needed to
    # mutate the score ledger. Match repeated rows occurrence-by-occurrence so
    # a decision for one duplicate cannot hide a stale timeout for another.
    checked_in_bank="${CHECKED_IN_BANK_DIR:+$CHECKED_IN_BANK_DIR/$cat.csv}"
    checked_in_verdict=""
    materialization_state="match"
    if [ "$res" = sat ] || [ "$res" = unsat ]; then
      if [ -z "$checked_in_bank" ]; then
        materialization_state="unknown_bank_route"
      elif [[ ! -e "$checked_in_bank" && ! -L "$checked_in_bank" ]]; then
        materialization_state="missing_bank"
      elif [[ -L "$checked_in_bank" || ! -f "$checked_in_bank" ]]; then
        materialization_state="nonregular_bank"
      elif ! _scorecard_source_git ls-files --error-unmatch \
        -- "$checked_in_bank" >/dev/null 2>&1; then
        materialization_state="untracked_bank"
      elif ! _scorecard_bank_matches_head "$checked_in_bank"; then
        materialization_state="dirty_bank"
      else
        checked_in_match=$(_scorecard_bank_lookup \
          "$checked_in_bank" "$cat" "$onnx" "$vnnlib" \
          "$wanted_occurrence") || checked_in_match="malformed:"
        case "$checked_in_match" in
          found:*) checked_in_verdict=${checked_in_match#found:} ;;
          missing:) materialization_state="missing_occurrence" ;;
          ambiguous:) materialization_state="ambiguous_verdict" ;;
          *) materialization_state="invalid_verdict" ;;
        esac
        if [ "$materialization_state" = match ]; then
          case "$checked_in_verdict" in
            "$res") ;;
            sat|unsat) materialization_state="verdict_conflict" ;;
            timeout|unknown|error) materialization_state="stale_verdict" ;;
            *) materialization_state="invalid_verdict" ;;
          esac
        fi
      fi
    fi
    if [ "$materialization_state" != match ]; then
      echo "================================================================" >&2
      if [ "$materialization_state" = verdict_conflict ]; then
        echo "ACTION REQUIRED: measured VNN-COMP verdict conflicts with checked-in bank" >&2
      else
        echo "ACTION REQUIRED: decided VNN-COMP result is not materialized" >&2
      fi
      echo "  instance: $cat row $n (pair occurrence $wanted_occurrence)" >&2
      echo "  measured: $res in $csv (run $RUN_ID)" >&2
      echo "  sealed artifacts: $ARTIFACT_ROOT" >&2
      case "$materialization_state" in
        unknown_bank_route)
          echo "  checked-in: no bank is mapped for benchmark root $NORMALIZED_BROOT" >&2
          ;;
        missing_bank)
          echo "  checked-in: $checked_in_bank is missing" >&2
          ;;
        nonregular_bank)
          echo "  checked-in: $checked_in_bank is not a regular non-symlink file" >&2
          ;;
        untracked_bank)
          echo "  checked-in: $checked_in_bank exists but is not tracked by Git" >&2
          ;;
        dirty_bank)
          echo "  checked-in: $checked_in_bank differs from HEAD" >&2
          ;;
        missing_occurrence)
          echo "  checked-in: $checked_in_bank has no row for pair occurrence $wanted_occurrence" >&2
          ;;
        invalid_verdict)
          echo "  checked-in: $checked_in_bank has a malformed row or no verdict token" >&2
          ;;
        ambiguous_verdict)
          echo "  checked-in: $checked_in_bank has multiple protocol verdict tokens" >&2
          ;;
        stale_verdict)
          echo "  checked-in: $checked_in_bank still says $checked_in_verdict" >&2
          ;;
        verdict_conflict)
          echo "  checked-in: $checked_in_bank says $checked_in_verdict, contradicting $res" >&2
          ;;
      esac
      if [ "$materialization_state" = verdict_conflict ]; then
        echo "  Investigate with the official checker and an independent sealed replay." >&2
        echo "  Do not promote either verdict until the conflict is resolved." >&2
      else
        if [ "$materialization_state" = invalid_verdict ] \
          || [ "$materialization_state" = ambiguous_verdict ]; then
          echo "  Audit or repair the malformed bank row before promotion." >&2
        fi
        case "$CHECKED_IN_BANK_DIR" in
          reports/measured)
            echo "  Validate and promote the sealed run with scripts/promote_regular_bank.py." >&2
            ;;
          reports/measured-2026)
            echo "  Validate the sealed run and use the reviewed 2026 bank evidence workflow." >&2
            ;;
          *)
            echo "  Validate the sealed run before selecting a corpus-specific bank workflow." >&2
            ;;
        esac
      fi
      echo "================================================================" >&2
    fi
  done
  exec 9<&-
  echo "  done $cat: $(wc -l < "$csv")/$total recorded"
done
if IFS= read -r -d '' unexpected_instances <&8; then
  echo "ERROR: sealed category instance paths contain an unexpected extra entry" >&2
  exit 1
fi
exec 8<&-
echo "=== ALL BENCHMARKS SWEPT $(date +%H:%M:%S) ==="
else
  /usr/bin/printf '%s\n' \
    "ERROR: scorecard measurement rejects BASH_ENV, ENV, shell-option injection, exported functions, and dynamic-loader injection." >&2
  /bin/sh -c 'exit 2'
fi
