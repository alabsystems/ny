#!/bin/bash
# Install a scorecard containment profile into the calling user's systemd
# instance, then attest what the kernel ACTUALLY applied.
#
#   install-user-slices.sh [gb10-80g|wsl24-20g]     (default: gb10-80g)
#
# Nothing in-tree previously shipped this unit — measure_ny_scorecard.sh only
# verified the policy, leaving the slice itself to be created by hand. The
# profile chosen here must match NY_MEASURE_CONTAINMENT_PROFILE at sweep time
# or the sweep refuses to run.
#
# Idempotent. Requires a systemd user session (`systemctl --user` must work);
# on WSL2 that means systemd is enabled and PID 1.
set -euo pipefail

profile="${1:-gb10-80g}"
case "$profile" in
    gb10-80g|wsl24-20g) ;;
    *)
        echo "usage: $0 [gb10-80g|wsl24-20g]" >&2
        exit 2
        ;;
esac

src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$src_dir/ny-build.slice.$profile"
[ -f "$src" ] || { echo "ERROR: missing profile unit $src" >&2; exit 1; }

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"
install -m 0644 "$src" "$unit_dir/ny-build.slice"
echo "installed $profile -> $unit_dir/ny-build.slice"

systemctl --user daemon-reload
systemctl --user restart ny-build.slice

# Report the kernel's view, not the unit's request: a typo, an unsupported
# directive, or a missing controller delegation all surface here as `max`
# rather than the intended cap.
cg="/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/ny.slice/ny-build.slice"
if [ ! -d "$cg" ]; then
    echo "ERROR: slice cgroup not found at $cg" >&2
    exit 1
fi
echo "effective policy at $cg:"
for control in memory.high memory.max memory.swap.max pids.max cpu.max; do
    printf '  %-16s %s\n' "$control" "$(cat "$cg/$control" 2>/dev/null || echo '<missing>')"
done

echo
# WSL always exports WSLENV, and the sweep's injection gate substring-matches
# `*ENV=*` — a deliberate conservative false rejection. Left set, the sweep
# aborts with "rejects BASH_ENV, ENV, shell-option injection" before doing any
# work. Unsetting it is safe: WSLENV only configures Win32<->WSL env sharing.
if [ -n "${WSLENV+x}" ] || grep -qi microsoft /proc/version 2>/dev/null; then
    echo "NOTE: WSL detected — WSLENV trips the sweep's ENV-injection gate."
    echo "sweep with: env -u WSLENV NY_MEASURE_CONTAINMENT_PROFILE=$profile \\"
    echo "              bash scripts/measure_ny_scorecard.sh"
else
    echo "sweep with: NY_MEASURE_CONTAINMENT_PROFILE=$profile bash scripts/measure_ny_scorecard.sh"
fi
