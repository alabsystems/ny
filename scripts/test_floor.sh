#!/bin/sh
# Run one `targo --unverified test` slice and require at least MIN passing tests.
#
#   sh scripts/test_floor.sh MIN <targo test arguments>...
#
# libtest exits 0 for a name filter that matches nothing, so the count is the
# check: the slice passes only when targo exits 0 and its first
# `test result: ok. N passed` line has N >= MIN. A compile error, a failing
# test, an empty selection or a short one fails. The whole log goes to stderr.
#
# publish/config.sh calls it as one simple command, so the release gate stays
# a plain `&&` chain that the publication engine runs step by step (a pipe in
# CHECK_CMD_DEFAULT is not a plain chain).
set -u

if [ "$#" -lt 2 ]; then
    echo "usage: sh scripts/test_floor.sh MIN <targo test arguments>..." >&2
    exit 2
fi
min=$1
shift
case $min in
    '' | *[!0-9]*)
        echo "test_floor: MIN must be a whole number, not '$min'" >&2
        exit 2
        ;;
esac

log=$(mktemp "${TMPDIR:-/tmp}/ny-test-floor.XXXXXX") || exit 2
rustup run trust targo --unverified test "$@" >"$log" 2>&1
rc=$?
cat "$log" >&2
passed=$(sed -n 's/^test result: ok\. \([0-9][0-9]*\) passed.*/\1/p' "$log" | head -n 1)
rm -f "$log"

if [ "$rc" -ne 0 ]; then
    echo "test_floor: targo test exited $rc" >&2
    exit "$rc"
fi
if [ -z "$passed" ]; then
    echo "test_floor: no 'test result: ok. N passed' line" >&2
    exit 1
fi
if [ "$passed" -lt "$min" ]; then
    echo "test_floor: $passed passed, below the floor of $min" >&2
    exit 1
fi
echo "test_floor: $passed passed (floor $min)" >&2
