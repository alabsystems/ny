# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Session setup for the transitional repository-tool suite (`pytest tests`).

Git's repository-local environment is removed before any test runs.

Many tests build throwaway fixture repositories with `git init`, `git config`
and `git commit` in a temporary directory. Git resolves the repository from
GIT_DIR before it looks at the working directory, and git itself exports
GIT_DIR to the commands it runs from a linked worktree: hooks, `git bisect
run`, `git rebase -x`. Under any of those, every fixture command acted on the
ENCLOSING repository instead. Measured 2026-10-04 by bisecting this suite with
`git bisect run` in a linked worktree of the shared checkout: the fixture
`git init` set `core.bare = true` and `git config` set `user.name = NY Test`
in the shared `.git/config` (every worktree of the clone then refused `git
status`, and every commit was authored by the fixture identity), and the
fixture's `git commit -m fixture` landed on the worktree's HEAD. No test needs
an inherited repository: each one names its own with `cwd` or `-C`.

The names are git's own list (`git rev-parse --local-env-vars`, git 2.50) plus
the `-c` pairs and the push-quarantine path, which carry configuration or
objects from the outer invocation into child git processes.
"""

from __future__ import annotations

import os

GIT_REPOSITORY_LOCAL_ENV = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
    "GIT_QUARANTINE_PATH",
)

for _name in list(os.environ):
    if _name in GIT_REPOSITORY_LOCAL_ENV or _name.startswith(
        ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
    ):
        del os.environ[_name]
