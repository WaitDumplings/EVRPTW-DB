"""Clean-revision provenance required before generating V2 artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class RevisionDisciplineError(RuntimeError):
    """The repository cannot identify one clean candidate implementation."""


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RevisionDisciplineError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def resolve_git_provenance(
    repo_root: str | Path,
    *,
    require_clean: bool = True,
    require_branch: str | None = None,
) -> dict[str, Any]:
    """Return the exact commit/branch, rejecting ambiguous generated state."""

    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD").lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RevisionDisciplineError(f"Invalid Git commit ID: {commit!r}")
    branch = _git(root, "branch", "--show-current")
    if not branch:
        raise RevisionDisciplineError("Artifact generation from detached HEAD is forbidden")
    if require_branch is not None and branch != require_branch:
        raise RevisionDisciplineError(
            f"Generation requires branch {require_branch!r}; current branch is {branch!r}"
        )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    clean = not bool(status)
    if require_clean and not clean:
        preview = status.splitlines()[:20]
        raise RevisionDisciplineError(
            "Artifact generation requires a clean working tree; dirty entries: "
            + repr(preview)
        )
    return {
        "schema": "evrptw_code_provenance_v1",
        "code_commit": commit,
        "code_branch": branch,
        "working_tree_clean": clean,
        "repository_root": str(root),
    }

