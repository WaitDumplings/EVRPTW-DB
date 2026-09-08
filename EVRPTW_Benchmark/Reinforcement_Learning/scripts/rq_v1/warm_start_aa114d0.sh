#!/usr/bin/env bash
set -euo pipefail

SERVER_SCRIPT_DIR="${SERVER_SCRIPT_DIR:?SERVER_SCRIPT_DIR must be set by the server wrapper}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SERVER_SCRIPT_DIR/jobs_warm_start_aa114d0.jsonl"
EXPECTED_COMMIT="aa114d06995cdd35429bcc793bee4cff14590eb5"
LAUNCHER_ID="warm-aa114d0"

export DRL_MANIFEST="$MANIFEST"
source "$SCRIPT_DIR/server_env.sh"

usage() {
  cat >&2 <<'EOF'
usage: warm_start_aa114d0.sh preflight|start|status|logs|resume [launcher arguments]

preflight  verify every exact-job source checkpoint without launching
start   fail-closed checkpoint preflight, then start a new weights-only run
status  report jobs selected by the aa114d0 warm-start manifest
logs    follow the dedicated launcher log
resume  recover an interrupted run created by this same launcher/revision
EOF
  exit 2
}

preflight_checkpoints() {
  python - "$MANIFEST" "$EVRPTW_OUTPUT_ROOT" "$EXPECTED_COMMIT" <<'PYCODE'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[2])
expected_commit = sys.argv[3]
if not manifest.is_file():
    raise SystemExit(f"missing special warm-start manifest: {manifest}")

rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"special warm-start manifest is empty: {manifest}")

missing = []
for row in rows:
    if row.get("warm_start_source_commit") != expected_commit:
        raise SystemExit(
            f"unexpected warm-start commit for {row.get('job_id')}: "
            f"{row.get('warm_start_source_commit')}"
        )
    if row.get("warm_start_scope") != "exact_job_only":
        raise SystemExit(f"warm-start scope is not exact_job_only: {row.get('job_id')}")
    if row.get("warm_start_missing_policy") != "error":
        raise SystemExit(f"warm-start missing policy is not fail-closed: {row.get('job_id')}")
    candidate = (
        output_root
        / str(row["representation"])
        / str(row["condition"])
        / str(row["method"])
        / str(row["scale"])
        / f"seed_{int(row['seed'])}"
        / expected_commit
        / str(row.get("warm_start_checkpoint_name", "best.ckpt"))
    )
    if not candidate.is_file():
        missing.append((str(row["job_id"]), candidate))

if missing:
    print("The aa114d0 weights-only warm-start is blocked; exact checkpoints are missing:", file=sys.stderr)
    for job_id, path in missing:
        print(f"  {job_id}: {path}", file=sys.stderr)
    raise SystemExit(4)

print(f"aa114d0 warm-start preflight passed: {len(rows)} exact-job checkpoint(s)")
PYCODE
}

ACTION="${1:-}"
[[ -n "$ACTION" ]] || usage
shift
case "$ACTION" in
  preflight)
    preflight_checkpoints
    ;;
  start)
    preflight_checkpoints
    exec bash "$SCRIPT_DIR/start_server.sh" full --launcher-id "$LAUNCHER_ID" "$@"
    ;;
  status)
    exec bash "$SCRIPT_DIR/run_server.sh" status --skip-gpu-preflight --launcher-id "$LAUNCHER_ID" "$@"
    ;;
  logs)
    LOG_DIR="$EVRPTW_OUTPUT_ROOT/launcher_logs/$DRL_SERVER_ID/launchers/$LAUNCHER_ID"
    [[ -f "$LOG_DIR/current.log.path" ]] || {
      echo "No aa114d0 warm-start launcher log yet: $LOG_DIR" >&2
      exit 2
    }
    exec tail -n 100 -f "$(cat "$LOG_DIR/current.log.path")"
    ;;
  resume)
    exec bash "$SCRIPT_DIR/start_server.sh" resume --launcher-id "$LAUNCHER_ID" "$@"
    ;;
  *) usage ;;
esac
