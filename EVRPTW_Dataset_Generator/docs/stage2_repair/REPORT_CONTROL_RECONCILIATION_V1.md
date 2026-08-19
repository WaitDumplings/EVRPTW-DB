# Stage-2 report control and reconciliation v1

## Scope

This procedure applies only when all planned Stage-2 families were already
materialized and verified, but the runner later recorded `BrokenPipeError`
while writing stdout. It does not authorize generation, seed changes,
materialization, archive/restore, or the 7,500-family release run.

The approved v12 root is:

```text
EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v12_c3_1b1a099
```

Its pre-reconciliation state is
`generation_complete_but_report_control_failed`. Keep every family, matrix,
manifest, seed, runtime ledger, and the original RED report unchanged.

## Correct terminal ordering

The runner now uses this order:

```text
verification completed
-> finalize the small progress file
-> persist run_manifest.json
-> atomically persist the terminal run report
-> set terminal_report_committed=true in process state
-> best-effort concise stdout summary
```

After the report commit, `BrokenPipeError` is an observability warning only.
The default stdout contains status, three counts, and the report path. The
complete JSON stays in the report file unless
`--debug-print-full-report` is explicitly supplied.

## Live progress

`stage2_progress.json` is atomically replaced after each family transition and
verification. It contains:

```text
planned, completed, materialized, verified, rejected,
timed_out, aborted, unresolved, active_family_ids,
not_started, last_completed_family_id, updated_at
```

`stage2_progress_events.jsonl` is the compact append-only event stream.
`stage2_observability_warnings.jsonl` is independent from generation outcome.

## Approved reconciliation command

Run only from a clean, pushed `stage2-repair-candidate` commit:

```bash
cd /data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset_Generator
PILOT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v12_c3_1b1a099
PYTHONPATH=src /home/npg/miniconda3/envs/maojie/bin/python scripts/reconcile_stage2_run_report.py --output-root "$PILOT_ROOT" --repository-root /data/Maojie/ICLR/EVRPTW-DB --require-branch stage2-repair-candidate --expected-generation-commit 1b1a09972f8b3ced9e08bf0f556aa6adc476db0f --expected-family-count 140 --workers 12
```

The tool performs no SHA256 or file-hash computation. To enforce read-only
family behavior it compares path, size, nanosecond modification time, and mode
before and after verification.

## Hard gates

The command refuses canonical PASS if any condition fails:

- plan, original planned, original materialized, original verified, and
  published family ID sets are not exactly equal and unique;
- any set has a count other than 140;
- any full family verifier fails;
- any rejection, timeout, abort, unresolved, not-started, hard-stop, or
  remaining-process-group evidence is nonzero;
- the original exception is not `BrokenPipeError` after verification;
- the original generation commit differs from the explicitly approved commit;
- any family artifact changes during reconciliation.

On success it creates or updates only these report-control files:

```text
stage2_run_report.broken_pipe_original.json
reports/report_reconciliation_v1.json
stage2_run_report.json
```

The preserved original is byte-for-byte copied before reconciliation. The new
canonical report records both the generation and reconciliation commits,
`reconciled=true`, the original failure, 140/140/140 counts, and
`family_artifacts_modified=false`.

## Tests and next boundary

The regression suite covers closed stdout after PASS, the warning ledger,
concise default output, monotonic and atomic progress, PASS/refusal
reconciliation paths, original RED preservation, family immutability, and
orphan cleanup.

After reconciliation PASS, existing families may be read for Phase-1,
M1-M5, Q90, charging sensitivity, and pilot acceptance. Stop for reviewer
after those reports. Full generation and archive/restore remain unapproved.
