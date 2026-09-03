from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from EVRPTW_Benchmark.Reinforcement_Learning.common.protocol_trainers import (
    paper_baseline_eval_due,
    paper_ema_baseline_due,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "drl_paper_alignment_v2.json"


def test_am_ema_warmup_matches_official_first_paper_epoch() -> None:
    args = SimpleNamespace(steps_per_epoch=2500, baseline_warmup_epochs=1)
    assert paper_ema_baseline_due("AM-EVRPTW", 0, args)
    assert paper_ema_baseline_due("AM-EVRPTW", 2499, args)
    assert not paper_ema_baseline_due("AM-EVRPTW", 2500, args)


def test_am_baseline_uses_paper_epoch_not_logical_update_name() -> None:
    args = SimpleNamespace(steps_per_epoch=2500)
    assert not paper_baseline_eval_due("AM-EVRPTW", 1, args)
    assert not paper_baseline_eval_due("AM-EVRPTW", 2499, args)
    assert paper_baseline_eval_due("AM-EVRPTW", 2500, args)
    assert paper_baseline_eval_due("AM-EVRPTW", 5000, args)


def test_evrptw_rl_baseline_uses_published_warmup_and_interval() -> None:
    args = SimpleNamespace(ema_warmup_steps=1000, baseline_eval_interval=100)
    assert paper_ema_baseline_due("EVRPTW-RL", 999, args)
    assert not paper_ema_baseline_due("EVRPTW-RL", 1000, args)
    assert not paper_baseline_eval_due("EVRPTW-RL", 1000, args)
    assert not paper_baseline_eval_due("EVRPTW-RL", 1099, args)
    assert paper_baseline_eval_due("EVRPTW-RL", 1100, args)
    assert paper_baseline_eval_due("EVRPTW-RL", 1200, args)


def test_paper_evidence_statuses_are_reported_without_overclaiming() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["formal_paper_fidelity_ready"] is False
    assert (
        report["models"]["drl_ts"]["status"]
        == "verified_paper_guided_adaptation"
    )
    assert report["models"]["drl_ts"]["blocking_issue"] is None
    assert (
        report["models"]["drl_ts"]["station_semantics"]
        == "stations_reusable_but_masked_after_depot_or_station"
    )
    assert (
        report["models"]["terran"]["status"]
        == "reference_code_verified_adaptation"
    )
    assert report["models"]["terran"]["blocking_issue"] is None
    assert report["models"]["edge_direct_h"]["status"] == "paper_fidelity_blocked"
    assert report["models"]["edge_direct_h"]["formal_method"] is False
