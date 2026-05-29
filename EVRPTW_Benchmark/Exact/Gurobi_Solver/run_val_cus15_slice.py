from __future__ import annotations

import argparse
from pathlib import Path

from run_gurobi import main as run_gurobi_main


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a restart-friendly val/Cus15 Gurobi slice. Defaults to val_Cus15_000750 through val_Cus15_000999."
    )
    parser.add_argument("--start_index", type=int, default=750, help="Inclusive instance suffix start. Default: 750.")
    parser.add_argument("--end_index", type=int, default=1000, help="Exclusive instance suffix end. Default: 1000.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker processes. Default: 16.")
    parser.add_argument("--time_limit_s", type=float, default=7200.0, help="Per-instance Gurobi time limit. Default: 7200.")
    parser.add_argument("--save_path", default=str(REPO_ROOT / "EVRPTW_Benchmark/results/dataset_v1/val/Gurobi_Solver"))
    parser.add_argument("--reference_save_path", default=str(REPO_ROOT / "EVRPTW_Dataset/dataset_v1/reference_solutions"))
    parser.add_argument("--dataset_path", default=str(REPO_ROOT / "EVRPTW_Dataset/dataset_v1/dataset/val"))
    parser.add_argument("--checkpoints_s", default="60,300,900,3600,7200")
    parser.add_argument("--cs_copies", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no_skip_completed", action="store_true", help="Re-solve completed rows instead of resuming.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gurobi_args = [
        "--dataset_path", args.dataset_path,
        "--save_path", args.save_path,
        "--reference_save_path", args.reference_save_path,
        "--reference_split", "val",
        "--scales", "Cus15",
        "--start_index", str(args.start_index),
        "--end_index", str(args.end_index),
        "--workers", str(args.workers),
        "--threads", str(args.threads),
        "--cs_copies", str(args.cs_copies),
        "--time_limit_s", str(args.time_limit_s),
        "--checkpoints_s", args.checkpoints_s,
    ]
    if not args.no_skip_completed:
        gurobi_args.append("--skip_completed")
    if args.verbose:
        gurobi_args.append("--verbose")

    run_gurobi_main(gurobi_args)


if __name__ == "__main__":
    main()
