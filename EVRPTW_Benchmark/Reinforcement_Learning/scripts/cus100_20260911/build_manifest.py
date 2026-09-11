#!/usr/bin/env python3
"""Build only this round's ten jobs; uncalibrated jobs remain nonlaunchable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .launch import ASSIGNMENTS, HERE, MANIFEST, REPO, sha256

METHOD_MODULES = {
    "am_evrptw": "AM_EVRPTW", "drl_ts": "DRL_TS", "terran": "TERRAN",
    "rrnco": "RRNCO_EVRPTW", "evrptw_rl": "EVRPTW_RL",
}
TESTS = {"TR02": "EV14", "TR01": "EV10", "TR06": "EV30", "TR05": "EV26",
         "TR04": "EV22", "TR03": "EV18", "TR18": "EV78", "TR17": "EV74",
         "TR10": "EV46", "TR09": "EV42"}
CONFIG = "EVRPTW_Benchmark/Reinforcement_Learning/configs/"


def build_jobs(profiles=None):
    profiles = profiles or {}
    jobs = []
    for experiment, (server, gpu, method, representation) in ASSIGNMENTS.items():
        synthetic = representation == "E"
        job = {
            "schema": "cus100_seed1234_job_v1", "experiment_id": experiment,
            "job_id": experiment, "server": server, "gpu": gpu, "global_slot": gpu,
            "method": method, "scale": "Cus100", "seed": 1234, "kind": "train",
            "representation": representation, "training_representation": representation,
            "source_kind": "terran_synthetic" if synthetic else "stage2_road",
            "dataset_root": "EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911" if synthetic else
                "EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823",
            "train_index": "train/view_index.parquet" if synthetic else "generation_plan/core/train/view_index.parquet",
            "validation_index": "val/view_index.parquet" if synthetic else "generation_plan/core/val/view_index.parquet",
            "train_pool_views": 50000, "validation_views": 500,
            "train_module": f"EVRPTW_Benchmark.Reinforcement_Learning.{METHOD_MODULES[method]}.train",
            "protocol_id": "cus100_terran_synthetic_road_20260911_v1",
            "training_epochs": 10000, "minimum_training_epochs": 5000,
            "physical_batch_size": None, "effective_batch_size": None,
            "training_trajectory_count": 30, "training_rollout_steps": 240,
            "validation_rollout_steps": 360, "validation_candidate_count": 30,
            "validation_decode_type": "sampling", "validation_seed": 910001234,
            "validation_every_epochs": 100, "post_minimum_validation_every_epochs": 100,
            "validation_checkpoints": 100, "early_stop_patience_validations": 5,
            "early_stop_start_epoch": 5000, "final_validation_views": 0,
            "objective_config_path": CONFIG + "rivian_energy_vehicle_cost_v2.json",
            "reward_contract_config_path": "EVRPTW_Benchmark/results/cus100_20260911/artifacts/reward_synthetic_feasible4/reward_contract.json" if synthetic else CONFIG + "drl_reward_contract_energy_vehicle_v3.json",
            "optimizer_name": "adamw", "optimizer_weight_decay": 0.01,
            "candidate_selection": "verifier_feasible_then_min_total_cost_usd",
            "training_stream_path": None, "customer_exposure_budget": None,
            "training_id_stream_policy": "shared_source_seed_full_pool_shuffle_cycles_exact_length_prefixes",
            "budget_policy": "same_epochs_and_ntraj_method_specific_physical_equals_effective_batch",
            "warm_start_source_commit": "", "enabled": False, "calibration_status": "pending",
            "target_gpu_memory_gib": [9.5, 10.3], "memory_definition": "nvidia_smi_process_peak_gib",
            "evaluation_id_after_checkpoint_freeze": TESTS[experiment],
            "test_target": "Road Cus100/T1", "automatic_test": False,
            "extra_args": [],
        }
        if method == "terran":
            job.update(terran_config_path=CONFIG + "2080ti/terran_cus100.yaml",
                       num_minibatches=4, ppo_step_chunk_size=64, terran_terminal_success_bonus=0.0)
            job["terran_config_sha256"] = sha256(REPO / job["terran_config_path"])
        if method == "drl_ts":
            job.update(method_auxiliary_profile_path=CONFIG + "drl_ts_soft_auxiliary_v1.json",
                       soft_stage_end_epoch=2500,
                       extra_args=["--batches-per-epoch", "250"],
                       resolved_training_method_fields={
                           "rollout_baseline_schedule_source": "native_adapter",
                           "rollout_baseline_interval_optimizer_updates": 250,
                           "rollout_baseline_probe_source": "training_pool_only",
                       })
        if method == "evrptw_rl":
            job["method_auxiliary_profile_path"] = CONFIG + "evrptw_rl_station_auxiliary_v1.json"
        if method == "rrnco":
            job["extra_args"] = ["--graph-mode", "full", "--aft-mode", "stable",
                "--distance-sampling", "nearest", "--relation-temperature", "5",
                "--relation-chunk-size", "32", "--checkpoint-bias", "--reinforce-baseline",
                "leave_one_out", "--learning-rate", "0.0001"]
        job["objective_config"] = json.loads((REPO / job["objective_config_path"]).read_text())["objective"]
        profile = {**profiles.get(method, {}), **profiles.get(experiment, {})}
        job.update(profile)
        if job["physical_batch_size"] is not None:
            batch = int(job["physical_batch_size"])
            job["effective_batch_size"] = int(job["effective_batch_size"] or batch)
            job["customer_exposure_budget"] = job["training_epochs"] * job["effective_batch_size"] * 100
            job["minimum_customer_exposure_budget"] = job["minimum_training_epochs"] * job["effective_batch_size"] * 100
        for field in ("train_index", "validation_index"):
            source = REPO / job["dataset_root"] / job[field]
            if source.is_file():
                job[field + "_sha256"] = sha256(source)
        for field in ("objective_config_path", "reward_contract_config_path", "method_auxiliary_profile_path"):
            if job.get(field):
                job[field + "_sha256"] = sha256(REPO / job[field])
        reward_path = REPO / job["reward_contract_config_path"]
        if reward_path.is_file():
            reward = json.loads(reward_path.read_text())
            job.update(reward_contract_id=reward["contract_id"], reward_contract_sha256=reward["sha256"],
                       reward_objective_scale=reward["scales"]["Cus100"]["objective_scale"],
                       reward_failure_base=reward["scales"]["Cus100"]["failure_base"],
                       reward_unserved_coefficient=reward["scales"]["Cus100"]["unserved_coefficient"])
        jobs.append(job)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, help="JSON mapping methods and/or TR IDs to measured field overrides")
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text()) if args.profiles else {}
    jobs = build_jobs(profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs))
    print(json.dumps({"manifest": str(args.output), "jobs": len(jobs),
                      "launchable_jobs": sum(bool(job["enabled"]) for job in jobs)}))


if __name__ == "__main__":
    main()
