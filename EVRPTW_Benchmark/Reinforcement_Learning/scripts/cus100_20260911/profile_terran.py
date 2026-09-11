#!/usr/bin/env python3
"""TERRAN uses the same monitored native-optimizer probe as the other models."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911 import profile_memory as probe
from EVRPTW_Benchmark.Reinforcement_Learning.scripts.cus100_20260911.launch import build_command

probe.METHODS['terran'] = 'TERRAN.train'

def command_for(args, output):
    dataset = Path(args.dataset_root)
    job = {
        'method': 'terran', 'source_kind': 'terran_synthetic' if args.representation == 'E' else 'stage2_road',
        'dataset_root': str(dataset), 'experiment_id': 'disposable_terran_probe',
        'train_module': 'EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train',
        'train_index': args.train_index or 'generation_plan/core/train/view_index.parquet',
        'validation_index': args.validation_index or 'generation_plan/core/val/view_index.parquet',
        'scale': 'Cus100', 'seed': 1234, 'physical_batch_size': args.batch, 'effective_batch_size': args.batch,
        'training_trajectory_count': args.n_traj, 'training_epochs': args.updates,
        'minimum_training_epochs': args.updates, 'training_rollout_steps': args.steps,
        'validation_rollout_steps': (3*args.steps+1)//2, 'validation_views': args.val_count,
        'validation_decode_type': 'sampling', 'validation_candidate_count': args.n_traj,
        'validation_checkpoints': 1, 'validation_every_epochs': args.updates,
        'protocol_id': 'cus100_20260911_disposable_memory_probe_v1',
        'objective_config_path': 'EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json',
        'reward_contract_config_path': args.reward_contract or 'EVRPTW_Benchmark/Reinforcement_Learning/configs/drl_reward_contract_energy_vehicle_v3.json',
        'optimizer_name': 'adamw', 'optimizer_weight_decay': .01,
        'training_representation': args.representation, 'terran_terminal_success_bonus': 0.,
        'ppo_step_chunk_size': int(os.environ.get('CUS100_TERRAN_CHUNK', '64')),
        'num_minibatches': 4,
        'extra_args': ['--pilot-mode', '--eval-batch-size', '1'],
    }
    return build_command(job, ROOT, output_root=output.parent, python=sys.executable, overrides={'output_dir': output})

probe.command_for = command_for
if __name__ == '__main__':
    probe.main()
