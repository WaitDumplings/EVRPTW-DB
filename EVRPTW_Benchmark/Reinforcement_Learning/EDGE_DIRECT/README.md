# Edge-DIRECT-H

Experimental homogeneous-fleet adapter based on Edge-DIRECT for the canonical
EVRPTW-DB Stage-2 environment. It is excluded from formal comparisons because
its current attention/normalization blocks do not match the paper. See
`ADAPTATION.md` for the exact mismatch, naming, and objective boundary.

Training:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.EDGE_DIRECT.train \
  --dataset-path EVRPTW_Dataset/Instances_v2/us_11city/generation_plan/views.jsonl \
  --family-root EVRPTW_Dataset/Instances_v2/us_11city/materialized/families \
  --scale Cus100 \
  --output-dir runs/edge_direct_h_cus100
```

Evaluation selects the verifier-passed, minimum directed-distance candidate:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.EDGE_DIRECT.eval \
  --dataset-path EVRPTW_Dataset/Instances_v2/us_11city/generation_plan/views.jsonl \
  --family-root EVRPTW_Dataset/Instances_v2/us_11city/materialized/families \
  --checkpoint runs/edge_direct_h_cus100/checkpoint_latest.pt \
  --scale Cus100 --output-dir results/edge_direct_h_cus100
```
