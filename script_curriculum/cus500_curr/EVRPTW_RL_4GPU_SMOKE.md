# EVRPTW-RL Road Cus500 four-GPU smoke (2026-09-22 UTC)

Source: Road Cus100 stage-1 best_overall epoch1700, val500/500,
cost482.87541062721596 USD. SHA256:
`d9afff1adbb1a5f79e1dafc68f7326f3d38d3cccf1656c9e26cc278046c5b497`.

Four RTX2080Ti, per-GPU24/global96 instances, 30 trajectories, mean aggregation,
D_time min-cost, caps600/700, activation checkpoint stride1.
Three EMA updates and an independent one-update greedy-baseline check completed.
Both smoke validations passed4/4; all training trajectories completed feasibly.
Peak allocated memory was about4.78GiB/rank. Sampled device memory was about
5.5–5.7GiB (including CUDA reservation/context and desktop). No OOM.
EMA updates took71.9/73.4/76.3s; greedy check took81.4s, excluding validation.
These small tests do not establish full500-instance validation quality or guarantee
all future random batches fit. Exact counters and artifact paths are in batch_profiles.json.

Production starts again from the staged Cus100 actor, resets optimizer/baseline/stream,
uses3000 new updates, full500-instance validation every100 updates, 30 candidates,
and the normal1000-step EMA then greedy-baseline schedule. Smoke weights are excluded.

Validation:49 curriculum launcher/source tests passed; shell syntax checks passed.
