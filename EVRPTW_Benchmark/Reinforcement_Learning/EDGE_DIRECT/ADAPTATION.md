# Edge-DIRECT-H Adaptation Record

## Published method and current status

This implementation is based on Mozhdehi, Mohammadizadeh, and Wang,
*Edge-DIRECT* (Canadian AI 2024, arXiv:2407.01615). The directed time-window
graph, two encoder stages, one-class vehicle context, node decoder,
autoregressive construction, and greedy-baseline REINFORCE are represented.

It is not currently paper-faithful. The paper specifies additive GAT attention,
linear projection followed by BatchNorm for initial node/edge embeddings, and
BatchNorm in encoder residual blocks. The current code uses scaled dot-product
attention and LayerNorm in those blocks. No verified public author code was
found, so this discrepancy cannot be resolved by source comparison.

## Benchmark adaptations and blocking boundary

The paper solves a finite heterogeneous-fleet problem. EVRPTW-DB defines an
unlimited homogeneous fleet and starts a fresh identical vehicle whenever a
route returns to the depot. Therefore vehicle identity is not identifiable.
The vehicle decoder is retained as a learned context module, but its choice set
has one homogeneous class and its categorical log-probability is exactly zero.
The method is consequently named **Edge-DIRECT-H**, not full Edge-DIRECT.

The paper's objective-facing reward is cumulative travel time. The benchmark
requires `min dist`, so both REINFORCE cost and final candidate selection use
the sum of canonical directed `distance_matrix_km` arcs. Canonical running time
still governs time windows and route duration; canonical path energy governs
battery feasibility; station-specific power governs full charging.

All final routes are replayed by the independent exact-model route verifier.
An incomplete-rollout penalty is training-only and never participates in final
solution ranking.

## Scale note

Both published graph encoders operate on dense directed node pairs, requiring
quadratic memory. Large-scale training therefore needs an explicit memory gate
or a separately reviewed sparse approximation. Edge-DIRECT-H remains excluded
from the frozen formal model set until the known encoder mismatch is corrected
and reviewed.
