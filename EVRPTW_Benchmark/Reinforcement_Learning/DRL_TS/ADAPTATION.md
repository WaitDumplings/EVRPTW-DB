# DRL-TS Adaptation Record

## Publication and code status

This baseline follows Jinbiao Chen, Huanhuan Huang, Zizhen Zhang, and Jiahai
Wang, *Deep Reinforcement Learning with Two-Stage Training Strategy for
Practical Electric Vehicle Routing Problem with Time Windows*, PPSN 2022,
pp. 356--370, DOI: <https://doi.org/10.1007/978-3-031-14714-2_25>.

The publisher page labels source code as available on request. No public
author-maintained repository was verified during implementation. This
directory is a paper-guided PyTorch reimplementation, not official code.

## Published design retained

- complete directed graph with asymmetric distance and travel-time features;
- node features for demand, time-window bounds, and node type;
- edge features for distance, time, and the `r`-nearest-neighbor indicator;
- two edge-aware graph-attention encoder layers by default;
- GRU route memory, dynamic time/cargo/energy context, multi-head glimpse, and
  tanh-clipped compatibility logits;
- Stage 1 with hard customer uniqueness and elementary depot/station rules but
  soft cargo, time-window, and energy constraints;
- Stage 2 with hard cargo, time-window, and safe-energy masking;
- total-distance objective plus normalized capacity, lateness, and energy
  violation penalties during Stage 1;
- REINFORCE with a statistically tested greedy rollout baseline;
- published defaults `r=10`, embedding size 128, two layers, eight heads,
  clipping constant 10, 200 epochs, and an equal Stage-1/Stage-2 split.

## Benchmark adaptations

| Component | Published DRL-TS | EVRPTW-B adapter |
|---|---|---|
| Objective-facing term | Total route distance on generated complete graphs | Sum of selected arcs in directed `distance_matrix_km` |
| Travel inputs | Generated asymmetric distance and time | Released directed-road distance and canonical running-time matrices |
| Energy | Scalar-consumption-derived transition cost | Released directed running-time-path energy matrix |
| Node input | demand, TW, node type | same inputs plus service duration and station power |
| Edge input | distance, time, nearest-neighbor indicator | same inputs plus explicit directed energy |
| Charging | constant station service time and full recharge | arrival-dependent, station-power full-charge duration |
| Stage-2 mask | paper feasibility rules | shared hard safe-continuation mask, including multi-station return paths |
| Fleet | paper maximum-vehicle representation | unlimited fleet; at most one route per customer is a nonbinding representation bound |
| Evaluation | paper instance generator and decoder | frozen Stage-2 splits, shared decoding budget, exported-route verifier |

The extra service, charging-power, and energy channels are input adapters; they
do not add encoder/decoder blocks or alter the two-stage training mechanism.
They expose distinctions in the benchmark that are absent from the paper's
fixed charging-time model.

## Reward adaptation

The paper already minimizes total travel distance, so no objective replacement
is necessary. We replace its generated-graph distance with the benchmark's
directed-road distance. The three auxiliary Stage-1 penalties are preserved.
Because their original normalized units cannot be reconstructed from physical
city instances, each violation is divided by its corresponding instance scale:

- excess volume by vehicle cargo capacity;
- lateness by the operating-horizon duration;
- energy deficit by battery capacity.

This keeps the published default penalty weights meaningful rather than adding
raw cubic centimetres, seconds, and kWh to kilometres. A separately named
incomplete-rollout guard supplies a finite signal only when a rollout reaches
the step limit; it never enters final ranking.

## Fidelity boundary

Architecture and curriculum correspondence are method-fidelity claims. We do
not claim numerical reproduction of the paper because its data generator,
fixed station service time, finite-fleet setting, and some implementation
details differ from the public benchmark contract.
