# RRNCO-EV adaptation contract

This module is an experimental EVRPTW-DB adaptation of the road-relation
architecture from **Real-World Routing Problems: Learning to Solve VRPs with
Arbitrary and Asymmetric Road Networks** (`ai4co/real-routing-nco`). It is not
an official upstream task and must be reported as **RRNCO-EV**, not as the
unmodified RRNCO benchmark.

## Preserved architectural ideas

- separate row/outgoing and column/incoming terminal embeddings;
- inverse-distance sampled directed-distance expert (`k=25` by default);
- coordinate/distance contextual gates;
- row and column attention-free blocks;
- neural adaptive relation bias using road distance, duration and bearing;
- current-node directed road-relation bias in the decoder.

## EVRPTW-DB extensions

- node order remains depot, customers, then charging stations;
- node attributes are normalized coordinate, demand, time window, service
  time, charging-time ratio and one-hot node type;
- relation bias uses normalized directed distance `D`, running time `T`, energy
  `E`, and bearing. Energy is the explicit EV extension to the upstream
  distance/time/bearing relation channels;
- decoder state uses previous terminal, normalized load, battery use and time;
- charging, time-window, capacity, return-to-depot and per-route station-revisit
  feasibility come only from the canonical shared environment action mask;
- training cost, terminal failure penalty, route export and verification come
  from the same shared EVRPTW-DB protocol used by AM-EVRPTW.

Matrix normalization is inherited from `DRL_TS.rollout.normalized_edge_matrices`:

- `D / dataset training-pool distance scale`;
- `T / instance working horizon`;
- `E / vehicle battery capacity`.

## Controlled-comparison interpretation

The first Cus100 experiment deliberately gives RRNCO-EV and AM-EVRPTW the same
Stage-2 train/validation pools, seed, logical batch, training trajectory count,
rollout cap, validation instances, validation candidate budget, objective,
reward contract and verifier. RRNCO-EV uses AM's EMA/rollout-baseline schedule
to isolate the encoder/decoder representation more cleanly. Consequently this
is an **architecture adaptation under a matched EVRPTW optimization protocol**,
not a reproduction of upstream RRNCO's POMO training recipe.

A 500-update screening run is an integration and learning-signal check only. It
cannot establish convergence or paper-level superiority; any such claim needs
the frozen long-budget experiment and multiple seeds.

## Provenance

Upstream: <https://github.com/ai4co/real-routing-nco>

The local source consulted for the adaptation was commit
`823d510dadf4dd711730ec4fbf337c356a0de6ae`. The upstream MIT license is
included under `third_party/real_routing_nco/LICENSE`.

## Explicit v2 optimization experiment

`run_optimized_long_training.sh` opts into joint-logit stable AFT, deterministic
nearest outgoing/incoming distance summaries, actual relation temperature 5,
chunked/checkpointed relation computation, and a leave-one-out same-instance
REINFORCE baseline. The original model defaults and original long launcher keep
their legacy semantics. `GRAPH_MODE=full` and `GRAPH_MODE=node_only` use the same
backbone, objective, hard environment masks and hashed ordered training stream;
the ablation removes explicit road inputs from ANE, encoder and decoder together.

The new launcher defaults to 10000 maximum / 5000 minimum logical epochs,
500 validation instances with 100 candidates every 100 epochs before the minimum
and every 250 afterwards. Physical/effective batch defaults are Cus50 128/128
and Cus100 32/32; the latter still requires a final-implementation hardware gate.
`DRY_RUN=1` prints the planned command, and `PREPARE_ONLY=1` freezes and verifies
the stream and writes run provenance without training.

See the [v2 implementation and resource report](../reports/RRNCO_EV_V2_OPTIMIZATION_20260909_ZH.md)
for precise ablation boundaries, the limited CaliRoute inspiration, measured
memory, numerical tests, and command examples. These implementation and memory
gates do not establish benchmark superiority.
