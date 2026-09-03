# EVRPTW-RL Adaptation Record

## Publication and implementation status

This baseline follows Bo Lin, Bissan Ghaddar, and Jatin Nathwani, *Deep
Reinforcement Learning for the Electric Vehicle Routing Problem With Time
Windows*, IEEE Transactions on Intelligent Transportation Systems 23(8),
11528--11538, DOI: <https://doi.org/10.1109/TITS.2021.3105232>.

No author-maintained implementation was found in the source search performed
for this benchmark.  This directory is therefore a paper-guided PyTorch
reimplementation, not official code and not a numerical reproduction of the
paper's tables.

## Published design retained

- shared one-dimensional projections of local and global state;
- iterative Structure2Vec graph embedding using node, global, neighboring-node,
  and edge-travel-time terms (paper Equation 6);
- context attention followed by next-node attention (Equations 7--11);
- an LSTM decoder carrying route history;
- stochastic policy-gradient training;
- an exponential moving-average baseline for the first 1,000 updates, followed
  by a greedy rollout baseline checked every 100 updates with a one-sided
  paired test;
- stochastic and greedy decoding;
- 128-dimensional default embeddings, Adam learning rate `1e-3`, batch size
  128, gradient clipping at 2.0, and the published station-visit penalty 0.3.

The paper denotes the number of Structure2Vec recursions by `p` but does not
report its experimental value.  The reimplementation therefore exposes it and
registers `p=3` as the default implementation choice.  This value must be
included in every run manifest.

## Benchmark adaptations

| Component | Published method | EVRPTW-B adapter |
|---|---|---|
| Objective-facing term | Negative normalized Euclidean route length | Directed `distance_matrix_km`, normalized only for policy-gradient scale |
| Edge input | Euclidean travel time | Canonical directed shortest-running-time matrix |
| Local node state | coordinates, TW, remaining demand | same core state plus service duration, station power, and node type required by the released task |
| Global state | time, battery, available finite EVs | time, battery, and remaining fraction of the nonbinding `N`-vehicle upper bound |
| Fleet assumption | fixed fleet | unlimited fleet; at most one useful route per customer makes `N` a nonbinding representation bound |
| Charging | constant linear full charging | station-dependent linear full charging from the shared environment |
| Feasibility | paper masking plus soft battery/fleet penalties | canonical hard safe-continuation mask and independent verifier |
| Data | fresh synthetic batch per update | sampling with replacement from the frozen training split |

The published training reward already uses route distance as its main term.
For physical instances, distance is divided by the per-instance median
depot-customer-depot repair distance before combining it with the published
station coefficient 0.3. This scale is an explicit benchmark adapter, not a
normalization stated in the paper. The station-visit penalty is retained as
method-specific auxiliary shaping.
The published excess-fleet penalty is structurally present but identically zero
under the benchmark's unlimited-fleet/nonbinding-`N` representation.  Negative
battery is prevented by the canonical hard mask, so its published soft penalty
is also identically zero.  A separately named incomplete-rollout penalty is
needed only to train on truncated trajectories; it never ranks benchmark
solutions.

## Fidelity boundary

The architecture and training logic above are claims about correspondence to
the publication.  Numerical equality with the paper is not claimed: the data,
fleet assumption, node attributes, physical charging model, and directed road
matrices differ, and the paper does not release all implementation details.
