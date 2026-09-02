# Edge-DIRECT Source and Adaptation Assessment

## Source checked

- Arash Mozhdehi, Mahdi Mohammadizadeh, and Xin Wang, *Edge-DIRECT: A Deep
  Reinforcement Learning-based Method for Solving Heterogeneous Electric
  Vehicle Routing Problem with Time Window Constraints*, Canadian AI 2024.
- Paper: <https://arxiv.org/abs/2407.01615>
- Conference PDF:
  <https://assets.pubpub.org/day222wb/XinWang-01716776157945.pdf>

No public author-maintained code repository was found in the source search.

## Published components relevant to implementation

1. A directed time-window graph connects `i` to `j` when some departure time in
   node `i`'s window reaches `j` within node `j`'s window using `t_ij`.
2. A GAT embeds this time-window graph.
3. A second edge-enhanced attention encoder uses node features and directed
   travel-time/energy edge features.
4. A vehicle decoder selects one vehicle from a finite heterogeneous fleet.
5. A node decoder selects the next node conditional on that vehicle's location,
   remaining energy, and remaining cargo.
6. Training uses REINFORCE with a greedy rollout baseline. The paper's
   objective-facing feedback is total travel time; a benchmark implementation
   would replace it with total directed-road distance while retaining the
   architecture and other state features.

## Homogeneous-fleet resolution

EVRPTW-B's canonical task has a homogeneous fleet and no hard fleet-size limit.
Consequently, the published heterogeneous vehicle-selection distribution has
no identifiable choice: all available vehicles have the same static features,
and a sequential route-construction environment starts a fresh vehicle at each
depot return. This is not a reader/mask-only mismatch; it removes one of the
paper's three main architectural contributions.

The reviewer-safe choices were:

- retain a one-class vehicle decoder and label the implementation explicitly as
  the homogeneous special case; or
- remove the vehicle decoder and label the result as an Edge-DIRECT ablation,
  not Edge-DIRECT itself.

The first choice is now frozen and implemented as `EDGE_DIRECT/`: the learned
vehicle-context module is retained with one admissible homogeneous class and
zero categorical log-probability. The method is named **Edge-DIRECT-H** and is
not represented as a full reproduction of heterogeneous vehicle selection.
Its objective-facing term is the requested canonical directed-road distance.
