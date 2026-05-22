# Road-Oriented Community Spatial Parameter Sweep

This directory keeps the calibration summary from the A/B/C/D spatial parameter sweep. The bulky per-variant generated instances, region pickles, and plots were removed after selecting the final A/B blend profile.

Kept artifacts:

- `spatial_sweep_summary.csv`
- `spatial_sweep_per_instance_metrics.csv`
- `configs/*.yaml`

| variant | depot mean | depot p90 | NN mean | community pairwise | bbox area | dispersion | active clusters | config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_compact_control | 18.201 | 30.443 | 0.088 | 1.235 | 10.161 | 0.889 | 14.4 | `configs/A_compact_control.yaml` |
| B_amazon_balanced | 18.136 | 30.832 | 0.111 | 1.821 | 18.360 | 1.317 | 14.2 | `configs/B_amazon_balanced.yaml` |
| C_corridor_elongated | 15.994 | 28.854 | 0.111 | 1.835 | 18.947 | 1.322 | 14.4 | `configs/C_corridor_elongated.yaml` |
| D_wide_community | 15.875 | 28.816 | 0.126 | 1.937 | 20.966 | 1.424 | 14.2 | `configs/D_wide_community.yaml` |

## Amazon Targets

| metric | target |
|---|---:|
| depot_stop_mean_km | 18.0526 |
| depot_stop_p50_km | 17.2825 |
| depot_stop_p90_km | 30.4566 |
| nearest_neighbor_mean_km | 0.0945 |
| community_pairwise_mean_km | 1.6223 |
| community_bbox_area_km2 | 12.7070 |
| community_dispersion_km | 1.2323 |
