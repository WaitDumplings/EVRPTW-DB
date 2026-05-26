# Amazon Depot-Day Scale Reference

Source files generated from `/data/aws/data/almrrc2021-data-training/model_build_inputs`:

- `/data/aws/amazon_daily_depot_analysis/depot_mother_board_summary.csv`
- `/data/aws/amazon_daily_depot_analysis/depot_daily_active_customers.csv`

The aggregation unit is `date x station_code`. A station/depot is interpreted as a region-level service territory graph; each date is interpreted as one active operating day.

## Station Mother-Board Size

Distinct dropoff customer IDs observed per station across the training horizon:

| rows | min | p25 | median | mean | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 7943 | 22300 | 35933 | 41949.3 | 51259 | 77174.8 | 119089 |

The generator default `mother_num_customers=5000` is the computational default used for benchmark training runs. For Amazon-scale station-territory studies, set this explicitly to 8000+; larger experiments can use 20k-50k for median-sized stations or 120k for DLA7-like upper-bound stress tests.

## Daily Active Customer Size

Unique active dropoff customers per `date x station_code`:

| rows | min | p25 | median | mean | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 521 | 56 | 633 | 1381 | 1719.0 | 2233 | 3633 | 8611 |

This supports fixed benchmark scales such as Cus50, Cus100, Cus1000, Cus1800, and Cus3000 as different operating-day demand levels sampled from a larger station territory. Cus8000-style studies should explicitly use an 8000+ service territory graph.

## Daily Route Count Diagnostic

Route count per `date x station_code` is reported as a diagnostic only. It is
not used as a required generated route count or vehicle count. In this generator,
vehicle count is produced by feasibility audit or solvers; Amazon route sequences
and realized route outcomes are excluded from generation rules.

Route count per `date x station_code`:

| rows | min | p25 | median | mean | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 521 | 1 | 5 | 10 | 11.7 | 15 | 23 | 56 |

Mean unique customers per historical route in each station-day has median 144.6 and mean 143.9. The default active-customer sampler treats this as a community-demand size proxy, uses a clipped lognormal model with median 141.6661 and sigma 0.1835, then maps a fixed `num_customers` to an active community count.
