# CLE cohort tables for the dataset and benchmark appendix

These tables are generated from the portable CLE manifests, not copied from logs.
Distance values of 200 m and 250 m are QA references, not deletion rules.
SCC-quarantined source locations remain in the provenance layer but are excluded
from the default benchmark candidate pool.

## City Scale

| city | road_nodes | directed_edges | latent_service_locations | default_service_pool | charger_candidates | depot_candidates |
| --- | --- | --- | --- | --- | --- | --- |
| Chicago | 35589 | 93678 | 412739 | 385521 | 289 | 153 |
| Dallas | 36401 | 92771 | 240608 | 204683 | 220 | 3627 |
| Fort Worth | 42801 | 110702 | 209659 | 203468 | 73 | 318 |
| Houston | 129583 | 312433 | 446536 | 426893 | 341 | 380 |
| Los Angeles | 52119 | 143055 | 624117 | 565610 | 1984 | 3654 |
| New York City | 55249 | 139085 | 399045 | 374417 | 801 | 160 |
| Philadelphia | 24994 | 61404 | 131347 | 111968 | 152 | 33 |
| Phoenix | 57324 | 145312 | 403118 | 386546 | 324 | 122 |
| San Antonio | 48713 | 124314 | 371856 | 362494 | 146 | 309 |
| San Diego | 52169 | 125743 | 267612 | 257939 | 817 | 112 |

## Road Connectivity

| city | buffer_km | node_coverage_pct | road_length_coverage_pct | largest_scc_node_pct | customer_scc_quarantine | depot_scc_quarantine | charger_scc_quarantine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chicago | 1.0000 | 99.9265 | 99.9489 | 99.3790 | 249 | 1 | 3 |
| Dallas | 0.0000 | 99.1664 | 99.5210 | 98.9808 | 943 | 14 | 4 |
| Fort Worth | 1.0000 | 99.7550 | 99.8900 | 99.4042 | 588 | 0 | 1 |
| Houston | 1.0000 | 99.3482 | 99.5616 | 99.3086 | 1308 | 0 | 2 |
| Los Angeles | 0.0000 | 99.3538 | 99.7002 | 94.1020 | 33942 | 501 | 69 |
| New York City | 0.0000 | 99.9204 | 99.9799 | 99.6742 | 577 | 0 | 0 |
| Philadelphia | 0.0000 | 99.9081 | 99.9556 | 99.5159 | 127 | 0 | 0 |
| Phoenix | 1.0000 | 99.6395 | 99.8310 | 98.6515 | 5897 | 0 | 11 |
| San Antonio | 1.0000 | 99.7389 | 99.9114 | 98.7313 | 2856 | 0 | 0 |
| San Diego | 5.0000 | 99.6363 | 99.6043 | 99.3061 | 506 | 8 | 15 |

## Service Locations

| city | nsi_residential_records | g1_locations | g2_pending_locations | road_distance_gt_200m | scc_quarantined_locations | house | manufactured_home | small_apt | medium_apt | large_apt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chicago | 483425 | 385756 | 26983 | 195 | 249 | 266456 | 2 | 107349 | 33652 | 5280 |
| Dallas | 279896 | 205529 | 35079 | 1396 | 943 | 213208 | 1356 | 13779 | 8520 | 3745 |
| Fort Worth | 266688 | 204049 | 5610 | 1380 | 588 | 192698 | 2217 | 9401 | 4205 | 1138 |
| Houston | 524078 | 428130 | 18406 | 3259 | 1308 | 388197 | 1864 | 31973 | 18391 | 6111 |
| Los Angeles | 655504 | 598626 | 25491 | 3770 | 33942 | 494969 | 921 | 84390 | 34399 | 9438 |
| New York City | 727990 | 374960 | 24085 | 1183 | 577 | 169657 | 0 | 151778 | 50896 | 26714 |
| Philadelphia | 470695 | 112082 | 19265 | 112 | 127 | 39581 | 0 | 57686 | 28233 | 5847 |
| Phoenix | 477277 | 392073 | 11045 | 4686 | 5897 | 356247 | 8803 | 24362 | 11392 | 2314 |
| San Antonio | 401243 | 365338 | 6518 | 13115 | 2856 | 340938 | 2892 | 16305 | 9255 | 2466 |
| San Diego | 283722 | 258433 | 9179 | 2326 | 506 | 223233 | 684 | 27242 | 14107 | 2346 |

## Facilities

| city | afdc_sites_inside_boundary | charger_exact_geometry | charger_address_corroborated | charger_uncorroborated | charger_distance_gt_250m | charger_candidate_pool | depot_retained | depot_strict | depot_optional | depot_candidate_pool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chicago | 346 | 5 | 285 | 56 | 0 | 289 | 154 | 19 | 134 | 153 |
| Dallas | 247 | 2 | 219 | 26 | 1 | 220 | 3641 | 5 | 3622 | 3627 |
| Fort Worth | 109 | 1 | 75 | 33 | 4 | 73 | 318 | 11 | 307 | 318 |
| Houston | 440 | 4 | 346 | 90 | 10 | 341 | 380 | 9 | 371 | 380 |
| Los Angeles | 2202 | 7 | 2065 | 130 | 3 | 1984 | 4155 | 6 | 3648 | 3654 |
| New York City | 884 | 7 | 798 | 79 | 2 | 801 | 160 | 13 | 147 | 160 |
| Philadelphia | 177 | 2 | 150 | 25 | 0 | 152 | 33 | 3 | 30 | 33 |
| Phoenix | 381 | 4 | 334 | 43 | 7 | 324 | 122 | 19 | 103 | 122 |
| San Antonio | 195 | 2 | 145 | 48 | 9 | 146 | 309 | 3 | 306 | 309 |
| San Diego | 953 | 18 | 815 | 120 | 12 | 817 | 120 | 5 | 107 | 112 |

## Speed Evidence

| city | directed_edges | observed_osm_maxspeed_pct | imputed_speed_edges | hpms_matched_edges | reference_profile |
| --- | --- | --- | --- | --- | --- |
| Chicago | 93678 | 10.2244 | 84100 | 0 | us_hpms_nrel_v1 |
| Dallas | 92771 | 19.6786 | 74515 | 0 | us_hpms_nrel_v1 |
| Fort Worth | 110702 | 13.6845 | 95553 | 0 | us_hpms_nrel_v1 |
| Houston | 312433 | 6.9637 | 290676 | 0 | us_hpms_nrel_v1 |
| Los Angeles | 143055 | 35.8086 | 91829 | 0 | us_hpms_nrel_v1 |
| New York City | 139085 | 30.9652 | 96017 | 0 | us_hpms_nrel_v1 |
| Philadelphia | 61404 | 15.0218 | 52180 | 0 | us_hpms_nrel_v1 |
| Phoenix | 145312 | 87.1945 | 18608 | 0 | us_hpms_nrel_v1 |
| San Antonio | 124314 | 11.5948 | 109900 | 0 | us_hpms_nrel_v1 |
| San Diego | 125743 | 15.2772 | 106533 | 0 | us_hpms_nrel_v1 |

## Validation Status

| city | technical_verification_passed | portable_package_verified | release_eligible | release_blocker_count | blocked_gates |
| --- | --- | --- | --- | --- | --- |
| Chicago | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| Dallas | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| Fort Worth | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| Houston | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| Los Angeles | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| New York City | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| Philadelphia | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| Phoenix | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| San Antonio | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
| San Diego | True | True | False | 6 | microsoft_nsi_geometry;customer_road_access_review;depot_release;charging_coordinate_validation;charging_release;delivery_communities |
