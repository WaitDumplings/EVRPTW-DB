#!/usr/bin/env bash

# Resolve release data relative to the checked-out repository. The returned
# path is canonicalized for downstream Python processes, but no machine-local
# absolute path is stored in the repository.
resolve_evrptw_dataset_root() {
  local repo_root="${1:?repository root is required}"
  local requested="${EVRPTW_DATASET_ROOT:-}"
  local candidate

  if [[ -n "$requested" ]]; then
    if [[ "$requested" = /* ]]; then
      candidate="$requested"
    else
      candidate="$repo_root/$requested"
    fi
    realpath -m "$candidate"
    return
  fi

  local relative_candidates=(
    "EVRPTW_Dataset/Instances_v2/us_11city"
    "EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
  )
  for candidate in "${relative_candidates[@]}"; do
    candidate="$repo_root/$candidate"
    if [[ -f "$candidate/generation_plan/core/train/view_index.parquet" ]]; then
      realpath -m "$candidate"
      return
    fi
  done

  # Return the portable canonical location so callers can report a useful,
  # deterministic missing-data path.
  realpath -m "$repo_root/${relative_candidates[0]}"
}
