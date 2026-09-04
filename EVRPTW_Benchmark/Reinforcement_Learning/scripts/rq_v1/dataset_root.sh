#!/usr/bin/env bash

# Resolve release data below the archive-restore destination, itself expressed
# relative to the checked-out repository. The returned path is canonicalized
# at runtime, but no machine-local absolute path is stored in the repository.
resolve_evrptw_dataset_root() {
  local repo_root="${1:?repository root is required}"
  local requested="${EVRPTW_DATASET_ROOT:-}"
  local restore_root="${EVRPTW_RESTORE_ROOT:-../../../evrptw_runtime}"
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

  if [[ "$restore_root" != /* ]]; then
    restore_root="$repo_root/$restore_root"
  fi
  restore_root="$(realpath -m "$restore_root")"

  local relative_candidates=(
    "EVRPTW_Dataset/Instances_v2/us_11city"
    "EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
  )
  local search_roots=("$restore_root" "$repo_root")
  local search_root
  for search_root in "${search_roots[@]}"; do
    for candidate in "${relative_candidates[@]}"; do
      candidate="$search_root/$candidate"
      if [[ -f "$candidate/generation_plan/core/train/view_index.parquet" ]]; then
        realpath -m "$candidate"
        return
      fi
    done
  done

  # Return the portable canonical location so callers can report a useful,
  # deterministic missing-data path.
  realpath -m "$restore_root/${relative_candidates[0]}"
}
