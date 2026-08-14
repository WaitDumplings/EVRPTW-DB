#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DESTINATION="${1:-$GENERATOR_DIR/data/sources/amazon-last-mile-2021}"
AWS_BIN="${AWS_BIN:-aws}"
SOURCE_ROOT="s3://amazon-last-mile-challenges/almrrc2021"
MODEL_SOURCE="$SOURCE_ROOT/almrrc2021-data-training/model_build_inputs"
MODEL_DESTINATION="$DESTINATION/model_build_inputs"

if ! command -v "$AWS_BIN" >/dev/null 2>&1; then
  echo "AWS CLI is required but was not found: $AWS_BIN" >&2
  echo "Install AWS CLI, then rerun this script. No AWS account is required." >&2
  exit 2
fi

mkdir -p "$MODEL_DESTINATION"

# Stage 2 needs only the three training-side model-build inputs below. Avoid
# downloading challenge scoring/apply outputs that are not part of the method.
"$AWS_BIN" s3 sync --no-sign-request \
  "$MODEL_SOURCE/" "$MODEL_DESTINATION/" \
  --exclude "*" \
  --include "route_data.json" \
  --include "package_data.json" \
  --include "travel_times.json"

# Preserve the upstream license and README beside the local source snapshot.
"$AWS_BIN" s3 cp --no-sign-request "$SOURCE_ROOT/License.txt" "$DESTINATION/License.txt"
"$AWS_BIN" s3 cp --no-sign-request "$SOURCE_ROOT/Readme.txt" "$DESTINATION/Readme.txt"

for required in route_data.json package_data.json travel_times.json; do
  if [[ ! -s "$MODEL_DESTINATION/$required" ]]; then
    echo "Amazon source download is incomplete: $MODEL_DESTINATION/$required" >&2
    exit 2
  fi
done
if [[ ! -s "$DESTINATION/License.txt" ]]; then
  echo "Amazon source license is missing: $DESTINATION/License.txt" >&2
  exit 2
fi

echo "Amazon model-build inputs: $MODEL_DESTINATION"
echo "Amazon license: $DESTINATION/License.txt"
echo "Use with: AMAZON_MODEL_BUILD_INPUTS=$MODEL_DESTINATION ./generate_instances.sh"
