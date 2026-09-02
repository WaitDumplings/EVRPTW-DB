#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="${SCRIPT_DIR}/papers"
REPO_DIR="${SCRIPT_DIR}/repos"

mkdir -p "${PAPER_DIR}" "${REPO_DIR}"

download_pdf() {
  local url="$1"
  local output="$2"
  if [[ -s "${output}" ]]; then
    printf 'reuse: %s\n' "${output}"
    return
  fi
  printf 'download: %s\n' "${output}"
  curl --fail --location --retry 3 --retry-delay 2 \
    --user-agent "Mozilla/5.0 EVRPTW-DB research audit" \
    --output "${output}.partial" "${url}"
  if [[ "$(LC_ALL=C head -c 4 "${output}.partial")" != "%PDF" ]]; then
    rm -f "${output}.partial"
    printf 'error: source did not return a PDF: %s\n' "${url}" >&2
    exit 1
  fi
  mv "${output}.partial" "${output}"
}

download_pdf_optional() {
  local url="$1"
  local output="$2"
  local label="$3"
  if [[ -s "${output}" ]]; then
    printf 'reuse: %s\n' "${output}"
    return
  fi
  printf 'download (optional): %s\n' "${output}"
  if ! curl --fail --location --retry 3 --retry-delay 2 \
    --user-agent "Mozilla/5.0 EVRPTW-DB research audit" \
    --output "${output}.partial" "${url}"; then
    rm -f "${output}.partial"
    printf 'warning: %s was not publicly downloadable; supply its local source path\n' "${label}" >&2
    return
  fi
  if [[ "$(LC_ALL=C head -c 4 "${output}.partial")" != "%PDF" ]]; then
    rm -f "${output}.partial"
    printf 'warning: %s endpoint did not return a PDF; supply its local source path\n' "${label}" >&2
    return
  fi
  mv "${output}.partial" "${output}"
}

copy_optional_pdf() {
  local source_path="$1"
  local output="$2"
  local label="$3"
  if [[ -z "${source_path}" ]]; then
    printf 'skip: %s source path was not supplied\n' "${label}"
    return
  fi
  if [[ ! -f "${source_path}" ]]; then
    printf 'error: %s source does not exist: %s\n' "${label}" "${source_path}" >&2
    exit 1
  fi
  cp "${source_path}" "${output}"
  printf 'copy: %s\n' "${output}"
}

download_pdf \
  "https://arxiv.org/pdf/1803.08475" \
  "${PAPER_DIR}/01_attention_model_iclr2019.pdf"

download_pdf \
  "https://repositorio.ie.edu/server/api/core/bitstreams/ed7554eb-ec96-41fe-ab6d-7386243eec6f/content" \
  "${PAPER_DIR}/02_evrptw_rl_tits2022.pdf"

if [[ -n "${DRL_TS_PAPER_SOURCE:-}" ]]; then
  copy_optional_pdf \
    "${DRL_TS_PAPER_SOURCE}" \
    "${PAPER_DIR}/03_drl_ts_ppsn2022.pdf" \
    "DRL-TS manuscript"
else
  download_pdf_optional \
    "https://link.springer.com/content/pdf/10.1007/978-3-031-14714-2_25.pdf" \
    "${PAPER_DIR}/03_drl_ts_ppsn2022.pdf" \
    "DRL-TS manuscript"
fi

copy_optional_pdf \
  "${TERRAN_PAPER_SOURCE:-}" \
  "${PAPER_DIR}/04_terran_project_manuscript.pdf" \
  "TERRAN manuscript"

download_pdf \
  "https://arxiv.org/pdf/2407.01615" \
  "${PAPER_DIR}/05_edge_direct_canadian_ai2024.pdf"

AM_REPO="${REPO_DIR}/attention-learn-to-route"
AM_COMMIT="c9abf41ac2f878a55b20dc7e829bc942bb999631"
if [[ ! -d "${AM_REPO}/.git" ]]; then
  git clone https://github.com/wouterkool/attention-learn-to-route.git "${AM_REPO}"
fi
if ! git -C "${AM_REPO}" cat-file -e "${AM_COMMIT}^{commit}"; then
  git -C "${AM_REPO}" fetch origin "${AM_COMMIT}"
fi
git -C "${AM_REPO}" checkout --detach "${AM_COMMIT}"

printf '\nReference cache ready: %s\n' "${SCRIPT_DIR}"
printf 'See README.md for provenance and comparison boundaries.\n'
