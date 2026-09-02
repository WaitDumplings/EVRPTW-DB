# Learning-baseline reference materials

This directory separates upstream evidence from the EVRPTW-B implementations.
Run `download_reference_materials.sh` to create the ignored `papers/` and
`repos/` caches.  Papers and complete third-party repositories are deliberately
not committed to EVRPTW-DB; the source registry and adaptation records are.

The registry was checked on 2026-09-02.  "No verified public repository" means
that no author-maintained implementation was identified by the recorded search;
it does not prove that no code exists privately or will be released later.

## Source registry and comparison order

| Order | Benchmark implementation | Publication | Paper source | Author code | Local comparison material | Implementation status |
|---:|---|---|---|---|---|---|
| 1 | AM-EVRPTW | Kool, van Hoof, and Welling, *Attention, Learn to Solve Routing Problems!*, ICLR 2019 | [OpenReview](https://openreview.net/forum?id=ByxBFsRqYm) | [Official repository](https://github.com/wouterkool/attention-learn-to-route), MIT; benchmark vendor snapshot records commit `c9abf41ac2f878a55b20dc7e829bc942bb999631` | `papers/01_attention_model_iclr2019.pdf`; `repos/attention-learn-to-route/` | Official architecture adapted to EVRPTW-B; not an author implementation of EVRPTW |
| 2 | EVRPTW-RL | Lin, Ghaddar, and Nathwani, *Deep Reinforcement Learning for the Electric Vehicle Routing Problem with Time Windows*, IEEE T-ITS 2022 | [DOI](https://doi.org/10.1109/TITS.2021.3105232); [open accepted manuscript](https://repositorio.ie.edu/entities/publication/c50e4c65-47c5-479e-a942-8f7d8203ea9b/full) | No verified public author repository | `papers/02_evrptw_rl_tits2022.pdf` | Paper-guided reimplementation |
| 3 | DRL-TS | Chen et al., *Deep Reinforcement Learning with Two-Stage Training Strategy for Practical Electric Vehicle Routing Problem with Time Windows*, PPSN 2022 | [DOI](https://doi.org/10.1007/978-3-031-14714-2_25); [Springer chapter PDF](https://link.springer.com/content/pdf/10.1007/978-3-031-14714-2_25.pdf) | Publisher states that source is available on request; no verified public author repository | `papers/03_drl_ts_ppsn2022.pdf` | Paper-guided reimplementation |
| 4 | TERRAN | Project method described in the EVRPTW-DB manuscript | Local manuscript supplied by the project authors | Existing EVRPTW-DB implementation | `papers/04_terran_project_manuscript.pdf` when `TERRAN_PAPER_SOURCE` is available | Existing implementation migrated to the canonical Stage-2 contract |
| 5 | Edge-DIRECT (conditional) | Mozhdehi, Mohammadizadeh, and Wang, *Edge-DIRECT*, Canadian AI 2024 | [conference PDF](https://assets.pubpub.org/day222wb/XinWang-01716776157945.pdf) | No verified public author repository | `papers/05_edge_direct_canadian_ai2024.pdf` | Source assessment only; homogeneous-fleet adaptation requires a separate naming decision |

## What to compare

For each method, compare the publication and any official source against its
benchmark `ADAPTATION.md`.  The canonical contract changes the objective-facing
term and final ranking to directed-road total distance while retaining the
published architecture, decoder, training procedure, and compatible auxiliary
shaping.  Training shaping is not part of the final evaluation objective.

The implementation records are:

- `../AM_EVRPTW/ADAPTATION.md`
- `../EVRPTW_RL/ADAPTATION.md`
- `../DRL_TS/ADAPTATION.md`
- `../TERRAN/ADAPTATION.md`
- `../EDGE_DIRECT_SOURCE_ASSESSMENT.md`

## Access boundary

The script only uses publisher, institutional-repository, conference, and author
repository URLs that were directly accessible at registry freeze time.  It does
not bypass login or paywall controls.  If a source later becomes unavailable,
use the DOI recorded above and provide a legally obtained local copy for manual
comparison.
