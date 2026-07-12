# When Best-of-K Becomes a Hindsight Oracle — benchmark, code, and result data

Code, instance pools, checkpoints, and per-seed result JSONs for the paper
*“When Best-of-K Becomes a Hindsight Oracle: Auditing Test-Time Search in
Dynamic Routing”* ([FILL: arXiv link when live]).

[![DOI](https://zenodo.org/badge/DOI/[FILL:ZENODO-DOI].svg)](https://doi.org/[FILL:ZENODO-DOI])
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)

**What this is.** A paired, seeded benchmark showing that best-of-K sampling under
disruption dynamics is an oracle upper bound, not a deployable policy: the paper
decomposes test-time search into an oracle component and a deployable online
counterpart at equal search width and measures the hindsight gap directly, against a
zero-compute repair heuristic and budget-matched rolling OR-Tools on real road
matrices (Damascus, London).

## Layout
Everything runnable lives flat in one directory so every script runs unchanged:
```
code/   evaluation harness, disruption env v2, matched-information experiment,
        suite runners, aggregation, equivalence/TOST analysis, figure scripts,
        PLUS the frozen model/baseline/utility modules they import (merged from
        the research repo's v6.2 dependency base — the only source modification
        is a one-line path patch so imports and checkpoint defaults resolve here)
        code/results/          instance pools + per-seed result JSONs
        code/checkpoints_*/    policy checkpoints (research_best.pt = headline)
```

**Large matched-information dumps.** The six per-seed files
`matched_information_h{4,8}_seed_*.json` (~2 GB total) exceed GitHub’s 100 MB
file limit, so they are not in the git tree. They ship as **GitHub/Zenodo release
assets** when you publish `v1.0.0`. The aggregate used for §6.3a tables is in
`code/results/matched_information_aggregate.json` and *is* in the repo.

## Quickstart (60 seconds)
```
pip install -r requirements.txt
cd code
python scenario_bucket_eval_v2.py --n-episodes 2 --buckets low --save-json results/smoke.json
```

## Reproduce the paper
- Main online suites (Table 3): `python run_stage5_suites.py`  (~1 GPU-night each on a
  single RTX 3060; resumable per seed)
- Review ablations (§6.4a, Table 3b, §6.8, budget sweep): `python run_review_experiments.py`
- Pool seeds → tables/figures: `python aggregate_stage2_seeds.py <variant>` →
  `python equivalence_analysis.py` → `python make_paper_assets.py`
- Most per-seed suite JSONs ship in `code/results/` (tables regenerate without
  re-running). The large matched-information per-seed dumps ship as release assets
  (see above); `matched_information_aggregate.json` is in-repo.

Seeds: 12345/13345/14345/15345/16345 · OR budget 30 ms/step · K=8 · H=8 h.

## Data attribution
The OSRM travel-time pools derive from OpenStreetMap data
(© OpenStreetMap contributors, ODbL) processed with OSRM.

## Cite
```bibtex
@software{[FILL:key],
  author  = {Al-Kaissi, Alaa and Sandouk, Obai},
  title   = {When Best-of-K Becomes a Hindsight Oracle: benchmark and code},
  year    = {2026},
  doi     = {[FILL:ZENODO-DOI]},
  url     = {https://github.com/[FILL:user/repo]}
}
```
Contact: [FILL: email] · ORCID: [FILL: orcid]
