# Hindsight Oracle Routing

Code, instance pools, checkpoints, and result data for the paper:

**When Best-of-K Becomes a Hindsight Oracle: Auditing Test-Time Search in Dynamic Routing**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21316177.svg)](https://doi.org/10.5281/zenodo.21316177)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![GitHub](https://img.shields.io/badge/GitHub-kaissi11%2Fhindsight--oracle--routing-blue?logo=github)](https://github.com/kaissi11/hindsight-oracle-routing)

## What this is

A paired, seeded benchmark showing that best-of-K sampling under disruption dynamics
is an **oracle upper bound**, not a deployable policy. The paper decomposes test-time
search into an oracle component and a deployable online counterpart at equal search
width, then measures the hindsight gap directly — against a zero-compute repair
heuristic and budget-matched rolling OR-Tools on real road matrices (Damascus, London).

## Repository layout

```text
hindsight-oracle-routing/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
└── code/
    ├── scenario_bucket_eval_v2.py      # main online evaluation harness
    ├── matched_information_eval.py     # §6.3a matched-information experiment
    ├── research_env_v2.py              # disruption environment
    ├── tsp_model_v2.py                 # policy model
    ├── rolling_horizon_or_baseline.py  # OR-Tools baseline
    ├── run_stage5_suites.py            # Table 3 suite runner
    ├── run_review_experiments.py       # ablations / review queue
    ├── aggregate_*.py                  # seed pooling
    ├── equivalence_analysis.py         # TOST / equivalence tests
    ├── make_paper_assets.py            # tables & figures
    ├── checkpoints_research_pomo/
    │   └── research_best.pt            # headline checkpoint
    ├── checkpoints_research_v2_pomo/
    │   └── research_v2_best.pt
    └── results/
        ├── osrm_instance_pool/         # Damascus OSRM pool
        ├── osrm_instance_pool_london/  # London OSRM pool
        ├── matched_information_aggregate.json
        ├── scenario_bucket_*.json      # per-seed suite results
        └── …                           # other aggregates / diagnostics
```

**Large matched-information dumps.** The six per-seed files
`matched_information_h{4,8}_seed_*.json` (~2 GB total) exceed GitHub’s 100 MB
limit, so they are **not** in the git tree. Attach them as **release assets** when
publishing `v1.0.0` (Zenodo will archive them with the release). The aggregate used
for §6.3a tables is in-repo at `code/results/matched_information_aggregate.json`.

## Quickstart

```bash
pip install -r requirements.txt
cd code
python scenario_bucket_eval_v2.py --n-episodes 2 --buckets low --save-json results/smoke.json
```

## Reproduce the paper

| Goal | Command |
|---|---|
| Main online suites (Table 3) | `python run_stage5_suites.py` |
| Review ablations (§6.4a, Table 3b, §6.8, budget sweep) | `python run_review_experiments.py` |
| Pool seeds → tables / figures | `python aggregate_stage2_seeds.py <variant>` → `python equivalence_analysis.py` → `python make_paper_assets.py` |

Hardware note: ~1 GPU-night per suite on an RTX 3060 Laptop GPU; suite runners are
resumable per seed.

Most per-seed suite JSONs ship in `code/results/` (tables regenerate without re-running).
The large matched-information per-seed dumps ship as release assets; the aggregate is in-repo.

**Protocol:** seeds `12345 / 13345 / 14345 / 15345 / 16345` · OR budget `30 ms/step` · `K=8` · `H=8 h`

## Data attribution

The OSRM travel-time pools derive from OpenStreetMap data
(© OpenStreetMap contributors, ODbL) processed with OSRM.

## Cite

```bibtex
@software{AlKaissi_hindsight_oracle,
  author  = {Al-Kaissi, Alaa},
  title   = {When Best-of-K Becomes a Hindsight Oracle: benchmark and code},
  year    = {2026},
  doi     = {10.5281/zenodo.21316177},
  url     = {https://github.com/kaissi11/hindsight-oracle-routing},
  version = {1.0.0}
}
```

**Contact:** alaakaissi11@gmail.com · [ORCID](https://orcid.org/0009-0000-7121-0089)
