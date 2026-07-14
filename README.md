# Hindsight Oracle Routing

Code, instance pools, checkpoints, and result data for the paper:

**When Best-of-K Becomes a Hindsight Oracle: Auditing Test-Time Search in Dynamic Routing**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21316176.svg)](https://doi.org/10.5281/zenodo.21316176)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![GitHub](https://img.shields.io/badge/GitHub-kaissi11%2Fhindsight--oracle--routing-blue?logo=github)](https://github.com/kaissi11/hindsight-oracle-routing)

## What this is

A paired, seeded benchmark auditing best-of-K sampling under disruption dynamics:
episode-level best-of-K after disruptions resolve is a **hindsight-selected** score
over its own trajectories, not a deployable policy. The paper separates that score
from a same-width online lookahead, then measures the gap against a zero-compute
repair heuristic and budget-matched rolling OR-Tools on real road matrices
(Damascus, London).

## Demo

![Online look-8 serving 19 stops on real Damascus roads while nodes block and reopen](media/policy_demo.gif)

One protocol episode (seed 32353, high-disruption bucket) on the real Damascus road
network: the deployable **online look-8** arm commits one step at a time while zonal
traffic drifts and nodes block/reopen mid-route (red = currently blocked). The
closing frame reports the paired outcomes of oracle-8, repair, and rolling OR-Tools
on the **same episode and disruption schedule** — the paper measures the gap between
the online and hindsight-selected arms. Regenerate with
`python code/make_demo_gif.py` (qualitative only; writes no result JSONs).
Map data © OpenStreetMap contributors (ODbL).

## Repository layout

```text
hindsight-oracle-routing/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── requirements.txt
├── media/
│   └── policy_demo.gif                 # README demo (regenerate: code/make_demo_gif.py)
└── code/
    ├── scenario_bucket_eval_v2.py      # main online evaluation harness
    ├── make_demo_gif.py                # qualitative real-map demo (media/)
    ├── matched_information_eval.py     # matched-information experiment (Table 8)
    ├── generic_hindsight_eval.py       # no-learning hindsight control (Table 7)
    ├── oracle_k_sweep.py               # retrospective oracle-K sweep (Figure 4)
    ├── kpi_aligned_or_eval.py          # completion-first rolling-OR control
    ├── analyze_dualcount_pilot.py      # deadline-rule sensitivity gate
    ├── research_env_v2.py              # disruption environment
    ├── tsp_model_v2.py                 # policy model
    ├── rolling_horizon_or_baseline.py  # OR-Tools baseline
    ├── run_stage5_suites.py            # online suite runner
    ├── aggregate_*.py                  # seed pooling (incl. aggregate_generic_hindsight)
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

**Large matched-information dumps.** The six raw per-episode trajectory files
`matched_information_h{4,8}_seed_*.json` (~2 GB total) are **not** shipped;
the archive includes `matched_information_aggregate.json` (Table 8 regenerates
from it). Raw files are available from the corresponding author, and the released
code re-derives them from the shipped pools and checkpoints.

## Quickstart

```bash
pip install -r requirements.txt
cd code
python scenario_bucket_eval_v2.py --n-episodes 2 --buckets low --save-json results/smoke.json
```

## Reproduce the paper

| Goal | Command |
|---|---|
| Main online suites | `python run_stage5_suites.py` |
| Generic hindsight control (Table 7) | `python generic_hindsight_eval.py` → `python aggregate_generic_hindsight.py` |
| Matched information (Table 8) | `python matched_information_eval.py` → `python aggregate_matched_information.py` |
| Deadline-rule / wait probes | `python run_dualcount_pilot.py` → `python analyze_dualcount_pilot.py` |
| Oracle-K sweep (Figure 4) | `python oracle_k_sweep.py` → `python make_fig_ksweep.py` |
| Pool seeds → tables / figures | `python aggregate_stage2_seeds.py <variant>` → `python equivalence_analysis.py` → `python make_paper_assets.py` |

Hardware note: ~1 GPU-night per suite on an RTX 3060 Laptop GPU; suite runners are
resumable per seed. Any suite with a wall-clock-budgeted rolling-OR arm must run
on an otherwise idle machine (one suite at a time).

Most per-seed suite JSONs ship in `code/results/` (tables regenerate without re-running).

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
  doi     = {10.5281/zenodo.21316176},
  url     = {https://github.com/kaissi11/hindsight-oracle-routing},
  version = {1.0.1}
}
```

**Contact:** alaakaissi11@gmail.com · [ORCID](https://orcid.org/0009-0000-7121-0089)
