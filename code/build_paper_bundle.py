#!/usr/bin/env python3
"""Build the submission bundle: paper draft + generated assets fused into
one document, exported as Markdown, DOCX, and PDF.

  python build_paper_bundle.py

Steps:
 1. Read paper/paper_draft_v3_submission.md (the master draft).
 2. Split main paper vs online supplement on the SI markers
    (`<!-- SI:BEGIN ... -->` / `<!-- SI:END -->`): the main build drops the
    blocks, the supplement build keeps only them (single source of truth —
    one master, two documents).
 3. Insert the generated figures (paper/assets/*.png) at their section
    anchors (falls back to an appended gallery if an anchor moved).
 4. Write  paper/bundle/paper_v3_bundle.md  +  paper_v3_supplement.md
 5. DOCX via pandoc (pypandoc-binary).
 6. PDF via pandoc standalone HTML (resources embedded) -> Edge headless.

Rerun after every draft/asset change — outputs are regenerated from scratch.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
PAPER = HERE / "paper"
ASSETS = PAPER / "assets"
OUT = PAPER / "bundle"
OUT.mkdir(parents=True, exist_ok=True)

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

SI_BEGIN = re.compile(r"^<!--\s*SI:BEGIN\b(.*?)-->\s*$")
SI_END = re.compile(r"^<!--\s*SI:END\s*-->\s*$")

SUPPLEMENT_HEADER = """# Supplementary Information

**for** *When Best-of-K Becomes a Hindsight Oracle: What Survives Online \
in Dynamic Routing* — Alaa Al-Kaissi, Damascus University.

Sections are numbered S1, S2, ... and are referenced from the main text. \
All numbers regenerate from the released result JSONs \
(<https://doi.org/10.5281/zenodo.21316176>).

---
"""


def split_master(src: str) -> tuple[str, str]:
    """-> (main_md, supplement_md); asserts balanced markers."""
    main, si = [], []
    in_si = False
    for n, ln in enumerate(src.splitlines(), 1):
        if SI_BEGIN.match(ln):
            if in_si:
                raise SystemExit(f"nested SI:BEGIN at master line {n}")
            in_si = True
            continue
        if SI_END.match(ln):
            if not in_si:
                raise SystemExit(f"unmatched SI:END at master line {n}")
            in_si = False
            si.append("")
            continue
        (si if in_si else main).append(ln)
    if in_si:
        raise SystemExit("unterminated SI block at end of master")
    return "\n".join(main), SUPPLEMENT_HEADER + "\n".join(si)

# (anchor substring in the draft, image, caption) — image inserted AFTER the
# first line containing the anchor.
FIGURES = [
    ("### 4.3 Decoding: the oracle–online split",
     "fig1_oracle_online_schematic.png",
     "Figure 1: Episode-level best-of-8 selects its winner after the realized "
     "disruptions — a hindsight-selected bound over its own sampled "
     "trajectory set, not a policy; the online lookahead commits each "
     "action using only the frozen current matrix (deployable)."),
    ("**Messages.** (i) The hindsight bound beats lightweight repair",
     "fig_forest_consistency.png",
     "Figure 2: Paired delivered-stop delta (policy oracle-8 − repair) with "
     "95% CIs — positive in all 18 recorded setting × bucket cells."),
    ("**The online result, in one sentence:",
     "fig_online_vs_oracle.png",
     "Figure 3: The headline result (N=100 online Stage-5 uses five "
     "evaluation seeds, n=200/bucket). Online lookahead − repair (upper panel) "
     "hugs zero in every bucket of every setting, while the protocol gap "
     "(online − oracle, lower panel) is significant everywhere and grows "
     "with disruption and scale (largest absolute gap −0.615 at N=100-high): "
     "most of the measured best-of-K advantage does not survive online "
     "selection."),
    ("**How the inflation scales with K.**",
     "fig_ksweep.png",
     "Figure 4: Retrospective oracle-K sweep — the hindsight-selected "
     "best-of-K minus deployable look-8, as a function of K on identical "
     "schedules and RNG streams (K=8 reproduces the recorded suite). At H=8 "
     "the hindsight-selected difference grows with K and is not a K=8 "
     "artifact; at H=4 the whole trajectory-hindsight curve sits below "
     "look-8, so the same K-axis traces a signed difference rather than "
     "an inflation above the deployable controller."),
    ("The tested checkpoint, trained around N≤20 on synthetic matrices only",
     "fig_effect_by_setting.png",
     "Figure 5: The completion effect out-of-distribution, raw (left) and "
     "per-100-customers (right): the raw-stop effect grows with route size "
     "while the per-customer rate stays comparable (~0.3-0.6 completion "
     "points at both scales) — under the hindsight-selected bound "
     "(Figure 3 shows the online counterpart)."),
    ("### 6.8 The price: completion is bought with makespan",
     "fig_tradeoff_v2.png",
     "Figure 6: Completion gain vs makespan premium relative to rolling-OR — "
     "every operating point buys stops with route time."),
    ("none of the compared methods dominates on both axes",
     "fig_pareto.png",
     "Figure 7: Absolute completion–makespan positions per method and bucket "
     "(dotted line = per-bucket Pareto frontier). No method dominates: "
     "rolling-OR holds the makespan frontier, only the hindsight-selected "
     "bound reaches the completion frontier, and deployable look-8 pays "
     "oracle-level makespan for repair-level completion."),
    ("Three reversals, all interpretable",
     "fig_horizon_inversion.png",
     "Figure 8: The horizon-stress inversion (Damascus; 5 seeds at H=8 h, "
     "1 at 6 h, 3 at 4 h; hierarchical seed-cluster bootstrap CIs). As the "
     "mission horizon tightens from 8 h to 4 h, the deployable lookahead "
     "pulls ahead of lightweight repair (left) and the look-8 − oracle-8 gap "
     "flips sign (right): the trajectory-hindsight metric bounds only "
     "selection among its own sampled trajectories — under binding deadlines "
     "it understates what deployable controllers achieve."),
]


def insert_figures(text: str) -> tuple[str, int]:
    out, pending = [], list(FIGURES)
    for ln in text.splitlines():
        out.append(ln)
        for anchor, img, cap in list(pending):
            if anchor in ln:
                out += ["", f"![{cap}](../assets/{img})", ""]
                pending.remove((anchor, img, cap))
    if pending:  # anchors that moved -> appended gallery so nothing is lost
        out += ["", "---", "", "## Figures (unanchored)"]
        for _, img, cap in pending:
            out += ["", f"![{cap}](../assets/{img})", ""]
    return "\n".join(out), len(pending)


def build_md() -> tuple[Path, Path]:
    src = (PAPER / "paper_draft_v3_submission.md").read_text(encoding="utf-8")
    main_md, si_md = split_master(src)
    main_md, unanchored = insert_figures(main_md)
    dst = OUT / "paper_v3_bundle.md"
    dst.write_text(main_md + "\n", encoding="utf-8")
    print(f"[md]   {dst}  ({unanchored} figure(s) unanchored)")
    sup = OUT / "paper_v3_supplement.md"
    sup.write_text(si_md + "\n", encoding="utf-8")
    n_si_words = len(re.findall(r"\S+", si_md))
    n_main_words = len(re.findall(r"\S+", main_md))
    print(f"[md]   {sup}  (main {n_main_words} words / SI {n_si_words} words)")
    return dst, sup


def build_docx(md: Path, stem: str) -> None:
    dst = OUT / f"{stem}.docx"
    pypandoc.convert_file(
        str(md), "docx", outputfile=str(dst),
        extra_args=[f"--resource-path={OUT};{ASSETS};{PAPER}"],
    )
    print(f"[docx] {dst}  ({dst.stat().st_size // 1024} KB)")


PRINT_CSS = """<style>
@page { size: A4; margin: 15mm 12mm; }
body { font-size: 10.5pt; line-height: 1.35; max-width: 100%; margin: 0; padding: 0 2mm; }
table { font-size: 8.2pt; width: 100%; border-collapse: collapse; margin: 8px 0; }
th, td { padding: 2px 4px; border-bottom: 0.5px solid #aaa; text-align: left;
         vertical-align: top; overflow-wrap: anywhere; hyphens: auto; }
thead th { border-bottom: 1.2px solid #333; border-top: 1.2px solid #333; }
table, tr, img { break-inside: avoid; }
img { max-width: 100%; height: auto; }
pre, code { font-size: 80%; white-space: pre-wrap; overflow-wrap: anywhere; }
h1 { font-size: 1.45em; } h2 { font-size: 1.2em; } h3 { font-size: 1.05em; }
blockquote { margin: 6px 0 6px 12px; padding-left: 8px; border-left: 3px solid #ccc; color: #444; }
</style>
"""


def build_pdf(md: Path, stem: str, title: str) -> None:
    css = OUT / "_print.css.html"
    css.write_text(PRINT_CSS, encoding="utf-8")
    html = OUT / f"{stem}.html"
    pypandoc.convert_file(
        str(md), "html", outputfile=str(html),
        extra_args=["--standalone", "--embed-resources",
                    f"--resource-path={OUT};{ASSETS};{PAPER}",
                    "--metadata", f"title={title}",
                    "-H", str(css)],
    )
    pdf = OUT / f"{stem}.pdf"
    subprocess.run(
        [EDGE, "--headless", "--disable-gpu",
         f"--print-to-pdf={pdf}", "--no-pdf-header-footer",
         html.as_uri()],
        check=True, timeout=120,
    )
    print(f"[pdf]  {pdf}  ({pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    md, sup = build_md()
    build_docx(md, "paper_v3_bundle")
    build_pdf(md, "paper_v3_bundle",
              "When Best-of-K Becomes a Hindsight Oracle")
    build_docx(sup, "paper_v3_supplement")
    build_pdf(sup, "paper_v3_supplement",
              "Supplementary Information - When Best-of-K Becomes a Hindsight Oracle")
    print("Bundle complete -> paper/bundle/")
