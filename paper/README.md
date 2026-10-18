# IEEE final report

The English report describes the completed Dry Bean label-randomization study.
It is a review version dated October 18, 2026. The scientific results are
complete; deployment acceptance, team contribution confirmation, and course
submission remain separate delivery tasks.

## Read and compile

- `ECS171_Final_Report.pdf`: compiled report, **10 body pages + 1 reference page**.
- `main.tex`: main source, IEEE conference two-column format on US Letter.
- `references.bib`: seven checked primary-source references.
- `IEEEtran.cls` and `IEEEtran.bst`: official class/style, with their original notices.
- `figures/`: eight vector PDF figures with matching PNG previews. Six are used
  in the manuscript; `study_flow` and `per_class_f1` are optional supporting assets.
- `generated/`: chart-ready tables, byte-exact source copies, source hashes,
  transformations, and figure metadata.
- `tools/build_figures.py`: regenerates plots without fitting models.

Upload the separately delivered Overleaf ZIP as a new project. Select
`main.tex` as the main document and **pdfLaTeX** as the compiler. Overleaf
should run BibTeX automatically. All figures, bibliography entries, and IEEE
style files are included; no API key, external image link, or training
environment is required to compile the report.

For a local TeX installation:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The delivered PDF was checked with the desktop editor's built-in compiler and
an independent Tectonic build. The ZIP was extracted into a new directory and
compiled separately before delivery.

## Regenerate figures

Use the repository's locked Python dependencies. In the original complete
artifact checkout, run from the repository root:

```powershell
.\.venv\Scripts\python.exe paper\tools\build_figures.py
```

In a clean Git clone, use the compact recorded source copies instead of the
ignored models and prediction artifacts:

```powershell
python paper\tools\build_figures.py --recorded-sources
```

That mode verifies source-copy hashes against the recorded provenance. It
recreates the historical paper figures, not a new training experiment. To
produce new scientific results, follow the repository-root
`REPRODUCIBILITY.md` and `Reproduce.ps1` instead. Do not mix a fresh run's
outputs with historical freezes or receipts.

## Interpretation checks retained in the manuscript

- Redraw probability `p` is distinct from actual changed fraction `q`;
  expected `q = 6p/7` because the original class can be redrawn.
- Error bars are sample standard deviations over three paired seeds on one
  fixed split, not confidence intervals or multi-split uncertainty.
- Training accuracy against noisy labels and clean-test accuracy refer to
  different targets.
- Uniform-guess expected accuracy `1/7` is not a universal Macro-F1 baseline.
- Fourteen main MLP runs reach the 500-iteration cap; this limits capacity claims.
- Future modeling after holdout inspection must be identified as exploratory.

The manuscript's final body page contains the actual GitHub URL, roadmap,
completion status, and an evidence-based contribution statement.
