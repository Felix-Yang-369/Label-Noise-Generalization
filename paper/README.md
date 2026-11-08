# Final report (IEEE conference format)

- `ECS171_Final_Report.pdf`: compiled report (9 body pages + 1 reference page).
- `main.tex`, `references.bib`: report source and bibliography.
- `figures/`: vector PDF figures with PNG previews (`per_class_f1` is not used in the report).
- `generated/`: chart-ready tables, copies of the source results, hashes, and figure metadata.
- `tools/build_figures.py`: rebuilds the figures from recorded results without fitting any model.
- `IEEEtran.cls`, `IEEEtran.bst`: official IEEE class and bibliography style (LPPL 1.3, unmodified).

## Compile

On Overleaf, upload this folder as a new project, set `main.tex` as the main document and **pdfLaTeX** as the compiler. Locally:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

No Python environment or training run is needed to compile the report.

## Rebuild figures

From the repository root, with the locked dependencies installed:

```bash
python paper/tools/build_figures.py --recorded-sources
```

This mode verifies the hashes of the recorded source copies in `generated/source_data/` and redraws the figures. It does not train models. To produce new results, follow `REPRODUCIBILITY.md` in a separate run directory.

## License

The report text, figures, and PDF are © 2026 Tianying Chen, Linjie Yang, and Elsey Chen and are not covered by the repository's Apache License 2.0. `tools/build_figures.py` is source code and is covered by it. The IEEE class and style files keep their own license notices. See `../NOTICE`.
