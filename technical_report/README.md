# Loreley evolution dynamics research

Four-page technical reports on wall-clock time to a useful candidate,
late-stage improvement, and reuse of retained branches.

| Language | Report | LaTeX source |
| --- | --- | --- |
| English | [PDF](loreley-evolution-dynamics-en.pdf) | [main_en.tex](main_en.tex) |
| 中文 | [PDF](loreley-evolution-dynamics-zh.pdf) | [main.tex](main.tex) |

## Build the PDFs

Run from the repository root with Tectonic installed. The checked-in figures
are sufficient to build either report; regenerating them is optional.

```bash
mkdir -p tmp/pdfs/technical-report-en tmp/pdfs/technical-report-zh output/pdf

# English: portable TeX Gyre Heros fonts.
(cd technical_report && tectonic --keep-logs --keep-intermediates \
  --outdir ../tmp/pdfs/technical-report-en main_en.tex)
cp tmp/pdfs/technical-report-en/main_en.pdf \
  output/pdf/loreley-evolution-dynamics-technical-report-en.pdf
cp tmp/pdfs/technical-report-en/main_en.pdf \
  technical_report/loreley-evolution-dynamics-en.pdf

# Chinese: requires Arial and Hiragino Sans GB on the build machine.
(cd technical_report && tectonic --keep-logs --keep-intermediates \
  --outdir ../tmp/pdfs/technical-report-zh main.tex)
cp tmp/pdfs/technical-report-zh/main.pdf \
  output/pdf/loreley-evolution-dynamics-technical-report.pdf
cp tmp/pdfs/technical-report-zh/main.pdf \
  technical_report/loreley-evolution-dynamics-zh.pdf
```

Fonts used by the final PDFs are embedded. The two analysis charts in each
report are embedded as approximately 300 dpi PNGs to avoid platform-specific
font-subset issues in Windows PDF readers; the first-page lineage diagrams and
all report text remain vector-based.

## Validate evidence and regenerate figures

```bash
python3 technical_report/evidence/validate_dynamics.py
uv run python paper/evidence/validate_zstd_method_efficacy.py
uv run --no-project --with matplotlib==3.11.1 \
  python technical_report/figures/generate_figures.py --language en
uv run --no-project --with matplotlib==3.11.1 \
  python technical_report/figures/generate_figures.py --language zh
```

The language option defaults to Chinese. English charts use Matplotlib's bundled
DejaVu Sans; Chinese charts use the configured CJK font fallback. Optional vector
PDF previews are generated alongside the PNGs but are not versioned or embedded
in the reports.

All plotting data are available in a clean checkout. The
[timing evidence](evidence/README.md) contains 1,008 sanitized task records and
126 validation-selected checkpoint results. The generator also reads the
[method evidence](../paper/evidence/zstd_method_efficacy.json) and the
[Pathspec lineage](../docs/research/2026-08-03-pathspec-deepseek-case-study.md).

After changing the source or figures, render all four pages and check labels,
page boundaries, font embedding, and README links before replacing the published
PDFs.
