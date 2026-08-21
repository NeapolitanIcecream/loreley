# Loreley paper

This directory contains the source and public evidence for the Loreley
preprint, *Loreley: Repository-Scale Program Evolution with Quality-Diversity
Search*.

The public version is available as
[arXiv:2608.19703](https://arxiv.org/abs/2608.19703), with direct links to the
[PDF](https://arxiv.org/pdf/2608.19703) and
[experimental HTML](https://arxiv.org/html/2608.19703). The arXiv v1 manuscript
source matches repository commit
[`d05392a4950fddee7a5719fa30f0a8db71e8fd4f`](https://github.com/NeapolitanIcecream/loreley/commit/d05392a4950fddee7a5719fa30f0a8db71e8fd4f).
The [verification manifest](arxiv_v1_manifest.json) records the source hashes
and the rendered equivalence of the figure PDF.

The paper names Mohan Chen as the sole author, without an affiliation. It
records no external funding, discloses the research and manuscript roles of
generative AI tools, and is released under CC BY 4.0.

## Citation

```bibtex
@misc{chen2026loreley,
  title         = {Loreley: Repository-Scale Program Evolution with Quality-Diversity Search},
  author        = {Mohan Chen},
  year          = {2026},
  eprint        = {2608.19703},
  archiveprefix = {arXiv},
  primaryclass  = {cs.SE},
  url           = {https://arxiv.org/abs/2608.19703}
}
```

The repository root also provides machine-readable
[`CITATION.cff`](../CITATION.cff) metadata.

## Build

From the repository root:

```bash
uv run python paper/evidence/validate_zstd_method_efficacy.py
uv run --no-project --with matplotlib==3.11.1 python paper/figures/generate_figures.py
mkdir -p tmp/pdfs output/pdf
(cd paper && tectonic --keep-logs --keep-intermediates --outdir ../tmp/pdfs main.tex)
cp tmp/pdfs/main.pdf output/pdf/loreley-paper.pdf
```

The figure generator reads checked-in Zstandard evidence rather than copied
plot values. The checked-in `figures/zstd_method_efficacy.pdf` lets the paper
build without first installing Matplotlib; running the generator should
reproduce it from `evidence/zstd_method_efficacy.json`.

Generated PDFs, submission bundles, rendered pages, and internal review
archives belong under `output/` or `tmp/` and are intentionally not versioned.

## Directory map

- `main.tex` and `references.bib` are the manuscript sources.
- `figures/generate_figures.py` regenerates the quantitative figures from the
  checked-in evidence.
- `evidence/` contains the public numerical records, treatment description,
  and validation scripts.
- `../tools/method_efficacy_experiment/zstd_target.py` contains the hash-bound
  Zstandard task and repository-state ignore text cited by the appendix.

## Evidence sources

- `docs/research/2026-08-07-loreley-case-study-evidence-report.md`
- `docs/research/2026-08-02-markdown-it-py-deepseek-case-study.md`
- `docs/research/2026-08-03-pathspec-deepseek-case-study.md`
- `docs/research/2026-08-07-zstandard-gpt-v19-case-study-report.md`
- `docs/research/2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md`
- `reports/zstandard-gpt-v19-evidence.json`
- `reports/zstandard-gpt-v19-top10-validation-supplement.json`
- `paper/evidence/python_uncertainty.json`
- `paper/evidence/python_qd_audit.json`
- `paper/evidence/python_generation_cost_audit.json`
- `paper/evidence/campaign_roots.json`
- `paper/evidence/zstd_candidate_split_records.json`
- `paper/evidence/zstd_registered_thresholds.json`
- `paper/evidence/zstd_qd_audit.json`
- `paper/evidence/zstd_method_efficacy.json`
- `paper/evidence/zstd_formal_records.json`
- `paper/evidence/zstd_formal_treatment.json`
- `paper/evidence/build_zstd_formal_records.py`
- `paper/evidence/validate_zstd_method_efficacy.py`

The operational report, private database dumps, hidden corpora, provider
state, and manuscript-review history are not public evidence. The checked-in
formal record, frozen treatment, and validator define the reproducible public
boundary.
