# Papers

This directory stores paper artifacts used by the benchmark and reproduction pipeline.

## Layout

- `papers/*.pdf`: reference PDFs used by `repro/papers/index.yaml` and per-paper metadata.
- `papers/paper_text/`: canonical location for plain-text extracts used for note-taking and protocol mapping.
- `papers/index.csv`: metadata index for implemented attacks and local artifact paths.

## Paper Text Policy (Conservative Default)

- Do not assume paper full text is redistributable unless license terms explicitly allow it.
- Keep this repository focused on metadata, mapping, and reproducibility scripts/configs.
- If a text extract is present, treat it as a local research aid and verify redistribution rights before publishing.
- When in doubt, prefer storing bibliographic metadata and retrieval instructions instead of full text.

## Sources and Provenance

- Existing text files were consolidated from historical local folders (`paper_text/`, `.paper_text/`, `temp/`, and legacy `papers/*.txt`).
- Exact duplicates were moved to `archive/papers_duplicates/` to keep a single canonical working set.
- Non-identical variants are preserved with a `__variant_<source>.txt` suffix.

## Regenerating / Refreshing Text Locally

- Use publisher/arXiv pages listed in `papers/index.csv` and extract locally.
- Keep resulting text in `papers/paper_text/` using deterministic names (lowercase slug, underscores).
- Record provenance and caveats in the `notes` column of `papers/index.csv`.
