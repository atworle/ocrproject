OCR Project
===========

This repository contains utilities to compare OCR outputs from several sources (Library of Congress OCR, Tesseract, and AI Vision OCR) using multiple quantitative and qualitative metrics. The main comparison tooling lives in `ocr_quality_metrics.py`.

Goals
-----
- Aggregate OCR outputs (per-page or multi-page) from different systems.
- Compute comparison metrics that highlight differences (token/phrase overlap, edit distances, WER, character similarity, diffs).
- Provide alignment helpers so aggregated files can be compared page-by-page.
- Produce JSON and CSV results that can be used for reporting or further analysis.

Project layout
--------------
- `ocr_quality_metrics.py` - Main analysis and comparison script. Offers CLI modes to analyze single files, aggregate vision OCR files, compare two files, and align+compare a directory of vision files against an aggregated reference.
- `imageprocessing.py` - (project helper) image pre-processing utilities (may be used before running OCR).
- `ocr.py` - (project helper) local Tesseract OCR wrapper (if present).
- `vision_api_ocr.py` - (project helper) code to call an AI Vision OCR API to obtain transcriptions.
- `original_ocr/` - the LOC aggregated OCR results (text files).
- `vision_results/` - AI Vision OCR per-page transcriptions (text files).
- `align_compares/` - example output directory where per-file comparison JSONs and `summary.csv` are written when using the align-and-compare mode.
- `ocr-results/` - other text outputs from experiments.
- `requirements.txt` - Python dependencies for running scripts.

Quick concepts
--------------
- Reference (ref): the ground (or primary) transcription used as the baseline for comparisons. In this repo `original_ocr/*` is treated as the reference.
- Hypothesis (hyp): the OCR output being evaluated against the reference (for example the AI Vision OCR outputs in `vision_results/`).
- Aggregation: concatenating multiple per-page transcriptions into a single file. If you compare aggregated files directly, you must align or compare per-page to avoid inflated errors.
- Normalization: optional lowercasing and punctuation removal to reduce noise from formatting differences. Use `--normalize` in the CLI for normalized metrics.

Metrics explained
-----------------
All metrics are produced by `ocr_quality_metrics.py` and are saved to JSON when you use the CLI `--compare-out` or the align-and-compare flow.

- ref_token_count / hyp_token_count
  - The number of tokens (word-like items) the tokenizer extracted from the reference and hypothesis texts. Tokenization is done with a regex that captures alphabetic sequences (including many accented characters) and apostrophes, then lowercases tokens.
  - If `hyp_token_count` is substantially larger than `ref_token_count`, many insertions or extra pages are present in the hypothesis; this will increase WER.

- token_jaccard
  - Jaccard similarity computed on the set of tokens in each text: |A ∩ B| / |A ∪ B|. Values range 0.0–1.0 where 1.0 means identical token sets.
  - Good for measuring vocabulary overlap; insensitive to ordering or duplicates.

- ngram_jaccard_3 / ngram_jaccard_5
  - Jaccard similarity of 3-grams or 5-grams (contiguous token sequences). Sensitive to phrasing and ordering. Lower values indicate different wording or large alignment shifts.

- word_error_rate (WER)
  - A token-level Levenshtein distance normalized by the reference token count: edits / |ref_tokens|. Edits = insertions + deletions + substitutions required to transform the hypothesis token sequence into the reference sequence.
  - Asymmetric: WER(ref, hyp) != WER(hyp, ref). Lower is better; 0.0 means perfect match.

- char_edit_distance and char_error_rate
  - Character-level Levenshtein distance (integer) and a normalized char error rate = char_edits / |ref_chars|.
  - Useful for capturing small spelling errors and character confusions that token-level WER can miss.

- char_similarity
  - difflib.SequenceMatcher ratio on raw characters (0.0–1.0). Quick measure of character overlap including whitespace/punctuation.

- diff_sample
  - A short unified diff (unified format) excerpt that shows `-` lines from the reference and `+` lines from the hypothesis. Useful for manual inspection to see the concrete differences.

- ref_keyword_counts / hyp_keyword_counts / keyword_count_diff
  - Counts for the keywords defined in `KEYWORDS` in the reference and hypothesis, and the difference (hyp - ref). Use to quickly see whether important terms (for your analysis) were preserved or lost.

How to use the main script
--------------------------
The script exposes several CLI modes. Run `python ocr_quality_metrics.py -h` for full usage. Examples below assume you are in the repo root.

1) Analyze a single file (keyword counts + optional spell-check):

```bash
python ocr_quality_metrics.py --analyze original_ocr/oct7-21-1756.txt --output analyze_oct7.json
```

2) Aggregate AI Vision OCR files into one text file (useful before a coarse compare):

```bash
python ocr_quality_metrics.py --aggregate-vision vision_results --vision-glob "*_vision_transcription.txt" --out-agg all_vision.txt
```

3) Compare two files and write JSON results (normalized):

```bash
python ocr_quality_metrics.py --compare original_ocr/oct7-21-1756.txt all_vision.txt --compare-out compare_results.json --normalize
```

4) Align each vision file to the best matching window in an aggregated reference, compare and save per-file JSONs plus a `summary.csv`:

```bash
python ocr_quality_metrics.py --align-compare --ref-agg original_ocr/oct7-21-1756.txt --vision-dir vision_results --out-dir align_compares --normalize
```






