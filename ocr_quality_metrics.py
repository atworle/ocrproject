#!/usr/bin/env python3
"""
ocr_quality_metrics.py

Utilities to analyze OCR text files and compare two OCR outputs (for example,
Library of Congress OCR vs an AI Vision OCR). New features added:

- Aggregate multiple AI Vision OCR text files into a single file.
- Compare two OCR transcriptions using multiple metrics:
  - Character-level similarity (difflib ratio)
  - Word Error Rate (WER) using token edit distance
  - Token Jaccard / n-gram Jaccard
  - Longest common subsequence length (approx)
  - Small unified diff sample for inspection

This keeps the original single-file analysis (keyword counts and optional
spell-check) and adds comparison helpers and a CLI for aggregation + comparison.

Usage examples:
  # Analyze a single file (existing behavior)
  python ocr_quality_metrics.py --analyze path/to/file.txt --output results.json

  # Aggregate all vision OCR files in a folder into one text file
  python ocr_quality_metrics.py --aggregate-vision vision_folder --vision-glob "*_vision_transcription.txt" --out-agg all_vision.txt

  # Compare LOC OCR file to aggregated AI vision OCR
  python ocr_quality_metrics.py --compare loc.txt all_vision.txt --compare-out compare.json

"""

import argparse
import json
import re
import os
import glob
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
import difflib

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None


KEYWORDS = ["liberty", "freedom", "rights", "king", "france", "french"]
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    return [t.lower() for t in tokens]


def keyword_counts(tokens: List[str], keywords=KEYWORDS) -> Dict[str, int]:
    c = Counter(tokens)
    return {k: c.get(k, 0) for k in keywords}


def spell_check(tokens: List[str], max_suggestions=10):
    if SpellChecker is None:
        raise RuntimeError("pyspellchecker is not installed. Install with: pip install pyspellchecker")

    spell = SpellChecker()
    unique_words = set(tokens)
    misspelled = spell.unknown(unique_words)

    freq = Counter(tokens)
    miss_freq = sorted(misspelled, key=lambda w: -freq[w])
    suggestions = {w: list(spell.candidates(w))[:5] for w in miss_freq[:max_suggestions]}

    return {
        "total_unique_words": len(unique_words),
        "misspelled_count": len(misspelled),
        "misspelled_samples": miss_freq[:max_suggestions],
        "suggestions": suggestions,
    }


def analyze_file(path: Path, run_spell: bool = True):
    text = load_text(path)
    tokens = tokenize(text)
    total_words = len(tokens)
    unique_words = len(set(tokens))
    keywords = keyword_counts(tokens)

    spell_info = None
    if run_spell:
        try:
            spell_info = spell_check(tokens)
        except RuntimeError as e:
            spell_info = {"error": str(e)}

    result = {
        "file": str(path),
        "total_words": total_words,
        "unique_words": unique_words,
        "keyword_counts": keywords,
        "spell_check": spell_info,
    }

    return result


def aggregate_texts_from_dir(dirpath: str, glob_pattern: str = "*.txt", outpath: str = None) -> str:
    """Concatenate all text files matching glob_pattern in dirpath (sorted)
    and write to outpath if provided. Returns aggregated text string.
    """
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Vision directory not found: {dirpath}")

    files = sorted(glob.glob(str(p / glob_pattern)))
    texts = []
    for f in files:
        try:
            t = Path(f).read_text(encoding="utf-8", errors="ignore")
            texts.append(t)
        except Exception:
            # skip unreadable files
            continue

    agg = "\n\n".join(texts)

    if outpath:
        outp = Path(outpath)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(agg, encoding="utf-8")

    return agg


def token_edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Compute Levenshtein edit distance between token sequences (reference, hypothesis)."""
    # classic DP
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1,      # deletion
                         cur[j - 1] + 1,    # insertion
                         prev[j - 1] + cost) # substitution
        prev = cur
    return prev[m]


def word_error_rate(ref_text: str, hyp_text: str) -> float:
    r_tokens = tokenize(ref_text)
    h_tokens = tokenize(hyp_text)
    if len(r_tokens) == 0:
        return float('inf') if len(h_tokens) > 0 else 0.0
    ed = token_edit_distance(r_tokens, h_tokens)
    return ed / max(1, len(r_tokens))


def char_edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings (character-level)."""
    # Use a memory-efficient DP: O(min(len(a), len(b))) space
    if a == b:
        return 0
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # Ensure n <= m to use less space
    if n > m:
        a, b = b, a
        n, m = m, n

    previous = list(range(n + 1))
    for j in range(1, m + 1):
        c = b[j - 1]
        current = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == c else 1
            current[i] = min(previous[i] + 1,      # deletion
                             current[i - 1] + 1,    # insertion
                             previous[i - 1] + cost) # substitution
        previous = current
    return previous[n]


def char_error_rate(ref_text: str, hyp_text: str) -> float:
    ed = char_edit_distance(ref_text, hyp_text)
    ref_len = len(ref_text)
    if ref_len == 0:
        return float('inf') if len(hyp_text) > 0 else 0.0
    return ed / max(1, ref_len)



def token_jaccard(a_text: str, b_text: str) -> float:
    a = set(tokenize(a_text))
    b = set(tokenize(b_text))
    if not a and not b:
        return 1.0
    inter = a & b
    uni = a | b
    return len(inter) / len(uni) if uni else 0.0


def ngram_jaccard(a_text: str, b_text: str, n: int = 3) -> float:
    def ngrams(tokens: List[str], n: int):
        return {" ".join(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}

    at = tokenize(a_text)
    bt = tokenize(b_text)
    an = ngrams(at, n)
    bn = ngrams(bt, n)
    if not an and not bn:
        return 1.0
    inter = an & bn
    uni = an | bn
    return len(inter) / len(uni) if uni else 0.0


def char_similarity(a_text: str, b_text: str) -> float:
    # difflib.SequenceMatcher ratio gives a quick char-level similarity
    return difflib.SequenceMatcher(None, a_text, b_text).ratio()


def normalize_text(text: str) -> str:
    """Simple normalization:
    - Lowercase
    - Remove punctuation (keep only letters/digits and whitespace)
    - Collapse whitespace
    """
    text = text.lower()
    # remove anything that's not a letter, digit or whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sample_unified_diff(a_text: str, b_text: str, nlines: int = 20) -> str:
    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, lineterm="")
    sample = []
    for i, line in enumerate(diff):
        if i >= nlines:
            break
        sample.append(line)
    return "\n".join(sample)


def compare_texts(ref_text: str, hyp_text: str) -> Dict:
    """Return a dictionary of comparison metrics between reference and hypothesis texts."""
    metrics = {}
    metrics["char_similarity"] = char_similarity(ref_text, hyp_text)
    metrics["word_error_rate"] = word_error_rate(ref_text, hyp_text)
    metrics["token_jaccard"] = token_jaccard(ref_text, hyp_text)
    metrics["ngram_jaccard_3"] = ngram_jaccard(ref_text, hyp_text, n=3)
    metrics["ngram_jaccard_5"] = ngram_jaccard(ref_text, hyp_text, n=5)
    metrics["ref_token_count"] = len(tokenize(ref_text))
    metrics["hyp_token_count"] = len(tokenize(hyp_text))
    metrics["diff_sample"] = sample_unified_diff(ref_text, hyp_text, nlines=60)
    # character-level edit distance and normalized char error rate
    metrics["char_edit_distance"] = char_edit_distance(ref_text, hyp_text)
    metrics["char_error_rate"] = char_error_rate(ref_text, hyp_text)
    # keyword counts for both texts
    try:
        metrics["ref_keyword_counts"] = keyword_counts(tokenize(ref_text))
        metrics["hyp_keyword_counts"] = keyword_counts(tokenize(hyp_text))
        # difference hyp - ref for each keyword
        metrics["keyword_count_diff"] = {k: metrics["hyp_keyword_counts"].get(k, 0) - metrics["ref_keyword_counts"].get(k, 0) for k in KEYWORDS}
    except Exception:
        metrics["ref_keyword_counts"] = {}
        metrics["hyp_keyword_counts"] = {}
        metrics["keyword_count_diff"] = {}
    return metrics


def compare_files(ref_path: Path, hyp_path: Path) -> Dict:
    ref_text = load_text(ref_path)
    hyp_text = load_text(hyp_path)
    return compare_texts(ref_text, hyp_text)


def main():
    parser = argparse.ArgumentParser(description="OCR quality metrics and comparison utilities")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--analyze", help="Path to a single text file to analyze")
    group.add_argument("--aggregate-vision", help="Directory with AI Vision OCR text files to aggregate")
    parser.add_argument("--vision-glob", default="*_vision_transcription.txt", help="Glob for vision files when aggregating")
    parser.add_argument("--out-agg", help="Output path for aggregated vision text")
    parser.add_argument("--compare", nargs=2, metavar=("REF","HYP"), help="Compare two text files: REF HYP")
    parser.add_argument("--compare-out", help="Optional JSON file to write comparison results")
    parser.add_argument("--normalize", action="store_true", help="Normalize texts before comparing (lowercase, strip punctuation)")
    parser.add_argument("--align-compare", action="store_true", help="Align each vision file to the aggregated reference and compare (writes per-file JSON + summary CSV)")
    parser.add_argument("--ref-agg", help="Reference aggregated text file (the LOC aggregated txt)")
    parser.add_argument("--vision-dir", help="Directory with per-vision text files to align and compare")
    parser.add_argument("--out-dir", help="Output directory for aligned comparisons (JSON per file + summary CSV)")
    parser.add_argument("--output", help="Optional JSON file to write single-file analysis results")
    parser.add_argument("--max-suggestions", type=int, default=10, help="Max misspelling suggestions to show")

    args = parser.parse_args()

    if args.analyze:
        p = Path(args.analyze)
        if not p.exists():
            print(f"Input file not found: {p}")
            raise SystemExit(2)
        metrics = analyze_file(p, run_spell=(SpellChecker is not None))

        # Print short report
        print("\nOCR Quality Metrics")
        print("=" * 40)
        print(f"File: {metrics['file']}")
        print(f"Total words: {metrics['total_words']}")
        print(f"Unique words: {metrics['unique_words']}")
        print("Keyword counts:")
        for k, v in metrics['keyword_counts'].items():
            print(f"  {k}: {v}")

        sc = metrics['spell_check']
        if isinstance(sc, dict) and sc.get('error'):
            print("\nSpell-check: NOT RUN")
            print(sc.get('error'))
        else:
            print("\nSpell-check summary:")
            if sc:
                print(f"  Unique words checked: {sc.get('total_unique_words')}")
                print(f"  Misspelled (unique): {sc.get('misspelled_count')}")
                if sc.get('misspelled_samples'):
                    print("  Top misspelled examples:")
                    for w in sc['misspelled_samples']:
                        s = sc['suggestions'].get(w, [])
                        print(f"    {w} -> suggestions: {', '.join(s) if s else 'none'}")

        if args.output:
            outp = Path(args.output)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nResults written to {outp}")

        return

    if args.aggregate_vision:
        agg = aggregate_texts_from_dir(args.aggregate_vision, glob_pattern=args.vision_glob, outpath=args.out_agg)
        if args.out_agg:
            print(f"Aggregated vision OCR written to: {args.out_agg}")
        else:
            print("Aggregated vision OCR (to stdout):\n")
            print(agg[:1000])
        return

    if args.compare:
        ref, hyp = args.compare
        refp = Path(ref)
        hypp = Path(hyp)
        if not refp.exists():
            print(f"Reference file not found: {refp}")
            raise SystemExit(2)
        if not hypp.exists():
            print(f"Hypothesis file not found: {hypp}")
            raise SystemExit(2)

        # Optionally normalize before comparison
        if args.normalize:
            ref_text = normalize_text(load_text(refp))
            hyp_text = normalize_text(load_text(hypp))
            comp = compare_texts(ref_text, hyp_text)
        else:
            comp = compare_files(refp, hypp)
        # Reduce diff sample size printed
        print("\nComparison summary")
        print("=" * 40)
        print(f"Char similarity (0-1): {comp['char_similarity']:.4f}")
        wer = comp['word_error_rate']
        print(f"Word Error Rate (WER): {wer:.4f}")
        print(f"Token Jaccard: {comp['token_jaccard']:.4f}")
        print(f"3-gram Jaccard: {comp['ngram_jaccard_3']:.4f}")
        print(f"Ref tokens: {comp['ref_token_count']}  Hyp tokens: {comp['hyp_token_count']}")

        # print small diff sample (first 60 lines)
        ds = comp.get('diff_sample', '')
        if ds:
            print("\nUnified diff (sample):")
            print(ds)

        if args.compare_out:
            outp = Path(args.compare_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nComparison results written to {outp}")

        return

    if args.align_compare:
        # Require ref-agg, vision-dir, out-dir
        if not args.ref_agg or not args.vision_dir or not args.out_dir:
            print("--align-compare requires --ref-agg, --vision-dir and --out-dir")
            raise SystemExit(2)

        refp = Path(args.ref_agg)
        vdir = Path(args.vision_dir)
        outd = Path(args.out_dir)
        if not refp.exists():
            print(f"Reference aggregated file not found: {refp}")
            raise SystemExit(2)
        if not vdir.exists() or not vdir.is_dir():
            print(f"Vision directory not found: {vdir}")
            raise SystemExit(2)

        outd.mkdir(parents=True, exist_ok=True)

        ref_text_full = load_text(refp)
        if args.normalize:
            ref_text_full = normalize_text(ref_text_full)
        ref_tokens = tokenize(ref_text_full)

        import csv

        summary_rows = []
        vision_files = sorted(glob.glob(str(vdir / args.vision_glob)))
        if not vision_files:
            print(f"No vision files found with glob: {vdir / args.vision_glob}")
            raise SystemExit(2)

        for vf in vision_files:
            vpath = Path(vf)
            hyp_text = load_text(vpath)
            if args.normalize:
                hyp_text_proc = normalize_text(hyp_text)
            else:
                hyp_text_proc = hyp_text
            hyp_tokens = tokenize(hyp_text_proc)
            m = len(hyp_tokens)
            if m == 0:
                print(f"Skipping empty vision file: {vf}")
                continue

            # sliding window search for best matching segment in ref_tokens
            best_score = -1.0
            best_start = 0
            step = max(1, m // 4)
            max_start = max(0, len(ref_tokens) - m)
            for start in range(0, max_start + 1, step):
                window = ref_tokens[start:start + m]
                win_text = " ".join(window)
                score = token_jaccard(win_text, hyp_text_proc)
                if score > best_score:
                    best_score = score
                    best_start = start

            # Extract best window text
            best_window_tokens = ref_tokens[best_start:best_start + m]
            best_window_text = " ".join(best_window_tokens)

            # Compute final metrics on these aligned texts
            metrics = compare_texts(best_window_text, hyp_text_proc)
            # Add info for traceability
            metrics['vision_file'] = str(vpath)
            metrics['ref_window_start_token'] = best_start
            metrics['ref_window_end_token'] = best_start + m
            # write per-file JSON
            outjson = outd / (vpath.stem + '.json')
            outjson.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

            summary_rows.append({
                'vision_file': vpath.name,
                'vision_tokens': len(hyp_tokens),
                'ref_window_start': best_start,
                'ref_window_end': best_start + m,
                'token_jaccard': metrics.get('token_jaccard'),
                'word_error_rate': metrics.get('word_error_rate'),
                'char_similarity': metrics.get('char_similarity')
            })

            print(f"Compared {vpath.name}: token_jaccard={metrics.get('token_jaccard'):.4f} wer={metrics.get('word_error_rate'):.4f}")

        # write summary CSV
        csvp = outd / 'summary.csv'
        with csvp.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)

        print(f"\nAlignment + comparison results written to: {outd} (summary.csv + per-file json)")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
