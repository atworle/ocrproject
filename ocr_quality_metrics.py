#!/usr/bin/env python3
"""
ocr_quality_metrics.py

Take a text file (OCR output) and:
- Perform a simple spell-check (using pyspellchecker)
- Count occurrences of the keywords: liberty, freedom, rights
- Print a short report and optionally save JSON output with metrics

Usage:
    python ocr_quality_metrics.py /path/to/file.txt --output results.json

The script tokenizes on letters/apostrophes, lowercases, and treats words
with basic normalization. It reports total words, unique words, misspelled
count and examples of misspellings.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None


KEYWORDS = ["liberty", "freedom", "rights"]
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def tokenize(text: str):
    tokens = TOKEN_RE.findall(text)
    return [t.lower() for t in tokens]


def keyword_counts(tokens, keywords=KEYWORDS):
    c = Counter(tokens)
    return {k: c.get(k, 0) for k in keywords}


def spell_check(tokens, max_suggestions=10):
    if SpellChecker is None:
        raise RuntimeError("pyspellchecker is not installed. Install with: pip install pyspellchecker")

    spell = SpellChecker()
    # feed the spellchecker the token list (it builds internal frequency map)
    # but pyspellchecker operates on word level; duplicates don't matter for .unknown
    unique_words = set(tokens)
    misspelled = spell.unknown(unique_words)

    # Prepare suggestions for the most frequent misspellings
    freq = Counter(tokens)
    miss_freq = sorted(misspelled, key=lambda w: -freq[w])
    suggestions = {w: list(spell.candidates(w))[:5] for w in miss_freq[:max_suggestions]}

    return {
        "total_unique_words": len(unique_words),
        "misspelled_count": len(misspelled),
        "misspelled_samples": miss_freq[:max_suggestions],
        "suggestions": suggestions,
    }


def analyze_file(path: Path):
    text = load_text(path)
    tokens = tokenize(text)
    total_words = len(tokens)
    unique_words = len(set(tokens))
    keywords = keyword_counts(tokens)

    spell_info = None
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


def main():
    parser = argparse.ArgumentParser(description="Compute simple OCR quality metrics from a text file")
    parser.add_argument("input", help="Path to text file to analyze")
    parser.add_argument("--output", help="Optional JSON file to write results to")
    parser.add_argument("--max-suggestions", type=int, default=10, help="Max misspelling suggestions to show")

    args = parser.parse_args()
    p = Path(args.input)
    if not p.exists():
        print(f"Input file not found: {p}")
        raise SystemExit(2)

    metrics = analyze_file(p)

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
        print(f"  Unique words checked: {sc['total_unique_words']}")
        print(f"  Misspelled (unique): {sc['misspelled_count']}")
        if sc['misspelled_samples']:
            print("  Top misspelled examples:")
            for w in sc['misspelled_samples']:
                s = sc['suggestions'].get(w, [])
                print(f"    {w} -> suggestions: {', '.join(s) if s else 'none'}")

    if args.output:
        outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {outp}")


if __name__ == '__main__':
    main()
