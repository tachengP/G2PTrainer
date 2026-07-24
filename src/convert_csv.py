"""Convert the three derived target columns of a G2P dataset CSV from the
separator-inserted string form into the compact *count* (segment-length)
sequences that the model actually predicts.

Original columns (separator form)::

    graphmes,phonemes,separated_graphmes,separated_phonemes,aligned_phonemes

where the last three are e.g. ``ㄴㅐ|ㄱ㏘``, ``n e|g a``, ``n/e g/a``.

Converted columns (count form)::

    graphmes,phonemes,separated_graphmes,separated_phonemes,aligned_phonemes

where the last three become space-separated integer segment lengths, e.g.
``2 2``, ``2 2``, ``1 2 1``. This matches the count-representation design
(``src/preprocessing.py`` ``CountCodec``): the model predicts these counts and
regroups the base graphmes / phonemes by them at inference, so the verbose
separator strings don't need to be stored.

Counting rules (identical to what ``record_targets`` used to do internally):

* ``separated_graphmes`` -- number of graphemes per ``|``-delimited segment
  (character-based; one grapheme per Hangul jamo/char).
* ``separated_phonemes`` -- number of phonemes per ``|``-delimited segment.
* ``aligned_phonemes`` -- number of phonemes per ``/``-delimited segment.

The script is idempotent: if a derived column already contains only space
separated integers, it is passed through unchanged.

Example::

    python -m src.convert_csv --input data/Korean/dataset-ko.csv --dry-run 5
    python -m src.convert_csv --input data/Korean/dataset-ko.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import PIPE, SLASH, parse_grapheme_counts, parse_phoneme_counts

# canonical column name -> list of accepted header variants (lower-cased)
_ALIASES = {
    "graphmes": ("graphmes", "graphemes"),
    "phonemes": ("phonemes",),
    "separated_graphmes": ("separated_graphmes",),
    "separated_phonemes": ("separated_phonemes",),
    "aligned_phonemes": ("aligned_phonemes",),
}

_COUNT_RE = re.compile(r"^\d+( \d+)*$")


def _is_count_string(s: str) -> bool:
    return bool(s) and _COUNT_RE.match(s.strip()) is not None


def convert_derived(sep_sg: str, sep_sp: str, sep_ap: str):
    """Return (sg_counts, sp_counts, ap_counts) as space-joined strings.

    Idempotent: already-count columns are passed through unchanged.
    """
    if _is_count_string(sep_sg):
        sg = sep_sg.strip()
    else:
        sg = " ".join(str(c) for c in parse_grapheme_counts(sep_sg))

    if _is_count_string(sep_sp):
        sp = sep_sp.strip()
    else:
        sp = " ".join(str(c) for c in parse_phoneme_counts(sep_sp, PIPE))

    if _is_count_string(sep_ap):
        ap = sep_ap.strip()
    else:
        ap = " ".join(str(c) for c in parse_phoneme_counts(sep_ap, SLASH))

    return sg, sp, ap


def _build_index(header):
    norm = {h.strip().lower(): i for i, h in enumerate(header)}
    idx = {}
    for canonical, variants in _ALIASES.items():
        for v in variants:
            if v in norm:
                idx[canonical] = norm[v]
                break
        else:
            raise SystemExit(f"[convert_csv] missing column for '{canonical}' in header: {header}")
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source dataset CSV")
    ap.add_argument("--output", default=None,
                    help="output CSV (default: overwrite --input in place)")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="print first N converted rows and exit without writing")
    args = ap.parse_args()

    out_path = args.output or args.input
    in_place = (out_path == args.input)

    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit("[convert_csv] empty input file")
        idx = _build_index(header)
        isg, isp, iap = idx["separated_graphmes"], idx["separated_phonemes"], idx["aligned_phonemes"]

        if args.dry_run:
            print("graphmes | phonemes | separated_graphmes | separated_phonemes | aligned_phonemes")
            print("-" * 80)
            for n, row in enumerate(reader):
                if n >= args.dry_run:
                    break
                sg, sp, ap = convert_derived(row[isg], row[isp], row[iap])
                print(f"{row[idx['graphmes']]} | {row[idx['phonemes']]} | {sg} | {sp} | {ap}")
            return

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(out_path)), suffix=".csv.tmp")
        os.close(tmp_fd)
        written = 0
        with open(tmp_path, "w", encoding="utf-8", newline="") as fo:
            writer = csv.writer(fo)
            writer.writerow(header)
            for row in reader:
                sg, sp, ap = convert_derived(row[isg], row[isp], row[iap])
                row[isg], row[isp], row[iap] = sg, sp, ap
                writer.writerow(row)
                written += 1
        # On Windows the target can be locked by a file watcher / IDE (WinError 32),
        # which blocks moving the original aside.  Try the in-place swap; if it
        # fails, fall back to a "*.conv" sibling so the conversion is never lost.
        try:
            if os.path.exists(out_path):
                backup = out_path + ".orig"
                if os.path.exists(backup):
                    os.remove(backup)
                os.rename(out_path, backup)
            os.rename(tmp_path, out_path)
            print(f"[convert_csv] converted {written} rows -> {out_path}")
        except OSError as e:
            conv = out_path + ".conv"
            if os.path.exists(conv):
                os.remove(conv)
            os.rename(tmp_path, conv)
            print(f"[convert_csv] target {out_path} is locked ({e}); wrote converted "
                  f"data to {conv} instead. Point file_glob at it, or close the file "
                  f"in your editor and re-run to overwrite in place.")


if __name__ == "__main__":
    main()
