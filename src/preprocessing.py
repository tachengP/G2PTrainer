"""Script-aware preprocessing, BPE sub-word tokenisation and vocabulary building.

The grapheme (source) side uses a BPE-trained sub-word vocabulary so that even
enormous orthographic inventories collapse to a small, fixed-size embedding
table.  CJK-like characters are kept as single-character units (already a small,
closed set) and are *not* BPE-merged.

Language labels are intentionally NOT handled here -- they are parsed from the
dataset file name by :mod:`src.data`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Special symbols (shared across all vocabularies)
# --------------------------------------------------------------------------- #
PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

SPECIALS = [PAD, SOS, EOS, UNK]

# Separators used by the multi-task targets
PIPE = "|"
SLASH = "/"

# The four multitask targets (order is fixed; the model builds one decoder each).
# The names below are EXACTLY the CSV header columns so the data layer stays in
# sync with the dataset.  Space, '|' and '/' are all just separators of varying
# granularity:
#   1. phonemes         -> phoneme sequence, no separator (from CSV `phonemes`)
#   2. separated_graphmes -> graphemes separated by '|' (from CSV `separated_graphmes`)
#   3. separated_phonemes -> phonemes separated by '|' (from CSV `separated_phonemes`)
#   4. aligned_phonemes   -> phonemes separated by '/' (from CSV `aligned_phonemes`)
#
# Consistency invariants enforced during training:
#   separated_graphmes (drop '|') == graphmes
#   separated_phonemes (drop '|') == phonemes
#   aligned_phonemes    (drop '/') == phonemes
TARGET_NAMES = ["phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes"]

# --------------------------------------------------------------------------- #
# Script detection
# --------------------------------------------------------------------------- #
# CJK unified ideographs, extension A/B, Hangul syllables/jamo, Hiragana,
# Katakana, Bopomofo, Kanbun.  Anything in these ranges is treated per-character.
_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\ua000-\ua48f\uac00-\ud7af"
    r"\uf900-\ufaff\ufb00-\ufb4f\uff00-\uffef]"
)

# Latin letters / digits / common ASCII punctuation -> BPE-able.
_LATIN_RE = re.compile(r"[A-Za-z0-9']")


def is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def is_latin(ch: str) -> bool:
    return bool(_LATIN_RE.match(ch))


def script_of(ch: str) -> str:
    if is_cjk(ch):
        return "cjk"
    if is_latin(ch):
        return "latin"
    return "other"


# --------------------------------------------------------------------------- #
# BPE sub-word tokenizer (word-based, GPT-2 style merges)
# --------------------------------------------------------------------------- #
class BPETokenizer:
    """Minimal word-based BPE.

    Only Latin/ASCII tokens are merged; CJK characters are never merged (they are
    emitted as single characters by the caller).  This bounds the vocabulary size
    and prevents the "dimension explosion" problem for huge grapheme sets.
    """

    def __init__(self) -> None:
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self.base_chars: Set[str] = set()
        self.cache: Dict[str, Tuple[str, ...]] = {}

    # ---- training -------------------------------------------------------- #
    def train(self, words: Iterable[str], num_merges: int) -> None:
        # words: a stream of whitespace-delimited latin tokens.
        word_freq: Counter = Counter()
        for w in words:
            w = w.strip()
            if not w:
                continue
            self.base_chars.update(w)
            word_freq[ tuple(w) ] += 1

        if not word_freq:
            return

        for _ in range(num_merges):
            pairs = self._get_pair_stats(word_freq)
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
            self.bpe_ranks[best] = len(self.bpe_ranks)
            word_freq = self._merge_pair(best, word_freq)

    @staticmethod
    def _get_pair_stats(word_freq: Counter) -> Dict[Tuple[str, str], int]:
        stats: Counter = Counter()
        for word, freq in word_freq.items():
            for pair in zip(word, word[1:]):
                stats[pair] += freq
        return stats

    def _merge_pair(self, pair: Tuple[str, str], word_freq: Counter) -> Counter:
        new_freq: Counter = Counter()
        pat = re.compile(r"(?<!^)" + re.escape(pair[0]) + r"(?!$)")
        # simple contiguous merge: combine only adjacent equal pair occurrences
        for word, freq in word_freq.items():
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freq[tuple(new_word)] += freq
        return new_freq

    # ---- inference ------------------------------------------------------- #
    def encode(self, word: str) -> List[str]:
        word = word.strip()
        if not word:
            return []
        if word in self.cache:
            return list(self.cache[word])

        symbols = list(word)
        if len(symbols) < 2:
            self.cache[word] = tuple(symbols)
            return symbols

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # pick the pair with the lowest merge rank
            min_rank = None
            min_idx = -1
            for idx, p in enumerate(pairs):
                r = self.bpe_ranks.get(p)
                if r is not None and (min_rank is None or r < min_rank):
                    min_rank = r
                    min_idx = idx
            if min_idx == -1:
                break
            # merge at min_idx
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i == min_idx:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        self.cache[word] = tuple(symbols)
        return symbols


# --------------------------------------------------------------------------- #
# Script tokenizer: produces source-side units (BPE subwords for latin,
# single chars for CJK).  These units are exactly what the grapheme-reconstruction
# target (task 2) also uses, keeping its vocabulary identical to the source.
# --------------------------------------------------------------------------- #
class ScriptTokenizer:
    def __init__(self, bpe: Optional[BPETokenizer] = None) -> None:
        self.bpe = bpe or BPETokenizer()

    def tokenize(self, text: str) -> List[str]:
        """Return a sequence of source units, one per character.

        The source is kept at character granularity on purpose: grapheme
        segmentation (``separated_graphmes``) is counted per character in
        convert_csv, so ``len(units)`` must equal the character count for the
        consistency check in ``bin_data`` to hold.  BPE sub-word merging would
        collapse non-CJK spans (e.g. romanized Korean ``ne ga`` -> 2 units)
        while ``separated_graphmes`` still sums to 4 characters, silently
        dropping every such row.
        """
        units: List[str] = []
        for token in text.split():
            for ch in token:
                units.append(ch)
        return units

    def train_bpe(self, texts: Iterable[str], num_merges: int) -> None:
        latin_words: List[str] = []
        for text in texts:
            for token in text.split():
                if any(is_latin(ch) for ch in token) and not any(is_cjk(ch) for ch in token):
                    latin_words.append(token)
        self.bpe.train(latin_words, num_merges)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, symbols: Iterable[str], specials: List[str] = SPECIALS):
        self.specials = list(specials)
        self._sym2idx: Dict[str, int] = {}
        self._idx2sym: List[str] = []
        for s in self.specials:
            self._add(s)
        for s in symbols:
            if s not in self._sym2idx:
                self._add(s)
        self.unk_idx = self._sym2idx[UNK]
        self.pad_idx = self._sym2idx[PAD]
        self.sos_idx = self._sym2idx[SOS]
        self.eos_idx = self._sym2idx[EOS]

    def _add(self, sym: str) -> None:
        self._sym2idx[sym] = len(self._idx2sym)
        self._idx2sym.append(sym)

    def __len__(self) -> int:
        return len(self._idx2sym)

    def stoi(self, sym: str) -> int:
        return self._sym2idx.get(sym, self.unk_idx)

    def itos(self, idx: int) -> str:
        if 0 <= idx < len(self._idx2sym):
            return self._idx2sym[idx]
        return UNK

    def encode(self, symbols: Iterable[str]) -> List[int]:
        return [self.stoi(s) for s in symbols]

    def decode(self, indices: Iterable[int], join: bool = False) -> List[str]:
        seq = [self.itos(i) for i in indices]
        return "".join(seq) if join else seq

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in self._idx2sym:
                f.write(s + "\n")

    @staticmethod
    def from_file(path: str, specials: List[str] = SPECIALS) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            syms = [line.rstrip("\n") for line in f]
        return Vocab(syms, specials=specials)


# --------------------------------------------------------------------------- #
# CSV column mapping (synchronised with the real dataset header).
#
# The training CSV has the header:
#   graphmes, phonemes, separated_graphmes, separated_phonemes, aligned_phonemes
#
# Each field maps to one of our four multitask targets (plus the source).  The
# keys of this dict ARE the canonical task names used everywhere downstream, so
# they match the CSV header columns exactly:
#   graphmes           -> source grapheme sequence (tokenised per-language)
#   phonemes           -> task "phonemes"         (no separator)
#   separated_graphmes -> task "separated_graphmes" (groups joined by '|')
#   separated_phonemes -> task "separated_phonemes" (groups joined by '|')
#   aligned_phonemes   -> task "aligned_phonemes"   (groups joined by '/')
#
# Note: the source column in the real data is spelled "graphmes" (typo); we
# also accept the canonical "graphemes" for robustness.
# --------------------------------------------------------------------------- #
CSV_COLUMNS = {
    "src": ["graphmes", "graphemes"],
    "phonemes": ["phonemes"],
    "separated_graphmes": ["separated_graphmes", "separated_graphemes"],
    "separated_phonemes": ["separated_phonemes"],
    "aligned_phonemes": ["aligned_phonemes"],
}


def parse_no_sep(text: str) -> List[str]:
    """Parse a space-separated symbol string into a flat symbol list.

    Used for the 'phonemes' (no separator) target, e.g. "n e g a" -> ['n','e','g','a'].
    """
    return text.split()


def parse_aligned_column(text: str, sep: str, split_within: bool) -> List[str]:
    """Parse a grouped CSV column into a symbol sequence.

    ``text`` looks like ``"group1<sep>group2<sep>group3"``. The separator ``sep``
    is emitted as a vocabulary symbol *between* groups. When ``split_within`` is
    True each group is further split on whitespace (phonemes, e.g. "n e"); when
    False the whole group is a single symbol (graphemes, e.g. "ㄴㅐ").

    Examples
    --------
    parse_aligned_column("ㄴㅐ|ㄱㅏ", '|', split_within=False) -> ['ㄴㅐ','|','ㄱㅏ']
    parse_aligned_column("n e|g a", '|', split_within=True)  -> ['n','e','|','g','a']
    parse_aligned_column("n/e g/a", '/', split_within=True)  -> ['n','/','e','g','/','a']
    """
    groups = text.split(sep)
    syms: List[str] = []
    for i, g in enumerate(groups):
        g = g.strip()
        if split_within:
            parts = g.split() if g else []
        else:
            parts = [g] if g else []
        syms.extend(parts)
        if i < len(groups) - 1:
            syms.append(sep)
    return syms


def parse_phoneme_counts(text: str, sep: str) -> List[int]:
    """Number of (whitespace-separated) phoneme units in each ``sep`` segment.

    Within a segment the phonemes are atomic, whitespace-separated tokens, so a
    segment's count is simply how many tokens it holds.  e.g. ``"n e|g a"`` with
    ``sep='|'`` -> ``[2, 2]``; ``"n/e g/a"`` with ``sep='/'`` -> ``[1, 2, 1]``.
    These counts always sum to the number of phoneme tokens in the base
    ``phonemes`` column, which is exactly what the model must reproduce at infer
    time when it regroups the predicted phonemes.
    """
    return [len(seg.split()) for seg in text.split(sep)]


def parse_grapheme_counts(text: str, src_units: Optional[List[str]] = None) -> List[int]:
    """Number of grapheme units per ``'|'`` segment of ``text``.

    ``|`` is just a visual stand-in for the space that normally separates grapheme
    groups, so each segment is a contiguous run of graphemes.  Two modes:

    * ``src_units`` given (the BPE-tokenised *source*) -> count source tokens per
      segment, so the counts sum *exactly* to ``len(src_units)`` and reconstruction
      can regroup the encoder's real input units.  This is the production path.
    * ``src_units`` is ``None`` -> fall back to raw characters (correct for CJK
      where one char == one grapheme and BPE keeps them as singletons).  Used by
      the offline CSV conversion helper that has no tokenizer.
    """
    segs = text.split(PIPE)
    if src_units is None:
        return [len(seg.replace(" ", "")) for seg in segs]
    # Align each source token to the segment of its starting character.  The source
    # is exactly ``text`` with every '|' removed, so char i of the source maps to
    # seg_of_char[i]; walking the tokens consumes those chars in order.
    seg_of_char: List[int] = []
    cur = 0
    for ch in text:
        if ch == PIPE:
            cur += 1
        else:
            seg_of_char.append(cur)
    counts = [0] * (cur + 1)
    cursor = 0
    for tok in src_units:
        s = seg_of_char[cursor] if cursor < len(seg_of_char) else cur
        counts[s] += 1
        cursor += len(tok)
    return counts


class CountCodec:
    """Compact codec for the three *derived* tasks (separated_*/aligned_*).

    Instead of emitting the full separator-inserted string the model predicts a
    flat sequence of *segment-length* integers -- e.g. the grapheme grouping
    ``ㄴㅐ|ㄱ㏘`` becomes ``[2, 2]``.  At inference we regroup the base sequence
    (graphmes / phonemes) by those counts, which removes the old KL consistency
    supervision entirely and shrinks the output vocabulary from the full
    phoneme/grapheme set down to ``max_count + 4``.

    Token ids reuse the phoneme vocab's PAD/SOS/EOS convention so the shared
    ``generate`` loop needs no special-casing::

        id 0 = PAD, 1 = SOS, 2 = EOS, and a count value ``c`` -> id ``c + 3``.

    ``c`` may be ``0`` (a leading/trailing empty alignment block, e.g. the silent
    initial ``ㅇ`` of ``아`` -> ``/a`` -> ``[0, 1]``), which maps to id ``3``.
    """

    def __init__(self, max_count: int):
        self.max_count = int(max_count)
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.vocab_size = self.max_count + 4  # 0..2 reserved, 3..max+3 = counts 0..max

    def encode(self, counts: List[int]) -> List[int]:
        ids = [self.sos_idx]
        for c in counts:
            c = int(c)
            if c < 0:
                c = 0
            if c > self.max_count:
                c = self.max_count
            ids.append(c + 3)
        ids.append(self.eos_idx)
        return ids

    def decode(self, ids: List[int]) -> List[int]:
        return [i - 3 for i in ids if i >= 3]


def reconstruct_groups(base: List[str], counts: List[int], seg_sep: str,
                       unit_sep: str) -> str:
    """Regroup ``base`` units into ``seg_sep``-joined segments per ``counts``.

    ``base`` is the base sequence: source units for ``separated_graphmes``,
    phoneme tokens for ``separated_/aligned_phonemes``.  ``unit_sep`` joins units
    *within* a segment (``""`` for graphemes, ``" "`` for phonemes); ``seg_sep``
    joins segments (``"|"`` for the separated tasks, ``"/"`` for aligned).
    """
    if not base:
        return ""
    if not counts:
        # no segment boundaries predicted -> the whole base is one group
        return unit_sep.join(base)
    groups: List[str] = []
    i = 0
    for c in counts:
        c = int(c)
        if c < 0:
            c = 0
        if i >= len(base) and c == 0:
            # trailing empty block beyond the base -> pure separator
            groups.append("")
            continue
        if i >= len(base):
            break
        groups.append(unit_sep.join(base[i:i + c]))
        i += c
    if i < len(base):
        groups.append(unit_sep.join(base[i:]))
    return seg_sep.join(groups)


def _fit_group_count(groups: List[List[str]], target: int) -> List[List[str]]:
    """Adjust ``groups`` to contain exactly ``target`` groups.

    Used to enforce structural length constraints -- e.g. ``aligned_phonemes``
    must be one group longer than ``separated_phonemes``.  When there are too
    many groups (the phoneme stream contained more vowels than syllables) we
    merge the trailing ones into the last kept group, preserving the leading
    onset.  When there are too few we split the last group into contiguous
    sub-groups until the target count is reached.
    """
    if not groups:
        return [[] for _ in range(target)]
    if len(groups) == target:
        return groups
    if len(groups) > target:
        head = groups[:target - 1]
        merged = [u for g in groups[target - 1:] for u in g]
        return head + [merged]
    # too few groups: split the last group into (deficit + 1) contiguous parts
    deficit = target - len(groups)
    last = groups[-1]
    groups = groups[:-1]
    k = deficit + 1
    n = len(last)
    if n == 0:
        groups.extend([[] for _ in range(k)])
        return groups[:target]
    step = max(1, n // k)
    parts = [last[i * step:(i + 1) * step] for i in range(k)]
    if sum(len(p) for p in parts) < n:
        parts[-1] = parts[-1] + last[step * k:]
    groups.extend(parts)
    return groups[:target]


def reconstruct_aligned_vowel_led(base_units: List[str], vowel_set: set,
                                  seg_sep: str, unit_sep: str,
                                  expected_n: Optional[int] = None) -> str:
    """Deterministic ``aligned_phonemes`` grouping: ``[leading-onset-run]``
    followed by ``[vowel + following-onset-run]*`` -- one group per vowel, with
    the consonant run *between* two vowels attached to the **previous** vowel's
    group.

    Unlike syllable (C*VC*) grouping, this rule is unambiguous straight from the
    phoneme stream + vowel set (it never has to decide coda-vs-onset), so it is
    safe to use at inference instead of trusting the count head.

    For one-char-one-syllable languages the number of groups is ``num_vowels + 1``
    (or ``num_vowels`` when the word starts with a vowel).  Because each syllable
    has exactly one vowel, ``num_vowels == len(separated_phonemes) == N`` (the
    syllable/character count), so the aligned length is ``N + 1`` when the word
    begins with a consonant and ``N`` when it begins with a vowel -- verified on
    the Korean gold data (92.8% are ``N+1``, 7.2% are ``N``).

    When ``expected_n`` (= the separated/syllable count ``N``) is supplied, the
    group count is *forced* to that relationship regardless of how many vowels
    the (possibly mis-predicted) phoneme stream contains: a leading onset group
    is added unless the word begins with a vowel, and any over/under-count is
    reconciled by :func:`_fit_group_count`.  This decouples the alignment length
    constraint from the phoneme-stream vowel count.
    """
    if not base_units:
        return ""
    vposs = [i for i, t in enumerate(base_units) if t in vowel_set]
    if not vposs:
        return unit_sep.join(base_units)
    groups: List[List[str]] = []
    # leading onset run before the first vowel (may be empty)
    groups.append(base_units[0:vposs[0]])
    for k, v in enumerate(vposs):
        nxt = vposs[k + 1] if k + 1 < len(vposs) else len(base_units)
        groups.append(base_units[v:nxt])
    groups = [g for g in groups if g]  # drop an empty leading group

    # ---- structural constraint: aligned length == separated length + (1 or 0) ----
    if expected_n is not None:
        starts_with_vowel = bool(base_units) and base_units[0] in vowel_set
        target = int(expected_n) + (0 if starts_with_vowel else 1)
        groups = _fit_group_count(groups, target)

    return seg_sep.join(unit_sep.join(g) for g in groups)


def reconstruct_separated_anchored(base_units: List[str], counts: List[int],
                                   n_groups: int, seg_sep: str,
                                   unit_sep: str) -> str:
    """Force ``separated_phonemes`` to have exactly ``n_groups`` groups, where
    ``n_groups`` is the number of source graphemes (== length of
    ``separated_graphmes``).  This is the hard structural constraint
    ``len(separated_phonemes) == len(separated_graphmes)``.

    Placement uses the model's predicted ``counts`` as *proportional weights* for
    where to cut the phoneme base, so the relative segmentation the count head
    learned is preserved while the total group count is pinned to ``n_groups``.
    The consonant between two vowels is treated as the onset of the next syllable
    (the ``separated_phonemes`` convention), which is what makes its length equal
    the syllable/character count.
    """
    if not base_units:
        return ""
    n_groups = max(1, int(n_groups))
    P = len(base_units)
    weights = [c for c in (counts or []) if isinstance(c, int) and c > 0]
    if not weights:
        weights = [1] * n_groups
    elif len(weights) < n_groups:
        # pad with the mean so every group gets a positive weight
        mean_w = max(1, sum(weights) // len(weights))
        weights = weights + [mean_w] * (n_groups - len(weights))
    else:
        weights = weights[:n_groups]
    total = sum(weights) or 1
    groups: List[List[str]] = []
    i = 0
    for k in range(n_groups):
        if k == n_groups - 1:
            c = P - i  # last group absorbs the exact remainder
        else:
            c = int(round(weights[k] / total * P))
            # keep >=1 token for this group and for every remaining group
            c = max(1, min(c, P - i - (n_groups - 1 - k)))
        c = max(1, min(c, P - i))
        if c <= 0:
            break
        groups.append(base_units[i:i + c])
        i += c
    if i < P:  # safety: append any leftover (shouldn't happen)
        groups.append(base_units[i:])
    return seg_sep.join(unit_sep.join(g) for g in groups)


def resegment_by_vowels(base_units: List[str], vowel_set: set, seg_sep: str,
                        unit_sep: str) -> str:
    """Regroup a flat phoneme-token sequence into syllables, each containing
    exactly one vowel, ignoring any prior count prediction.

    For CJK-like languages (``syllable_is_char``) the structural rule is
    syllable = ``C* V C*`` with at most one vowel nucleus.  We cut a new group at
    each vowel boundary, placing the cut *after the first consonant* of every
    consonant run that lies between two vowels (Korean codas are <= 1 consonant,
    so the first consonant of the run is the coda of the previous syllable and
    the remainder are onsets of the next).  Leading onset consonants attach to
    the first syllable; trailing coda consonants attach to the last.  The result
    always has exactly one group per vowel (== per syllable), which is what the
    count heads fail to learn on their own.

    Note: a single consonant between two vowels (K==1) is genuinely ambiguous
    (coda of the previous syllable vs onset of the next) from phonemes alone; the
    rule treats it as the previous syllable's coda, which is right for the common
    CVC case.  Perfect resolution needs the grapheme side (Hangul coda info).
    """
    if not base_units:
        return ""
    n = len(base_units)
    vposs = [i for i, t in enumerate(base_units) if t in vowel_set]
    if not vposs:
        return unit_sep.join(base_units)
    cuts = [0]
    for j in range(1, len(vposs)):
        run_start = vposs[j - 1] + 1
        run_end = vposs[j]            # exclusive
        if run_end > run_start:       # there is a consonant run between the vowels
            cuts.append(run_start + 1)  # coda <= 1: cut after the first consonant
        else:
            cuts.append(vposs[j])       # vowels adjacent: cut at the second vowel
    groups = []
    for k in range(len(cuts)):
        start = cuts[k]
        end = cuts[k + 1] if k + 1 < len(cuts) else n
        groups.append(base_units[start:end])
    return seg_sep.join(unit_sep.join(g) for g in groups)


def parse_counts(text: str) -> List[int]:
    """Parse an already-flat count column (space-separated integers) into a list.

    After ``src/convert_csv.py`` has run, the three derived CSV columns
    (``separated_graphmes`` / ``separated_phonemes`` / ``aligned_phonemes``) store
    the segment-length sequences directly, so they are read as-is instead of being
    re-split on separators.
    """
    return [int(x) for x in text.split() if x.strip() != ""]


def record_targets(rec: Dict[str, str], src_units=None) -> Dict[str, list]:
    """Convert a raw CSV record (dict with canonical keys) into the four targets.

    ``phonemes`` stays a symbol sequence framed with ``<sos>``/``<eos>`` (embedded
    via the phoneme table).  The three derived tasks are stored as flat *count*
    sequences (segment lengths, space-separated integers) -- exactly the
    representation the model predicts -- and are read directly with
    :func:`parse_counts`.  ``src_units`` is kept for call-site compatibility but is
    no longer needed once the CSV has been converted by ``src/convert_csv.py``.
    """
    return {
        "phonemes": [SOS] + parse_no_sep(rec["phonemes"]) + [EOS],
        "separated_graphmes": parse_counts(rec["separated_graphmes"]),
        "separated_phonemes": parse_counts(rec["separated_phonemes"]),
        "aligned_phonemes": parse_counts(rec["aligned_phonemes"]),
    }


def build_source_vocab(
    texts: Iterable[str],
    bpe_merges: int,
    min_freq: int = 1,
) -> Tuple[ScriptTokenizer, Vocab]:
    """Train a :class:`ScriptTokenizer` and build the reduced source vocab."""
    texts = list(texts)  # materialise: the corpus is iterated more than once below
    tokenizer = ScriptTokenizer()
    tokenizer.train_bpe(texts, bpe_merges)

    # base chars = all single characters seen in the data
    char_counter: Counter = Counter()
    for text in texts:
        for ch in text:
            if ch.isspace():
                continue
            char_counter[ch] += 1

    symbols: List[str] = []
    for ch, freq in char_counter.most_common():
        if freq >= min_freq:
            symbols.append(ch)
    # add any BPE merge products
    for (a, b) in sorted(tokenizer.bpe.bpe_ranks.keys(), key=lambda kv: tokenizer.bpe.bpe_ranks[kv]):
        merged = a + b
        if merged not in symbols:
            symbols.append(merged)

    vocab = Vocab(symbols)
    return tokenizer, vocab


# --------------------------------------------------------------------------- #
# BPE (de)serialisation -- required so inference can tokenise unseen text
# --------------------------------------------------------------------------- #
def save_bpe(tokenizer: "ScriptTokenizer", path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ch in sorted(tokenizer.bpe.base_chars):
            f.write(f"<char>\t{ch}\n")
        for (a, b), _r in sorted(
            tokenizer.bpe.bpe_ranks.items(), key=lambda kv: kv[1]
        ):
            f.write(f"{a}\t{b}\n")


def load_bpe(path: str) -> "ScriptTokenizer":
    bpe = BPETokenizer()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "<char>":
                bpe.base_chars.add(parts[1])
            elif len(parts) == 2:
                bpe.bpe_ranks[(parts[0], parts[1])] = len(bpe.bpe_ranks)
    return ScriptTokenizer(bpe)
