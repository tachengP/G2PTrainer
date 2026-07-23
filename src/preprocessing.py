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
        """Return a sequence of source units.

        * Latin-family spans are BPE-sub-word tokenised.
        * CJK characters are emitted one per unit.
        * Other characters (punctuation, digits) are emitted as single units.
        """
        units: List[str] = []
        for token in text.split():
            # split a whitespace token into runs by script
            buf: List[str] = []
            for ch in token:
                if is_cjk(ch):
                    if buf:
                        units.extend(self.bpe.encode("".join(buf)))
                        buf = []
                    units.append(ch)
                else:
                    buf.append(ch)
            if buf:
                units.extend(self.bpe.encode("".join(buf)))
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


def record_targets(rec: Dict[str, str]) -> Dict[str, List[str]]:
    """Convert a raw CSV record (dict with canonical keys) into the four targets.

    Every target is framed with ``<sos>`` / ``<eos>`` so that teacher-forcing
    training and greedy inference share the *exact* same start/stop symbols.
    Without these, the decoder is never fed ``<sos>`` during training (its
    embedding row stays randomly initialised) while ``generate`` starts from
    ``<sos>`` -- the train/infer mismatch makes greedy decoding collapse into
    repetitive loops that run to ``max_len``.
    """
    return {
        "phonemes": [SOS] + parse_no_sep(rec["phonemes"]) + [EOS],
        "separated_graphmes": [SOS] + parse_aligned_column(rec["separated_graphmes"], PIPE, split_within=False) + [EOS],
        "separated_phonemes": [SOS] + parse_aligned_column(rec["separated_phonemes"], PIPE, split_within=True) + [EOS],
        "aligned_phonemes": [SOS] + parse_aligned_column(rec["aligned_phonemes"], SLASH, split_within=True) + [EOS],
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
