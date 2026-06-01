"""Lightweight English-language filter — no extra pip dependencies.

Strategy (three complementary checks, any one failure rejects the text):
1. Script check  — reject if > 15 % of alphabetic characters are non-Latin.
2. ASCII ratio   — reject if < 70 % of characters are plain ASCII printable.
3. Stopword ratio — reject if fewer than 1 common English stopword appears per
                    10 words (catches romanised non-English or gibberish).

No langdetect, no fasttext, no extra downloads beyond what is already in
requirements.txt (nltk stopwords are already used by preprocessor.py).
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Build a fast lookup set of common English stopwords + function words.
# We avoid importing nltk here so the filter works even before corpora
# are downloaded — we carry a minimal hardcoded set as fallback.
# ---------------------------------------------------------------------------

_FALLBACK_STOPWORDS: frozenset[str] = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am is are
    was were be been being have has had having do does did doing a an the
    and but if or because as until while of at by for with about against
    between into through during before after above below to from up down
    in out on off over under again further then once here there when where
    why how all both each few more most other some such no nor not only
    own same so than too very s t can will just don should now d ll m o
    re ve y ain aren couldn didn doesn hadn hasn haven isn mightn mustn
    needn shan shouldn wasn weren won wouldn
    i'm you're he's she's it's we're they're i've you've we've they've
    i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll
    that's what's let's
    """.split()
)


def _load_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords as _sw
        return frozenset(_sw.words("english"))
    except Exception:
        return _FALLBACK_STOPWORDS


_STOPWORDS: frozenset[str] = _load_stopwords()

# Regex: matches one or more alphabetic characters in any script
_ALPHA_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Unicode block ranges that are NOT Latin/Basic-Latin/Latin-Extended
# We check by Unicode category: 'L' = letter; then script via name
_NON_LATIN_SCRIPTS = {
    "CJK",       # Chinese, Japanese, Korean
    "HANGUL",    # Korean
    "HIRAGANA",  # Japanese
    "KATAKANA",  # Japanese
    "ARABIC",    # Arabic
    "HEBREW",    # Hebrew
    "DEVANAGARI",# Hindi etc.
    "CYRILLIC",  # Russian etc.
    "THAI",
    "GREEK",
    "GEORGIAN",
    "ARMENIAN",
    "ETHIOPIC",
}


def _char_script(ch: str) -> str:
    """Return the Unicode script name (upper-case) for a single character."""
    try:
        name = unicodedata.name(ch, "")
        # name is like "LATIN SMALL LETTER A" or "CJK UNIFIED IDEOGRAPH-4E16"
        return name.split()[0] if name else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def is_english_line(line: str, min_stopword_density: float = 0.08) -> bool:
    """Return True if *line* is likely English.

    Parameters
    ----------
    line:
        A single lyric line.
    min_stopword_density:
        Minimum fraction of tokens that must be English stopwords.
        Default 0.08 (≈ 1 stopword per 12 words) — intentionally lenient
        to allow poetic / fragmented lines.
    """
    if not line or not line.strip():
        return False

    # --- 1. Script check ------------------------------------------------
    letters = [ch for ch in line if unicodedata.category(ch).startswith("L")]
    if letters:
        non_latin = sum(
            1 for ch in letters
            if _char_script(ch) in _NON_LATIN_SCRIPTS
        )
        if non_latin / len(letters) > 0.15:
            return False

    # --- 2. ASCII printable ratio ---------------------------------------
    printable_ascii = sum(1 for ch in line if 0x20 <= ord(ch) <= 0x7E)
    if len(line) > 0 and printable_ascii / len(line) < 0.70:
        return False

    # --- 3. Vowel ratio — English text is ~38-42% vowels ---------------
    letters_lower = re.findall(r"[a-z]", line.lower())
    if letters_lower:
        vowels = sum(1 for ch in letters_lower if ch in "aeiou")
        if vowels / len(letters_lower) < 0.18:
            return False

    # --- 4. English stopword density ------------------------------------
    tokens = re.findall(r"[a-z']+", line.lower())
    if not tokens:
        return False
    stopword_hits = sum(1 for t in tokens if t in _STOPWORDS or t.strip("'") in _STOPWORDS)
    density = stopword_hits / len(tokens)
    if density < min_stopword_density and len(tokens) >= 4:
        # No stopwords AND > 4 tokens → very likely non-English
        return False

    return True


def filter_english_lines(lines: list[str]) -> list[str]:
    """Remove non-English lines from a list; return filtered list."""
    return [ln for ln in lines if is_english_line(ln)]


def filter_english_corpus(series: "pd.Series") -> "pd.Series":  # noqa: F821
    """Filter a pandas Series of lyric strings to rows that are likely English.

    A lyric entry is kept if at least 60 % of its non-empty lines pass the
    English check.  This handles songs that mix a few non-English ad-libs
    into otherwise English lyrics (common in K-pop / Reggaeton crossovers).
    """
    import pandas as pd  # local import so module stays importable without pandas

    def _keep(lyric: str) -> bool:
        if not isinstance(lyric, str) or not lyric.strip():
            return False
        raw_lines = [ln.strip() for ln in lyric.splitlines() if ln.strip()]
        if not raw_lines:
            return False
        english_count = sum(1 for ln in raw_lines if is_english_line(ln))
        return english_count / len(raw_lines) >= 0.60

    return series[series.apply(_keep)]
