"""Mood-based and artist-style lyrics generation using Markov chains.

All output is English-only.  A lightweight Unicode/stopword filter strips
non-English lines from training corpora and from generated output before
they reach the user.

Artist selection is style/dataset-based only — the generator does NOT clone
or impersonate the real artist.

Optional theme keywords (e.g. "heartbreak", "summer", "dreams") bias the
corpus towards songs that contain those words, strengthening thematic focus.

Lines within each section are sorted so that adjacent pairs prefer end-rhymes,
making the output feel more song-like without any extra dependencies.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import markovify
import pandas as pd

from src.data_loader import LyricsDataLoader
from src.english_filter import is_english_line
from src.preprocessor import LyricsPreprocessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOOD_VOCAB: dict[str, list[str]] = {
    "Happy": [
        "shine", "light", "love", "joy", "dance", "smile", "dream", "bright",
        "gold", "fly", "free", "alive", "sun", "rise", "heart", "glow",
    ],
    "Sad": [
        "cry", "pain", "tears", "lost", "gone", "alone", "dark", "night",
        "fall", "cold", "broken", "fade", "rain", "miss", "shadow", "void",
    ],
    "Angry": [
        "fire", "rage", "fight", "war", "burn", "scream", "fury", "storm",
        "break", "blood", "clash", "power", "roar", "rise", "force", "wall",
    ],
    "Neutral": [
        "time", "road", "walk", "move", "change", "find", "know", "feel",
        "day", "life", "way", "mind", "words", "place", "truth", "ground",
    ],
}

_ANGRY_PATTERN = re.compile(
    r"\b(?:fight|rage|hate|angry|mad|burn|war|kill|scream|fury|violent|blood|"
    r"destroy|blast|crash|smash|shatter|explode|attack|strike|rebel)\b",
    re.IGNORECASE,
)

_MIN_CORPUS_WORDS = 500


# ---------------------------------------------------------------------------
# Rhyme helpers  (no extra dependencies)
# ---------------------------------------------------------------------------

def _rhyme_tail(word: str, n: int = 3) -> str:
    """Return the last *n* letters of *word* as a rough rhyme key."""
    w = re.sub(r"[^a-z]", "", word.lower())
    return w[-n:] if len(w) >= n else w


def _last_word(line: str) -> str:
    words = re.findall(r"[a-zA-Z]+", line)
    return words[-1] if words else ""


def _rhyme_score(a: str, b: str) -> int:
    """Simple rhyme score 0-3: longer matching suffix = better rhyme."""
    wa = _rhyme_tail(_last_word(a), 4)
    wb = _rhyme_tail(_last_word(b), 4)
    if not wa or not wb:
        return 0
    score = 0
    for i in range(1, 5):
        if wa[-i:] == wb[-i:]:
            score = i
        else:
            break
    # Penalise identical last words (too repetitive)
    if _last_word(a).lower() == _last_word(b).lower():
        score = 0
    return score


def _arrange_for_rhyme(lines: list[str]) -> list[str]:
    """Greedy reorder so adjacent pairs prefer rhyming end-words (AABB pattern)."""
    if len(lines) <= 2:
        return lines

    result: list[str] = []
    remaining = list(lines)

    while remaining:
        if len(remaining) == 1:
            result.append(remaining.pop())
            break

        anchor = remaining.pop(0)
        # Find best rhyme partner among the rest
        best_idx, best_score = 0, -1
        for i, candidate in enumerate(remaining):
            s = _rhyme_score(anchor, candidate)
            if s > best_score:
                best_score, best_idx = s, i

        partner = remaining.pop(best_idx)
        result.append(anchor)
        result.append(partner)

    return result


# ---------------------------------------------------------------------------
# LyricsGenerator
# ---------------------------------------------------------------------------

class LyricsGenerator:
    """Generate English-only structured song lyrics from Markov chains."""

    def __init__(self, data_dir: str = "csv", model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self._mood_models:    dict[str, markovify.Text] = {}
        self._artist_models:  dict[str, markovify.Text] = {}
        self._mood_top_words: dict[str, list[str]] = {}
        self._available_artists: list[str] = []
        self._fallback: markovify.Text | None = None
        self._df: pd.DataFrame | None = None

        self._build_all(data_dir)

    # ------------------------------------------------------------------
    # Corpus helpers
    # ------------------------------------------------------------------

    def _lyrics_to_lines(self, series: pd.Series) -> str:
        lines: list[str] = []
        for lyric in series.dropna().astype(str):
            chunks = re.split(r"\s{2,}|[\n\r]+|(?<=[.!?])\s+", lyric)
            for chunk in chunks:
                chunk = chunk.strip()
                if (
                    len(chunk.split()) >= 4
                    and not re.match(r"^\[", chunk)
                    and is_english_line(chunk)
                ):
                    lines.append(chunk)
        return "\n".join(lines)

    def _make_model(self, text: str, state_size: int = 2) -> markovify.Text:
        if not text or len(text.split()) < _MIN_CORPUS_WORDS:
            raise ValueError("Corpus too small")
        try:
            return markovify.NewlineText(text, state_size=state_size, well_formed=False)
        except Exception:
            return markovify.Text(text, state_size=state_size, well_formed=False)

    def _best_model(self, text: str) -> markovify.Text:
        model = self._make_model(text, state_size=2)
        if len(text.split()) > 20_000:
            try:
                model = self._make_model(text, state_size=3)
            except Exception:
                pass
        return model

    def _extract_top_words(self, series: pd.Series, n: int = 30) -> list[str]:
        from nltk.corpus import stopwords as _sw
        stops = set(_sw.words("english")) | {
            "oh", "yeah", "gonna", "wanna", "gotta", "like", "know",
            "got", "get", "let", "come", "go", "said", "say", "ooh", "ah",
        }
        counter: Counter = Counter()
        for lyric in series.dropna().astype(str):
            for word in re.findall(r"[a-z]+", lyric.lower()):
                if word not in stops and len(word) > 3:
                    counter[word] += 1
        return [w for w, _ in counter.most_common(n)]

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------

    def _build_all(self, data_dir: str) -> None:
        loader = LyricsDataLoader(data_dir)
        preprocessor = LyricsPreprocessor()
        df = loader.load_all_artists()
        df = preprocessor.preprocess_lyrics(df)
        df = preprocessor.create_mood_labels(df)
        self._df = df

        self._available_artists = sorted(df["Artist"].dropna().unique().tolist())

        angry_mask = (
            df["cleaned_lyrics"].str.contains(_ANGRY_PATTERN, na=False)
            | (df["sentiment"] < -0.2)
        )
        mood_corpora = {
            "Happy":   df[df["mood_category"] == "Happy"],
            "Sad":     df[df["mood_category"] == "Sad"],
            "Neutral": df[df["mood_category"] == "Neutral"],
            "Angry":   df[angry_mask],
        }

        all_text = self._lyrics_to_lines(df["Lyric"])
        try:
            self._fallback = self._best_model(all_text)
        except ValueError:
            pass

        for mood, sub_df in mood_corpora.items():
            text = self._lyrics_to_lines(sub_df["Lyric"])
            try:
                self._mood_models[mood] = self._best_model(text)
            except ValueError:
                if self._fallback:
                    self._mood_models[mood] = self._fallback
            self._mood_top_words[mood] = self._extract_top_words(sub_df["cleaned_lyrics"])

    def _get_or_build_artist_model(self, artist: str) -> markovify.Text | None:
        if artist in self._artist_models:
            return self._artist_models[artist]
        if self._df is None:
            return None
        adf = self._df[self._df["Artist"] == artist]
        if adf.empty:
            return None
        text = self._lyrics_to_lines(adf["Lyric"])
        try:
            model = self._best_model(text)
            self._artist_models[artist] = model
            return model
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Public metadata
    # ------------------------------------------------------------------

    @property
    def available_artists(self) -> list[str]:
        return self._available_artists

    # ------------------------------------------------------------------
    # Line generation helpers
    # ------------------------------------------------------------------

    def _raw_lines_from_model(
        self,
        model: markovify.Text,
        target: int,
        temperature: float,
        max_chars: int = 120,
    ) -> list[str]:
        tries = int(40 + temperature * 120)
        seen: set[str] = set()
        lines: list[str] = []
        for _ in range(target * 10):
            if len(lines) >= target:
                break
            line = model.make_short_sentence(
                max_chars=max_chars, tries=tries, test_output=False
            )
            if not line:
                line = model.make_sentence(tries=tries, test_output=False)
            if line:
                line = _clean_line(line)
                if line and line.lower() not in seen and is_english_line(line):
                    seen.add(line.lower())
                    lines.append(line)
        return lines

    def _generate_pool(
        self,
        mood: str,
        artist: str | None,
        theme_words: list[str],
        target: int,
        temperature: float,
    ) -> list[str]:
        """Generate candidate lines, blending artist + mood models."""
        lines: list[str] = []
        if artist:
            am = self._get_or_build_artist_model(artist)
            if am:
                lines += self._raw_lines_from_model(am, int(target * 0.70), temperature)
        mm = self._mood_models.get(mood) or self._fallback
        if mm:
            remaining = target - len(lines)
            lines += self._raw_lines_from_model(mm, max(remaining, int(target * 0.30)), temperature)

        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for ln in lines:
            if ln.lower() not in seen:
                seen.add(ln.lower())
                unique.append(ln)

        # Theme filter: if theme words supplied, prefer lines containing any
        if theme_words:
            theme_set = {w.lower() for w in theme_words}
            themed   = [ln for ln in unique if any(w in ln.lower() for w in theme_set)]
            unthemed = [ln for ln in unique if ln not in themed]
            # at least half themed, rest fill
            unique = themed + unthemed

        return unique

    def _generate_chorus_lines(
        self, mood: str, artist: str | None, temperature: float
    ) -> list[str]:
        if artist:
            model = self._get_or_build_artist_model(artist) or self._mood_models.get(mood) or self._fallback
        else:
            model = self._mood_models.get(mood) or self._fallback
        if not model:
            return ["Keep on singing"] * 4

        tries = int(40 + temperature * 80)
        seen: set[str] = set()
        lines: list[str] = []
        for _ in range(80):
            if len(lines) >= 4:
                break
            line = model.make_short_sentence(max_chars=70, tries=tries, test_output=False)
            if not line:
                line = model.make_sentence(tries=tries, test_output=False)
            if line:
                line = _clean_line(line)
                if line and line.lower() not in seen and is_english_line(line):
                    seen.add(line.lower())
                    lines.append(line)
        while len(lines) < 4:
            lines.append(lines[0] if lines else "Keep on singing")
        return _arrange_for_rhyme(lines[:4])

    def _score_and_arrange(self, lines: list[str], mood: str, n: int) -> list[str]:
        """Score by mood vocab, take top-n, then arrange adjacent pairs for rhyme."""
        vocab = set(_MOOD_VOCAB.get(mood, []))
        scored = [(sum(1 for w in vocab if w in ln.lower()), ln) for ln in lines]
        scored.sort(key=lambda x: -x[0])
        top  = [l for _, l in scored[:max(1, len(scored) // 2)]]
        rest = [l for _, l in scored[len(top):]]
        random.shuffle(rest)
        pool = (top + rest)[:n]
        return _arrange_for_rhyme(pool)

    def _make_title(self, mood: str, chorus: list[str], theme_words: list[str]) -> str:
        # If theme supplied, use first theme word + a chorus word
        if theme_words:
            theme_cap = theme_words[0].capitalize()
            if chorus:
                words = chorus[0].split()
                for w in words:
                    if len(w) > 3 and w.lower() not in {"that", "this", "with", "from", "have", "been"}:
                        return f"{theme_cap} {w.capitalize()}"
            return theme_cap

        for line in chorus:
            words = line.split()
            if len(words) >= 4:
                return " ".join(words[:4])
            if len(words) >= 2:
                return " ".join(words)

        top = self._mood_top_words.get(mood, _MOOD_VOCAB.get(mood, ["Song"]))
        chosen = random.sample(top[:15], min(3, len(top[:15])))
        return " ".join(w.capitalize() for w in chosen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_lyrics(
        self,
        mood: str,
        max_length: int = 150,
        temperature: float = 0.8,
        artist: str | None = None,
        theme: str = "",
    ) -> str:
        """Return a fully structured English-only song.

        Parameters
        ----------
        mood:       "Happy" | "Sad" | "Angry" | "Neutral"
        max_length: approximate word budget
        temperature: 0.1–1.0  (higher = more varied)
        artist:     optional artist name for style corpus
        theme:      optional comma-separated theme keywords
                    e.g. "heartbreak, rain, midnight"
        """
        mood = mood if mood in self._mood_models else "Neutral"

        # Parse theme words
        theme_words = [w.strip().lower() for w in theme.split(",") if w.strip()] if theme else []

        lines_needed = max(16, max_length // 8)
        candidates   = self._generate_pool(mood, artist, theme_words, lines_needed * 3, temperature)
        pool         = self._score_and_arrange(candidates, mood, lines_needed + 4)

        chorus = self._generate_chorus_lines(mood, artist, temperature)
        title  = self._make_title(mood, chorus, theme_words)

        # Split pool into sections; arrange each section for rhyme
        verse1  = _arrange_for_rhyme(pool[:4])
        verse2  = _arrange_for_rhyme(pool[4:8]  if len(pool) >= 8  else pool[:4])
        bridge  = pool[8:10] if len(pool) >= 10 else pool[:2]
        outro   = pool[10:12]if len(pool) >= 12 else (pool[-2:] if len(pool) >= 2 else pool[:1])

        parts = [
            f"♪ {title} ♪",
            "",
            _section("Verse 1", verse1),
            "",
            _section("Chorus", chorus),
            "",
            _section("Verse 2", verse2),
            "",
            _section("Chorus", chorus),
        ]
        if bridge:
            parts += ["", _section("Bridge", bridge)]
        if outro:
            parts += ["", _section("Outro", outro)]

        return "\n".join(parts)

    def get_music_metadata(self, mood: str) -> dict:
        """Return BPM, key name, and scale type for the given mood (for UI display)."""
        from src.music_generator import _MOOD_SETTINGS, _KEY_NAMES
        cfg = _MOOD_SETTINGS.get(mood, _MOOD_SETTINGS["Neutral"])
        return {
            "tempo": cfg["tempo"],
            "key":   _KEY_NAMES.get(cfg["key"], str(cfg["key"])),
            "scale": "Major" if cfg["scale"] == [0, 2, 4, 5, 7, 9, 11] else "Minor",
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    line = line[0].upper() + line[1:]
    return re.sub(r"[,;]+$", "", line).strip()


def _section(label: str, lines: list[str]) -> str:
    return f"[{label}]\n" + "\n".join(lines)
