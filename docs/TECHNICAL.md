# RZNation — Technical Reference

## Architecture

```
app.py (Streamlit)
│
├── load_components()          @st.cache_resource — built once per session
│   ├── LyricsAnalyzer
│   ├── LyricsVisualizer
│   ├── LyricsGenerator        builds mood models + per-artist lazy models
│   └── MusicGenerator         three-voice MIDI
│
└── load_and_preprocess()      @st.cache_data — shared by Analyze page
    ├── LyricsDataLoader
    └── LyricsPreprocessor
```

---

## English-Only Filtering  (`src/english_filter.py`)

All lyric lines pass three checks before entering the training corpus or the
generated output.  The filter is entirely based on the standard library
(`unicodedata`) and NLTK stopwords — no extra pip packages.

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Script | < 15 % non-Latin letters | Rejects CJK, Korean, Arabic, Cyrillic, etc. |
| ASCII ratio | ≥ 70 % plain ASCII | Rejects heavily diacritical or mixed-script text |
| Stopword density | ≥ 1 English stopword per 12 words | Rejects romanised non-English or gibberish |

A lyric *entry* (whole song) is kept if ≥ 60 % of its lines pass the line-level
check.  This retains English songs that contain a few non-English ad-libs
(common in K-pop / Reggaeton crossovers).

The filter is applied:
1. In `LyricsGenerator._lyrics_to_lines()` — strips non-English lines from
   training corpora before the Markov model is built.
2. In `LyricsGenerator._raw_lines_from_model()` — rejects non-English sentences
   returned by the model at generation time.

---

## Lyrics Pipeline  (`src/generator.py`)

### 1. Loading & preprocessing
`LyricsDataLoader` reads all `csv/*.csv`; `LyricsPreprocessor` adds
`cleaned_lyrics`, `sentiment` (TextBlob polarity), `mood_category`.

### 2. Mood label mapping

| Bucket | Polarity |
|--------|----------|
| Sad | < −0.33 |
| Neutral | −0.33 to 0.33 |
| Happy | > 0.33 |
| Angry | regex match OR polarity < −0.2 (overlaps Sad) |

### 3. Markov model building
- Trains on the **raw `Lyric` column** (not cleaned), preserving capitalization
  and punctuation for more natural output.
- Stage-direction lines (`[Chorus]`, `[Verse]`, etc.) stripped before training.
- `markovify.NewlineText(state_size=2)` per mood;
  corpora > 20 000 words automatically try `state_size=3`.
- Full-dataset English fallback model handles empty corpora.

### 4. Artist-style models (lazy)
When the user selects an artist, `_get_or_build_artist_model()` builds a
model on that artist's English-filtered corpus on demand and caches it.
This produces lyrics *inspired by* the artist's vocabulary — no identity
claims, no voice cloning.

### 5. Generation algorithm
1. Generate ~50 × 3 candidate lines from the mood (+ optionally artist) model.
2. English-filter each line.
3. Score lines for mood-vocabulary alignment.
4. Deduplicate (case-insensitive set).
5. Generate a separate 4-line chorus block (shorter lines preferred).
6. Derive title from first chorus line (first 4 words).
7. Assemble: `♪ Title ♪ → [Verse 1] → [Chorus] → [Verse 2] → [Chorus] → [Bridge] → [Outro]`.

---

## Music Pipeline  (`src/music_generator.py`)

### Three MIDI voices

| Track | Name | Purpose |
|-------|------|---------|
| 0 | Chord pads | Block chord on beat 1; softer repeat on beat 3 |
| 1 | Lead melody | Scale-walking single notes; 60 % chord tones |
| 2 | Vocal melody | Slow, singable MIDI line (Choir Aahs / Strings) |

The vocal melody is a **musical suggestion** of a singing line — it is an
instrument track, NOT a synthesised or cloned human voice.

### Mood settings

| Mood | BPM | Key | Scale | Chord voice | Lead voice | Vocal voice |
|------|-----|-----|-------|-------------|------------|-------------|
| Happy | 128 | C4 | Major | Piano (0) | Flute (73) | Choir Aahs (52) |
| Sad | 72 | A3 | Minor | Strings (48) | Violin (40) | Choir Aahs (52) |
| Angry | 160 | D4 | Minor | Overdrive Guitar (29) | Distortion (30) | Strings (49) |
| Neutral | 100 | G3 | Major | Acoustic Guitar (25) | Acoustic Guitar (24) | Choir Aahs (52) |

### Song structure
Bars distributed proportionally across: intro → verse → chorus → verse → chorus → outro.

Velocity levels:
- Chorus: `vel_hi − 4` (loudest)
- Verse: `(vel_lo + vel_hi) / 2` (medium)
- Intro / Outro: `vel_lo + 4` (softest)

### Chord progressions

| Mood | Progression | Roman numerals |
|------|-------------|----------------|
| Happy / Neutral | I – IV – V – vi | Major feel |
| Sad | i – VII – VI – III | Natural minor |
| Angry | i – v – VI – VII | Minor with tension |

---

## Analysis Pipeline  (`src/analyzer.py`)

| Method | Description |
|--------|-------------|
| `get_word_frequencies` | Top-N words per artist (Counter) |
| `perform_topic_modeling` | LDA with TF-IDF (5 topics, 1 000 features) |
| `calculate_artist_similarity` | Cosine similarity on per-artist TF-IDF; NetworkX graph |
| `analyze_temporal_trends` | Group-by-year aggregation of sentiment, subjectivity, word count |

---

## ML Model  (`src/model.py`)  — optional

`LyricsModel` provides two RandomForest classifiers (mood, artist).
They are **not loaded on startup** — training is optional and separate:

```python
from src.model import LyricsModel
from src.data_loader import LyricsDataLoader
from src.preprocessor import LyricsPreprocessor

df = LyricsDataLoader().load_all_artists()
df = LyricsPreprocessor().preprocess_lyrics(df)
df = LyricsPreprocessor().create_mood_labels(df)

m = LyricsModel()
m.train_mood_model(df)     # saves models/mood_model.joblib
m.train_artist_model(df)   # saves models/artist_model.joblib
```

---

## Voice Cloning — Not Implemented

**RZNation does not implement, use, or support artist voice cloning.**

Cloning a real artist's voice (Taylor Swift, Drake, Eminem, BTS members, etc.)
requires:
- The artist's explicit written consent.
- Compliance with relevant copyright and personality-rights law.
- Technical infrastructure (TTS models trained on that artist's recordings).

None of these conditions are met in this educational project.

### Safer alternatives implemented

| Option | Status |
|--------|--------|
| A: MIDI instrumental output only | ✅ Implemented (default) |
| B: Generic vocal melody via MIDI Choir Aahs instrument | ✅ Implemented (Track 2) |
| C: User-provided voice sample support | ❌ Not implemented (out of scope) |
| D: Optional text-to-singing tools (documented only) | See note below |

**Note on external TTS/singing tools:** Tools such as Suno AI, Udio, or
open-source projects (e.g. `so-vits-svc`) exist but require artist permission
for likeness/voice data.  They are outside this project's scope and are not
referenced in the app.

---

## Standalone Scripts

| Script | Purpose |
|--------|---------|
| `visualization_graphs.py` | Build 5 interactive HTML charts in `plots/` |
| `visualization_demo.py` | Smaller run; PNG export (needs `kaleido`) |
| `verify_setup.py` | Pre-flight check; smoke-tests the lyrics generator |
| `22i_1854_&&_22l_7554_project_ai.py` | Historical Colab export (needs `Lyrics_.csv`) |
