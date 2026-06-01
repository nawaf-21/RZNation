# RZNation

**AI-powered mood-based song generator** — produces structured English lyrics and MIDI music from a 21-artist dataset, with an interactive lyrics-analysis dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What it does

- **Generate Song** — pick a mood (Happy / Sad / Angry / Neutral), optional artist style, optional theme keywords; get a fully structured song (Verse / Chorus / Bridge / Outro) with rhyme-sorted lines, plus a downloadable MIDI file and a ZIP package.
- **Analyze Lyrics** — per-artist mood distribution chart, word cloud, sentiment timeline, and quick stats across 21 artists and ~4 000 songs.
- **Predict Mood** — paste any lyrics; a trained Random Forest classifier returns Happy / Sad / Neutral with a confidence score.
- **Visualizations** — five pre-built interactive charts embedded directly in the app (violin plot, word counts, bubble chart, pie chart, sentiment timeline).

---

## Tech stack

| Layer | Libraries |
|-------|-----------|
| Web UI | `streamlit`, `streamlit-option-menu` |
| Lyrics generation | `markovify` (Markov chains, state size 2–3) |
| NLP / sentiment | `textblob`, `nltk` |
| English filter | `unicodedata` + NLTK stopwords (stdlib only) |
| ML classifier | `scikit-learn` RandomForest, `joblib` |
| Music generation | `midiutil` (procedural MIDI, 3 voices) |
| Visualizations | `plotly`, `matplotlib`, `seaborn`, `wordcloud` |
| Data | `pandas`, `numpy`, `networkx` |

**Not used:** PyTorch · TensorFlow · GPT-2 · transformers · pydub · langdetect.
Works on Python 3.10 through 3.14 without a GPU.

---

## Requirements

- Python **3.10 or newer**
- pip
- **Optional** (for in-browser MP3 playback): [FluidSynth](https://www.fluidsynth.org/) + [FFmpeg](https://ffmpeg.org/) on your system `PATH`. The app works fine without them — MIDI download is always available.

---

## Installation & run

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/RZNation.git
cd RZNation

# Create virtual environment
python -m venv .venv

# Activate  (Windows)
.venv\Scripts\activate
# Activate  (macOS / Linux)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m textblob.download_corpora

# Verify everything is wired up
python verify_setup.py

# Launch
streamlit run app.py
```

Open **http://localhost:8501**.

> First launch takes **30–60 seconds** while Markov models are built from the dataset.
> A progress bar shows each step. Every subsequent run is instant (Streamlit caches the models).

---

## Enable Predict Mood

The Predict Mood page requires a trained RandomForest model. Run this once after installation:

```python
from src.model import LyricsModel
from src.data_loader import LyricsDataLoader
from src.preprocessor import LyricsPreprocessor

df = LyricsDataLoader().load_all_artists()
df = LyricsPreprocessor().preprocess_lyrics(df)
df = LyricsPreprocessor().create_mood_labels(df)
LyricsModel().train_mood_model(df)   # saves models/mood_model.joblib
```

Restart the app — the Predict Mood page will work automatically.

---

## Regenerate visualizations

```bash
python visualization_graphs.py
```

This writes five interactive HTML files to `plots/`, which the Visualizations page embeds.

---

## Project structure

```
RZNation/
├── app.py                  ← Streamlit app  (four pages)
├── requirements.txt
├── verify_setup.py
├── visualization_graphs.py ← builds plots/*.html
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py     ← TextBlob sentiment + mood labels
│   ├── english_filter.py   ← Unicode + stopword English filter
│   ├── analyzer.py         ← TF-IDF, LDA, artist similarity
│   ├── visualizer.py       ← chart factory
│   ├── model.py            ← RandomForest mood classifier
│   ├── generator.py        ← Markov lyrics + artist style + theme + rhyme
│   └── music_generator.py  ← 3-voice MIDI composition
├── csv/                    ← 21 artist lyric datasets
├── plots/                  ← pre-built HTML charts
├── models/                 ← runtime output (gitignored except .gitkeep)
├── docs/
│   ├── TECHNICAL.md        ← full pipeline documentation
│   └── screenshots/        ← add UI screenshots here
└── .github/
    └── workflows/ci.yml    ← GitHub Actions CI
```

---

## Screenshots

> Screenshots coming soon — see [`docs/screenshots/`](docs/screenshots/) for instructions.

---

## How lyrics are generated

1. All 21 artist CSVs are loaded and scored with TextBlob sentiment.
2. Songs are split into mood corpora (Happy / Sad / Neutral / Angry).
3. Optional theme keywords and artist-style selection further filter the corpus.
4. A `markovify.NewlineText` model (state size 2–3) is trained on raw English-only lyrics.
5. ~50 candidate lines are generated, English-filtered, scored for mood vocabulary, deduplicated, and sorted so adjacent pairs prefer end-rhymes (AABB).
6. Lines are assembled into a structured song: Verse 1 → Chorus → Verse 2 → Chorus → Bridge → Outro.

**Artist style note:** selecting an artist trains the model on that artist's corpus only. Output is *inspired by* their vocabulary — it is wholly original computer-generated text.

---

## How music is generated

Three-voice procedural MIDI — no samples, no soundfonts required:

| Mood | BPM | Key | Scale |
|------|-----|-----|-------|
| Happy | 128 | C Major | Major |
| Sad | 72 | A Minor | Natural Minor |
| Angry | 160 | D Minor | Natural Minor |
| Neutral | 100 | G Major | Major |

Song structure: intro → verse → chorus → verse → chorus → outro.
Track 0: chord pads · Track 1: lead melody · Track 2: vocal melody suggestion (Choir Aahs).

---

## License

[MIT](LICENSE)
