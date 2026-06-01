import streamlit as st

st.set_page_config(
    page_title="RZNation - AI Song Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import io
import os
import zipfile
from pathlib import Path

from streamlit_option_menu import option_menu

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────── */
.stApp { background-color: #0a0a0a; color: #f0f0f0; }
.main  { background-color: #0a0a0a; color: #f0f0f0; }

/* ── Home typography ───────────────────────────────────────────── */
.main-title {
    font-size: 5.5rem; font-weight: 900; color: #ffffff;
    text-align: center; letter-spacing: 3px;
    font-family: 'Arial Black', sans-serif;
    text-shadow: 0 0 50px rgba(76,175,80,0.45);
    margin-bottom: 0.4rem;
}
.subtitle   { font-size: 1.8rem; color: #cccccc; text-align: center; font-style: italic; margin-bottom: 1rem; }
.tagline    { font-size: 1.15rem; color: #999; text-align: center; margin-bottom: 3rem; letter-spacing: 1px; }
.team-credit{ text-align: center; color: #666; font-size: 0.88rem; margin-top: 3rem; }

/* ── Buttons ───────────────────────────────────────────────────── */
.stElementContainer.element-container {
    width: 100% !important;
    min-width: 100% !important;
}
.stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}
.stButton > button,
.stButton button {
    margin: 0 auto !important;
    display: block !important;
    background-color: #4CAF50; color: #fff;
    padding: 0.9rem 2.5rem; border-radius: 50px; border: none;
    font-size: 1.1rem; font-weight: bold; letter-spacing: 1px;
    text-transform: uppercase; transition: all 0.25s ease;
}
.stButton > button:hover,
.stButton button:hover {
    background-color: #45a049;
    box-shadow: 0 0 22px rgba(76,175,80,0.55);
    transform: scale(1.04);
}

/* ── Song card ─────────────────────────────────────────────────── */
.song-box {
    background: #111827; border: 1px solid #2d3748; border-radius: 12px;
    padding: 1.6rem 2rem; font-family: 'Georgia', serif;
    font-size: 1.05rem; line-height: 1.9; color: #e2e8f0;
}
.song-title    { font-size: 1.6rem; font-weight: bold; color: #4CAF50; text-align: center; margin-bottom: 1.4rem; }
.section-label { color: #718096; font-size: 0.75rem; letter-spacing: 2.5px; text-transform: uppercase; margin-top: 1rem; }
.lyric-line    { margin: 0; padding: 0; }

/* ── Mood badge ────────────────────────────────────────────────── */
.mood-badge  { display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.82rem; font-weight: bold; letter-spacing: 1px; margin-bottom: 0.8rem; }
.mood-Happy  { background: #1a3a1a; color: #4ade80; border: 1px solid #4ade80; }
.mood-Sad    { background: #1a1a3a; color: #60a5fa; border: 1px solid #60a5fa; }
.mood-Angry  { background: #3a1a1a; color: #f87171; border: 1px solid #f87171; }
.mood-Neutral{ background: #2a2a2a; color: #d1d5db; border: 1px solid #d1d5db; }

/* ── Metadata pills ────────────────────────────────────────────── */
.meta-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.meta-pill {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 0.25rem 0.75rem; font-size: 0.78rem; color: #94a3b8;
}
.meta-pill strong { color: #e2e8f0; }

/* ── Disclaimer ────────────────────────────────────────────────── */
.disclaimer {
    background: #0f172a; border-left: 3px solid #4CAF50;
    padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0;
    font-size: 0.82rem; color: #94a3b8; line-height: 1.7;
}

/* ── Plots gallery ─────────────────────────────────────────────── */
.plot-card {
    background: #111827; border: 1px solid #2d3748; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 1rem;
}
.plot-title { color: #4ade80; font-weight: bold; margin-bottom: 0.4rem; }

/* ── Nav ───────────────────────────────────────────────────────── */
label, .stMarkdown { color: #f0f0f0 !important; }
#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INSTRUMENT_MAP = {
    "Happy":   "Acoustic Piano (chords) · Flute (lead melody) · Choir Aahs (vocal line)",
    "Sad":     "String Ensemble (chords) · Violin (lead melody) · Choir Aahs (vocal line)",
    "Angry":   "Overdriven Guitar (chords) · Distortion Guitar (lead) · Strings (vocal line)",
    "Neutral": "Acoustic Guitar rhythm · Acoustic Guitar lead · Choir Aahs (vocal line)",
}
_SCALE_MAP = {
    "Happy":   ("C Major",  128),
    "Sad":     ("A Minor",   72),
    "Angry":   ("D Minor",  160),
    "Neutral": ("G Major",  100),
}
_PLOT_META = [
    ("sentiment_distribution.html",  "Sentiment Distribution",   "Violin plot of sentiment scores across all 21 artists"),
    ("word_count_analysis.html",     "Word Count Analysis",      "Average words per song, ranked by artist"),
    ("common_words_bubble.html",     "Common Words Bubble",      "Top 50 words across the whole dataset"),
    ("artist_song_distribution.html","Artist Song Distribution", "Pie chart showing each artist's share of the dataset"),
    ("sentiment_timeline.html",      "Sentiment Timeline",       "Rolling sentiment score for each artist over their discography"),
]


# ---------------------------------------------------------------------------
# Cached loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_components():
    from src.analyzer import LyricsAnalyzer
    from src.visualizer import LyricsVisualizer
    from src.generator import LyricsGenerator
    from src.music_generator import MusicGenerator

    bar = st.progress(0, text="Loading dataset…")
    bar.progress(20, text="Initialising analysers…")
    analyzer   = LyricsAnalyzer()
    visualizer = LyricsVisualizer()
    bar.progress(35, text="Building Markov language models — first run ~30-60 s…")
    generator  = LyricsGenerator()
    bar.progress(85, text="Initialising music generator…")
    music_gen  = MusicGenerator(require_soundfont=False)
    bar.progress(100, text="Ready!")
    bar.empty()
    return analyzer, visualizer, generator, music_gen


@st.cache_data(show_spinner=False)
def load_and_preprocess():
    from src.data_loader import LyricsDataLoader
    from src.preprocessor import LyricsPreprocessor
    df = LyricsDataLoader().load_all_artists()
    df = LyricsPreprocessor().preprocess_lyrics(df)
    df = LyricsPreprocessor().create_mood_labels(df)
    return df


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

def show_home_page() -> None:
    st.markdown("<div style='text-align:center; padding-top:8vh;'>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>RZNation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your Mood &middot; Your Music &middot; Our Rizz</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='tagline'>AI-generated English lyrics &amp; MIDI music &mdash; "
        "21 artists, 4 000+ songs</p>",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        if st.button("Let's Get Started", key="start_btn"):
            st.session_state.page = "app"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Generate page
# ---------------------------------------------------------------------------

def show_generate_page(generator, music_generator) -> None:
    st.header("Generate a New Song")

    artists = ["None (mood-only)"] + generator.available_artists

    # Row 1: mood / artist / length
    c1, c2, c3 = st.columns(3)
    with c1:
        mood = st.selectbox("Mood", ["Happy", "Sad", "Angry", "Neutral"])
    with c2:
        artist_choice = st.selectbox(
            "Artist style (optional)",
            artists,
            help="Uses that artist's dataset as the style source. Lyrics are original — no voice cloning.",
        )
    with c3:
        max_length = st.slider("Lyric length (words)", 60, 300, 150, step=10)

    # Row 2: theme / creativity / music duration
    c4, c5, c6 = st.columns(3)
    with c4:
        theme = st.text_input(
            "Theme keywords (optional)",
            placeholder="e.g. heartbreak, midnight, dreams",
            help="Comma-separated words that bias the corpus towards matching songs.",
        )
    with c5:
        temperature = st.slider("Creativity", 0.1, 1.0, 0.75, step=0.05,
                                help="Higher = more varied word choices")
    with c6:
        music_duration = st.slider("Music duration (s)", 15, 120, 30, step=5)

    selected_artist = None if artist_choice == "None (mood-only)" else artist_choice

    gen_col, _ = st.columns([1, 3])
    with gen_col:
        clicked = st.button("Generate Song", use_container_width=True)

    # Initialise session state
    for k in ("last_lyrics", "last_midi_path", "last_mood", "last_artist", "last_theme"):
        if k not in st.session_state:
            st.session_state[k] = None

    if clicked:
        with st.spinner("Composing your song…"):
            lyrics = generator.generate_lyrics(
                mood=mood,
                max_length=max_length,
                temperature=temperature,
                artist=selected_artist,
                theme=theme,
            )
            music_path = music_generator.generate_music(
                mood=mood,
                duration_seconds=music_duration,
                temperature=temperature,
            )
        st.session_state.last_lyrics    = lyrics
        st.session_state.last_midi_path = str(music_path)
        st.session_state.last_mood      = mood
        st.session_state.last_artist    = selected_artist
        st.session_state.last_theme     = theme

    # ── Results ───────────────────────────────────────────────────────
    if st.session_state.last_lyrics:
        d_mood   = st.session_state.last_mood or mood
        d_artist = st.session_state.last_artist
        d_theme  = st.session_state.last_theme or ""
        key_str, bpm = _SCALE_MAP.get(d_mood, ("C Major", 100))

        st.markdown(
            f"<div class='mood-badge mood-{d_mood}'>{d_mood.upper()}</div>",
            unsafe_allow_html=True,
        )

        artist_label = f"Inspired by {d_artist} dataset" if d_artist else "All artists (mood-filtered)"
        theme_label  = d_theme if d_theme else "None"
        _meta_pills({
            "Language":     "English",
            "Style source": artist_label,
            "Theme":        theme_label,
            "Key":          key_str,
            "Tempo":        f"{bpm} BPM",
            "Output":       "MIDI instrumental",
        })

        tab_lyrics, tab_music, tab_pkg, tab_about = st.tabs(
            ["Lyrics", "Music", "Download Package", "About"]
        )

        with tab_lyrics:
            text = st.session_state.last_lyrics
            st.markdown(
                f"<div class='song-box'>{_lyrics_to_html(text)}</div>",
                unsafe_allow_html=True,
            )
            fname = f"rznation_{d_mood.lower()}"
            if d_artist:
                fname += f"_{d_artist.replace(' ','_').lower()}"
            st.download_button(
                "Download Lyrics (.txt)",
                data=text,
                file_name=fname + "_lyrics.txt",
                mime="text/plain",
            )

        with tab_music:
            mp = st.session_state.last_midi_path
            if mp and Path(mp).exists():
                _render_music_player(mp)
            else:
                st.warning("Music file not found — click Generate Song again.")
            st.info(f"Instruments: {_INSTRUMENT_MAP.get(d_mood, '—')}")
            st.caption(
                "The vocal melody track (Choir Aahs / Strings) is a MIDI musical "
                "suggestion — NOT a synthesised or cloned human voice."
            )

        with tab_pkg:
            _show_package_download(
                st.session_state.last_lyrics,
                st.session_state.last_midi_path,
                d_mood, d_artist,
            )

        with tab_about:
            _show_disclaimer()


def _meta_pills(info: dict) -> None:
    pills = "".join(
        f"<span class='meta-pill'><strong>{k}:</strong> {v}</span>"
        for k, v in info.items()
    )
    st.markdown(f"<div class='meta-row'>{pills}</div>", unsafe_allow_html=True)


def _lyrics_to_html(text: str) -> str:
    import html as h
    out = []
    for line in text.splitlines():
        esc = h.escape(line)
        if esc.startswith("&#9834;") or esc.startswith("♪"):
            out.append(f"<div class='song-title'>{esc}</div>")
        elif esc.startswith("[") and esc.endswith("]"):
            out.append(f"<div class='section-label'>{esc}</div>")
        elif esc == "":
            out.append("<br>")
        else:
            out.append(f"<div class='lyric-line'>{esc}</div>")
    return "\n".join(out)


def _render_music_player(path_str: str) -> None:
    if path_str.endswith(".mid"):
        st.markdown("**MIDI generated.** Download below and open in Windows Media Player, GarageBand, or any MIDI player.")
        with open(path_str, "rb") as f:
            st.download_button(
                "Download MIDI",
                data=f.read(),
                file_name=os.path.basename(path_str),
                mime="audio/midi",
            )
        st.caption("Tip: install FluidSynth + FFmpeg and add to PATH for in-browser MP3 playback.")
    else:
        with open(path_str, "rb") as f:
            audio = f.read()
        st.audio(audio, format="audio/mp3")
        st.download_button(
            "Download MP3",
            data=audio,
            file_name=os.path.basename(path_str),
            mime="audio/mp3",
        )


def _show_package_download(lyrics: str, midi_path: str | None, mood: str, artist: str | None) -> None:
    """Build a ZIP containing lyrics + MIDI and offer a single download button."""
    st.subheader("Download Song Package")
    st.write("Download your complete song as a single ZIP file containing the lyrics text and MIDI music file.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Lyrics
        zf.writestr("lyrics.txt", lyrics or "")
        # MIDI
        if midi_path and Path(midi_path).exists():
            zf.write(midi_path, os.path.basename(midi_path))
        # Readme inside zip
        info_lines = [
            "RZNation — AI-Generated Song Package",
            "=" * 38,
            f"Mood   : {mood}",
            f"Style  : {'Inspired by ' + artist + ' dataset' if artist else 'All artists (mood-filtered)'}",
            "",
            "Lyrics are generated in English using Markov chains trained on the",
            "RZNation artist dataset.",
            "",
            "Music is procedural MIDI — three voices: chords, lead melody, vocal melody.",
            "Open the .mid file in any MIDI-capable player (Windows Media Player,",
            "GarageBand, VLC, etc.).",
            "",
            "IMPORTANT: These lyrics are wholly original computer-generated text.",
            "Artist style selection affects training corpus only — not authorship.",
            "No real artist voices are cloned or reproduced.",
            "",
            "Generated by RZNation — https://github.com/RZNation",
        ]
        zf.writestr("README.txt", "\n".join(info_lines))
    buf.seek(0)

    fname = f"rznation_{mood.lower()}"
    if artist:
        fname += f"_{artist.replace(' ', '_').lower()}"

    st.download_button(
        "Download Song Package (.zip)",
        data=buf.getvalue(),
        file_name=fname + "_package.zip",
        mime="application/zip",
    )


def _show_disclaimer() -> None:
    st.markdown("""
<div class='disclaimer'>
<strong>How this song was generated</strong><br>
Lyrics are generated in <strong>English</strong> using Markov chains trained on the selected
mood/artist corpus. Non-English lines are filtered out before training and before display
using a Unicode script + stopword density filter.<br><br>
<strong>Artist style</strong> selection trains the Markov model on that artist's dataset only.
Output is inspired by their vocabulary and phrasing — it is <em>wholly original text</em>,
not written by, attributed to, or endorsed by the real artist.<br><br>
<strong>Theme keywords</strong> bias corpus selection towards songs that contain those words,
strengthening thematic consistency across the generated lines.<br><br>
<strong>Music</strong> is procedurally generated MIDI with three voices: chord pads, lead melody,
and a vocal melody track (Choir Aahs / Strings). This vocal track is a <em>musical
representation</em> of a singing line — it is NOT a synthesised or cloned human voice.<br><br>
<strong>Voice cloning</strong> of real artists is <strong>not implemented</strong>.
Such technology requires explicit permission from the rights holder and is outside the scope
of this educational project.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Analyze page
# ---------------------------------------------------------------------------

def show_analyze_page() -> None:
    st.header("Analyze Artist Lyrics")

    try:
        df = load_and_preprocess()
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.info("Ensure the `csv/` directory exists and all requirements are installed.")
        return

    from src.analyzer import LyricsAnalyzer
    from src.visualizer import LyricsVisualizer
    analyzer   = LyricsAnalyzer()
    visualizer = LyricsVisualizer()

    artists = sorted(df["Artist"].dropna().unique().tolist())
    artist  = st.selectbox("Select an artist", artists)
    adf     = df[df["Artist"] == artist]

    if adf.empty:
        st.warning(f"No data found for {artist}.")
        return

    st.subheader(f"{artist} — {len(adf)} songs")

    t_mood, t_cloud, t_trend, t_stats = st.tabs(
        ["Mood Distribution", "Word Cloud", "Sentiment Timeline", "Quick Stats"]
    )

    with t_mood:
        st.plotly_chart(
            visualizer.plot_sentiment_distribution(adf, artist),
            use_container_width=True,
        )
        counts = adf["mood_category"].value_counts()
        for col, (label, n) in zip(st.columns(len(counts)), counts.items()):
            col.metric(str(label), n)

    with t_cloud:
        text = " ".join(adf["cleaned_lyrics"].dropna())
        if text.strip():
            st.pyplot(visualizer.create_wordcloud(text, f"Common Words — {artist}"))
        else:
            st.info("Not enough text for a word cloud.")

    with t_trend:
        try:
            yearly = analyzer.analyze_temporal_trends(adf)
            st.plotly_chart(visualizer.plot_temporal_trends(yearly), use_container_width=True)
        except ValueError as exc:
            st.info(f"Timeline unavailable: {exc}")

    with t_stats:
        avg_sent  = adf["sentiment"].mean()
        avg_words = adf["word_count"].mean() if "word_count" in adf.columns else 0
        years     = adf["Year"].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Songs",          len(adf))
        c2.metric("Avg Sentiment",  f"{avg_sent:.2f}")
        c3.metric("Avg Words/Song", f"{avg_words:.0f}")
        c4.metric("Year span",
                  f"{int(years.min())}-{int(years.max())}" if not years.empty else "-")


# ---------------------------------------------------------------------------
# Predict Mood page
# ---------------------------------------------------------------------------

def show_predict_page() -> None:
    st.header("Predict Mood from Lyrics")
    st.write(
        "Paste any song lyrics below. A Random Forest classifier trained on the "
        "RZNation dataset will predict the mood."
    )

    lyrics_input = st.text_area(
        "Paste lyrics here",
        height=200,
        placeholder="Type or paste some lyrics…",
    )

    if st.button("Predict Mood", key="predict_btn"):
        if not lyrics_input.strip():
            st.warning("Please enter some lyrics first.")
            return
        with st.spinner("Analysing…"):
            result = _run_mood_prediction(lyrics_input)
        if result is None:
            st.error(
                "Model not trained yet. Train it first by running:\n\n"
                "```python\n"
                "from src.model import LyricsModel\n"
                "from src.data_loader import LyricsDataLoader\n"
                "from src.preprocessor import LyricsPreprocessor\n\n"
                "df = LyricsDataLoader().load_all_artists()\n"
                "df = LyricsPreprocessor().preprocess_lyrics(df)\n"
                "df = LyricsPreprocessor().create_mood_labels(df)\n"
                "LyricsModel().train_mood_model(df)\n"
                "```\n\n"
                "This saves `models/mood_model.joblib`. Restart the app afterwards."
            )
        else:
            mood, confidence = result
            key_str, bpm = _SCALE_MAP.get(mood, ("C Major", 100))
            st.markdown(
                f"<div class='mood-badge mood-{mood}' style='font-size:1.1rem; padding:0.5rem 1.5rem;'>"
                f"Predicted Mood: {mood.upper()}</div>",
                unsafe_allow_html=True,
            )
            _meta_pills({
                "Confidence": f"{confidence:.0%}",
                "Typical Key": key_str,
                "Typical BPM": str(bpm),
            })
            st.caption(
                "Confidence is the Random Forest's class probability. "
                "The model was trained on TextBlob sentiment labels, not human annotations."
            )


def _run_mood_prediction(lyrics: str):
    """Return (mood_label, confidence) or None if model not available."""
    try:
        from src.model import LyricsModel
        from src.preprocessor import LyricsPreprocessor
        preprocessor = LyricsPreprocessor()
        cleaned = preprocessor.clean_text(lyrics)
        model = LyricsModel()
        # Try to load saved model
        import joblib
        from pathlib import Path
        mood_path = Path("models") / "mood_model.joblib"
        vec_path  = Path("models") / "mood_vectorizer.joblib"
        if not mood_path.exists() or not vec_path.exists():
            return None
        model.mood_model  = joblib.load(mood_path)
        model.vectorizer  = joblib.load(vec_path)
        X    = model.vectorizer.transform([cleaned])
        pred = model.mood_model.predict(X)[0]
        proba = model.mood_model.predict_proba(X).max()
        return str(pred), float(proba)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Visualizations gallery page
# ---------------------------------------------------------------------------

def show_visualizations_page() -> None:
    st.header("Dataset Visualizations")
    st.write(
        "Five pre-built interactive charts generated from the full 21-artist dataset. "
        "If charts appear blank, run `python visualization_graphs.py` first."
    )

    plots_dir = Path("plots")
    if not plots_dir.exists():
        st.warning("`plots/` directory not found. Run `python visualization_graphs.py` to generate charts.")
        return

    for filename, title, description in _PLOT_META:
        html_path = plots_dir / filename
        with st.expander(f"{title} — {description}", expanded=False):
            if html_path.exists():
                html_content = html_path.read_text(encoding="utf-8", errors="replace")
                st.components.v1.html(html_content, height=600, scrolling=True)
            else:
                st.info(
                    f"`{filename}` not found. "
                    "Run `python visualization_graphs.py` to generate it."
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        show_home_page()
        return

    # Load heavy components (with progress bar on first run)
    try:
        analyzer, visualizer, generator, music_gen = load_components()
    except Exception as exc:
        st.error(f"Failed to initialise app: {exc}")
        st.info("Run `python verify_setup.py` to diagnose missing dependencies.")
        st.stop()

    selected = option_menu(
        menu_title=None,
        options=["Generate Song", "Analyze Lyrics", "Predict Mood", "Visualizations"],
        icons=["music-note-beamed", "bar-chart-line", "emoji-smile", "graph-up-arrow"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container":         {"padding": "0!important", "background-color": "#0d1117"},
            "icon":              {"color": "#4CAF50", "font-size": "16px"},
            "nav-link":          {"font-size": "14px", "text-align": "center", "--hover-color": "#1a2a1a"},
            "nav-link-selected": {"background-color": "#1a3a1a", "color": "#4ade80"},
        },
    )

    if selected == "Generate Song":
        show_generate_page(generator, music_gen)
    elif selected == "Analyze Lyrics":
        show_analyze_page()
    elif selected == "Predict Mood":
        show_predict_page()
    else:
        show_visualizations_page()


if __name__ == "__main__":
    main()
