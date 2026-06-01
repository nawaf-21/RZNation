#!/usr/bin/env python3
"""Check that RZNation can run before starting Streamlit."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "nltk",
    "textblob",
    "sklearn",
    "plotly",
    "wordcloud",
    "networkx",
    "joblib",
    "markovify",
    "streamlit",
    "streamlit_option_menu",
    "midiutil",
]


def check_packages() -> list[str]:
    missing = []
    for name in REQUIRED_PACKAGES:
        module = "sklearn" if name == "sklearn" else name
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(name)
    if missing:
        return [f"Missing packages: {', '.join(missing)}. Run: pip install -r requirements.txt"]
    return []


def check_data() -> list[str]:
    if not CSV_DIR.is_dir():
        return [f"Missing data folder: {CSV_DIR}"]
    csv_files = list(CSV_DIR.glob("*.csv"))
    if not csv_files:
        return [f"No CSV files in {CSV_DIR}"]
    return []


def check_generator() -> list[str]:
    try:
        from src.generator import LyricsGenerator

        gen = LyricsGenerator()
        sample = gen.generate_lyrics("Happy", max_length=40, temperature=0.7)
        if not sample or len(sample.split()) < 5:
            return ["Lyrics generator returned empty or very short output."]
    except Exception as exc:
        return [f"Lyrics generator smoke test failed: {exc}"]
    return []


def check_optional_tools() -> list[str]:
    import shutil

    notes = []
    if not shutil.which("fluidsynth"):
        notes.append("fluidsynth not on PATH - MIDI to MP3 disabled (MIDI download still works).")
    if not shutil.which("ffmpeg"):
        notes.append("ffmpeg not on PATH - MP3 export disabled when conversion runs.")
    return notes


def main() -> int:
    print("RZNation setup check\n" + "=" * 40)
    print(f"Python {sys.version.split()[0]}")

    issues: list[str] = []
    issues.extend(check_packages())
    issues.extend(check_data())

    if issues:
        print("\nBLOCKERS:")
        for item in issues:
            print(f"  - {item}")
        print("\nFix the items above, then run: streamlit run app.py")
        return 1

    print("\nPackages and dataset OK. Testing lyrics generator (may take ~30s)...")
    gen_issues = check_generator()
    if gen_issues:
        print("\nBLOCKERS:")
        for item in gen_issues:
            print(f"  - {item}")
        return 1

    print("\nAll checks passed.")
    notes = check_optional_tools()
    if notes:
        print("\nOptional notes:")
        for note in notes:
            print(f"  - {note}")

    print(f"\nDataset: {len(list(CSV_DIR.glob('*.csv')))} artist CSV files")
    print("\nStart the app:\n  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
