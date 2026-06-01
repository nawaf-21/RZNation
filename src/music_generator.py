"""MIDI music generation — mood-specific, structured, three-voice arrangement.

Voices
------
Track 0  Chord pads       — block chords + arpeggiation
Track 1  Melody / lead    — scale-walking single notes (represents vocal line)
Track 2  Vocal melody     — slower, wider-interval "singing" line on chorus;
                            follows chord roots on verses

No voice cloning of real artists is performed.  All output is procedural MIDI
using General MIDI instrument numbers.  The "vocal melody" is represented by a
soft instrument (e.g. Choir Aahs on GM #53) — it is a musical suggestion, not
a synthesised human voice.

For real audio rendering:
  * Download FluidSynth + a General MIDI soundfont → converts to WAV/MP3.
  * We do NOT support text-to-singing or artist voice cloning.  Any such
    feature would require the artist's permission and is outside the scope of
    this educational project.
"""

from __future__ import annotations

from midiutil import MIDIFile
from pathlib import Path
import random
import subprocess
import urllib.request
import zipfile


# ---------------------------------------------------------------------------
# Mood configuration
# ---------------------------------------------------------------------------

# Human-readable key names for display in the UI
_KEY_NAMES: dict[int, str] = {
    48: "C3", 50: "D3", 52: "E3", 53: "F3", 55: "G3", 57: "A3", 59: "B3",
    60: "C4", 62: "D4", 64: "E4", 65: "F4", 67: "G4", 69: "A4", 71: "B4",
}

# fmt: off
_MOOD_SETTINGS: dict[str, dict] = {
    "Happy": {
        "key": 60,                          # C4
        "scale": [0, 2, 4, 5, 7, 9, 11],   # Major
        "tempo": 128,
        "instrument_chords": 0,             # Acoustic Grand Piano
        "instrument_melody": 73,            # Flute  (lead melody)
        "instrument_vocal":  52,            # Choir Aahs  (vocal suggestion)
        # I – IV – V – vi  (C F G Am)
        "progressions": [
            [0, 4, 7],   # I
            [5, 9, 0],   # IV
            [7, 11, 2],  # V
            [9, 0, 4],   # vi
        ],
        "velocity_range": (85, 110),
    },
    "Sad": {
        "key": 57,                          # A3
        "scale": [0, 2, 3, 5, 7, 8, 10],   # Natural Minor
        "tempo": 72,
        "instrument_chords": 48,            # String Ensemble 1
        "instrument_melody": 40,            # Violin
        "instrument_vocal":  52,            # Choir Aahs
        # i – VII – VI – III  (Am G F C)
        "progressions": [
            [0, 3, 7],
            [10, 2, 5],
            [8, 0, 3],
            [3, 7, 10],
        ],
        "velocity_range": (55, 80),
    },
    "Angry": {
        "key": 62,                          # D4
        "scale": [0, 2, 3, 5, 7, 8, 10],   # Natural Minor
        "tempo": 160,
        "instrument_chords": 29,            # Overdriven Guitar
        "instrument_melody": 30,            # Distortion Guitar
        "instrument_vocal":  49,            # String Ensemble 2 (aggressive stabs)
        # i – v – VI – VII  (power-chord feel)
        "progressions": [
            [0, 3, 7],
            [7, 10, 2],
            [8, 0, 3],
            [10, 2, 5],
        ],
        "velocity_range": (100, 127),
    },
    "Neutral": {
        "key": 55,                          # G3
        "scale": [0, 2, 4, 5, 7, 9, 11],   # Major
        "tempo": 100,
        "instrument_chords": 25,            # Acoustic Guitar (nylon)
        "instrument_melody": 24,            # Acoustic Guitar (steel)
        "instrument_vocal":  52,            # Choir Aahs
        # I – vi – IV – V  (G Em C D)
        "progressions": [
            [0, 4, 7],
            [9, 0, 4],
            [5, 9, 0],
            [7, 11, 2],
        ],
        "velocity_range": (70, 95),
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# MusicGenerator
# ---------------------------------------------------------------------------

class MusicGenerator:
    """Generate structured MIDI music files for a given mood.

    Output is always MIDI (.mid).  Optional MP3 conversion requires FluidSynth
    and FFmpeg to be installed and on PATH.

    Artist voice cloning is NOT implemented and is NOT planned.  The "vocal
    melody" track uses GM instrument #52 (Choir Aahs) as a placeholder — a
    stylistic suggestion only.
    """

    def __init__(self, model_dir: str = "models", require_soundfont: bool = False):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.require_soundfont = require_soundfont
        self._setup_soundfont()

    # ------------------------------------------------------------------
    # Soundfont setup
    # ------------------------------------------------------------------

    def _setup_soundfont(self) -> None:
        soundfont_dir = self.model_dir / "soundfonts"
        soundfont_dir.mkdir(exist_ok=True)
        self.soundfont_path = soundfont_dir / "GeneralUser GS v1.471.sf2"

        if self.soundfont_path.exists():
            return

        urls = [
            "https://schristiancollins.com/soundfonts/GeneralUser_GS_1.471.zip",
            "https://www.schristiancollins.com/generaluser/GeneralUser_GS_1.471.zip",
            "https://archive.org/download/GeneralUserGSv1.471/GeneralUser_GS_1.471.zip",
        ]
        zip_path = soundfont_dir / "GeneralUser_GS_1.471.zip"

        for url in urls:
            try:
                print(f"Downloading soundfont from {url}…")
                urllib.request.urlretrieve(url, str(zip_path))
                if not zipfile.is_zipfile(zip_path):
                    zip_path.unlink()
                    continue
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for name in zf.namelist():
                        if name.endswith(".sf2"):
                            zf.extract(name, soundfont_dir)
                            (soundfont_dir / name).rename(self.soundfont_path)
                            break
                zip_path.unlink()
                print("Soundfont ready.")
                return
            except Exception as exc:
                print(f"Download failed ({url}): {exc}")
                if zip_path.exists():
                    zip_path.unlink()

        if self.require_soundfont:
            raise RuntimeError("Soundfont unavailable. Place it manually in models/soundfonts/.")
        print("Warning: soundfont unavailable — MIDI only, no MP3.")
        self.soundfont_path = None

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_music(
        self,
        mood: str,
        duration_seconds: int = 30,
        temperature: float = 0.8,
    ) -> Path:
        """Build a structured three-voice MIDI for the given mood.

        Tracks
        ------
        0  Chord pads
        1  Lead melody (flute / guitar / violin)
        2  Vocal melody suggestion (choir aahs / strings) — represents singing line
        """
        if mood not in _MOOD_SETTINGS:
            mood = "Neutral"

        cfg = _MOOD_SETTINGS[mood]
        key         = cfg["key"]
        scale       = cfg["scale"]
        tempo       = cfg["tempo"]
        progressions= cfg["progressions"]
        vel_lo, vel_hi = cfg["velocity_range"]

        # Three tracks
        midi = MIDIFile(3)
        track_names = ["Chords", "Lead Melody", "Vocal Melody"]
        for t, name in enumerate(track_names):
            midi.addTrackName(t, 0, f"{mood} - {name}")
            midi.addTempo(t, 0, tempo)

        midi.addProgramChange(0, 0, 0, cfg["instrument_chords"])
        midi.addProgramChange(1, 1, 0, cfg["instrument_melody"])
        midi.addProgramChange(2, 2, 0, cfg["instrument_vocal"])

        # Section plan
        beats_total = int(duration_seconds * tempo / 60)
        bars_total  = max(4, beats_total // 4)
        sections    = _plan_sections(bars_total)

        chord_time  = 0.0
        lead_time   = 0.0
        vocal_time  = 0.0

        for section_name, bar_count in sections:
            is_chorus = (section_name == "chorus")
            prog_cycle = _section_progression(progressions, bar_count)

            for chord in prog_cycle:
                base_vel = (vel_lo + vel_hi) // 2
                if is_chorus:
                    base_vel = vel_hi - 4
                elif section_name in ("intro", "outro"):
                    base_vel = vel_lo + 4

                # --- Track 0: Chord pads ---
                chord_notes = [max(36, min(84, key + i)) for i in chord]
                for note in chord_notes:
                    # Beat 1: full chord block
                    midi.addNote(0, 0, note, chord_time,       2.0,
                                 min(127, base_vel + random.randint(-4, 4)))
                    # Beat 3: softer repeat
                    midi.addNote(0, 0, note, chord_time + 2.0, 2.0,
                                 min(127, base_vel - 12 + random.randint(-4, 4)))

                # --- Track 1: Lead melody ---
                lead_time = _add_lead_bar(
                    midi, key, scale, chord, lead_time,
                    vel_lo, vel_hi, temperature, is_chorus,
                )

                # --- Track 2: Vocal melody ---
                vocal_time = _add_vocal_bar(
                    midi, key, scale, chord, vocal_time,
                    vel_lo, vel_hi, is_chorus,
                )

                chord_time += 4

        midi_path = self._save_midi(midi, mood)

        try:
            if self.soundfont_path and self.soundfont_path.exists():
                mp3_path = self._convert_to_mp3(midi_path, mood)
                if str(mp3_path).endswith(".mp3"):
                    midi_path.unlink()
                    return mp3_path
        except Exception as exc:
            print(f"MP3 conversion failed: {exc}")

        return midi_path

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _save_midi(self, midi: MIDIFile, mood: str) -> Path:
        out_dir = self.model_dir / "generated_music"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"generated_music_{mood.lower()}.mid"
        with open(path, "wb") as f:
            midi.writeFile(f)
        return path

    def _convert_to_mp3(self, midi_path: Path, mood: str) -> Path:
        mp3_dir = self.model_dir / "generated_music" / "mp3"
        mp3_dir.mkdir(exist_ok=True)
        wav  = mp3_dir / f"temp_{mood.lower()}.wav"
        mp3  = mp3_dir / f"generated_music_{mood.lower()}.mp3"

        subprocess.run(
            ["fluidsynth", "-ni", "-F", str(wav), "-r", "44100",
             "-g", "1.0", "-q", str(self.soundfont_path), str(midi_path)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
             "-qscale:a", "2", "-ar", "44100", str(mp3)],
            check=True, capture_output=True,
        )
        if wav.exists():
            wav.unlink()
        return mp3 if (mp3.exists() and mp3.stat().st_size > 0) else midi_path

    def combine_lyrics_and_music(
        self, lyrics: str, mood: str, output_path: Path | None = None
    ) -> dict:
        music_path = self.generate_music(mood)
        if output_path is None:
            out_dir = self.model_dir / "generated_songs"
            out_dir.mkdir(exist_ok=True)
            output_path = out_dir / f"song_{mood.lower()}"
        lyrics_path = str(output_path) + "_lyrics.txt"
        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write(lyrics)
        return {
            "lyrics_path": lyrics_path,
            "music_path":  str(music_path),
            "output_path": str(output_path),
        }


# ---------------------------------------------------------------------------
# Section planning
# ---------------------------------------------------------------------------

def _plan_sections(bars_total: int) -> list[tuple[str, int]]:
    if bars_total <= 4:
        return [("verse", bars_total)]
    intro  = max(2, bars_total // 8)
    outro  = max(2, bars_total // 8)
    middle = bars_total - intro - outro
    chorus = max(4, middle // 3)
    verse1 = max(4, (middle - chorus * 2) // 2)
    verse2 = max(2, middle - chorus * 2 - verse1)
    plan = [
        ("intro",  intro),
        ("verse",  verse1),
        ("chorus", chorus),
        ("verse",  verse2),
        ("chorus", chorus),
        ("outro",  outro),
    ]
    return [(n, b) for n, b in plan if b > 0]


def _section_progression(
    progressions: list[list[int]], bar_count: int
) -> list[list[int]]:
    return [progressions[i % len(progressions)] for i in range(bar_count)]


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------

def _scale_notes(key: int, scale: list[int], octaves: tuple = (-1, 0, 1)) -> list[int]:
    notes = [
        max(36, min(84, key + interval + 12 * oct_))
        for oct_ in octaves
        for interval in scale
    ]
    return sorted(set(notes))


def _add_lead_bar(
    midi: MIDIFile,
    key: int,
    scale: list[int],
    chord: list[int],
    start: float,
    vel_lo: int,
    vel_hi: int,
    temperature: float,
    is_chorus: bool,
) -> float:
    """Fill 4 beats on track 1 with a single-note lead melody."""
    notes     = _scale_notes(key, scale)
    chord_set = {max(36, min(84, key + i)) for i in chord}

    t = start
    remaining = 4.0

    while remaining > 0.01:
        dur = random.choice([1.0, 2.0] if is_chorus else [0.5, 0.5, 1.0, 1.0, 0.5])
        dur = min(dur, remaining)

        if random.random() < (0.60 + temperature * 0.10):
            candidates = [n for n in notes if n in chord_set]
            note = random.choice(candidates) if candidates else random.choice(notes)
        else:
            note = random.choice(notes)

        vel = max(30, min(127, random.randint(vel_lo - 10, vel_hi - 10)))
        midi.addNote(1, 1, note, t, dur, vel)
        t         += dur
        remaining -= dur

    return t


def _add_vocal_bar(
    midi: MIDIFile,
    key: int,
    scale: list[int],
    chord: list[int],
    start: float,
    vel_lo: int,
    vel_hi: int,
    is_chorus: bool,
) -> float:
    """Add a slow, singable vocal-melody suggestion on track 2.

    Chorus: long held notes on chord roots (feels like sung phrases).
    Verse: shorter call-and-response phrases; rests leave space.

    This is a purely instrumental MIDI representation of a vocal line —
    NOT a synthesised or cloned human voice.
    """
    # Vocal melody stays in a narrower range: one octave above the key
    vocal_notes = _scale_notes(key, scale, octaves=(0, 1))
    chord_tones = sorted({max(36, min(84, key + i)) for i in chord})
    # Prefer the root and third of the chord for "singing"
    root  = max(36, min(84, key + chord[0]))
    third = max(36, min(84, key + chord[1])) if len(chord) > 1 else root

    t = start

    if is_chorus:
        # Long held notes — 2 notes over 4 beats, louder
        for note, dur in [(third, 2.0), (root, 2.0)]:
            vel = max(40, min(127, random.randint(vel_lo, vel_hi) - 5))
            midi.addNote(2, 2, note, t, dur, vel)
            t += dur
    else:
        # Verse: shorter phrase (1 beat), then rest (1 beat), repeat
        # Gives a "call / breath" feel
        phrase = [
            random.choice(chord_tones if chord_tones else vocal_notes),
            random.choice(vocal_notes),
        ]
        durations = [1.0, 1.0, 0.5, 0.5]  # note, rest, note, rest
        for i, dur in enumerate(durations):
            if i % 2 == 0:   # note on even steps
                note = phrase[i // 2 % len(phrase)]
                vel  = max(30, min(110, random.randint(vel_lo - 15, vel_hi - 15)))
                midi.addNote(2, 2, note, t, dur * 0.85, vel)  # slight staccato
            # odd steps = rest (no note added)
            t += dur

    return t
