from typing import List, Dict, Tuple
from dataclasses import dataclass
import csv

GENRE_WEIGHT = 4.0
MOOD_WEIGHT = 2.0
ENERGY_WEIGHT = 3.0
ACOUSTIC_WEIGHT = 1.0


def _closeness(value: float, target: float) -> float:
    """
    Returns a 0..1 closeness score where 1 means exact match.
    """
    return max(0.0, min(1.0, 1.0 - abs(value - target)))


def _song_score_components(
    genre: str,
    mood: str,
    energy: float,
    acousticness: float,
    favorite_genre: str,
    favorite_mood: str,
    target_energy: float,
    likes_acoustic: bool,
) -> Dict[str, float]:
    """
    Weighted scoring components used by both functional and OOP APIs.

    - Categorical matches (genre, mood) are binary (0 or 1) then weighted.
    - Numeric energy uses closeness-to-target, not "higher/lower is better".
    """
    genre_match = float(genre.strip().lower() == favorite_genre.strip().lower())
    mood_match = float(mood.strip().lower() == favorite_mood.strip().lower())
    energy_closeness = _closeness(energy, target_energy)
    acoustic_alignment = acousticness if likes_acoustic else 1.0 - acousticness

    return {
        "genre": genre_match * GENRE_WEIGHT,
        "mood": mood_match * MOOD_WEIGHT,
        "energy": energy_closeness * ENERGY_WEIGHT,
        "acoustic": acoustic_alignment * ACOUSTIC_WEIGHT,
    }


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """
    Scores one song dictionary against a user preference dictionary.
    """
    favorite_genre = str(user_prefs.get("favorite_genre", user_prefs.get("genre", "")))
    favorite_mood = str(user_prefs.get("favorite_mood", user_prefs.get("mood", "")))
    target_energy = float(user_prefs.get("target_energy", user_prefs.get("energy", 0.5)))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    components = _song_score_components(
        genre=str(song["genre"]),
        mood=str(song["mood"]),
        energy=float(song["energy"]),
        acousticness=float(song.get("acousticness", 0.0)),
        favorite_genre=favorite_genre,
        favorite_mood=favorite_mood,
        target_energy=target_energy,
        likes_acoustic=likes_acoustic,
    )
    total = sum(components.values())

    explanation = (
        f"genre={components['genre']:.2f}/{GENRE_WEIGHT:.1f}, "
        f"mood={components['mood']:.2f}/{MOOD_WEIGHT:.1f}, "
        f"energy={components['energy']:.2f}/{ENERGY_WEIGHT:.1f}, "
        f"acoustic={components['acoustic']:.2f}/{ACOUSTIC_WEIGHT:.1f}, "
        f"total={total:.2f}"
    )
    return total, explanation

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score_song(self, user: UserProfile, song: Song) -> float:
        components = _song_score_components(
            genre=song.genre,
            mood=song.mood,
            energy=song.energy,
            acousticness=song.acousticness,
            favorite_genre=user.favorite_genre,
            favorite_mood=user.favorite_mood,
            target_energy=user.target_energy,
            likes_acoustic=user.likes_acoustic,
        )
        return sum(components.values())

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        ranked = sorted(self.songs, key=lambda song: self._score_song(user, song), reverse=True)
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        components = _song_score_components(
            genre=song.genre,
            mood=song.mood,
            energy=song.energy,
            acousticness=song.acousticness,
            favorite_genre=user.favorite_genre,
            favorite_mood=user.favorite_mood,
            target_energy=user.target_energy,
            likes_acoustic=user.likes_acoustic,
        )

        parts = [
            f"genre={components['genre']:.2f}/{GENRE_WEIGHT:.1f}",
            f"mood={components['mood']:.2f}/{MOOD_WEIGHT:.1f}",
            f"energy={components['energy']:.2f}/{ENERGY_WEIGHT:.1f}",
            f"acoustic={components['acoustic']:.2f}/{ACOUSTIC_WEIGHT:.1f}",
        ]

        return ", ".join(parts)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, explanation = score_song(song, user_prefs)
        scored.append((song, score, explanation))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
