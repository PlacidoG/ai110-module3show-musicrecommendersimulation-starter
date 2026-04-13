from src.recommender import Song, UserProfile, Recommender, score_song

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_energy_scoring_rewards_closeness_not_just_higher_value():
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    }

    close_but_lower_energy = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.75,
        "acousticness": 0.2,
    }
    farther_but_higher_energy = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.98,
        "acousticness": 0.2,
    }

    close_score, _ = score_song(close_but_lower_energy, user_prefs)
    far_score, _ = score_song(farther_but_higher_energy, user_prefs)

    assert close_score > far_score


def test_matching_genre_scores_more_than_matching_mood():
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.6,
        "likes_acoustic": False,
    }

    genre_only_match = {
        "genre": "pop",
        "mood": "sad",
        "energy": 0.6,
        "acousticness": 0.4,
    }
    mood_only_match = {
        "genre": "rock",
        "mood": "happy",
        "energy": 0.6,
        "acousticness": 0.4,
    }

    genre_score, _ = score_song(genre_only_match, user_prefs)
    mood_score, _ = score_song(mood_only_match, user_prefs)

    assert genre_score > mood_score


def test_taste_profile_key_names_are_supported():
    taste_profile = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }

    strong_match_song = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "acousticness": 0.2,
    }
    weak_match_song = {
        "genre": "rock",
        "mood": "moody",
        "energy": 0.3,
        "acousticness": 0.8,
    }

    strong_score, _ = score_song(strong_match_song, taste_profile)
    weak_score, _ = score_song(weak_match_song, taste_profile)

    assert strong_score > weak_score
