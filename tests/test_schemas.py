import pytest

from sparkle_motion.schemas import MoviePlan


def test_movieplan_minimal_valid():
    data = {
        "title": "Test Movie",
        "characters": [{"id": "c1", "name": "Alice"}],
        "shots": [
            {
                "id": "shot_1",
                "duration_sec": 5,
                "visual_description": "A calm park",
                "start_frame_prompt": "park at dawn",
                "end_frame_prompt": "park closeup",
            }
        ],
    }

    mp = MoviePlan(**data)
    assert mp.title == "Test Movie"
    assert len(mp.shots) == 1


def test_shot_missing_required_field_fails():
    bad = {
        "title": "Bad Movie",
        "shots": [
            {
                "id": "shot_1",
                "duration_sec": 5,
                "visual_description": "A calm park",
                # missing start_frame_prompt and end_frame_prompt
            }
        ],
    }

    with pytest.raises(Exception):
        MoviePlan(**bad)
