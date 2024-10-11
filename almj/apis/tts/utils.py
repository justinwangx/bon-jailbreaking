import json
import os
from pathlib import Path
from typing import Dict, List

import requests


def load_voice_dict():
    # Construct the path to the voice_dict.json file using pathlib
    voices_file_path = Path(__file__).resolve().parent / "voice_dict.json"
    with open(voices_file_path, "r") as file:
        voice_dict = json.load(file)

    return voice_dict, voices_file_path


def query_voices() -> List[Dict]:
    url = "https://api.elevenlabs.io/v1/voices"

    headers = {
        "Accept": "application/json",
        "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    return data["voices"]


def update_voice_dict(pulled_voices: List[Dict]):
    """
    Updates the voice_dict.json file with the latest voices from the ElevenLabs API
    This is then used to help populate voices.py
    """
    voice_dict, voices_file_path = load_voice_dict()
    current_ids = [v.get("id") for v in voice_dict.values()]

    for voice in pulled_voices:
        # If voice id is already in our VOICE_DICT we don't want to re-add it or overwrite
        if voice["voice_id"] in current_ids:
            continue

        # If its not then we want to add
        else:
            voice_dict[voice["name"]] = {
                "id": voice["voice_id"],
                "gender": voice["labels"]["gender"],
                "age": voice["labels"]["age"],
                "nationality": voice["labels"]["accent"],
                "voice": (
                    voice["labels"]["description"]
                    if "description" in voice["labels"]
                    else voice["labels"]["descriptive"]
                ),
                "use": voice["labels"]["use_case"] if "use_case" in voice["labels"] else None,
                "description": voice["description"],
            }

    with open(voices_file_path, "w") as f:
        json.dump(voice_dict, f)
