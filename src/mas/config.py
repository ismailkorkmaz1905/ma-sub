from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "EPISODES"


def episode_dir(episode):
    return EPISODES / f"Muhtemel Ask {episode}.Bolum"
