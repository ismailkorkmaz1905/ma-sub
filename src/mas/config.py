from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EPISODES=ROOT/'EPISODES'
def episode_dir(n): return EPISODES/f'Muhtemel Ask {n}.Bolum'
