"""
Convenience launcher for Street Fighter II on stable-retro.

    python play_sf2.py                 # random-input demo with a window
    python play_sf2.py --model runs/sf2_ppo/sf2_ppo_final.zip
    python play_sf2.py --steps 600     # random demo, headless

Training lives in train_sf2.py:

    python train_sf2.py --timesteps 1000000
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# If invoked from another Python without dependencies, transparently switch to Miniconda
MINICONDA_PY = r"C:\Users\HEC\miniconda3\python.exe"
if os.path.isfile(MINICONDA_PY) and os.path.abspath(sys.executable).lower() != os.path.abspath(MINICONDA_PY).lower():
    try:
        import stable_retro  # noqa: F401
    except ImportError:
        import subprocess
        print(f"[*] Switching to Miniconda Python: {MINICONDA_PY}")
        sys.exit(subprocess.call([MINICONDA_PY, os.path.abspath(__file__)] + sys.argv[1:]))


def check_rom() -> bool:
    try:
        import stable_retro as retro
    except ImportError:
        print("[X] stable_retro is not installed.")
        print("    Build it with: powershell -ExecutionPolicy Bypass -File build_stable_retro.ps1")
        return False

    import train_sf2

    try:
        rom = retro.data.get_romfile_path(train_sf2.SF2_GAME)
    except FileNotFoundError:
        print("[X] No Street Fighter II ROM has been imported.")
        print("    Put the Sega Genesis ROM in this folder and run: python setup_rom.py")
        return False

    if retro.data.verify_hash(train_sf2.SF2_GAME):
        print(f"[X] The installed ROM is not the expected one: {rom}")
        print("    Run: python setup_rom.py    (it will replace it)")
        return False

    print(f"[OK] ROM: {rom} ({os.path.getsize(rom):,} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="trained .zip to play with")
    parser.add_argument("--steps", type=int, default=2000, help="random-demo step budget")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--scenario", default="bundled")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not check_rom():
        return 1

    import time

    import train_sf2

    if args.model:
        return train_sf2.main_with(
            [
                "--play",
                "--render",
                "--model",
                args.model,
                "--episodes",
                "1",
                "--scenario",
                args.scenario,
            ],
        )

    env = train_sf2.make_env(
        scenario=args.scenario,
        seed=args.seed,
        render_mode="human",
        frame_skip=1,
    )
    obs, info = env.reset(seed=args.seed)
    print("[*] Random-input demo. Press Ctrl+C to stop.")
    try:
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if reward:
                print(f"    reward {reward:+.2f}  {info}")
            time.sleep(max(0.0, 1.0 / args.fps) - 0.004)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print("[OK] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
