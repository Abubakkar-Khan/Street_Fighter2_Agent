"""
Create a save state for the *installed* Street Fighter II ROM at match start.

stable-retro's SF2 integration ships one save state that was captured from the
USA release. Other releases (e.g. the Europe build) boot fine but crash if that
state is restored, because the saved CPU context belongs to different code.

This script boots whatever ROM is installed, plays the menus with scripted
inputs until a match is live, and writes the resulting emulator state next to
the integration data, gzipped exactly the way ``RetroEnv.load_state`` expects.

Usage
-----
    python make_match_state.py                     # -> MatchStart.Europe.state
    python make_match_state.py --name MyState      # custom name
    python make_match_state.py --settle 120        # frames to run once live

The name is then wired up automatically: ``train_sf2.py`` maps the SHA1 of the
installed ROM to this state (see ``ROM_STATE_MAP`` in train_sf2.py).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SF2_GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"

# Sequence that walks: SEGA logo -> CAPCOM -> title -> mode menu -> main menu
# -> PLAYER SELECT -> confirm with an attack button -> VS -> ROUND 1.
PRESS_FRAMES = 6
GAP_FRAMES = 34
MAX_MENU_PRESSES = 40


def sha1_of(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def drive_to_match(env, verbose=True):
    """Press through the menus until a match with full health is running."""
    import stable_retro as retro  # noqa: F401  (imported for Actions by caller)

    buttons = env.unwrapped.buttons
    start = buttons.index("START")
    attack = buttons.index("C")

    def step(pressed=None):
        action = np.zeros(env.action_space.n, np.uint8)
        if pressed is not None:
            action[pressed] = 1
        return env.step(action)

    env.reset(seed=0)
    for attempt in range(MAX_MENU_PRESSES):
        for _ in range(PRESS_FRAMES):
            obs, reward, terminated, truncated, info = step(start)
        for _ in range(GAP_FRAMES):
            obs, reward, terminated, truncated, info = step()
        for _ in range(PRESS_FRAMES):
            obs, reward, terminated, truncated, info = step(attack)
        for _ in range(GAP_FRAMES):
            obs, reward, terminated, truncated, info = step()

        health = info.get("health", 0)
        enemy = info.get("enemy_health", 0)
        if verbose:
            print(
                f"    menu cycle {attempt + 1:2d}: health={health:4d} "
                f"enemy_health={enemy:4d}",
            )
        if health and health > 0 and enemy and enemy > 0:
            return info
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default=SF2_GAME)
    parser.add_argument("--name", default="MatchStart.Europe")
    parser.add_argument(
        "--settle",
        type=int,
        default=90,
        help="frames to run once the match is live, before snapshotting",
    )
    parser.add_argument("--screenshot", default=os.path.join(HERE, "match_state_frame.png"))
    args = parser.parse_args()

    try:
        import stable_retro as retro
    except ImportError:
        print("[X] stable_retro is not installed")
        return 1

    try:
        rom = retro.data.get_romfile_path(args.game)
    except FileNotFoundError:
        print("[X] no ROM imported for", args.game)
        return 1
    digest = sha1_of(rom)
    print(f"[*] ROM : {rom}")
    print(f"[*] SHA1: {digest}")

    print("[*] Booting and driving the menus...")
    env = retro.make(
        game=args.game,
        state=retro.State.NONE,
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.ALL,
    )
    try:
        info = drive_to_match(env)
        if info is None:
            print("[X] never reached a live match")
            return 1

        print(f"[*] Match live (health={info['health']}, "
              f"enemy_health={info['enemy_health']}); settling {args.settle} frames")
        blank = np.zeros(env.action_space.n, np.uint8)
        for _ in range(args.settle):
            obs, reward, terminated, truncated, info = env.step(blank)

        if info.get("health", 0) <= 0 or info.get("enemy_health", 0) <= 0:
            print("[X] the match ended while settling; lower --settle")
            return 1

        state = bytes(env.unwrapped.em.get_state())
        if not state:
            print("[X] the emulator returned an empty state")
            return 1

        game_dir = os.path.dirname(
            retro.data.get_file_path(args.game, "data.json", retro.data.Integrations.STABLE),
        )
        os.makedirs(game_dir, exist_ok=True)
        target = os.path.join(game_dir, f"{args.name}.state")
        with open(target, "wb") as handle:
            handle.write(gzip.compress(state, mtime=0))
        print(f"[OK] wrote {target} ({os.path.getsize(target):,} bytes gzip, "
              f"{len(state):,} bytes raw)")

        try:
            from PIL import Image

            Image.fromarray(obs).save(args.screenshot)
            print(f"[OK] screenshot {args.screenshot}")
        except ImportError:
            pass

        print()
        print("Next steps:")
        print(f"  python train_sf2.py --smoke     # should show health={info['health']}")
        print("  (train_sf2.py maps the installed ROM's SHA1 to this state automatically)")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
