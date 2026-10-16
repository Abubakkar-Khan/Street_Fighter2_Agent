r"""
Watch the trained Street Fighter II agent play, live, in its own window.

    python watch_model_play.py                                   # runs\sf2_ppo\sf2_ppo_final.zip
    python watch_model_play.py --model runs\quick\sf2_ppo_final.zip
    python watch_model_play.py --speed 2                         # 2x faster
    python watch_model_play.py --episodes 3

The window title shows the live episode / return / health. Close the window or
press Ctrl+C to stop. `--speed 1` plays at real time (60 emulated frames per
second, matching the training frame-skip of 4).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# If invoked from another Python without dependencies (e.g. Windows Python 3.14),
# transparently forward to the Miniconda Python where stable-retro and torch are installed.
MINICONDA_PY = r"C:\Users\HEC\miniconda3\python.exe"
if os.path.isfile(MINICONDA_PY) and os.path.abspath(sys.executable).lower() != os.path.abspath(MINICONDA_PY).lower():
    try:
        import stable_baselines3  # noqa: F401
        import stable_retro      # noqa: F401
    except ImportError:
        import subprocess
        print(f"[*] Detected Python environment missing dependencies: {sys.executable}")
        print(f"[*] Switching to Miniconda Python: {MINICONDA_PY}")
        sys.exit(subprocess.call([MINICONDA_PY, os.path.abspath(__file__)] + sys.argv[1:]))

MULTI_MODEL = os.path.join(HERE, "runs", "sf2_multi", "sf2_ppo_final.zip")
SPECIAL_MODEL = os.path.join(HERE, "runs", "sf2_special", "sf2_ppo_final.zip")
CHAMPION_MODEL = os.path.join(HERE, "runs", "sf2_champion", "sf2_ppo_final.zip")
LEGACY_MODEL = os.path.join(HERE, "runs", "sf2_ppo", "sf2_ppo_final.zip")

if os.path.exists(MULTI_MODEL):
    DEFAULT_MODEL = MULTI_MODEL
elif os.path.exists(SPECIAL_MODEL):
    DEFAULT_MODEL = SPECIAL_MODEL
elif os.path.exists(CHAMPION_MODEL):
    DEFAULT_MODEL = CHAMPION_MODEL
else:
    DEFAULT_MODEL = LEGACY_MODEL


def set_caption(env, text):
    """Update the pyglet window title, if the viewer exists yet."""
    try:
        viewer = env.unwrapped.viewer
        if viewer is not None and viewer.window is not None:
            viewer.window.set_caption(text)
    except Exception:  # noqa: BLE001
        pass


def viewer_closed(env):
    try:
        viewer = env.unwrapped.viewer
        return viewer is not None and not viewer.isopen
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default="health")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=0, help="0 = keep playing")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="1.0 = real time (60 emulated frames/s)",
    )
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic greedy actions (default is dynamic policy sampling)",
    )
    parser.add_argument(
        "--char",
        default=None,
        help="character specialist (e.g. guile, chunli, honda, blanka, zangief, dhalsim, ryu, ken)",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="save state name (e.g. MatchStart.Europe.Ken or shorthand: Ken, ChunLi, Honda, etc.)",
    )
    parser.add_argument(
        "--multi-enemy",
        action="store_true",
        help="cycle across various opponents/characters each episode",
    )
    args = parser.parse_args()

    import train_sf2

    # Resolve character specialist
    norm_char = None
    if args.char:
        norm_char = train_sf2.normalize_character_name(args.char)
        # If user didn't explicitly override model, check for specialist model
        if args.model == DEFAULT_MODEL:
            char_model = os.path.join(HERE, "runs", "characters", norm_char, "sf2_ppo_final.zip")
            if os.path.exists(char_model):
                args.model = char_model

    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        print(f"[X] model not found: {args.model}")
        return 1
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[X] stable-baselines3 is not installed")
        return 1

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # Resolve state / character shorthand
    chosen_state = args.state
    if chosen_state:
        if not chosen_state.startswith("MatchStart.Europe"):
            candidate = f"MatchStart.Europe.{chosen_state}"
            available = train_sf2.get_available_states()
            if candidate in available:
                chosen_state = candidate
    elif norm_char:
        chosen_state = train_sf2.get_character_state(norm_char)
    else:
        chosen_state = train_sf2.default_state_for()

    states_pool = None
    if args.multi_enemy:
        states_pool = train_sf2.get_available_states()

    print("=" * 66)
    print("Street Fighter II - live agent view")
    print("=" * 66)
    print(f"[*] specialist: {norm_char.upper() if norm_char else 'Universal'}")
    print(f"[*] model     : {args.model}")
    print(f"[*] ROM       : {train_sf2.installed_rom_sha1()}")
    print(f"[*] state     : {chosen_state or 'integration default'}")
    if states_pool:
        print(f"[*] multi     : active across {len(states_pool)} roster states")
    print(f"[*] speed     : {args.speed}x   (close the window or Ctrl+C to stop)")
    print()

    model = PPO.load(args.model, device="cpu")

    action_set = "champion"
    if hasattr(model, "action_space") and hasattr(model.action_space, "n"):
        if model.action_space.n == 60:
            action_set = "full"
        elif model.action_space.n in (15, 16):
            action_set = "champion"
    print(f"[*] actions: {action_set} ({getattr(model.action_space, 'n', '?')} discrete moves)")

    env = train_sf2.make_model_env(
        state=chosen_state,
        states=states_pool,
        character=norm_char,
        seed=args.seed,
        scenario=args.scenario,
        render_mode="human",          # this is what opens the window
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        action_set=action_set,
    )

    # frame_skip emulated frames per env step; 60 frames/s is real time.
    step_seconds = args.frame_skip / (60.0 * max(args.speed, 0.05))

    episode = 0
    closed = False
    try:
        while not closed and (args.episodes == 0 or episode < args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            episode += 1
            total, steps, done = 0.0, 0, False

            while not done:
                started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, term, trunc, info = env.step(action)
                total += float(reward)
                steps += 1
                done = term or trunc

                if steps % 15 == 0:
                    set_caption(
                        env,
                        f"SF2 episode {episode} | step {steps} | return {total:+.1f} "
                        f"| HP {info.get('health')} vs {info.get('enemy_health')} "
                        f"| score {info.get('score')}",
                    )
                if viewer_closed(env):
                    closed = True
                    break

                sleep_for = step_seconds - (time.perf_counter() - started)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            if not closed:
                print(
                    f"episode {episode:>3}: return {total:+7.1f} | steps {steps:>5} | "
                    f"health {info.get('health')} vs {info.get('enemy_health')} | "
                    f"score {info.get('score')}",
                )
    except KeyboardInterrupt:
        print("\n[*] interrupted")
    finally:
        env.close()
        print("[OK] window closed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
