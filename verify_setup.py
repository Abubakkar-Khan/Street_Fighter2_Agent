"""
End-to-end verification for the Street Fighter II RL setup.

It checks, in order:

1. the Python environment and the compiled stable-retro core,
2. that the SF2 integration exists and its ROM is present and valid,
3. that the environment runs (frames advance, rewards/info come back),
4. that Stable-Baselines3 can actually take gradient steps on it.

If the SF2 ROM is missing, the script still proves steps 1, 3 and 4 by using
the small homebrew *Sega Genesis* ROM that ships with stable-retro's own test
suite (``stable-retro/tests/roms/Dekadence-Dekadrive.md``), registered as a
custom integration. That verifies the whole RL pipeline without needing the
copyrighted Sega ROM.

Usage
-----
    python verify_setup.py                 # everything
    python verify_setup.py --no-homebrew   # only the SF2 checks
    python verify_setup.py --rl-steps 512  # longer PPO smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# If invoked from another Python without dependencies, transparently switch to Miniconda
MINICONDA_PY = r"C:\Users\HEC\miniconda3\python.exe"
if os.path.isfile(MINICONDA_PY) and os.path.abspath(sys.executable).lower() != os.path.abspath(MINICONDA_PY).lower():
    try:
        import stable_baselines3  # noqa: F401
        import stable_retro      # noqa: F401
    except ImportError:
        import subprocess
        print(f"[*] Switching to Miniconda Python: {MINICONDA_PY}")
        sys.exit(subprocess.call([MINICONDA_PY, os.path.abspath(__file__)] + sys.argv[1:]))

DEFAULT_REPO = os.path.join(HERE, "stable-retro")
SF2_GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"

# Freely redistributable homebrew ROMs from stable-retro's own test suite. The
# GameGear and Master System ones run on the same genesis_plus_gx core that SF2
# uses, so they exercise the identical code path.
HOMEBREW_CANDIDATES = (
    {
        "game": "SegaTween-GameGear",
        "rom": ("tests", "roms", "benryves-SegaTween.gg"),
        "target": "rom.gg",
        "address": 49152,  # GameGear/Master System RAM base
    },
    {
        "game": "Happy10-Sms",
        "rom": ("tests", "roms", "blind-happy10.sms"),
        "target": "rom.sms",
        "address": 49152,
    },
)
HOMEBREW_DIR = os.path.join(HERE, "homebrew_test")

PASS = "[OK]"
FAIL = "[X]"
INFO = "[*]"


def step(title):
    print()
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))


def check_import():
    step("1. stable-retro install")
    try:
        import stable_retro
    except ImportError as exc:
        print(f"{FAIL} cannot import stable_retro: {exc}")
        print("     Build it with:  powershell -ExecutionPolicy Bypass -File build_stable_retro.ps1")
        return None

    version = stable_retro.__version__.strip()
    package_dir = os.path.dirname(stable_retro.__file__)
    print(f"{PASS} stable_retro {version} ({package_dir})")

    try:
        from stable_retro._retro import core_path

        cores = core_path()
        print(f"{PASS} _retro extension loaded, cores in {cores}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} _retro C++ extension not usable: {exc}")
        return None

    try:
        systems = sorted(stable_retro.data.EMU_INFO)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} core metadata not loaded: {exc}")
        return None
    print(f"{PASS} {len(systems)} system(s) available: {', '.join(systems)}")

    needed = "Genesis"
    if needed not in systems:
        print(f"{FAIL} the Genesis core was not built (needed for SF2)")
        return None
    core_file = os.path.join(cores, stable_retro.data.EMU_CORES[needed])
    if os.path.exists(core_file):
        print(f"{PASS} Genesis core present: {os.path.basename(core_file)}")
    else:
        print(f"{FAIL} Genesis core file missing: {core_file}")
        return None
    return stable_retro


def check_sf2_rom(retro):
    step("2. Street Fighter II integration + ROM")
    try:
        games = retro.data.list_games()
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} could not list games: {exc}")
        return False
    print(f"{PASS} {len(games)} integrations found")
    if SF2_GAME not in games:
        print(f"{FAIL} {SF2_GAME} is not in the stable integrations")
        return False
    print(f"{PASS} {SF2_GAME} present")

    try:
        rom = retro.data.get_romfile_path(SF2_GAME)
    except FileNotFoundError:
        print(f"{FAIL} no ROM imported yet")
        print("     Put the Sega Genesis ROM in this folder and run: python setup_rom.py")
        return False

    size = os.path.getsize(rom)
    errors = retro.data.verify_hash(SF2_GAME)
    if errors:
        print(f"{FAIL} ROM hash mismatch: {rom} ({size:,} bytes)")
        print("     Run: python setup_rom.py    (it will replace the bad file)")
        return False
    print(f"{PASS} ROM verified: {rom} ({size:,} bytes)")

    import train_sf2

    digest = train_sf2.installed_rom_sha1()
    print(f"{PASS} ROM SHA1: {digest}")
    state = train_sf2.default_state_for()
    if state:
        print(f"{PASS} save state for this release: {state}")
    else:
        print(f"{PASS} save state for this release: integration default "
              "(Champion.Level1.RyuVsGuile)")
    return True


def check_integration_files(retro, root):
    """Validate the SF2 integration's data/scenario/state files (no ROM needed)."""
    step("2b. SF2 integration files")
    stable = retro.data.Integrations.STABLE
    info_path = retro.data.get_file_path(SF2_GAME, "data.json", stable)
    scenario_path = retro.data.get_file_path(SF2_GAME, "scenario.json", stable)
    metadata_path = retro.data.get_file_path(SF2_GAME, "metadata.json", stable)
    if not (info_path and scenario_path and metadata_path):
        print(f"{FAIL} SF2 integration is incomplete")
        return False

    state_path = None
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            default_state = json.load(handle).get("default_state")
        if default_state:
            state_path = retro.data.get_file_path(
                SF2_GAME,
                f"{default_state}.state",
                stable,
            )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{FAIL} metadata.json unreadable: {exc}")
        return False

    if state_path and os.path.exists(state_path):
        print(f"{PASS} default save state: {os.path.basename(state_path)}")
    else:
        print(f"{FAIL} default save state missing: {default_state}")
        return False

    checks = [("bundled", scenario_path)]
    health = os.path.join(root, "scenarios", "health.json")
    if os.path.exists(health):
        checks.append(("health", health))
    for label, path in checks:
        data = retro.data.GameData()
        try:
            ok = data.load(info_path, path)
        except Exception as exc:  # noqa: BLE001
            print(f"{FAIL} scenario '{label}' failed to load: {exc}")
            return False
        if not ok:
            print(f"{FAIL} scenario '{label}' failed to load")
            return False
        variables = sorted(data.list_variables())
    print(f"{PASS} data.json + {len(checks)} scenario(s) load; variables: {', '.join(variables)}")
    return True


def run_env(retro, game, steps, info_keys=None):
    """Create an env, run random steps, and report what came back."""
    import hashlib

    import train_sf2

    # A save state only works on the ROM release it was recorded from.
    state = train_sf2.default_state_for(game) or retro.State.DEFAULT
    env = retro.make(
        game=game,
        state=state,
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.FILTERED,
    )
    try:
        obs, info = env.reset(seed=0)
        print(f"{PASS} reset(): obs {obs.shape} {obs.dtype}, action_space {env.action_space}")
        seen = set()
        rewards = []
        infos = {}
        brightest = 0.0
        for _ in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            seen.add(hashlib.md5(obs.tobytes()).hexdigest())
            rewards.append(float(reward))
            infos.update(info)
            if obs.size:
                brightest = max(brightest, float(obs.mean()))
            if terminated or truncated:
                obs, info = env.reset()
        print(
            f"{PASS} {steps} random steps: rewards "
            f"[{min(rewards):.2f}, {max(rewards):.2f}], "
            f"{len(seen)} distinct frames, mean pixel {brightest:.1f}",
        )
        if infos:
            preview = {k: infos[k] for k in sorted(infos)[:6]}
            print(f"{PASS} info variables: {preview}")
        if len(seen) < 2:
            print(f"{FAIL} the emulator is not producing new frames")
            return False, infos
        return True, infos
    finally:
        env.close()


def setup_homebrew(retro, repo):
    """Register a bundled homebrew ROM as a custom integration."""
    for candidate in HOMEBREW_CANDIDATES:
        rom = os.path.join(repo, *candidate["rom"])
        if not os.path.exists(rom):
            continue
        game_dir = os.path.join(HOMEBREW_DIR, candidate["game"])
        os.makedirs(game_dir, exist_ok=True)
        shutil.copyfile(rom, os.path.join(game_dir, candidate["target"]))
        with open(os.path.join(game_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({}, handle)
        with open(os.path.join(game_dir, "data.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"info": {"ram": {"address": candidate["address"], "type": "|u1"}}},
                handle,
            )
        with open(os.path.join(game_dir, "scenario.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "done": {},
                    "reward": {
                        "variables": {"ram": {"reward": 0.01, "penalty": -0.01}},
                    },
                    "crop": [0, 0, 0, 0],
                },
                handle,
            )
        retro.data.add_custom_integration(HOMEBREW_DIR)
        print(f"{PASS} homebrew integration registered: {game_dir}")
        return candidate["game"]
    print(f"{FAIL} no bundled homebrew ROM found under {os.path.join(repo, 'tests', 'roms')}")
    return None


def check_rl(game, steps):
    step("4. Stable-Baselines3 smoke run")
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecFrameStack,
            VecTransposeImage,
        )
    except ImportError as exc:
        print(f"{FAIL} stable-baselines3 not installed: {exc}")
        print("     pip install stable-baselines3")
        return False

    import train_sf2

    def _make():
        return train_sf2.make_env(game=game, seed=0)

    vec = VecTransposeImage(VecFrameStack(DummyVecEnv([_make]), n_stack=4))
    print(f"{PASS} stable-baselines3 {stable_baselines3.__version__}, obs {vec.observation_space}")
    model = PPO(
        "CnnPolicy",
        vec,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=steps, progress_bar=False)
    print(f"{PASS} PPO took {steps} gradient-collecting timesteps on {game}")
    vec.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-homebrew", action="store_true", help="skip the homebrew fallback")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="path to the stable-retro checkout")
    parser.add_argument("--smoke-steps", type=int, default=600)
    parser.add_argument("--rl-steps", type=int, default=256)
    args = parser.parse_args()

    print("=" * 68)
    print("Street Fighter II RL - setup verification")
    print("=" * 68)

    retro = check_import()
    if retro is None:
        return 1

    sf2_ready = check_sf2_rom(retro)

    if sf2_ready:
        step("3. SF2 environment")
        ok, infos = run_env(retro, SF2_GAME, args.smoke_steps)
        if not ok:
            return 1
        for key in ("health", "enemy_health", "score", "matches_won"):
            if key in infos:
                print(f"    {key} = {infos[key]}")
    else:
        print()
        print(f"{INFO} SF2 cannot run yet (ROM missing). Verifying the pipeline another way.")
        if not check_integration_files(retro, HERE):
            return 1

    if not sf2_ready and args.no_homebrew:
        return 1

    if not sf2_ready or not args.no_homebrew:
        step("3b. Homebrew Sega environment (pipeline proof)")
        homebrew_game = setup_homebrew(retro, args.repo)
        if homebrew_game is None:
            return 1
        ok, _ = run_env(retro, homebrew_game, args.smoke_steps)
        if not ok:
            return 1
        rl_game = SF2_GAME if sf2_ready else homebrew_game
    else:
        rl_game = SF2_GAME

    if not check_rl(rl_game, args.rl_steps):
        return 1

    print()
    print("=" * 68)
    if sf2_ready:
        print(f"{PASS} SF2 RL is fully working.")
        print("     Train:  python train_sf2.py --timesteps 1000000")
    else:
        print(f"{PASS} stable-retro, the Genesis core and the RL stack all work.")
        print(f"{FAIL} SF2 is still missing its Sega Genesis ROM.")
        print("     Add 'Street Fighter II - Special Champion Edition (USA)' to this")
        print("     folder and run: python setup_rom.py")
    print("=" * 68)
    return 0 if sf2_ready else 2


if __name__ == "__main__":
    sys.exit(main())
