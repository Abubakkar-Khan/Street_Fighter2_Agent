import os
import sys
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MINICONDA_PY = r"C:\Users\HEC\miniconda3\python.exe"
if os.path.isfile(MINICONDA_PY) and os.path.abspath(sys.executable).lower() != os.path.abspath(MINICONDA_PY).lower():
    import subprocess
    sys.exit(subprocess.call([MINICONDA_PY, os.path.abspath(__file__)] + sys.argv[1:]))

import stable_retro as retro
import gymnasium as gym
import train_sf2

BUTTONS = train_sf2.GENESIS_BUTTONS

HADOUKEN_MACRO = [
    (["DOWN"], 3),
    (["DOWN", "RIGHT"], 3),
    (["RIGHT", "C"], 4),
    ([], 2),
]
SHORYUKEN_MACRO = [
    (["RIGHT"], 3),
    (["DOWN"], 3),
    (["DOWN", "RIGHT", "C"], 4),
    ([], 2),
]
TATSUMAKI_MACRO = [
    (["DOWN"], 3),
    (["DOWN", "LEFT"], 3),
    (["LEFT", "Z"], 4),
    ([], 2),
]

COMBAT_ACTIONS = [
    ["RIGHT"],                     # 0: Advance
    ["RIGHT", "A"],                # 1: Advance + Light Punch (Jab)
    ["RIGHT", "Z"],                # 2: Advance + Heavy Kick (Roundhouse)
    ["UP", "RIGHT"],               # 3: Jump forward
    ["UP", "RIGHT", "C"],          # 4: Jump-in Fierce Punch
    ["UP", "RIGHT", "Z"],          # 5: Jump-in Heavy Roundhouse
    ["DOWN", "Z"],                 # 6: Knockdown Sweep (Trip)
    ["DOWN", "C"],                 # 7: Anti-Air Uppercut
    ["A"],                         # 8: Standing Light Attack
    ["C"],                         # 9: Standing Fierce Punch
    ["Z"],                         # 10: Standing Roundhouse Kick
    ["DOWN", "LEFT"],              # 11: Low Guard (Crouch Block)
    ["LEFT"],                      # 12: Stand Guard (High Block)
    HADOUKEN_MACRO,                # 13: HADOUKEN (Fireball)
    SHORYUKEN_MACRO,               # 14: SHORYUKEN (Dragon Punch)
    TATSUMAKI_MACRO,               # 15: TATSUMAKI (Hurricane Kick)
]

def btn_to_mask(names):
    mask = np.zeros(len(BUTTONS), dtype=np.int8)
    for name in names:
        if name in BUTTONS:
            mask[BUTTONS.index(name)] = 1
    return mask

class SF2MacroActionWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip=4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.action_space = gym.spaces.Discrete(len(COMBAT_ACTIONS))

    def step(self, action_idx):
        action_spec = COMBAT_ACTIONS[action_idx]
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info = {}

        if isinstance(action_spec, list) and action_spec and isinstance(action_spec[0], tuple):
            # Special Move Macro Sequence
            for button_list, frames in action_spec:
                mask = btn_to_mask(button_list)
                for _ in range(frames):
                    obs, reward, term, trunc, info = self.env.step(mask)
                    total_reward += reward
                    if term or trunc:
                        return obs, total_reward, term, trunc, info
        else:
            # Standard single move held for frame_skip frames
            mask = btn_to_mask(action_spec)
            for _ in range(self.frame_skip):
                obs, reward, term, trunc, info = self.env.step(mask)
                total_reward += reward
                if term or trunc:
                    return obs, total_reward, term, trunc, info

        return obs, total_reward, terminated, truncated, info

def main():
    state = train_sf2.default_state_for() or retro.State.DEFAULT
    print(f"[*] Testing SF2MacroActionWrapper with state: {state}")
    raw_env = retro.make(
        game=train_sf2.SF2_GAME,
        state=state,
        scenario=train_sf2.resolve_scenario("health"),
        render_mode="human",
    )
    env = SF2MacroActionWrapper(raw_env)
    env.reset()

    empty = btn_to_mask([])
    for _ in range(120):
        raw_env.step(empty)

    print("[*] Testing Action 13 (HADOUKEN)...")
    for _ in range(3):
        env.step(13)
        for _ in range(30):
            raw_env.step(empty)

    print("[*] Testing Action 14 (SHORYUKEN)...")
    for _ in range(2):
        env.step(14)
        for _ in range(40):
            raw_env.step(empty)

    print("[*] Testing Action 15 (TATSUMAKI)...")
    for _ in range(2):
        env.step(15)
        for _ in range(40):
            raw_env.step(empty)

    env.close()
    print("[OK] Macro action wrapper test successful!")

if __name__ == "__main__":
    main()
