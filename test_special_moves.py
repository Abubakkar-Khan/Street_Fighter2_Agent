import os
import sys
import time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MINICONDA_PY = r"C:\Users\HEC\miniconda3\python.exe"
if os.path.isfile(MINICONDA_PY) and os.path.abspath(sys.executable).lower() != os.path.abspath(MINICONDA_PY).lower():
    import subprocess
    sys.exit(subprocess.call([MINICONDA_PY, os.path.abspath(__file__)] + sys.argv[1:]))

import stable_retro as retro
import train_sf2

# Button names: ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
BUTTONS = train_sf2.GENESIS_BUTTONS

def btn_to_mask(names):
    mask = np.zeros(len(BUTTONS), dtype=np.int8)
    for name in names:
        if name in BUTTONS:
            mask[BUTTONS.index(name)] = 1
    return mask

def play_sequence(env, sequence, wait_between=30):
    for buttons, frames in sequence:
        mask = btn_to_mask(buttons)
        for _ in range(frames):
            env.step(mask)
    # Neutral recovery frames
    empty = btn_to_mask([])
    for _ in range(wait_between):
        env.step(empty)

def main():
    state = train_sf2.default_state_for() or retro.State.DEFAULT
    print(f"[*] Testing Special Moves with state: {state}")
    env = retro.make(
        game=train_sf2.SF2_GAME,
        state=state,
        scenario=train_sf2.resolve_scenario("health"),
        render_mode="human",
    )
    env.reset()

    # Allow intro / match start animation to finish
    empty = btn_to_mask([])
    print("[*] Waiting for match start...")
    for _ in range(120):
        env.step(empty)

    print("[*] Executing HADOUKEN (Fireball)...")
    # Hadouken: D -> D+R -> R + Punch
    hadouken = [
        (["DOWN"], 3),
        (["DOWN", "RIGHT"], 3),
        (["RIGHT", "C"], 4),
        ([], 2),
    ]
    play_sequence(env, hadouken, wait_between=45)

    print("[*] Executing SHORYUKEN (Dragon Punch)...")
    # Shoryuken: R -> D -> D+R + Punch
    shoryuken = [
        (["RIGHT"], 3),
        (["DOWN"], 3),
        (["DOWN", "RIGHT", "C"], 4),
        ([], 2),
    ]
    play_sequence(env, shoryuken, wait_between=50)

    print("[*] Executing TATSUMAKI (Hurricane Kick)...")
    # Tatsumaki: D -> D+L -> L + Kick
    tatsumaki = [
        (["DOWN"], 3),
        (["DOWN", "LEFT"], 3),
        (["LEFT", "Z"], 4),
        ([], 2),
    ]
    play_sequence(env, tatsumaki, wait_between=60)

    print("[*] Repeating HADOUKEN barrage...")
    for i in range(3):
        play_sequence(env, hadouken, wait_between=35)

    env.close()
    print("[OK] Special moves test finished cleanly.")

if __name__ == "__main__":
    main()
