"""
Generate match start save states for the entire SF2 roster (12 characters).
Saves them directly into the stable-retro game data directory:
  MatchStart.Europe.Ryu.state (= MatchStart.Europe.state)
  MatchStart.Europe.Honda.state
  MatchStart.Europe.Blanka.state
  MatchStart.Europe.Guile.state
  MatchStart.Europe.Balrog.state
  MatchStart.Europe.Vega.state
  MatchStart.Europe.Ken.state
  MatchStart.Europe.ChunLi.state
  MatchStart.Europe.Zangief.state
  MatchStart.Europe.Dhalsim.state
  MatchStart.Europe.Sagat.state
  MatchStart.Europe.Bison.state
"""

import os
import sys
import gzip
import numpy as np
import stable_retro as retro
import PIL.Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SF2_GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"

ROSTER = {
    "Ryu": [],
    "Honda": ["RIGHT"],
    "Blanka": ["RIGHT", "RIGHT"],
    "Guile": ["RIGHT", "RIGHT", "RIGHT"],
    "Balrog": ["RIGHT", "RIGHT", "RIGHT", "RIGHT"],
    "Vega": ["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
    "Ken": ["DOWN"],
    "ChunLi": ["DOWN", "RIGHT"],
    "Zangief": ["DOWN", "RIGHT", "RIGHT"],
    "Dhalsim": ["DOWN", "RIGHT", "RIGHT", "RIGHT"],
    "Sagat": ["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
    "Bison": ["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
}

def generate_state(char_name, cursor_inputs, data_dir):
    print(f"\n[*] Generating match state for: {char_name}...")
    env = retro.make(
        SF2_GAME,
        state=retro.State.NONE,
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.ALL
    )
    env.reset(seed=0)
    b = env.unwrapped.buttons
    start = b.index("START")
    attack = b.index("C")

    def step(btn=None, frames=1):
        act = np.zeros(env.action_space.n, np.uint8)
        if btn is not None:
            act[btn] = 1
        obs = None
        for _ in range(frames):
            obs, r, term, trunc, info = env.step(act)
        return obs, info

    # Step through Sega logo, Capcom logo, title screen, Champion mode
    for _ in range(9):
        step(start, 6)
        step(None, 34)
        step(attack, 6)
        step(None, 34)

    # Wait 40 frames for Player Select screen fade-in
    step(None, 40)

    # Move cursor to target character
    for d in cursor_inputs:
        d_idx = b.index(d)
        step(d_idx, 8)
        step(None, 14)

    # Confirm character
    step(attack, 10)
    step(None, 30)

    # Advance until match is live (both fighters have 176 HP)
    obs = None
    for cycle in range(15):
        step(start, 6)
        step(None, 34)
        obs, info = step(attack, 6)
        step(None, 34)
        h = info.get("health", 0)
        eh = info.get("enemy_health", 0)
        if h == 176 and eh == 176:
            # Settle 90 frames for 'ROUND 1 FIGHT' banner
            for _ in range(90):
                obs, r, term, trunc, info = env.step(np.zeros(env.action_space.n, np.uint8))
            break

    # Save state directly into retro game data folder
    state_file = os.path.join(data_dir, f"MatchStart.Europe.{char_name}.state")
    raw_state = bytes(env.unwrapped.em.get_state())
    with open(state_file, "wb") as f:
        f.write(gzip.compress(raw_state, mtime=0))

    # Also save screenshot
    img_dir = os.path.join("assets", "roster_states")
    os.makedirs(img_dir, exist_ok=True)
    img_file = os.path.join(img_dir, f"{char_name}.png")
    PIL.Image.fromarray(obs).save(img_file)

    print(f"[OK] Wrote {state_file} ({os.path.getsize(state_file):,} bytes)")
    env.close()

def main():
    data_dir = os.path.dirname(
        retro.data.get_file_path(SF2_GAME, "data.json", retro.data.Integrations.STABLE)
    )
    print(f"[*] Target data directory: {data_dir}")

    for char_name, moves in ROSTER.items():
        state_file = os.path.join(data_dir, f"MatchStart.Europe.{char_name}.state")
        # Check if already generated
        if os.path.isfile(state_file) and os.path.getsize(state_file) > 10000:
            print(f"[+] Already exists: {char_name}")
            continue
        generate_state(char_name, moves, data_dir)

    print("\n[DONE] All roster match start states ready!")

if __name__ == "__main__":
    main()
