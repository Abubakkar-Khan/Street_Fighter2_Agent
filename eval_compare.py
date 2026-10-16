"""Head-to-head evaluation of two SF2 checkpoints on identical seeds."""

import sys

import numpy as np
from stable_baselines3 import PPO

import train_sf2

MODELS = [
    ("10k steps ", r"runs\sf2_ppo\sf2_ppo_10000_steps.zip"),
    ("50k steps ", r"runs\sf2_ppo\sf2_ppo_50000_steps.zip"),
    ("100k steps", r"runs\sf2_ppo\sf2_ppo_final.zip"),
]
SEEDS = list(range(100, 118))  # 18 matches, identical for every model


def evaluate(path, seeds, max_steps=1700):
    model = PPO.load(path, device="cpu")
    returns = []
    for seed in seeds:
        env = train_sf2.make_model_env(seed=seed, max_episode_steps=max_steps)
        obs, _ = env.reset(seed=seed)
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        env.close()
        returns.append(total)
    return np.array(returns)


def main():
    print(f"evaluating on {len(SEEDS)} identical seeds, 1700-step episodes\n")
    results = {}
    for name, path in MODELS:
        r = evaluate(path, SEEDS)
        results[name] = r
        print(
            f"{name}: mean {r.mean():6.2f} | median {np.median(r):6.1f} | "
            f"best {r.max():5.1f} | worst {r.min():6.1f} | "
            f"positive {int((r > 0).sum())}/{len(r)}",
        )

    first = list(results.values())[0]
    last = list(results.values())[-1]
    wins = int((last > first).sum())
    print(f"\n100k beat 10k on {wins}/{len(SEEDS)} matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
