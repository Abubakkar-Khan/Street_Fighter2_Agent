"""
Watch a training run live, from progress.csv.

    python watch_progress.py                          # default run
    python watch_progress.py runs\\quick\\progress.csv
    python watch_progress.py --once                   # print one line and exit
    python watch_progress.py --window 20 --interval 3

The CSV is appended by train_sf2.py once per finished episode:
    timestep,episode,episode_return,episode_length
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "runs", "sf2_ppo", "progress.csv")

# ASCII only, so it renders in any Windows console.
BAR = " .:-=+*#%@"


def sparkline(values):
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        return BAR[0] * len(values)
    return "".join(
        BAR[int((value - low) / (high - low) * (len(BAR) - 1))] for value in values
    )


def summarise(csv_path, window):
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None
    rolling = frame.episode_return.rolling(window, min_periods=1).mean()
    return {
        "timesteps": int(frame.timestep.max()),
        "episodes": len(frame),
        "last": float(frame.episode_return.iloc[-1]),
        "mean_window": float(rolling.iloc[-1]),
        "mean_first": float(frame.episode_return.head(window).mean()),
        "best": float(frame.episode_return.max()),
        "length": int(frame.episode_length.iloc[-1]),
        "bar": sparkline(frame.episode_return.tail(30).tolist()),
    }


def line(csv_path, window):
    stats = summarise(csv_path, window)
    if stats is None:
        return "no episodes recorded yet..."
    return (
        f"steps {stats['timesteps']:>7,} | "
        f"episodes {stats['episodes']:>4} | "
        f"last {stats['last']:+6.1f} | "
        f"mean{window} {stats['mean_window']:+6.2f} | "
        f"first{window} {stats['mean_first']:+6.2f} | "
        f"best {stats['best']:+6.1f} | "
        f"len {stats['length']:>5} | {stats['bar']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="print one line and exit")
    args = parser.parse_args()

    if args.once:
        print(line(args.csv, args.window))
        return 0

    print(f"watching {args.csv}")
    print("(Ctrl+C to stop; the last column is the last 30 episode returns)\n")
    last_episodes = -1
    try:
        while True:
            if os.path.exists(args.csv):
                stats = summarise(args.csv, args.window)
                if stats and stats["episodes"] != last_episodes:
                    last_episodes = stats["episodes"]
                    print(line(args.csv, args.window), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
