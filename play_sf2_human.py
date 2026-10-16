"""
Play Street Fighter II yourself, through stable-retro.

Opens a window and maps your keyboard onto the Genesis 3/6-button pad.

Controls
--------
    Arrow keys   move / jump / crouch
    Z  X  C      A  B  C   (Genesis punches & kicks)
    A  S  D      X  Y  Z   (6-button mode)
    Enter        START
    Tab          MODE
    Esc          quit

The save state is chosen for whichever ROM release is installed
(see ROM_STATE_MAP in train_sf2.py).
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# If invoked from another Python without dependencies, transparently switch to Miniconda
MINICONDA_PY = r"C:\Users\HEC\miniconda3\python.exe"
if os.path.isfile(MINICONDA_PY) and os.path.abspath(sys.executable).lower() != os.path.abspath(MINICONDA_PY).lower():
    try:
        import stable_retro  # noqa: F401
        import pyglet        # noqa: F401
    except ImportError:
        import subprocess
        print(f"[*] Switching to Miniconda Python: {MINICONDA_PY}")
        sys.exit(subprocess.call([MINICONDA_PY, os.path.abspath(__file__)] + sys.argv[1:]))

WINDOW_TITLE = "Street Fighter II' - Special Champion Edition (stable-retro)"


def build_interactive_class(retro_interactive):
    """Wrap stable-retro's example with a shutdown that actually works.

    ``RetroInteractive._on_close`` calls ``sys.exit(0)`` from inside a Win32
    message callback. ``SystemExit`` cannot cross that ctypes boundary, pyglet
    ignores it, the render loop keeps running and every later ``env.step()``
    then raises ``'RetroEnv' object has no attribute 'em'``. Closing the window
    should instead stop the loop and release the emulator once.
    """

    class SF2Interactive(retro_interactive):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._quit = False
            self._win.on_close = self._request_quit
            try:
                self._win.set_caption(WINDOW_TITLE)
            except Exception:  # noqa: BLE001
                pass

        def _request_quit(self):
            print("[*] close requested: window close event (X / WM_CLOSE)", flush=True)
            self._quit = True

        def _on_close(self):
            # Called by the base class when Esc is pressed.
            print("[*] close requested: Esc key", flush=True)
            self._quit = True

        def _draw(self):
            if self._quit:
                return
            super()._draw()

        def run(self):
            print(
                f"[*] window opened: {self._win.width}x{self._win.height} "
                f"({self._win.get_size()})",
                flush=True,
            )
            prev_frame_time = time.time()
            last_report = 0
            while not self._quit:
                self._win.switch_to()
                self._win.dispatch_events()
                now = time.time()
                self._update(now - prev_frame_time)
                prev_frame_time = now
                self._draw()
                self._win.flip()
                if self._steps - last_report >= 1800:
                    last_report = self._steps
                    print(f"[*] alive: {self._steps} steps", flush=True)

            print("\n[*] closing...", flush=True)
            for action in (self._env.close, self._win.close):
                try:
                    action()
                except Exception:  # noqa: BLE001
                    pass

    return SF2Interactive


def main() -> int:
    try:
        import stable_retro as retro
        from stable_retro.examples.interactive import RetroInteractive
    except ImportError as exc:
        print(f"[X] stable_retro is not usable: {exc}")
        return 1

    import train_sf2

    try:
        rom = retro.data.get_romfile_path(train_sf2.SF2_GAME)
    except FileNotFoundError:
        print("[X] no Street Fighter II ROM imported; run: python setup_rom.py")
        return 1

    state = train_sf2.default_state_for() or retro.State.DEFAULT
    print("=" * 64)
    print("Street Fighter II - playable window (stable-retro)")
    print("=" * 64)
    print(f"[*] ROM   : {rom} ({os.path.getsize(rom):,} bytes)")
    print(f"[*] state : {state}")
    print()
    print("    Arrows      move / jump / crouch")
    print("    Z X C       A B C")
    print("    A S D       X Y Z  (6-button)")
    print("    Enter       START")
    print("    Esc         quit")
    print()

    interactive = build_interactive_class(RetroInteractive)(
        game=train_sf2.SF2_GAME,
        state=state,
        scenario=None,
        # False, not None: stable-retro's example starts movie recording for any
        # value that is not literally False, which litters the folder with .bk2.
        record=False,
    )
    interactive.run()
    print("[OK] window closed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
