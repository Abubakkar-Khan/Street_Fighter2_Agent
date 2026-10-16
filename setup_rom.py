"""
Street Fighter II ROM importer for stable-retro.

stable-retro ships an RL-ready environment for the *Sega Genesis* release:

    StreetFighterIISpecialChampionEdition-Genesis-v0

This script finds the matching ROM file anywhere under the given folders,
hash-checks it, and lets stable-retro import it into the environment folder.

It never installs an unverified file: the previous version of this script
blindly copied the first large candidate it found (which silently installed an
arcade ROM as ``rom.md``) and that is exactly what made the environment
unusable.

Usage
-----
    python setup_rom.py                      # scan this folder for the ROM
    python setup_rom.py --source D:\\roms     # scan additional folders
    python setup_rom.py --check-only         # verify, import nothing
    python setup_rom.py --remove-bad         # also delete a rejected rom.md

Exit code 0 means "the environment has a valid ROM", 1 means "not yet".
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile

GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"

# Genesis ROMs are picked up with any of these extensions; "md" is the one
# stable-retro itself uses for the Genesis env.
ROM_EXTENSIONS = (".md", ".bin", ".gen", ".smd")
ARCHIVE_EXTENSIONS = (".zip",)

# 32 MiB is stable-retro's own upper bound for a supported ROM.
MAX_ROM_BYTES = 0x2000000

HERE = os.path.dirname(os.path.abspath(__file__))


def _fail(message: str) -> None:
    print(f"[X] {message}")


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _info(message: str) -> None:
    print(f"[*] {message}")


def import_stable_retro():
    try:
        import stable_retro  # noqa: PLC0415
    except ImportError as exc:
        _fail(f"stable_retro is not installed/importable: {exc}")
        print()
        print("Build it first, e.g. with the bundled PowerShell script:")
        print("    powershell -ExecutionPolicy Bypass -File build_stable_retro.ps1")
        print()
        print("Note: stable-retro publishes no Windows wheels, so the C++ core")
        print("must be compiled locally (MinGW-w64 + MSYS2 make).")
        return None
    return stable_retro


def expected_shas(retro) -> list[str]:
    """Return the SHA1 values stable-retro accepts for this game."""
    sha_path = retro.data.get_file_path(
        GAME, "rom.sha", retro.data.Integrations.STABLE
    )
    if not sha_path:
        return []
    with open(sha_path, "r", encoding="utf-8") as handle:
        return [line.strip().lower() for line in handle if line.strip()]


def game_dir(retro) -> str | None:
    sha_path = retro.data.get_file_path(
        GAME, "rom.sha", retro.data.Integrations.STABLE
    )
    return os.path.dirname(sha_path) if sha_path else None


def installed_rom(retro) -> str | None:
    """Return the installed rom.md path, or None."""
    directory = game_dir(retro)
    if not directory:
        return None
    for extension in ROM_EXTENSIONS:
        candidate = os.path.join(directory, "rom" + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def sha1_of_file(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_rom_file(retro, path: str) -> str | None:
    """SHA1 of a candidate ROM, normalised the way stable-retro does it."""
    try:
        if path.lower().endswith(".smd"):
            # .smd files carry a 512 byte copier header that must be stripped.
            with open(path, "rb") as handle:
                _, digest = retro.data.groom_rom(path, handle)
                return digest
        if os.path.getsize(path) > MAX_ROM_BYTES:
            return None
        return sha1_of_file(path)
    except OSError:
        return None


# Folders that never contain the ROM we are looking for. Walking the whole
# stable-retro checkout would otherwise hash every game's rom.md (~2.5 GB).
SKIP_DIRS = (
    ".git",
    "__pycache__",
    "build",
    "cmakefiles",
    "stable-retro",
    "stable_retro",
    "homebrew_test",
    "runs",
    "scenarios",
    "cores",
    "third-party",
)


def _skip_dir(path: str) -> bool:
    parts = [part.lower() for part in path.replace("\\", "/").split("/")]
    return any(part in SKIP_DIRS for part in parts)


def iter_candidates(sources) -> list[tuple[str, str, str]]:
    """Yield ``(path, label, kind)`` for ROM-looking files in the sources.

    ``kind`` is ``""`` for a plain file and ``"zip"`` for an archive member.
    """
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for source in sources:
        if not os.path.isdir(source):
            continue
        for root, _dirs, files in os.walk(source):
            if _skip_dir(root):
                _dirs[:] = []
                continue
            for name in files:
                full = os.path.join(root, name)
                lower = name.lower()
                if lower.endswith(ROM_EXTENSIONS):
                    if os.path.getsize(full) == 0 or full in seen:
                        continue
                    seen.add(full)
                    found.append((full, name, ""))
                elif lower.endswith(ARCHIVE_EXTENSIONS):
                    try:
                        with zipfile.ZipFile(full) as archive:
                            for entry in archive.namelist():
                                if entry.lower().endswith(ROM_EXTENSIONS):
                                    found.append((full, f"{name}!{entry}", "zip"))
                    except (zipfile.BadZipFile, OSError):
                        continue
    return found


def find_matching_rom(retro, sources, expected) -> tuple[str, str] | None:
    """Return ``(path, label)`` of the first candidate whose hash matches."""
    candidates = iter_candidates(sources)
    if not candidates:
        return None

    _info(f"Scanning {len(candidates)} candidate file(s)...")
    for path, label, kind in candidates:
        if kind == "zip":
            entry = label.split("!", 1)[1]
            try:
                with zipfile.ZipFile(path) as archive:
                    raw = archive.read(entry)
            except (zipfile.BadZipFile, KeyError, OSError):
                continue
            if len(raw) > MAX_ROM_BYTES:
                continue
            digest = hashlib.sha1(raw).hexdigest()
            if digest in expected:
                # stable-retro imports from a real file path, so unpack first.
                tmp_dir = tempfile.mkdtemp(prefix="sf2rom_")
                unpacked = os.path.join(tmp_dir, os.path.basename(entry))
                with open(unpacked, "wb") as handle:
                    handle.write(raw)
                return unpacked, label
            continue

        digest = hash_rom_file(retro, path)
        if digest and digest in expected:
            return path, label
    return None


def quarantine_bad_rom(retro, remove: bool) -> None:
    """Move (or delete) a rom.md whose hash is not accepted."""
    path = installed_rom(retro)
    if not path:
        return
    digest = sha1_of_file(path)
    if remove:
        os.remove(path)
        _info(f"Removed invalid ROM {path}")
    else:
        target = path + f".invalid-{digest[:8]}"
        shutil.move(path, target)
        _info(f"Moved invalid ROM aside: {os.path.basename(target)}")


def main() -> int:
    global GAME

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="extra folder to scan (may be given more than once)",
    )
    parser.add_argument(
        "--game",
        default=GAME,
        help=f"stable-retro game id (default: {GAME})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only report the current state, never import or move files",
    )
    parser.add_argument(
        "--remove-bad",
        action="store_true",
        help="delete a rejected rom.md instead of renaming it",
    )
    args = parser.parse_args()

    GAME = args.game

    print("=" * 68)
    print("Street Fighter II ROM setup for stable-retro")
    print("=" * 68)

    retro = import_stable_retro()
    if retro is None:
        return 1

    _ok(f"stable_retro {retro.__version__.strip()} at {os.path.dirname(retro.__file__)}")

    expected = expected_shas(retro)
    if not expected:
        _fail(f"unknown game {GAME!r}: no rom.sha found. Is the game id correct?")
        return 1
    _info(f"Accepted SHA1: {', '.join(expected)}")

    directory = game_dir(retro)
    _info(f"Environment directory: {directory}")

    current = installed_rom(retro)
    if current:
        digest = sha1_of_file(current)
        if digest in expected:
            _ok(f"Valid ROM already installed: {current}")
            _info(f"SHA1 {digest}")
            return 0
        _fail(f"Installed ROM is NOT the expected one: {current}")
        _info(f"  found    SHA1 {digest}")
        _info(f"  expected SHA1 {expected[0]}")
        if args.check_only:
            print()
            print("Re-run without --check-only to replace it with a valid ROM.")
            return 1
        quarantine_bad_rom(retro, args.remove_bad)
    else:
        _info("No ROM installed yet.")

    sources = [HERE, os.path.join(HERE, "roms"), *args.source]
    match = find_matching_rom(retro, sources, expected)
    if not match:
        print()
        _fail("No matching ROM found.")
        print()
        print("What is needed")
        print("--------------")
        print("Stable-retro's SF2 environment is the SEGA GENESIS release:")
        print("    Street Fighter II' - Special Champion Edition (USA)")
        print(f"    SHA1 {expected[0]}")
        print("    ~2.5 MB, file names like 'Street Fighter II ... (USA).md/.bin/.gen'")
        print("    A .zip containing that file also works.")
        print()
        print("Where to put it")
        print("---------------")
        print(f"    {HERE}")
        print("    then run this script again.")
        print()
        print("Not supported")
        print("-------------")
        print("sfight2b.zip is an arcade bootleg set. Its ROM CRCs are not present in")
        print("the FBNeo core bundled with stable-retro, and stable-retro has no SF2")
        print("arcade integration, so it cannot drive this environment.")
        if args.check_only:
            return 1
        return 1

    path, label = match
    _ok(f"Found matching ROM: {label}")
    if args.check_only:
        return 0

    _info("Importing with stable_retro.data.merge() ...")
    before = installed_rom(retro)
    retro.data.merge(path, quiet=False)
    after = installed_rom(retro)
    if not after or (not before and not after):
        _fail("merge() did not install a ROM for this game.")
        return 1

    digest = sha1_of_file(after)
    if digest not in expected:
        _fail(f"Imported file has unexpected SHA1 {digest}")
        return 1

    _ok(f"ROM installed: {after}")
    _info(f"SHA1 {digest}")

    # A save state only works on the release it was recorded from.
    try:
        import train_sf2  # noqa: PLC0415

        state = train_sf2.default_state_for(GAME)
    except Exception:  # noqa: BLE001
        state = None
    if state:
        _ok(f"save state for this release: {state}")
    elif digest != "a5aad1d108046d9388e33247610dafb4c6516e0b":
        print()
        print("[!] This is not the USA release the integration was authored for,")
        print("    so its save state will crash this ROM. Record one for it once:")
        print("        python make_match_state.py")
        print("    then add the SHA1 -> state mapping to ROM_STATE_MAP in train_sf2.py.")
    else:
        _info("using the integration's own save state (USA release)")

    print()
    print("Next step:")
    print("    python train_sf2.py --smoke     # verify the environment")
    print("    python train_sf2.py             # train a PPO agent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
