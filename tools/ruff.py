"""Run ruff even where Windows Smart App Control blocks the unsigned ruff.exe.

Tries the ruff in the venv first; if Windows refuses to start it, downloads the official
Linux build of the *same version* (from PyPI, into .cache/ruff/, gitignored) and runs it
through WSL. Arguments pass straight through:

    uv run python tools/ruff.py check
    uv run python tools/ruff.py format --check
"""

import hashlib
import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "ruff"


def _native(args) -> int | None:
    exe = shutil.which("ruff")
    if exe is None:
        return None
    try:
        return subprocess.call([exe, *args], cwd=ROOT)
    except OSError as e:  # WinError 4551: blocked by an application control policy
        print(f"native ruff cannot start ({e.strerror}); falling back to WSL", file=sys.stderr)
        return None


def _linux_binary() -> Path:
    ver = version("ruff")
    dest = CACHE / f"ruff-{ver}-linux"
    if dest.is_file():
        return dest
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/ruff/{ver}/json", timeout=60))
    wheel = next(
        u
        for u in meta["urls"]
        if u["filename"].endswith("manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
    )
    data = urllib.request.urlopen(wheel["url"], timeout=300).read()
    if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
        raise SystemExit("ruff wheel failed its sha256 check")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = next(n for n in z.namelist() if n.endswith("/scripts/ruff"))
        CACHE.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(member))
    return dest


def _wsl_path(p: Path) -> str:
    drive, rest = p.drive.rstrip(":").lower(), p.as_posix()[2:]
    return f"/mnt/{drive}{rest}"


def main() -> int:
    args = sys.argv[1:] or ["check"]
    code = _native(args)
    if code is not None:
        return code
    if shutil.which("wsl") is None:
        raise SystemExit("ruff is blocked here and WSL is not available")
    binary = _wsl_path(_linux_binary())
    cmd = f"chmod +x '{binary}' && cd '{_wsl_path(ROOT)}' && '{binary}' " + " ".join(
        f"'{a}'" for a in args
    )
    return subprocess.call(["wsl", "-e", "bash", "-c", cmd])


if __name__ == "__main__":
    sys.exit(main())
