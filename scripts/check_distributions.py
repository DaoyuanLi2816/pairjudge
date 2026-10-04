"""Check or record the exact wheel/sdist pair, without rebuilding them."""

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(folder, verify=False):
    files = sorted([*folder.glob("*.whl"), *folder.glob("*.tar.gz")])
    if len(files) != 2 or len(list(folder.glob("*.whl"))) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    wheel = next(folder.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if not any(name.endswith("licenses/LICENSE") for name in names):
            raise ValueError("wheel is missing MIT license")
        for name in ("artifacts.py", "catalog.py", "cli.py", "demo.py", "training.py"):
            if "pairjudge/" + name not in names:
                raise ValueError("missing installed module: " + name)
        if any(name.endswith((".safetensors", ".bin", ".pt")) for name in names):
            raise ValueError("weights do not belong in PyPI")
    with tarfile.open(next(folder.glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        for suffix in (
            "/LICENSE",
            "/README.md",
            "/pyproject.toml",
            "/src/pairjudge/cli.py",
            "/tests/test_real_artifacts.py",
        ):
            if not any(name.endswith(suffix) for name in names):
                raise ValueError("sdist missing " + suffix)
        if any("/.cache/" in name or "/.venv" in name for name in names):
            raise ValueError("private build files included")
    checksums = "".join(f"{digest(path)}  {path.name}\n" for path in files)
    manifest = folder / "SHA256SUMS"
    if verify:
        if manifest.read_text(encoding="utf-8") != checksums:
            raise ValueError("candidate hashes differ from SHA256SUMS")
    else:
        if manifest.exists() and manifest.read_text(encoding="utf-8") != checksums:
            raise ValueError("refusing to replace a qualified candidate hash list")
        manifest.write_text(checksums, encoding="utf-8")
    print(checksums, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    check(args.folder, args.verify)
