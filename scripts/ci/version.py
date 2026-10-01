"""Read, stamp and bump the version: `[project] version` in pyproject.toml is the source of
truth; `[package] version` in Cargo.toml follows it (as SemVer).

Usage:
    python scripts/ci/version.py read
    python scripts/ci/version.py nightly            # X.Y.(Z+1).devYYYYmmddHHMMSS
    python scripts/ci/version.py bump patch|minor|major
    python scripts/ci/version.py write <version>    # stamp pyproject.toml and Cargo.toml
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
CARGO = ROOT / "Cargo.toml"
# the first `version = "..."` line: [project] / [package] come before any dependency table
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')


def read() -> str:
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version found in pyproject.toml")
    return match.group(2)


def semver(version: str) -> str:
    """PEP 440 -> SemVer for Cargo: 1.0.1.dev20261001 -> 1.0.1-dev20261001."""
    core = re.match(r"^(\d+\.\d+\.\d+)(.*)$", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    rest = core.group(2).lstrip(".")
    return f"{core.group(1)}-{rest}" if rest else core.group(1)


def stamp(path: Path, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, count = PATTERN.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise SystemExit(f"no version found in {path.name}")
    path.write_text(new, encoding="utf-8")


def write(version: str) -> None:
    stamp(PYPROJECT, version)
    stamp(CARGO, semver(version))


def base(version: str) -> tuple[int, int, int]:
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    major, minor, patch = (int(g) for g in core.groups())
    return major, minor, patch


def bump(version: str, part: str) -> str:
    major, minor, patch = base(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version: str) -> str:
    """PEP 440 dev release of the next patch, ordered by its UTC timestamp."""
    stamp_ = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    major, minor, patch = base(version)
    return f"{major}.{minor}.{patch + 1}.dev{stamp_}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("read", "nightly", "bump", "write"))
    ap.add_argument("value", nargs="?")
    args = ap.parse_args()
    current = read()

    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        print(bump(current, args.value or "patch"))
    else:
        if not args.value:
            raise SystemExit("write needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
