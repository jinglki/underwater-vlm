"""Verify the immutable count pipeline and enhancement files from the supplied ZIPs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    hashes = json.loads((ROOT / "docs/preserved_sha256.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        print("Changed/missing protected files:\n" + "\n".join(failures))
        return 1
    print(f"PASS: {len(hashes)} protected files match the supplied originals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
