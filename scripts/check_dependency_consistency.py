from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def main() -> int:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    constraints: dict[str, object] = {}

    for dep in project["project"]["dependencies"]:
        req = Requirement(dep)
        constraints[req.name.lower()] = req.specifier

    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)")
    violations: list[tuple[str, str, str]] = []

    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue

        name, version = match.group(1).lower(), match.group(2)
        spec = constraints.get(name)
        if spec and Version(version) not in spec:
            violations.append((name, version, str(spec)))

    if violations:
        print("Dependency mismatch between pyproject.toml and requirements.txt:")
        for name, version, spec in violations:
            print(f" - {name}=={version} violates {spec}")
        return 1

    print("Dependency constraints are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
