from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import unquote


MARKDOWN_TARGET_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def test_readme_local_links_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    missing: list[str] = []

    for match in MARKDOWN_TARGET_PATTERN.finditer(readme):
        target = match.group(1).strip().strip("<>")
        if (
            not target
            or target.startswith("#")
            or target.startswith(("http://", "https://", "mailto:"))
        ):
            continue

        local_path = unquote(target.split("#", maxsplit=1)[0])
        assert not local_path.startswith("docs/figures"), (
            f"Legacy docs/figures link found in README.md: {target}"
        )
        if not (repo_root / local_path).exists():
            missing.append(target)

    assert not missing, f"Missing local README.md targets: {missing}"


def test_release_version_and_current_readme_claims_are_consistent() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "1.0.0"
    assert "The authoritative current status is [`v1.0.0`]" in readme
    assert "Full real VinDr/VinBigData inference on 15,000 prepared images" in readme
    assert "It does not present a completed benchmark, external validation study" not in readme
    assert "No completed MIMIC-CXR-JPG or VinDr-CXR external validation" not in readme
