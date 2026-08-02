#!/usr/bin/env python3
"""Build the browser-readable PyTorch reference implementation asset.

The PyTorch files are the source of truth.  This script copies each complete
file and extracts its numbered teaching blocks without importing PyTorch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "assets" / "implementations.js"

CHAPTER_SOURCES = {
    "mha": "pytorch/mha.py",
    "mqa": "pytorch/mqa.py",
    "gqa": "pytorch/gqa.py",
    "mla": "pytorch/mla.py",
    "dsa": "pytorch/dsa.py",
    "csa": "pytorch/csa.py",
    "hca": "pytorch/hca.py",
    "linear": "pytorch/linear_attention.py",
    "gated-delta": "pytorch/gated_delta.py",
    "kda": "pytorch/kda.py",
}

_OPEN_MARKER = re.compile(r"^\s*# \[Block (\d{2})\] (.+?)\s*$")
_CLOSE_MARKER = re.compile(r"^\s*# \[/Block (\d{2})\]\s*$")
_MARKER_FRAGMENT = re.compile(r"# \[/?Block\b")


class MarkerError(ValueError):
    """Raised when a source file has invalid teaching-block markers."""


def _read_source(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as source_file:
        return source_file.read()


def extract_blocks(source: str, source_name: str = "<source>") -> list[dict[str, Any]]:
    """Extract and validate numbered teaching blocks from ``source``.

    Block ids must start at ``01`` and increase without gaps.  Markers must be
    paired and cannot nest.  ``start`` and ``end`` are one-based, inclusive
    line numbers for ``code`` and therefore exclude the two marker lines.
    """

    lines = source.splitlines(keepends=True)
    blocks: list[dict[str, Any]] = []
    active: tuple[str, str, int] | None = None

    for line_number, line in enumerate(lines, start=1):
        marker_line = line.rstrip("\r\n")
        opening = _OPEN_MARKER.fullmatch(marker_line)
        closing = _CLOSE_MARKER.fullmatch(marker_line)

        if opening:
            if active is not None:
                raise MarkerError(
                    f"{source_name}:{line_number}: nested block {opening.group(1)} "
                    f"inside block {active[0]}"
                )
            expected_id = f"{len(blocks) + 1:02d}"
            block_id, title = opening.groups()
            if block_id != expected_id:
                raise MarkerError(
                    f"{source_name}:{line_number}: expected Block {expected_id}, "
                    f"found Block {block_id}"
                )
            active = (block_id, title, line_number)
            continue

        if closing:
            if active is None:
                raise MarkerError(
                    f"{source_name}:{line_number}: closing Block "
                    f"{closing.group(1)} has no opening marker"
                )
            block_id, title, opening_line = active
            if closing.group(1) != block_id:
                raise MarkerError(
                    f"{source_name}:{line_number}: Block {block_id} closed by "
                    f"Block {closing.group(1)}"
                )
            code_start = opening_line + 1
            code_end = line_number - 1
            blocks.append(
                {
                    "id": block_id,
                    "title": title,
                    "code": "".join(lines[opening_line:line_number - 1]),
                    "start": code_start,
                    "end": code_end,
                }
            )
            active = None
            continue

        if _MARKER_FRAGMENT.search(marker_line):
            raise MarkerError(
                f"{source_name}:{line_number}: malformed teaching-block marker"
            )

    if active is not None:
        block_id, _, opening_line = active
        raise MarkerError(
            f"{source_name}:{opening_line}: Block {block_id} has no closing marker"
        )
    if not blocks:
        raise MarkerError(f"{source_name}: no teaching blocks found")
    return blocks


def build_implementations(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Read every mapped source and return deterministic browser data."""

    implementations: dict[str, dict[str, Any]] = {}
    for chapter_id, relative_path in CHAPTER_SOURCES.items():
        source_path = root / relative_path
        source = _read_source(source_path)
        implementations[chapter_id] = {
            "path": relative_path,
            "source": source,
            "blocks": extract_blocks(source, relative_path),
        }
    return implementations


def render_asset(implementations: dict[str, dict[str, Any]]) -> str:
    """Serialize implementations as an executable, deterministic JS asset."""

    payload = json.dumps(
        implementations,
        ensure_ascii=False,
        indent=2,
        separators=(",", ": "),
    )
    return f"window.ATTENTION_IMPLEMENTATIONS = {payload};\n"


def synchronize(*, check: bool = False, root: Path = ROOT) -> bool:
    """Check or update the generated asset.

    Returns ``True`` when the checked asset is current or a write succeeds.
    """

    output_path = root / "assets" / "implementations.js"
    expected = render_asset(build_implementations(root))
    current = _read_source(output_path) if output_path.exists() else None

    if check:
        if current != expected:
            print(
                f"{output_path} is stale; run {Path(__file__).name}",
                file=sys.stderr,
            )
            return False
        return True

    if current != expected:
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            output_file.write(expected)
        print(f"updated {output_path}")
    else:
        print(f"{output_path} is already current")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero instead of writing when the generated asset is stale",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return 0 if synchronize(check=args.check) else 1
    except (MarkerError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
