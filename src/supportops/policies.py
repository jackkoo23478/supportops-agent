"""Load policy documents into identifiable sections."""

import re
from pathlib import Path

from supportops.schemas import PolicyChunk
from supportops.store import default_data_dir


def load_policy_chunks(
    policy_dir: Path | None = None,
) -> list[PolicyChunk]:
    """Read Markdown policies and split them at level-two headings."""
    if policy_dir is None:
        policy_dir = default_data_dir() / "policies"

    if not policy_dir.is_dir():
        raise FileNotFoundError(
            f"Policy directory does not exist: {policy_dir}"
        )

    paths = sorted(policy_dir.glob("*.md"))

    if not paths:
        raise ValueError(
            f"No Markdown policy files found in: {policy_dir}"
        )

    chunks: list[PolicyChunk] = []
    seen_ids: set[str] = set()

    for path in paths:
        text = path.read_text(encoding="utf-8")

        sections = re.split(
            r"^##[ \t]+",
            text,
            flags=re.MULTILINE,
        )[1:]

        if not sections:
            raise ValueError(
                f"No policy sections found in: {path.name}"
            )

        for section in sections:
            heading, _, body = section.partition("\n")
            source_id = heading.strip()
            content = body.strip()

            if not source_id or not content:
                raise ValueError(
                    f"Empty policy ID or content in: {path.name}"
                )

            if source_id in seen_ids:
                raise ValueError(
                    f"Duplicate policy source ID: {source_id}"
                )

            chunks.append(
                PolicyChunk(
                    source_id=source_id,
                    document=path.name,
                    content=content,
                )
            )
            seen_ids.add(source_id)

    return chunks