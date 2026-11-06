import pytest

from supportops.policies import load_policy_chunks


def test_loads_sections_with_source_metadata(tmp_path):
    (tmp_path / "returns.md").write_text(
        "# Return Policy\n\n"
        "## returns.eligibility\n\n"
        "Unused items may be returned within 30 days.\n\n"
        "## returns.exclusions\n\n"
        "Digital products cannot be returned.\n",
        encoding="utf-8",
    )

    chunks = load_policy_chunks(tmp_path)

    assert len(chunks) == 2

    assert chunks[0].source_id == "returns.eligibility"
    assert chunks[0].document == "returns.md"
    assert chunks[0].content == (
        "Unused items may be returned within 30 days."
    )

    assert chunks[1].source_id == "returns.exclusions"
    assert chunks[1].content == (
        "Digital products cannot be returned."
    )


def test_rejects_duplicate_ids_across_documents(tmp_path):
    for filename in ["returns.md", "refunds.md"]:
        (tmp_path / filename).write_text(
            "# Policy\n\n"
            "## shared.rule\n\n"
            "A policy rule.\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Duplicate policy source ID"):
        load_policy_chunks(tmp_path)


@pytest.mark.parametrize(
    ("text", "expected_error"),
    [
        (
            "# Return Policy\n\nNo sections here.\n",
            "No policy sections found",
        ),
        (
            "# Return Policy\n\n## returns.eligibility\n",
            "Empty policy ID or content",
        ),
    ],
)
def test_rejects_invalid_documents(tmp_path, text, expected_error):
    (tmp_path / "returns.md").write_text(
        text,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=expected_error):
        load_policy_chunks(tmp_path)


def test_loads_sample_policy_documents():
    chunks = load_policy_chunks()
    by_id = {chunk.source_id: chunk for chunk in chunks}

    assert "returns.eligibility" in by_id
    assert "refunds.approval" in by_id
    assert "shipping.delivery" in by_id
    assert "warranty.coverage" in by_id

    assert "30 calendar days" in by_id["returns.eligibility"].content
    assert by_id["returns.eligibility"].document == "returns.md"