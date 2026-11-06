"""Numeric inference must preserve threshold direction, quantity roles, and units."""

from unittest.mock import Mock

import pytest

from supportops.quantity_checks import check_quantities
from supportops.schemas import PolicyAnswer
from supportops.verification import PolicyEvidence, PolicyVerification, PolicyVerifier


RETURN = {"source_id": "returns.eligibility", "content": (
    "Unused items in their original packaging may be returned within "
    "30 calendar days of delivery. Proof of purchase is required."
)}
APPROVAL = {"source_id": "refunds.approval", "content": (
    "Refunds must satisfy the applicable return policy. "
    "Eligible refunds of USD 50.00 or less may be approved automatically. "
    "Refunds above USD 50.00 require human approval before execution."
)}
PROCESSING = {"source_id": "refunds.processing", "content": (
    "Refund processing takes 5 to 10 business days after inspection."
)}


@pytest.mark.parametrize("answer", [
    "The 30-day return window starts from delivery.",
    "Returns are allowed within 30 days of delivery.",
    "Returns are allowed within 30 calendar days of delivery.",
    "Returns have a 30-calendar-day window from delivery.",
])
def test_omitted_or_hyphenated_qualifiers_preserve_the_source_duration(answer):
    issues, _ = check_quantities(answer, [RETURN])
    assert issues == []


@pytest.mark.parametrize("answer", [
    "Returns are allowed within 60 days of delivery.",
    "Returns are allowed within 30 business days of delivery.",
    "Returns have a 30-business-day window.",
    "Returns are allowed within 30 working days of delivery.",
    "Returns are allowed within 30 months of delivery.",
])
def test_wrong_duration_values_units_and_explicit_qualifiers_remain_blocked(answer):
    assert check_quantities(answer, [RETURN])[0]


def test_ambiguous_omitted_qualifier_requires_clarification():
    other = {"source_id": "other", "content": "The shipping window is 30 business days."}
    issues, _ = check_quantities("The window is 30 days.", [RETURN, other])
    assert issues


@pytest.mark.parametrize("answer,passed", [
    ("Processing takes 5 to 10 business days.", True),
    ("Processing takes 5-10 days.", True),
    ("Processing takes 5 to 10 working days.", True),
    ("Processing takes 10 business days.", False),
    ("Processing takes 5 to 10 calendar days.", False),
    ("Processing takes 5 to 11 business days.", False),
])
def test_processing_range_bounds_cannot_be_silently_changed(answer, passed):
    assert (not check_quantities(answer, [PROCESSING])[0]) is passed


def test_question_amount_is_verified_against_source_threshold():
    issues, context = check_quantities(
        "An eligible USD 50.01 refund requires human approval.", [APPROVAL],
        "Does an eligible USD 50.01 refund require human approval?",
    )
    assert issues == []
    assert context[0]["amount"] == "50.01"
    assert context[0]["threshold"] == "50.00"
    assert context[0]["comparison"] == ">"
    assert context[0]["approval"] == "manual_required"


@pytest.mark.parametrize("value,claim,passed", [
    ("49.99", "may be approved automatically", True),
    ("49.99", "does not require human approval", True),
    ("49.99", "requires human approval", False),
    ("50.01", "may be approved automatically", False),
    ("50.01", "does not require human approval", False),
    ("50.01", "requires human approval", True),
    ("50.001", "requires human approval", False),
])
def test_derived_approval_values_preserve_direction_and_currency_precision(value, claim, passed):
    issues, _ = check_quantities(
        f"An eligible USD {value} refund {claim}.", [APPROVAL], f"Refund policy for USD {value}?",
    )
    assert (not issues) is passed


def test_arithmetic_uses_actual_source_threshold_not_a_hardcoded_fifty():
    source = {**APPROVAL, "content": APPROVAL["content"].replace("50.00", "75.00")}
    issues, context = check_quantities(
        "An eligible USD 75.01 refund requires human approval.", [source], "Policy for USD 75.01?",
    )
    assert issues == []
    assert context[0]["threshold"] == "75.00"


@pytest.mark.parametrize("answer,question,sources", [
    ("An eligible USD 50.01 refund requires human approval.", "What is the refund policy?", [APPROVAL]),
    ("An eligible USD 60.01 refund requires human approval.", "Policy for USD 50.01?", [APPROVAL]),
    ("Refunds above USD 50.01 require human approval.", "Policy for USD 50.01?", [APPROVAL]),
    ("Refunds above USD50.01 require human approval.", "Policy for USD50.01?", [APPROVAL]),
    ("Refunds greater than USD 50.01 require human approval.", "Policy for USD 50.01?", [APPROVAL]),
    ("The return fee is USD 50.01.", "Policy for USD 50.01?", [APPROVAL]),
    ("The refund fee of USD 50.01 requires human approval.", "Policy for USD 50.01?", [APPROVAL]),
    ("An eligible USD 50.01 refund requires human approval.", "Policy for USD 50.01?", [RETURN]),
])
def test_new_thresholds_fees_or_unsupplied_amounts_cannot_use_the_arithmetic_exception(
    answer, question, sources,
):
    assert check_quantities(answer, sources, question)[0]


def test_repeated_amounts_are_checked_in_their_own_sentences():
    text = (
        "An eligible USD 50.01 refund requires human approval. "
        "An eligible USD 50.01 refund may be approved automatically."
    )
    assert check_quantities(text, [APPROVAL], "Policy for USD 50.01?")[0]


def test_known_amount_cannot_legitimize_a_new_duration_with_the_same_numeric_value():
    issues, _ = check_quantities(
        "An eligible USD 50.01 refund requires human approval. Processing takes 50.01 days.",
        [APPROVAL], "Policy for USD 50.01?",
    )
    assert issues


@pytest.mark.parametrize("answer,question,source", [
    ("The 30-day return window starts from delivery.", "Does it start from purchase?", RETURN),
    ("An eligible USD 50.01 refund requires human approval.", "Policy for USD 50.01?", APPROVAL),
])
def test_quantity_context_goes_to_semantic_verifier_instead_of_bypassing_it(answer, question, source):
    llm = Mock()
    llm.chat_json.return_value = PolicyVerification(supported=True, evidence=[PolicyEvidence(
        sentence=answer, source_id=source["source_id"], quote=source["content"],
    )])
    draft = PolicyAnswer(answer=answer, source_ids=[source["source_id"]], insufficient_evidence=False)
    verdict = PolicyVerifier().verify(question, draft, [source], llm)
    assert verdict.supported
    llm.chat_json.assert_called_once()
    import json
    payload = json.loads(llm.chat_json.call_args.args[0][1]["content"])
    assert payload["quantity_checks"]
