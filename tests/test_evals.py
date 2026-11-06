"""Evaluation fixtures and graders must detect errors without changing the agent."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from evals.run import load_suites, run_case, step_checks, summarize
from evals.schemas import EvalSuite, ReferenceFact
from supportops.llm import OllamaClient
from supportops.schemas import ConversationState, IntentResult


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "evals" / "cases" / "smoke.json"
EXTENDED = ROOT / "evals" / "cases" / "extended.json"
VERIFIER_QUESTIONS = ROOT / "evals" / "cases" / "verifier_questions.json"
DATA = ROOT / "data" / "sample"


def test_combined_suites_keep_twenty_unique_cases_and_separate_groups():
    suite, groups, manifest = load_suites([SMOKE, EXTENDED])
    assert len(suite.cases) == 20
    assert len({case.id for case in suite.cases}) == 20
    assert list(groups.values()).count("smoke") == 10
    assert list(groups.values()).count("extended") == 10
    assert len(manifest) == 2
    assert suite.requested_on == date(2026, 11, 6)


def test_combining_a_suite_with_itself_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="unique"):
        load_suites([SMOKE, SMOKE])


def test_three_suites_preserve_baseline_and_add_eight_policy_questions():
    suite, groups, manifest = load_suites([SMOKE, EXTENDED, VERIFIER_QUESTIONS])
    assert len(suite.cases) == 28
    assert list(groups.values()).count("verifier_questions") == 8
    assert len(manifest) == 3
    assert all(step.expect.intent == "policy_question" for case in suite.cases[-8:]
               for step in case.steps)


@pytest.mark.parametrize("case_id,correct,incorrect", [
    ("verify_packaging_positive", "Original packaging is required for a return.",
     "Original packaging is not required."),
    ("verify_packaging_negative", "No. Unused items must be in their original packaging.",
     "Unused items can be returned without original packaging."),
    ("verify_purchase_proof_optional", "Proof of purchase is required for returns.",
     "Proof of purchase is optional for returns."),
    ("verify_deadline_purchase_premise", "No. The return window starts from delivery.",
     "Yes. The return window starts from the purchase date."),
    ("verify_fifty_approval_policy", "Eligible USD 50 refunds may be approved automatically.",
     "Eligible USD 50 refunds require human approval."),
    ("verify_fifty_one_cent_approval_policy",
     "Yes. Refunds above USD 50 require human approval; eligible USD 50 refunds may be approved automatically.",
     "No. Human approval is not required for USD 50.01."),
    ("verify_processing_start_event", "No. Processing starts after the item is received and inspected.",
     "Yes. Processing starts when you submit the request."),
    ("verify_alternative_refund_account", "No. Refunds are issued to the original payment method.",
     "Yes. Refunds can be paid into a different bank account."),
])
def test_new_policy_reference_facts_distinguish_correct_and_incorrect_claims(
    store, case_id, correct, incorrect,
):
    suite = EvalSuite.model_validate_json(VERIFIER_QUESTIONS.read_text(encoding="utf-8"))
    expect = next(case for case in suite.cases if case.id == case_id).steps[0].expect
    for reply, expected in [(correct, True), (incorrect, False)]:
        checks, _ = step_checks(
            expect, reply, ConversationState(), {"tickets": [], "refunds": [], "transactions": []},
            [], "C001", store,
        )
        fact = next(check for check in checks if check["category"] == "fact")
        assert fact["passed"] is expected, (case_id, reply, checks)


@pytest.mark.parametrize("pattern", ["", " ", "["])
def test_invalid_fact_patterns_fail_before_evaluation(pattern):
    with pytest.raises(ValidationError):
        ReferenceFact(id="bad", description="Invalid pattern", any_patterns=[pattern])


@pytest.mark.parametrize("case_id", [
    "refund_exactly_fifty", "refund_fifty_and_one_cent", "refund_expired_day_thirty_one",
    "refund_human_rejection", "refund_change_order_resets_amount",
])
def test_refund_fixtures_check_real_persisted_results_without_model_variance(tmp_path, case_id):
    suite = EvalSuite.model_validate_json(EXTENDED.read_text(encoding="utf-8"))
    case = next(case for case in suite.cases if case.id == case_id)
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(intent="refund")
    result = run_case(
        case, requested_on=suite.requested_on, llm=llm, policy_answers=Mock(),
        data_dir=DATA, directory=tmp_path / case.id, group="extended",
    )
    assert result["passed"], result
    assert result["group"] == "extended"
    assert result["requested_on"] == (case.requested_on or suite.requested_on).isoformat()
    assert summarize([result])["cases"]["rate"] == 1


@pytest.mark.parametrize("reply,expected", [
    ("No, standard returns require original packaging.", True),
    ("Original packaging is required for returns.", True),
    ("No, original packaging is not required.", False),
    ("You can return items without original packaging.", False),
    ("I do not have enough policy information to answer that question.", False),
    ("Source: returns.eligibility (original packaging is required)", False),
])
def test_packaging_fact_grader_checks_meaning_markers_and_ignores_citation_labels(store, reply, expected):
    suite = EvalSuite.model_validate_json(EXTENDED.read_text(encoding="utf-8"))
    expect = next(case for case in suite.cases if case.id == "policy_without_packaging").steps[0].expect
    checks, _ = step_checks(
        expect, reply, ConversationState(), {"tickets": [], "refunds": [], "transactions": []},
        [], "C001", store,
    )
    fact = next(check for check in checks if check["category"] == "fact")
    assert fact["passed"] is expected


def test_missing_state_path_is_not_equivalent_to_a_null_value(store):
    suite = EvalSuite.model_validate_json(EXTENDED.read_text(encoding="utf-8"))
    expect = next(case for case in suite.cases if case.id == "refund_change_order_resets_amount").steps[1].expect
    checks, _ = step_checks(
        expect, "", ConversationState(), {"tickets": [], "refunds": [], "transactions": []},
        [], "C001", store,
    )
    amount_check = next(check for check in checks if check["name"] == "state:pending_refund.amount")
    assert not amount_check["passed"]


def test_deadline_reference_does_not_misread_purchase_proof_as_starting_event(store):
    suite = EvalSuite.model_validate_json(EXTENDED.read_text(encoding="utf-8"))
    expect = next(case for case in suite.cases if case.id == "policy_return_deadline").steps[0].expect
    checks, _ = step_checks(
        expect, "Return within 30 calendar days of delivery. Proof of purchase is required.",
        ConversationState(), {"tickets": [], "refunds": [], "transactions": []}, [], "C001", store,
    )
    assert all(check["passed"] for check in checks if check["category"] == "fact")
