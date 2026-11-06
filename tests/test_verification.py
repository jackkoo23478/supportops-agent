"""Block unsupported policy drafts, require evidence, and bound rewrites."""

import json
from unittest.mock import Mock

import pytest

from scripts.show_trace import read_events
from supportops.llm import LLMError, OllamaClient
from supportops.loop import run_message
from supportops.policy_answers import NO_POLICY_EVIDENCE, PolicyAnswerService
from supportops.schemas import IntentResult, PolicyAnswer
from supportops.tracing import JsonlTracer
from supportops.verification import (
    PolicyEvidence, PolicyVerification, PolicyVerifier, check_policy_draft,
)


RETURN_TEXT = (
    "Unused items in their original packaging may be returned within\n"
    "30 calendar days of delivery. Proof of purchase is required."
)
SOURCE = {"source_id": "returns.eligibility", "document": "returns.md", "content": RETURN_TEXT}
GOOD_TEXT = "Unused items in their original packaging may be returned within 30 calendar days of delivery."


def draft(text=GOOD_TEXT, ids=None):
    return PolicyAnswer(
        answer=text, source_ids=ids if ids is not None else [SOURCE["source_id"]],
        insufficient_evidence=False,
    )


def supported(text=GOOD_TEXT, quote=RETURN_TEXT, source_id="returns.eligibility"):
    return PolicyVerification(supported=True, evidence=[PolicyEvidence(
        sentence=text, quote=quote, source_id=source_id,
    )])


@pytest.fixture
def searcher():
    index = Mock()
    index.search.return_value = [SOURCE.copy()]
    return index


@pytest.fixture
def llm():
    return Mock(spec=OllamaClient)


@pytest.mark.parametrize("text", [
    "Returns are accepted within 60 calendar days of delivery.",
    "Returns are accepted within 30 business days of delivery.",
    "Returns are accepted within 30 months of delivery.",
    "We have issued your refund.",
    "Your refund has been approved.",
    "Order O1001 is eligible for a return.",
    "Returns have a USD 30 fee.",
    "30% of purchases can be returned.",
])
def test_local_checks_reject_observable_contradictions_without_model(text, llm):
    verdict = PolicyVerifier().verify("Return policy?", draft(text), [SOURCE], llm)
    assert not verdict.supported
    assert verdict.issues
    llm.chat_json.assert_not_called()


def test_numeric_and_whitespace_normalization_do_not_invent_contradictions():
    source = {"source_id": "refunds.approval", "content": "Eligible refunds of USD 50.00 or less."}
    assert check_policy_draft(draft("Eligible refunds of USD 50 or less.", [source["source_id"]]), [source]) == []


def test_wrong_number_is_rewritten_using_same_retrieval(searcher, llm):
    llm.chat_json.side_effect = [
        draft("Return within 60 calendar days of delivery."), draft(), supported(),
    ]
    answer = PolicyAnswerService(searcher).answer("Return policy?", llm)
    assert GOOD_TEXT in answer and "60" not in answer
    searcher.search.assert_called_once()
    rewrite = json.loads(llm.chat_json.call_args_list[1].args[0][1]["content"])
    assert rewrite["verification_issues"]
    assert "previous_draft" not in rewrite
    assert rewrite["sources"] == [SOURCE]


@pytest.mark.parametrize("text,issue", [
    ("Unused items can be returned within 30 calendar days of purchase.", "Use delivery, not purchase."),
    ("All items can be returned within 30 calendar days of delivery.", "Preserve unused and packaging conditions."),
])
def test_semantic_verdict_triggers_rewrite_even_when_numbers_match(searcher, llm, text, issue):
    assert check_policy_draft(draft(text), [SOURCE]) == []
    llm.chat_json.side_effect = [
        draft(text), PolicyVerification(
            supported=False, issues=[issue], evidence=supported(text=text).evidence,
        ), draft(), supported(),
    ]
    assert GOOD_TEXT in PolicyAnswerService(searcher).answer("Return policy?", llm)
    rewrite = json.loads(llm.chat_json.call_args_list[2].args[0][1]["content"])
    assert issue in rewrite["verification_issues"]
    verification = json.loads(llm.chat_json.call_args_list[1].args[0][1]["content"])
    assert verification["cited_sources"] == [SOURCE]
    assert verification["sentences"] == [text]


def test_failure_limit_withholds_draft_and_does_not_rerun_tools(searcher, llm, tmp_path):
    invalid = draft("Return within 60 calendar days of delivery.")
    llm.chat_json.side_effect = [IntentResult(intent="policy_question"), invalid, invalid, invalid]
    store = Mock()
    trace = tmp_path / "verify.jsonl"
    reply = run_message(
        "What is the return policy?", llm=llm, store=store,
        policy_answers=PolicyAnswerService(searcher), tracer=JsonlTracer(trace),
    )
    assert reply == NO_POLICY_EVIDENCE
    assert llm.chat_json.call_count == 4
    searcher.search.assert_called_once()
    store.get_order.assert_not_called()
    events = read_events(trace)
    completed = [event for event in events if event["phase"] == "end"]
    assert sum(event["name"] == "policy.verification" for event in completed) == 3
    assert sum(event["name"] == "policy.rewrite" for event in completed) == 2
    assert any(event["name"] == "policy.fallback" for event in completed)
    assert not any(event["name"] in {"tool.create_ticket", "tool.execute_refund"} for event in events)


@pytest.mark.parametrize("verdict", [
    PolicyVerification(supported=True),
    supported(quote="All purchases qualify for returns without conditions."),
    supported(source_id="invented.source"),
    supported(text="A different answer sentence."),
    supported(quote="of"),
    PolicyVerification(supported=True, issues=["Actually unsupported."], evidence=[]),
])
def test_verifier_pass_is_insufficient_without_valid_evidence(llm, verdict):
    llm.chat_json.return_value = verdict
    result = PolicyVerifier().verify("Return policy?", draft(), [SOURCE], llm)
    assert not result.supported
    assert result.issues


def test_every_answer_sentence_requires_evidence(llm):
    text = GOOD_TEXT + " Proof of purchase is required."
    llm.chat_json.return_value = supported()
    result = PolicyVerifier().verify("Return policy?", draft(text), [SOURCE], llm)
    assert not result.supported
    assert any("Every answer sentence" in issue for issue in result.issues)


def test_verbatim_quotes_allow_markdown_line_wrapping(llm):
    llm.chat_json.return_value = supported(quote=" ".join(RETURN_TEXT.split()))
    assert PolicyVerifier().verify("Return policy?", draft(), [SOURCE], llm).supported


def test_uncited_source_cannot_justify_answer(llm):
    extra = {"source_id": "other", "content": "Unused items can be returned."}
    llm.chat_json.return_value = supported(source_id="other", quote=extra["content"])
    verdict = PolicyVerifier().verify("Return policy?", draft(), [SOURCE, extra], llm)
    assert not verdict.supported
    payload = json.loads(llm.chat_json.call_args.args[0][1]["content"])
    assert payload["cited_sources"] == [SOURCE]


def test_changed_approval_boundary_reaches_semantic_verifier(llm):
    source = {
        "source_id": "refunds.approval", "document": "refunds.md",
        "content": (
            "Eligible refunds of USD 50.00 or less may be approved automatically. "
            "Refunds above USD 50.00 require human approval before execution."
        ),
    }
    incorrect = draft("Refunds of USD 50.00 or more require human approval.", [source["source_id"]])
    assert check_policy_draft(incorrect, [source]) == []
    llm.chat_json.return_value = PolicyVerification(
        supported=False, issues=["Exactly USD 50 may be approved automatically."],
        evidence=[PolicyEvidence(
            sentence=incorrect.answer, source_id=source["source_id"], quote=source["content"],
        )],
    )
    verdict = PolicyVerifier().verify("When is human approval needed?", incorrect, [source], llm)
    assert not verdict.supported
    assert "Exactly USD 50" in verdict.issues[0]


def test_missing_citation_can_be_fixed_before_rendering(searcher, llm):
    llm.chat_json.side_effect = [draft(ids=["invented"]), draft(), supported()]
    answer = PolicyAnswerService(searcher).answer("Return policy?", llm)
    assert "Source: returns.eligibility" in answer
    assert "invented" not in answer


def test_verifier_outage_withholds_unverified_draft(searcher, llm):
    llm.chat_json.side_effect = [draft(), LLMError("Verifier timed out")]
    assert PolicyAnswerService(searcher).answer("Return policy?", llm) == NO_POLICY_EVIDENCE
    assert llm.chat_json.call_count == 2


def test_ungrounded_rejection_rechecks_judge_without_rewriting_answer(searcher, llm, tmp_path):
    text = "Returns require original packaging."
    llm.chat_json.side_effect = [
        draft(text),
        PolicyVerification(supported=False, issues=["Proof of purchase is absent from the policy."]),
        supported(text=text),
    ]
    trace = tmp_path / "judge-retry.jsonl"
    with JsonlTracer(trace).trace("focused-question"):
        answer = PolicyAnswerService(searcher).answer("Do returns require original packaging?", llm)
    assert text in answer
    schemas = [call.args[1] for call in llm.chat_json.call_args_list]
    assert schemas == [PolicyAnswer, PolicyVerification, PolicyVerification]
    events = read_events(trace)
    assert any(event["name"] == "policy.judge_retry" for event in events)
    assert not any(event["name"] == "policy.rewrite" for event in events)
    searcher.search.assert_called_once()


@pytest.mark.parametrize("evidence", [
    [],
    [PolicyEvidence(sentence=GOOD_TEXT, source_id="invented", quote=RETURN_TEXT)],
    [PolicyEvidence(sentence=GOOD_TEXT, source_id=SOURCE["source_id"], quote="A fake source quote.")],
])
def test_invalid_rejection_never_becomes_generator_feedback(searcher, llm, evidence):
    invalid = PolicyVerification(supported=False, issues=["An ungrounded criticism."], evidence=evidence)
    llm.chat_json.side_effect = [draft(), invalid, invalid]
    assert PolicyAnswerService(searcher).answer("Return policy?", llm) == NO_POLICY_EVIDENCE
    assert [call.args[1] for call in llm.chat_json.call_args_list] == [
        PolicyAnswer, PolicyVerification, PolicyVerification,
    ]


def test_judge_receives_normalized_source_sentences(llm):
    text = "Proof of purchase is required."
    llm.chat_json.return_value = supported(text=text, quote=text)
    assert PolicyVerifier().verify("Is proof of purchase required?", draft(text), [SOURCE], llm).supported
    payload = json.loads(llm.chat_json.call_args.args[0][1]["content"])
    assert payload["source_sentences"] == [{
        "source_id": SOURCE["source_id"],
        "sentences": [GOOD_TEXT, "Proof of purchase is required."],
    }]


def test_repeated_semantic_failure_is_bounded(searcher, llm):
    results = []
    for _ in range(3):
        results += [draft(), PolicyVerification(
            supported=False, issues=["Unsupported claim."], evidence=supported().evidence,
        )]

    llm.chat_json.side_effect = results
    assert PolicyAnswerService(searcher).answer("Return policy?", llm) == NO_POLICY_EVIDENCE
    assert llm.chat_json.call_count == 6


def test_generator_can_refuse_without_verifier_or_rewrites(searcher, llm):
    refused = draft("An unsupported guess.")
    refused.insufficient_evidence = True
    llm.chat_json.return_value = refused
    assert PolicyAnswerService(searcher).answer("Unknown policy?", llm) == NO_POLICY_EVIDENCE
    llm.chat_json.assert_called_once()


@pytest.mark.parametrize("attempts", [0, 4])
def test_attempt_limit_cannot_disable_bound_or_create_unlimited_loop(attempts):
    with pytest.raises(ValueError):
        PolicyAnswerService(max_attempts=attempts)
