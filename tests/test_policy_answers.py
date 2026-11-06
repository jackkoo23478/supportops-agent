"""Policy-answer orchestration and citation checks without live models."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from supportops import cli
from supportops.llm import LLMError, OllamaClient
from supportops.loop import run_message
from supportops.policy_answers import NO_POLICY_EVIDENCE, PolicyAnswerService, PolicySearchError
from supportops.schemas import IntentResult, PolicyAnswer
from supportops.verification import PolicyEvidence, PolicyVerification


def supported_verdict(draft):
    return PolicyVerification(supported=True, evidence=[PolicyEvidence(
        sentence=draft.answer, source_id="returns.eligibility",
        quote="Unused items in original packaging may be returned within 30 calendar days.",
    )])


@pytest.fixture
def searcher():
    index = Mock()
    index.search.return_value = [{
        "source_id": "returns.eligibility", "document": "returns.md",
        "content": "Unused items in original packaging may be returned within 30 calendar days.",
        "bm25_score": 2.0, "semantic_score": 0.3, "rrf_score": 0.03, "rerank_score": 2.0,
    }]
    return index


@pytest.fixture
def llm():
    model = Mock(spec=OllamaClient)
    model.chat_json.return_value = PolicyAnswer(
        answer="Unused items must be in their original packaging.",
        source_ids=["returns.eligibility"], insufficient_evidence=False,
    )
    def respond(messages, schema):
        if schema is PolicyVerification:
            return supported_verdict(model.chat_json.return_value)
        return model.chat_json.return_value

    model.chat_json.side_effect = respond
    return model


def test_answer_uses_only_retrieved_sources_and_renders_citation(searcher, llm):
    answer = PolicyAnswerService(searcher).answer("Do I need original packaging?", llm)
    assert "Unused items must be in their original packaging." in answer
    assert "Source: returns.eligibility (returns.md)" in answer
    messages, schema = llm.chat_json.call_args_list[0].args
    assert schema is PolicyAnswer
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == "Do I need original packaging?"
    assert payload["sources"] == [{
        key: searcher.search.return_value[0][key]
        for key in ("source_id", "document", "content")
    }]
    assert "semantic_score" not in payload["sources"][0]
    assert searcher.search.call_args.kwargs["top_k"] == 3
    assert searcher.search.call_args.kwargs["candidate_k"] == 5


def test_no_search_results_skip_answer_generation(searcher, llm):
    searcher.search.return_value = []
    assert PolicyAnswerService(searcher).answer("weather", llm) == NO_POLICY_EVIDENCE
    llm.chat_json.assert_not_called()


def test_blank_question_skips_search_and_generation(searcher, llm):
    assert PolicyAnswerService(searcher).answer("  ", llm) == NO_POLICY_EVIDENCE
    searcher.search.assert_not_called()
    llm.chat_json.assert_not_called()


@pytest.mark.parametrize("ids", [[], ["invented.rule"], ["returns.eligibility", "invented.rule"]])
def test_missing_or_invented_citations_refuse_the_entire_answer(searcher, llm, ids):
    llm.chat_json.return_value = PolicyAnswer(
        answer="An unverified claim.", source_ids=ids, insufficient_evidence=False,
    )
    assert PolicyAnswerService(searcher).answer("return rules", llm) == NO_POLICY_EVIDENCE


def test_model_refusal_does_not_show_its_draft_or_citations(searcher, llm):
    llm.chat_json.return_value = PolicyAnswer(
        answer="An unsupported guess.", source_ids=["returns.eligibility"],
        insufficient_evidence=True,
    )
    assert PolicyAnswerService(searcher).answer("weather", llm) == NO_POLICY_EVIDENCE


def test_duplicate_source_ids_are_displayed_once(searcher, llm):
    llm.chat_json.return_value.source_ids *= 2
    answer = PolicyAnswerService(searcher).answer("packaging", llm)
    assert answer.count("Source:") == 1


def test_policy_searcher_is_created_lazily_and_reused(monkeypatch, searcher, llm):
    from supportops import policies, retrieval

    loader = Mock(return_value=[])
    factory = Mock(return_value=searcher)
    monkeypatch.setattr(policies, "load_policy_chunks", loader)
    monkeypatch.setattr(retrieval, "HybridPolicySearcher", factory)
    service = PolicyAnswerService()
    factory.assert_not_called()
    service.answer("packaging", llm)
    service.answer("returns", llm)
    loader.assert_called_once()
    factory.assert_called_once_with([])
    assert searcher.search.call_count == 2


def test_search_error_is_distinct_from_model_failure(searcher, llm):
    searcher.search.side_effect = RuntimeError("Cannot load retrieval model")
    with pytest.raises(PolicySearchError, match="Cannot load retrieval model"):
        PolicyAnswerService(searcher).answer("packaging", llm)
    llm.chat_json.assert_not_called()


def test_answer_model_failure_propagates(searcher, llm):
    llm.chat_json.side_effect = LLMError("offline")
    with pytest.raises(LLMError, match="offline"):
        PolicyAnswerService(searcher).answer("packaging", llm)


def test_answer_schema_rejects_whitespace_only_answer():
    with pytest.raises(ValidationError):
        PolicyAnswer(answer="  ", source_ids=[], insufficient_evidence=False)


def test_policy_intent_routes_to_policy_service_without_order_lookup(llm):
    llm.chat_json.return_value = IntentResult(intent="policy_question")
    service = Mock(spec=PolicyAnswerService)
    service.answer.return_value = "A policy answer."
    store = Mock()
    assert run_message(
        "What is the return policy?", llm=llm, store=store, policy_answers=service
    ) == "A policy answer."
    service.answer.assert_called_once_with("What is the return policy?", llm)
    store.get_order.assert_not_called()


def test_unrelated_intent_does_not_search_policies(store, llm):
    llm.chat_json.return_value = IntentResult(intent="other")
    service = Mock(spec=PolicyAnswerService)
    run_message("What is the weather?", llm=llm, store=store, policy_answers=service)
    service.answer.assert_not_called()


def test_cli_integrates_policy_answers_and_preserves_order_context(
    monkeypatch, capsys, store, searcher, llm,
):
    draft = llm.chat_json.return_value
    llm.chat_json.side_effect = [
        IntentResult(intent="order_status", order_id="O1003"),
        IntentResult(intent="policy_question"),
        draft,
        supported_verdict(draft),
        IntentResult(intent="order_status", use_previous_order=True),
    ]
    service = PolicyAnswerService(searcher)
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: store)
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    monkeypatch.setattr(cli, "PolicyAnswerService", lambda: service)
    inputs = iter([
        "Where is order O1003?", "Do I need original packaging?",
        "When was it ordered?", "exit",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert output.count("Order O1003 status:") == 2
    assert "Source: returns.eligibility (returns.md)" in output
    searcher.search.assert_called_once()


def test_cli_single_policy_question(monkeypatch, capsys, store, searcher, llm):
    draft = llm.chat_json.return_value
    llm.chat_json.side_effect = [
        IntentResult(intent="policy_question"), draft, supported_verdict(draft),
    ]
    monkeypatch.setattr(cli, "load_settings", lambda: object())
    monkeypatch.setattr(cli, "Store", lambda: store)
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    monkeypatch.setattr(cli, "PolicyAnswerService", lambda: PolicyAnswerService(searcher))
    assert cli.main(["Do I need original packaging?"]) == 0
    assert "Source: returns.eligibility" in capsys.readouterr().out
