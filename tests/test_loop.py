from unittest.mock import Mock

import pytest

from supportops.llm import LLMError, OllamaClient
from supportops.loop import run_message
from supportops.schemas import IntentResult
from supportops.store import Store

def make_llm_result(
    intent: str,
    order_id: str | None = None,
) -> Mock:
    llm = Mock(spec=OllamaClient)
    llm.chat_json.return_value = IntentResult(
        intent=intent,
        order_id=order_id,
    )
    return llm


def test_order_status_uses_store_data(store):
    llm = make_llm_result("order_status", "O1001")
    order = store.get_order("O1001")
    assert order is not None

    reply = run_message(
        "Where is my order O1001?",
        llm=llm,
        store=store,
    )

    assert "Order O1001 status: Delivered." in reply
    assert f"Order date: {order.order_date}." in reply
    assert f"Delivery date: {order.delivered_date}." in reply


def test_order_id_is_normalized(store):
    llm = make_llm_result("order_status", "  o1001 ")

    reply = run_message(
        "Where is my order o1001?",
        llm=llm,
        store=store,
    )

    assert "Order O1001 status:" in reply


def test_missing_order_id_asks_customer(store):
    llm = make_llm_result("order_status")

    reply = run_message(
        "Where is my order?",
        llm=llm,
        store=store,
    )

    assert reply == "Please provide an order ID, such as O1003."


def test_unknown_order_returns_clear_reply(store):
    llm = make_llm_result("order_status", "O9999")

    reply = run_message(
        "Where is my order O9999?",
        llm=llm,
        store=store,
    )

    assert reply == (
        "Order O9999 was not found. Please check the order ID."
    )


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("refund", "No refund has been issued."),
        ("complaint", "No ticket has been created."),
        ("other", "I can currently help with order status inquiries."),
    ],
)
def test_other_intents_do_not_lookup_order(intent, expected):
    llm = make_llm_result(intent, "O1001")
    store = Mock(spec=Store)

    reply = run_message(
        "A customer message",
        llm=llm,
        store=store,
    )

    assert expected in reply
    store.get_order.assert_not_called()


def test_empty_message_does_not_call_model(store):
    llm = Mock(spec=OllamaClient)

    reply = run_message(
        "   ",
        llm=llm,
        store=store,
    )

    assert reply == "Please enter a message."
    llm.chat_json.assert_not_called()


def test_model_failure_is_propagated(store):
    llm = Mock(spec=OllamaClient)
    llm.chat_json.side_effect = LLMError("Ollama unavailable")

    with pytest.raises(LLMError, match="Ollama unavailable"):
        run_message(
            "Where is my order O1001?",
            llm=llm,
            store=store,
        )