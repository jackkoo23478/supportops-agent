"""Manual check: classify a few customer messages with the local model."""

from typing import Literal

from pydantic import BaseModel

from supportops.config import load_settings
from supportops.llm import make_llm


class Intent(BaseModel):
    intent: Literal["order_status", "refund", "complaint", "other"]
    order_id: str | None = None


MESSAGES = [
    "Where is my order O1003? It has not arrived.",
    "The monitor from order O1005 is broken, I want my money back.",
    "Your delivery driver was very rude to me.",
]

llm = make_llm(load_settings())
for text in MESSAGES:
    result = llm.chat_json(
        [
            {
                "role": "system",
                "content": "Classify the customer message and extract the order id if any.",
            },
            {"role": "user", "content": text},
        ],
        Intent,
    )
    print(f"{text}\n  -> {result}\n")
