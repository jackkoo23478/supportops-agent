"""Demonstrate refund decisions with verified, fictional facts; no payments."""

from datetime import date
from decimal import Decimal

from supportops.refunds import RefundFacts, evaluate_refund


def main() -> None:
    facts = RefundFacts(
        delivered_date=date(2026, 10, 20), item_type="standard", unused=True,
        original_packaging=True, proof_of_purchase=True, damaged_or_defective=False,
        return_received=True, inspection_passed=True,
    )
    for value in ("30.00", "50.00", "80.00"):
        decision = evaluate_refund(
            facts, amount=Decimal(value), refundable_balance=Decimal("100.00"),
            requested_on=date(2026, 11, 5),
        )
        print(f"USD {value}: {decision.model_dump_json()}")


if __name__ == "__main__":
    main()
