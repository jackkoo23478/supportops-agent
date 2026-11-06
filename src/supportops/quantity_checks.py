"""Check policy quantities while preserving their roles, bounds, and units."""

import re
from decimal import Decimal


NUMBER = r"\d[\d,]*(?:\.\d+)?"
USD = re.compile(rf"\bUSD\s*(?P<amount>{NUMBER})(?![\w.])", re.IGNORECASE)
DURATION = re.compile(
    rf"(?P<start>{NUMBER})(?:\s*(?:to|[-–])\s*(?P<end>{NUMBER}))?"
    r"(?:\s*[-–]\s*|\s+)(?:(?P<qualifier>calendar|business|working)(?:\s*[-–]\s*|\s+))?"
    r"(?P<unit>days?|months?|years?)\b", re.IGNORECASE,
)
NO_HUMAN = (
    r"\b(?:does not|do not|doesn't|don't|not|never)\s+(?:require|need)\s+"
    r"(?:human|manual)\s+approval\b|\b(?:human|manual)\s+approval\s+is\s+not\s+"
    r"(?:required|needed)\b"
)
HUMAN = (
    r"\b(?:requires?|needs?)\s+(?:human|manual)\s+approval\b|"
    r"\b(?:human|manual)\s+approval\s+is\s+(?:required|needed)\b|"
    r"\bmust\s+(?:be\s+)?approved\s+by\s+(?:a\s+)?human\b"
)
AUTOMATIC = (
    r"\b(?:can|may|will|is|are)\s+(?:be\s+)?approved\s+automatically\b|"
    r"\b(?:is|are)\s+eligible\s+for\s+automatic\s+approval\b|"
    r"\bqualif(?:y|ies)\s+for\s+automatic\s+approval\b"
)


def _decimal(value: str) -> Decimal:
    return Decimal(value.replace(",", ""))


def _numbers(text: str) -> set[Decimal]:
    return {_decimal(value) for value in re.findall(NUMBER, text)}


def _sentence_at(text: str, position: int) -> str:
    left = 0
    for separator in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if separator.start() > position:
            return text[left:separator.start()]
        left = separator.end()
    return text[left:]


def _approval_rules(sources: list[dict]) -> list[dict]:
    patterns = {
        "manual_required": rf"\brefunds\s+above\s+USD\s*({NUMBER})\s+require\s+human\s+approval\b",
        "automatic_allowed": (
            rf"\beligible\s+refunds\s+of\s+USD\s*({NUMBER})\s+or\s+less\s+"
            r"may\s+be\s+approved\s+automatically\b"
        ),
    }
    rules = []
    for source in sources:
        content = " ".join(source["content"].split())
        for kind, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                rules.append({"kind": kind, "threshold": _decimal(match.group(1)),
                              "source_id": source["source_id"], "quote": match.group(0)})
    return rules


def _approval_claim(sentence: str) -> str | None:
    no_human = bool(re.search(NO_HUMAN, sentence, re.IGNORECASE))
    without_negatives = re.sub(NO_HUMAN, "", sentence, flags=re.IGNORECASE)
    human = bool(re.search(HUMAN, without_negatives, re.IGNORECASE))
    automatic = bool(re.search(AUTOMATIC, sentence, re.IGNORECASE))
    if human and (no_human or automatic):
        return None
    if human:
        return "manual_required"
    if no_human or automatic:
        return "automatic_allowed"
    return None


def _derived_amount(value: Decimal, sentence: str, question: str, rules: list[dict]) -> dict | None:
    question_amounts = {_decimal(match.group("amount")) for match in USD.finditer(question)}
    if value not in question_amounts or value <= 0 or value.normalize().as_tuple().exponent < -2:
        return None
    # Keep this parser deliberately narrow: one subject amount per statement.
    if len(list(USD.finditer(sentence))) != 1:
        return None
    if not re.search(r"\brefunds?\b", sentence, re.IGNORECASE) or re.search(
        r"\bfees?\b", sentence, re.IGNORECASE,
    ):
        return None
    if re.search(
        r"\b(?:above|over|below|under|at least|at most|up to|greater than|more than|"
        r"less than|exceeding)\s+USD(?=\s|\d)|(?:>=|<=|>|<|=)\s*USD(?=\s|\d)|"
        rf"\bUSD\s*{NUMBER}\s+(?:or more|or less|and above|and below)\b",
        sentence, re.IGNORECASE,
    ):
        return None  # A new policy boundary is not an application to a user amount.
    applicable = [rule for rule in rules if (
        value > rule["threshold"] if rule["kind"] == "manual_required"
        else value <= rule["threshold"]
    )]
    if not applicable or len({rule["kind"] for rule in applicable}) != 1:
        return None
    claim = _approval_claim(sentence)
    if claim != applicable[0]["kind"]:
        return None
    rule = applicable[0]
    return {
        "kind": "approval_threshold_application", "currency": "USD", "amount": str(value),
        "threshold": str(rule["threshold"]), "comparison": ">" if claim == "manual_required" else "<=",
        "approval": claim, "source_id": rule["source_id"], "policy_quote": rule["quote"],
        "note": "This verifies the amount comparison only, not refund eligibility or execution.",
    }


def _durations(text: str) -> list[tuple[Decimal, Decimal | None, str, str]]:
    return [(
        _decimal(match.group("start")),
        _decimal(match.group("end")) if match.group("end") else None,
        (match.group("qualifier") or "").lower().replace("working", "business"),
        match.group("unit").lower().rstrip("s"),
    ) for match in DURATION.finditer(text)]


def check_quantities(answer: str, sources: list[dict], question: str = "") -> tuple[list[str], list[dict]]:
    """Return contradictions and source-grounded numeric context for the judge.

    Unlisted numbers remain blocked except narrow, correct applications of a
    quoted USD approval rule to an amount explicitly present in the question.
    Missing day qualifiers inherit only an unambiguous matching cited duration;
    explicit qualifiers and complete range boundaries must still agree.
    """
    content = "\n".join(source["content"] for source in sources)
    known = _numbers(content)
    rules = _approval_rules(sources)
    issues = []
    context = []
    numeric_text = answer
    replacements = []
    for match in USD.finditer(answer):
        value = _decimal(match.group("amount"))
        if value in known:
            continue
        sentence = _sentence_at(answer, match.start())
        derived = _derived_amount(value, sentence, question, rules)
        if derived is None:
            issues.append("An unlisted USD amount needs a correct, source-backed approval comparison.")
        else:
            context.append(derived)
            replacements.append(match.span("amount"))
    for start, end in reversed(replacements):
        numeric_text = numeric_text[:start] + "amount" + numeric_text[end:]
    if not _numbers(numeric_text) <= known:
        issues.append("The answer contains numbers unsupported by the cited policy or a verified comparison.")

    source_durations = _durations(content)
    for start, end, qualifier, unit in _durations(answer):
        candidates = {item[2] for item in source_durations
                      if item[0] == start and item[1] == end and item[3] == unit}
        if qualifier:
            accepted = qualifier in candidates
        else:
            accepted = len(candidates) == 1
            if accepted and next(iter(candidates)):
                context.append({
                    "kind": "duration_qualifier_from_source", "start": str(start),
                    "end": str(end) if end is not None else None, "unit": unit,
                    "interpreted_qualifier": next(iter(candidates)),
                    "note": "The answer omitted the qualifier; it did not state a conflicting unit.",
                })
        if not accepted:
            issues.append("Preserve duration bounds and explicit calendar/business units; clarify ambiguous units.")
    return list(dict.fromkeys(issues)), context
