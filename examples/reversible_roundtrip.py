"""
Reversible masking — the LLM round trip, regex-only (no external services).

Mask on the way out with Action.TOKENIZE, send the placeholder text to a model,
then put the real values back into its answer. The "model" here is a stub so the
example runs offline; swap `fake_llm` for a real call and nothing else changes.
"""

import re

from wardcat import Action, Entity, Wardcat

PROMPT = (
    "Ali Veli wrote from ali@example.com about the charge on "
    "4532 0151 1283 0366. Draft a reply to ali@example.com."
)


def fake_llm(prompt: str) -> str:
    """Stand-in for the model: it only ever sees the placeholders.

    It copies them out of the prompt rather than hard-coding them, because each
    scan stamps its own context id into its tokens — that is what stops one
    request's answer from being restored against another request's result.
    """
    assert "ali@example.com" not in prompt, "the model must never see the real value"
    person, card, email = (
        re.search(rf"\[{t}_\d+_[0-9a-f]+\]", prompt) for t in ("PERSON", "CREDIT_CARD", "EMAIL")
    )
    return (
        f"Hi {person.group()}, I have checked the charge on {card.group()} "
        f"and emailed the details to {email.group()}."
    )


def main() -> None:
    guard = Wardcat(salt="example-salt").add_entities(
        [Entity.EMAIL, Entity.CREDIT_CARD, Entity.PERSON], action=Action.TOKENIZE
    )
    # PERSON needs the NER layer (wardcat[ner]); the denylist keeps this example
    # regex-only so it runs with no model downloads.
    guard.add_denylist([{"value": "Ali Veli", "entity_type": "PERSON"}])

    result = guard.scan(PROMPT)
    print("== sent to the model ==")
    print(f"  {result.sanitized_text}\n")

    answer = fake_llm(result.sanitized_text)
    print("== raw answer ==")
    print(f"  {answer}\n")

    print("== restored, with its sources ==")
    print(result.restore(answer))

    restored = result.restore(answer)
    print(f"\n(complete={restored.is_complete}, {len(restored.substitutions)} value(s) put back)")
    print(f"(this scan's context id: {result.context_id} — another scan's tokens would be refused)")


if __name__ == "__main__":
    main()
