"""
FastAPI / ASGI middleware — scan request bodies before they reach your handler.

Modes (on_pii_detected):
    "block"    → reject requests containing PII with HTTP 422
    "sanitize" → replace PII in the body before it reaches the route
    "warn"     → pass through unchanged; findings on request.state.ai_guard_result

Run:
    pip install fastapi uvicorn
    uvicorn examples.fastapi_middleware:app --reload
    curl -s -X POST localhost:8000/echo -H 'content-type: application/json' \
         -d '{"text":"my card is 4111 1111 1111 1111"}'
"""

from fastapi import FastAPI, Request

from ai_guard import LLMGuard
from ai_guard.integrations.fastapi import AIGuardMiddleware

guard = (
    LLMGuard(use_ner=False, salt="example-salt")
    .configure_entity("EMAIL", enabled=True, action="warn")
    .configure_entity("CREDIT_CARD", enabled=True, action="hash")
)

app = FastAPI()
app.add_middleware(AIGuardMiddleware, guard=guard, on_pii_detected="sanitize")


@app.post("/echo")
async def echo(request: Request) -> dict:
    body = await request.body()
    result = getattr(request.state, "ai_guard_result", None)
    return {
        "received": body.decode("utf-8", errors="replace"),
        "had_pii": bool(result and not result.is_clean),
    }
