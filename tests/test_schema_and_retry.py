import asyncio

import pytest

from agentops.retry import RetryLimit, call_with_retry
from agentops.schema_check import SchemaError, validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {
        "sheet": {"type": "string"},
        "values": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sheet", "values"],
}


def test_schema_accepts_a_matching_object():
    validate_arguments(SCHEMA, {"sheet": "budget", "values": ["Q4", "180", "100"]})


def test_schema_rejects_missing_and_mistyped_fields():
    with pytest.raises(SchemaError):
        validate_arguments(SCHEMA, {"sheet": "budget"})
    with pytest.raises(SchemaError):
        validate_arguments(SCHEMA, {"sheet": "budget", "values": "Q4"})
    with pytest.raises(SchemaError):
        validate_arguments(SCHEMA, ["not", "an", "object"])


class _Result:
    def __init__(self, ok: bool, transient: bool = False, error: str | None = None) -> None:
        self.ok = ok
        self.transient = transient
        self.error = error
        self.output = {"ok": ok}


def test_retry_returns_the_first_success():
    async def scenario():
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            return _Result(True)

        result, attempts = await call_with_retry(fn, timeout=1, retries=2, backoff=0, on_retry=None)
        assert result.ok
        assert attempts == 1
        assert calls["n"] == 1

    asyncio.run(scenario())


def test_retry_repeats_a_transient_failure():
    async def scenario():
        calls = {"n": 0}
        reasons = []

        async def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                return _Result(False, transient=True, error="transient spreadsheet backend error")
            return _Result(True)

        async def on_retry(attempt, reason):
            reasons.append((attempt, reason))

        result, attempts = await call_with_retry(
            fn, timeout=1, retries=2, backoff=0, on_retry=on_retry
        )
        assert result.ok
        assert attempts == 2
        assert reasons == [(1, "transient spreadsheet backend error")]

    asyncio.run(scenario())


def test_retry_does_not_repeat_a_normal_failure():
    async def scenario():
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            return _Result(False, transient=False, error="invalid range")

        result, attempts = await call_with_retry(fn, timeout=1, retries=3, backoff=0, on_retry=None)
        assert not result.ok
        assert attempts == 1
        assert calls["n"] == 1

    asyncio.run(scenario())


def test_timeout_is_retried_then_raised():
    async def scenario():
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.sleep(0.2)
                return _Result(True)
            return _Result(True)

        result, attempts = await call_with_retry(
            fn, timeout=0.05, retries=1, backoff=0, on_retry=None
        )
        assert result.ok
        assert attempts == 2

        async def always_slow():
            await asyncio.sleep(0.2)
            return _Result(True)

        with pytest.raises(RetryLimit) as raised:
            await call_with_retry(always_slow, timeout=0.05, retries=1, backoff=0, on_retry=None)
        assert raised.value.attempts == 2

    asyncio.run(scenario())
