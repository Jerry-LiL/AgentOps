"""Timeouts and retries for one tool call."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class RetryLimit(Exception):
    """The call timed out on every attempt."""

    def __init__(self, reason: str, attempts: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts


async def call_with_retry(
    fn: Callable[[], Awaitable],
    *,
    timeout: float,
    retries: int,
    backoff: float,
    on_retry: Callable[[int, str], Awaitable[None]] | None = None,
):
    """Call `fn` until it returns a non-transient result, or the attempt budget runs out.

    `retries` is the number of extra attempts after the first. A timeout is always
    retried. A result is retried only when `result.transient` is set. Other failures
    return immediately so a bad argument is not hammered.
    """
    max_attempts = max(retries, 0) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            result = await asyncio.wait_for(fn(), timeout)
        except TimeoutError:
            if attempt == max_attempts:
                raise RetryLimit("timed out", attempt) from None
            reason = "timeout"
        else:
            if result.ok or not result.transient:
                return result, attempt
            if attempt == max_attempts:
                return result, attempt
            reason = result.error or "transient error"
        if on_retry is not None:
            await on_retry(attempt, reason)
        await asyncio.sleep(backoff * (2 ** (attempt - 1)))
    raise RetryLimit("retry budget exhausted", max_attempts)
