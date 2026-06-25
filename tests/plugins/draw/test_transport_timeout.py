from pallas_plugin_draw.image_api import (
    cffi_error_is_timeout,
    effective_request_timeout,
    httpx_cap_after_cffi_timeout,
    request_timeout_for_backend_attempt,
    request_timeout_for_deadline,
)


def test_cffi_error_is_timeout_curl_28() -> None:
    err = "Failed to perform, curl: (28) Operation timed out after 180000 milliseconds"
    assert cffi_error_is_timeout(RuntimeError(err))


def test_effective_request_timeout_capped_by_remaining() -> None:
    assert effective_request_timeout(30.0) == 30.0
    assert request_timeout_for_deadline(9999.0) == effective_request_timeout(None)


def test_httpx_cap_after_cffi_timeout_when_budget_enough() -> None:
    cap = httpx_cap_after_cffi_timeout(180.0)
    assert cap is not None
    assert cap >= 45.0


def test_httpx_cap_after_cffi_timeout_when_budget_too_small() -> None:
    assert httpx_cap_after_cffi_timeout(40.0) is None


def test_request_timeout_for_backend_attempt_single_slot_uses_full_budget() -> None:
    assert request_timeout_for_backend_attempt(
        240.0,
        backends_remaining=1,
        param_attempts_remaining=1,
    ) == effective_request_timeout(240.0)


def test_request_timeout_for_backend_attempt_primary_uses_full_budget() -> None:
    assert request_timeout_for_backend_attempt(
        480.0,
        backends_remaining=4,
        param_attempts_remaining=2,
        primary_backend=True,
    ) == effective_request_timeout(480.0)


def test_request_timeout_for_backend_attempt_fallback_splits_budget() -> None:
    cap = request_timeout_for_backend_attempt(
        240.0,
        backends_remaining=3,
        param_attempts_remaining=2,
        primary_backend=False,
    )
    assert cap == 40.0


def test_request_timeout_for_backend_attempt_fallback_respects_hard_cap(monkeypatch) -> None:
    from pallas_plugin_draw.config import image_gen_config

    monkeypatch.setattr(image_gen_config._c, "pallas_image_backend_attempt_timeout", 45.0)
    cap = request_timeout_for_backend_attempt(
        240.0,
        backends_remaining=2,
        param_attempts_remaining=1,
        primary_backend=False,
    )
    assert cap == 45.0
