from pallas_plugin_draw.image_api import (
    cffi_error_is_timeout,
    effective_request_timeout,
    httpx_cap_after_cffi_timeout,
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
