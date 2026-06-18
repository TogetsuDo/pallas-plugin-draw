from pallas_plugin_draw.image_request_options import (
    ImageGenRequestOptions,
    capped_param_attempts,
    image_gen_fast_attempts,
    image_gen_slow_attempts,
    response_format_attempts,
)


def test_response_format_attempts_b64_json() -> None:
    assert response_format_attempts("b64_json") == ["b64_json", "url"]


def test_capped_param_attempts_starts_with_config() -> None:
    attempts = capped_param_attempts(with_ref_urls=False)
    assert attempts
    assert attempts[0] == ImageGenRequestOptions.from_config()


def test_fast_attempts_smaller_than_capped() -> None:
    fast = image_gen_fast_attempts(with_ref_urls=False)
    capped = capped_param_attempts(with_ref_urls=False)
    assert len(fast) <= len(capped)


def test_capped_param_attempts_includes_alt_response_format() -> None:
    attempts = capped_param_attempts(with_ref_urls=False)
    formats = {a.response_format for a in attempts if a.response_format}
    assert "b64_json" in formats
    assert "url" in formats


def test_capped_param_attempts_omit_response_format() -> None:
    attempts = capped_param_attempts(with_ref_urls=False, omit_response_format=True)
    assert attempts
    assert all(not a.response_format for a in attempts)


def test_capped_param_attempts_ref_image_variant() -> None:
    attempts = capped_param_attempts(with_ref_urls=True)
    assert any(not a.include_ref_images for a in attempts)


def test_capped_without_slow_fallback_is_fast_only(monkeypatch) -> None:
    from pallas_plugin_draw import config as cfg_mod

    cfg = cfg_mod.get_draw_config()
    cfg_mod.image_gen_config.reload(cfg.model_copy(update={"pallas_image_slow_param_fallback": False}))
    try:
        fast = image_gen_fast_attempts(with_ref_urls=False)
        capped = capped_param_attempts(with_ref_urls=False)
        assert capped == fast or len(capped) <= len(fast)
    finally:
        cfg_mod.image_gen_config.reload(cfg)


def test_slow_attempts_not_in_fast_only() -> None:
    fast = image_gen_fast_attempts(with_ref_urls=False)
    slow = image_gen_slow_attempts(with_ref_urls=False)
    assert slow
    fast_keys = {(o.size, o.aspect_ratio, o.quality, o.response_format, o.include_ref_images) for o in fast}
    assert any(
        (o.size, o.aspect_ratio, o.quality, o.response_format, o.include_ref_images) not in fast_keys for o in slow
    )
