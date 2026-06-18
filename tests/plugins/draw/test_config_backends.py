from pallas_plugin_draw.config import Config, ImageGenSettings, migrate_legacy_gateway_config


def test_migrate_legacy_gateway_promotes_first_backend() -> None:
    cfg = Config(
        pallas_image_base_url="",
        pallas_image_api_key="",
        pallas_image_model="gpt-image-2",
        pallas_image_api_backends=[
            {
                "base_url": "https://router.example/api/",
                "api_key": "sk-first",
                "model": "openai/gpt-image-2",
                "name": "主站",
            },
            {"base_url": "https://b.example/", "api_key": "sk-b"},
        ],
    )
    out = migrate_legacy_gateway_config(cfg)
    assert out.pallas_image_base_url == "https://router.example/api/"
    assert out.pallas_image_api_key == "sk-first"
    assert out.pallas_image_model == "openai/gpt-image-2"
    assert out.pallas_image_primary_name == "主站"
    assert len(out.pallas_image_api_backends) == 1
    assert out.pallas_image_api_backends[0].base_url == "https://b.example/"

    backends = ImageGenSettings(out).api_backends()
    assert len(backends) == 2
    assert backends[0].label == "primary"
    assert backends[0].name == "主站"


def test_migrate_legacy_gateway_skips_when_primary_set() -> None:
    cfg = Config(
        pallas_image_base_url="https://primary.example/",
        pallas_image_api_key="sk-p",
        pallas_image_api_backends=[{"base_url": "https://b.example/", "api_key": "sk-b"}],
    )
    out = migrate_legacy_gateway_config(cfg)
    assert out.pallas_image_base_url == "https://primary.example/"
    assert len(out.pallas_image_api_backends) == 1


def test_api_backends_dedupe_same_url_key_model() -> None:
    cfg = Config(
        pallas_image_base_url="https://api.example.com/",
        pallas_image_api_key="sk-a",
        pallas_image_model="m1",
        pallas_image_api_backends=[
            {"base_url": "https://api.example.com/", "api_key": "sk-a", "model": "m1"},
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert len(backends) == 1
    assert backends[0].label == "primary"


def test_api_backends_propagate_omit_response_format() -> None:
    cfg = Config(
        pallas_image_base_url="https://api.example.com/",
        pallas_image_api_key="sk-a",
        pallas_image_model="m1",
        pallas_image_api_backends=[
            {
                "base_url": "https://gateway.example.net/api/",
                "api_key": "sk-b",
                "model": "m2",
                "omit_response_format": True,
            },
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert len(backends) == 2
    assert backends[0].omit_response_format is False
    assert backends[1].omit_response_format is True


def test_api_backends_propagate_display_name() -> None:
    cfg = Config(
        pallas_image_base_url="https://api.example.com/",
        pallas_image_api_key="sk-a",
        pallas_image_model="m1",
        pallas_image_primary_name="主站",
        pallas_image_api_backends=[
            {"name": "备线 A", "base_url": "https://b.example/", "api_key": "sk-b"},
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert backends[0].name == "主站"
    assert backends[1].name == "备线 A"
