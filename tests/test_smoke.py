"""Smoke test ensuring the package imports and exposes its version."""


def test_import_voussoir() -> None:
    import voussoir

    assert voussoir.__version__
