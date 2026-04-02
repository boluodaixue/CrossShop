from globex_agent import __version__


def test_package_is_importable() -> None:
    assert __version__ == "0.1.0"
