"""Verify that the project package is installed and importable."""

from globex_agent import __version__


def main() -> None:
    print(f"globex_agent {__version__} imported successfully")


if __name__ == "__main__":
    main()
