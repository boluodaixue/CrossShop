"""Load, validate, and group the controlled demo catalog."""

from pathlib import Path

from globex_agent.catalog import LocalCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


def main() -> None:
    result = LocalCatalog.from_jsonl(CATALOG_PATH, strict=True)
    platforms = sorted({offer.platform.value for offer in result.catalog.offers})

    print(f"input records: {result.total_records}")
    print(f"accepted records: {result.accepted_records}")
    print(f"standard products: {len(result.catalog)}")
    print(f"platform offers: {len(result.catalog.offers)}")
    print(f"platforms: {', '.join(platforms)}")

    example = result.catalog.get("hp-aurora-quietpro")
    if example is None:
        raise RuntimeError("expected demo product hp-aurora-quietpro is missing")
    offer_summary = ", ".join(
        f"{offer.platform.value}=CNY {offer.price}" for offer in example.offers
    )
    print(f"example: {example.title}")
    print(f"grouped offers: {offer_summary}")


if __name__ == "__main__":
    main()
