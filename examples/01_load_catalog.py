"""Load and validate the chapter 09-1 StandardItem demo catalog."""

from pathlib import Path

from globex_agent.catalog import LocalCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


def main() -> None:
    result = LocalCatalog.from_jsonl(CATALOG_PATH, strict=True)
    platforms = [platform.value for platform in result.catalog.platforms]

    print(f"input records: {result.total_records}")
    print(f"accepted records: {result.accepted_records}")
    print(f"standard items: {len(result.catalog)}")
    print(f"same-product groups: {result.catalog.group_count}")
    print(f"platforms: {', '.join(platforms)}")

    example_group = result.catalog.items_for_group("hp-aurora-quietpro")
    if not example_group:
        raise RuntimeError("expected same_group_id hp-aurora-quietpro is missing")
    item_summary = ", ".join(
        f"{item.item_id}=CNY {item.price_cny}" for item in example_group
    )
    print(f"example: {example_group[0].title}")
    print("same_group_id: hp-aurora-quietpro")
    print(f"platform items: {item_summary}")


if __name__ == "__main__":
    main()
