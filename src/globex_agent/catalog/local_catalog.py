"""JSONL-backed local catalog used by the first offline implementation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median

from pydantic import ValidationError

from globex_agent.domain.models import Platform, StandardItem

MAX_CATEGORY_PRICE_RATIO = Decimal("50")


@dataclass(frozen=True)
class CatalogLoadError:
    line_number: int
    reason: str
    raw_line: str


@dataclass(frozen=True)
class CatalogLoadResult:
    catalog: LocalCatalog
    total_records: int
    accepted_records: int
    errors: tuple[CatalogLoadError, ...]
    deduplicated_records: int = 0

    @property
    def rejected_records(self) -> int:
        return len(self.errors)


class CatalogValidationError(ValueError):
    def __init__(self, errors: tuple[CatalogLoadError, ...]) -> None:
        self.errors = errors
        super().__init__(f"catalog contains {len(errors)} invalid record(s)")


@dataclass(frozen=True)
class _LoadedItem:
    line_number: int
    raw_line: str
    item: StandardItem


class LocalCatalog:
    """Read-only in-memory view of platform-level ``StandardItem`` rows."""

    def __init__(self, items: list[StandardItem]) -> None:
        item_map = {item.item_id: item for item in items}
        if len(item_map) != len(items):
            raise ValueError("item_id must be globally unique")
        self._items = item_map

    def __len__(self) -> int:
        """Return the number of platform-level items, not cross-platform groups."""

        return len(self._items)

    @property
    def items(self) -> tuple[StandardItem, ...]:
        return tuple(self._items[item_id] for item_id in sorted(self._items))

    @property
    def same_group_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.same_group_id for item in self.items}))

    @property
    def group_count(self) -> int:
        return len(self.same_group_ids)

    @property
    def platforms(self) -> tuple[Platform, ...]:
        return tuple(sorted({item.platform for item in self.items}, key=lambda value: value.value))

    def get(self, item_id: str) -> StandardItem | None:
        return self._items.get(item_id)

    def items_for_group(self, same_group_id: str) -> tuple[StandardItem, ...]:
        return tuple(item for item in self.items if item.same_group_id == same_group_id)

    @classmethod
    def from_jsonl(cls, path: str | Path, *, strict: bool = False) -> CatalogLoadResult:
        """Load and validate the chapter 09-1 platform-level item rows.

        Invalid schemas, duplicate globally unique ``item_id`` values, and extreme
        category price outliers are rejected. Cross-platform rows sharing a
        ``same_group_id`` remain separate records for the later PriceCompare step.
        """

        source_path = Path(path)
        loaded: list[_LoadedItem] = []
        errors: list[CatalogLoadError] = []
        total_records = 0

        with source_path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                total_records += 1
                try:
                    item = StandardItem.model_validate_json(stripped)
                except ValidationError as exc:
                    errors.append(
                        CatalogLoadError(
                            line_number=line_number,
                            reason=_summarize_validation_error(exc),
                            raw_line=stripped[:500],
                        )
                    )
                    continue
                loaded.append(_LoadedItem(line_number, stripped[:500], item))

        loaded, duplicate_errors = _reject_duplicate_item_ids(loaded)
        errors.extend(duplicate_errors)
        loaded, deduplicated_records = _deduplicate_same_platform_items(loaded)
        loaded, price_errors = _reject_price_outliers(loaded)
        errors.extend(price_errors)

        catalog = cls([entry.item for entry in loaded])
        ordered_errors = tuple(sorted(errors, key=lambda error: error.line_number))
        if strict and ordered_errors:
            raise CatalogValidationError(ordered_errors)
        return CatalogLoadResult(
            catalog=catalog,
            total_records=total_records,
            accepted_records=len(loaded),
            errors=ordered_errors,
            deduplicated_records=deduplicated_records,
        )


def _summarize_validation_error(exc: ValidationError) -> str:
    first = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else str(first["msg"])


def _reject_duplicate_item_ids(
    loaded: list[_LoadedItem],
) -> tuple[list[_LoadedItem], list[CatalogLoadError]]:
    accepted: list[_LoadedItem] = []
    errors: list[CatalogLoadError] = []
    item_ids: set[str] = set()
    for entry in loaded:
        if entry.item.item_id in item_ids:
            errors.append(
                CatalogLoadError(
                    entry.line_number,
                    f"duplicate item_id: {entry.item.item_id}",
                    entry.raw_line,
                )
            )
            continue
        item_ids.add(entry.item.item_id)
        accepted.append(entry)
    return accepted, errors


def _reject_price_outliers(
    loaded: list[_LoadedItem],
) -> tuple[list[_LoadedItem], list[CatalogLoadError]]:
    prices_by_category: dict[tuple[str, ...], list[Decimal]] = defaultdict(list)
    for entry in loaded:
        if entry.item.price_cny is not None:
            prices_by_category[tuple(entry.item.category_path)].append(entry.item.price_cny)

    medians = {
        category: median(prices) for category, prices in prices_by_category.items() if prices
    }
    accepted: list[_LoadedItem] = []
    errors: list[CatalogLoadError] = []
    for entry in loaded:
        price = entry.item.price_cny
        if price is None:
            accepted.append(entry)
            continue
        category_median = medians[tuple(entry.item.category_path)]
        ratio = max(price / category_median, category_median / price)
        if ratio > MAX_CATEGORY_PRICE_RATIO:
            errors.append(
                CatalogLoadError(
                    entry.line_number,
                    (
                        f"price {price} is more than {MAX_CATEGORY_PRICE_RATIO}x away "
                        f"from category median {category_median}"
                    ),
                    entry.raw_line,
                )
            )
            continue
        accepted.append(entry)
    return accepted, errors


def _deduplicate_same_platform_items(
    loaded: list[_LoadedItem],
) -> tuple[list[_LoadedItem], int]:
    """Keep the higher-rated/newer row for one same item on one platform."""

    best: dict[tuple[Platform, str, str | None], _LoadedItem] = {}
    for entry in loaded:
        # Parallel EN/ZH listings of one canonical product remain available for
        # post-recall display selection. Duplicate rows in the same language
        # still collapse deterministically.
        key = (entry.item.platform, entry.item.same_group_id, entry.item.language)
        current = best.get(key)
        if current is None or _dedupe_rank(entry) > _dedupe_rank(current):
            best[key] = entry
    accepted = sorted(best.values(), key=lambda entry: entry.line_number)
    return accepted, len(loaded) - len(accepted)


def _dedupe_rank(entry: _LoadedItem) -> tuple[float, float, str]:
    return (
        entry.item.rating if entry.item.rating is not None else -1.0,
        (
            entry.item.source_updated_at.timestamp()
            if entry.item.source_updated_at is not None
            else float("-inf")
        ),
        entry.item.item_id,
    )
