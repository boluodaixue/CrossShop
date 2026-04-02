"""JSONL-backed local catalog used by the first offline implementation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median

from pydantic import ValidationError

from globex_agent.domain.models import CatalogRecord, Offer, StandardItem

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

    @property
    def rejected_records(self) -> int:
        return len(self.errors)


class CatalogValidationError(ValueError):
    def __init__(self, errors: tuple[CatalogLoadError, ...]) -> None:
        self.errors = errors
        super().__init__(f"catalog contains {len(errors)} invalid record(s)")


@dataclass(frozen=True)
class _LoadedRecord:
    line_number: int
    raw_line: str
    record: CatalogRecord


class LocalCatalog:
    """Read-only in-memory view of normalized demo products."""

    def __init__(self, items: list[StandardItem]) -> None:
        item_map = {item.canonical_product_id: item for item in items}
        if len(item_map) != len(items):
            raise ValueError("canonical_product_id must be unique")
        self._items = item_map

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[StandardItem, ...]:
        return tuple(self._items[item_id] for item_id in sorted(self._items))

    @property
    def offers(self) -> tuple[Offer, ...]:
        return tuple(offer for item in self.items for offer in item.offers)

    def get(self, canonical_product_id: str) -> StandardItem | None:
        return self._items.get(canonical_product_id)

    def offers_for(self, canonical_product_id: str) -> tuple[Offer, ...]:
        item = self.get(canonical_product_id)
        return tuple(item.offers) if item is not None else ()

    @classmethod
    def from_jsonl(cls, path: str | Path, *, strict: bool = False) -> CatalogLoadResult:
        """Load records, report invalid rows, and group cross-platform offers.

        A row can be rejected for schema violations, duplicate offer IDs, conflicting
        product master data, or a price more than 50x away from its category median.
        Valid rows remain usable unless ``strict=True``.
        """

        source_path = Path(path)
        loaded: list[_LoadedRecord] = []
        errors: list[CatalogLoadError] = []
        total_records = 0

        with source_path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                total_records += 1
                try:
                    record = CatalogRecord.model_validate_json(stripped)
                except ValidationError as exc:
                    errors.append(
                        CatalogLoadError(
                            line_number=line_number,
                            reason=_summarize_validation_error(exc),
                            raw_line=stripped[:500],
                        )
                    )
                    continue
                loaded.append(_LoadedRecord(line_number, stripped[:500], record))

        loaded, structural_errors = _reject_structural_conflicts(loaded)
        errors.extend(structural_errors)
        loaded, price_errors = _reject_price_outliers(loaded)
        errors.extend(price_errors)

        catalog = cls(_group_records(loaded))
        ordered_errors = tuple(sorted(errors, key=lambda error: error.line_number))
        if strict and ordered_errors:
            raise CatalogValidationError(ordered_errors)
        return CatalogLoadResult(
            catalog=catalog,
            total_records=total_records,
            accepted_records=len(loaded),
            errors=ordered_errors,
        )


def _summarize_validation_error(exc: ValidationError) -> str:
    first = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else str(first["msg"])


def _product_signature(record: CatalogRecord) -> tuple[object, ...]:
    return (
        record.title,
        record.brand,
        tuple(record.category_path),
        record.description,
        tuple(sorted(record.attributes.items())),
        record.provenance,
    )


def _reject_structural_conflicts(
    loaded: list[_LoadedRecord],
) -> tuple[list[_LoadedRecord], list[CatalogLoadError]]:
    accepted: list[_LoadedRecord] = []
    errors: list[CatalogLoadError] = []
    product_signatures: dict[str, tuple[object, ...]] = {}
    offer_ids: set[str] = set()

    for entry in loaded:
        record = entry.record
        expected_signature = product_signatures.get(record.canonical_product_id)
        signature = _product_signature(record)
        if expected_signature is not None and expected_signature != signature:
            errors.append(
                CatalogLoadError(
                    entry.line_number,
                    f"conflicting product data for {record.canonical_product_id}",
                    entry.raw_line,
                )
            )
            continue
        if record.offer.offer_id in offer_ids:
            errors.append(
                CatalogLoadError(
                    entry.line_number,
                    f"duplicate offer_id: {record.offer.offer_id}",
                    entry.raw_line,
                )
            )
            continue
        product_signatures.setdefault(record.canonical_product_id, signature)
        offer_ids.add(record.offer.offer_id)
        accepted.append(entry)
    return accepted, errors


def _reject_price_outliers(
    loaded: list[_LoadedRecord],
) -> tuple[list[_LoadedRecord], list[CatalogLoadError]]:
    prices_by_category: dict[tuple[str, ...], list[Decimal]] = defaultdict(list)
    for entry in loaded:
        prices_by_category[tuple(entry.record.category_path)].append(entry.record.offer.price)

    medians = {
        category: median(prices) for category, prices in prices_by_category.items() if prices
    }
    accepted: list[_LoadedRecord] = []
    errors: list[CatalogLoadError] = []
    for entry in loaded:
        price = entry.record.offer.price
        category_median = medians[tuple(entry.record.category_path)]
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


def _group_records(loaded: list[_LoadedRecord]) -> list[StandardItem]:
    grouped: dict[str, list[CatalogRecord]] = defaultdict(list)
    for entry in loaded:
        grouped[entry.record.canonical_product_id].append(entry.record)

    items: list[StandardItem] = []
    for product_id in sorted(grouped):
        records = grouped[product_id]
        first = records[0]
        offers = sorted((record.offer for record in records), key=lambda offer: offer.offer_id)
        items.append(
            StandardItem(
                canonical_product_id=first.canonical_product_id,
                title=first.title,
                brand=first.brand,
                category_path=first.category_path,
                description=first.description,
                attributes=first.attributes,
                offers=offers,
                provenance=first.provenance,
            )
        )
    return items
