"""Strict offline representation of the V2 Product/Sku contract.

This module is intentionally used only by the offline catalog builder.  It
mirrors the small dataclasses in the V2 reference application without wiring
the learning repository's runtime to that application.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.catalog.money import Money


@dataclass(frozen=True)
class ProductHighlight:
    label: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("ProductHighlight.label required")

    def to_dict(self) -> dict[str, str]:
        return {"detail": self.detail, "label": self.label}


@dataclass(frozen=True)
class Sku:
    sku_id: str
    spec: str
    price: Money
    stock: int

    def __post_init__(self) -> None:
        if not self.sku_id.strip():
            raise ValueError("Sku.sku_id required")
        if not isinstance(self.price, Money):
            raise TypeError("Sku.price must be Money")
        if not isinstance(self.stock, int) or isinstance(self.stock, bool) or self.stock < 0:
            raise ValueError("Sku.stock must be a non-negative integer")

    def to_dict(self) -> dict:
        return {
            "price": {
                "amount_in_minor_units": self.price.amount_in_minor_units,
                "currency": self.price.currency,
            },
            "sku_id": self.sku_id,
            "spec": self.spec,
            "stock": self.stock,
        }


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    description: str
    highlights: list[ProductHighlight] = field(default_factory=list)
    ships_to: list[str] = field(default_factory=list)
    skus: list[Sku] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("product_id", "title", "brand", "category", "origin_country"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Product.{name} required")
        if not isinstance(self.ships_to, list) or any(
            not str(value).strip() for value in self.ships_to
        ):
            raise ValueError("Product.ships_to must contain non-empty country codes")
        if not self.skus:
            raise ValueError(f"Product requires at least one Sku: {self.product_id}")

    def to_dict(self) -> dict:
        return {
            "brand": self.brand,
            "category": self.category,
            "description": self.description,
            "highlights": [highlight.to_dict() for highlight in self.highlights],
            "origin_country": self.origin_country,
            "product_id": self.product_id,
            "ships_to": list(self.ships_to),
            "skus": [sku.to_dict() for sku in self.skus],
            "title": self.title,
        }
