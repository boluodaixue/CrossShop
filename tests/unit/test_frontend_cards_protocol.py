from __future__ import annotations

from pathlib import Path


def test_product_cards_only_reads_supported_final_result() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "components"
        / "ProductCards.tsx"
    )
    source = path.read_text(encoding="utf-8")
    assert 'event.type !== "final.result"' in source
    assert 'event.payload?.verification_status !== "supported"' in source
    assert 'event.type === "tool.result"' not in source
    assert "recommended_cards" in source
