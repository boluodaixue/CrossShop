import type { DisplayedProduct, TradeEvent } from "../types";

/** 只展示 Main 最终明确选择且经后端真实候选校验过的商品。 */
function latestDisplayed(events: TradeEvent[]): DisplayedProduct[] {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.type !== "final.result") continue;
    return (event.payload?.displayed_products as DisplayedProduct[] | undefined) ?? [];
  }
  return [];
}

export default function ProductCards({ events }: { events: TradeEvent[] }) {
  const displayed = latestDisplayed(events);
  if (!displayed.length) return null;

  return (
    <div className="cards">
      {displayed.map(({ rank, platform, site_locale: siteLocale, card }) => (
        <article key={`${platform}:${card.product_id}`} className="card">
          <header>
            <strong>
              {rank}. {card.title}
            </strong>
            <span className="brand">
              {platform}
              {siteLocale ? `/${siteLocale}` : ""} · {card.brand} · {card.origin_country}
            </span>
          </header>
          <div className="price">
            {card.price_major} {card.currency}
          </div>
          {card.landed_price && !card.landed_price.unavailable_reason && (
            <div className="landed">
              <div className="landed-total">
                到手价 {card.landed_price.landed_total_major} {card.landed_price.currency}
              </div>
              <div className="landed-detail">
                小计 {card.landed_price.subtotal_major} + 运费 {card.landed_price.freight_major} + 关税{" "}
                {card.landed_price.tariff_major}
                {card.landed_price.de_minimis_applied ? "（免税额度内）" : ""}
              </div>
            </div>
          )}
          <ul className="highlights">
            {card.highlights.map((highlight) => (
              <li key={highlight}>{highlight}</li>
            ))}
          </ul>
          <div className="skus">
            {card.skus.map((sku) => (
              <span key={sku.sku_id} className="sku">
                {sku.spec} · {sku.price_major} {sku.currency} · 库存 {sku.stock}
              </span>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
