import type { ProductCard, TradeEvent } from "../types";

/** 只从在线验证通过的 final.result 读取后端 hydrate 的卡片。 */
function latestCards(events: TradeEvent[]): ProductCard[] {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.type !== "final.result") continue;
    if (event.payload?.verification_status !== "supported") return [];
    const cards = event.payload?.recommended_cards as ProductCard[] | undefined;
    if (cards && cards.length) return cards;
  }
  return [];
}

export default function ProductCards({ events }: { events: TradeEvent[] }) {
  const cards = latestCards(events);
  if (!cards.length) return null;

  return (
    <div className="cards">
      {cards.map((card) => (
        <article key={card.item_id} className="card">
          <header>
            <strong>{card.title}</strong>
            <span className="brand">
              {card.brand} · {card.origin_country}
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
            {card.variants.map((variant) => (
              <span key={variant.variant_id} className="sku">
                {variant.display_name} · {variant.price_major ?? "暂无"} {variant.currency}
                {variant.landed_price && !variant.landed_price.unavailable_reason
                  ? ` · 到手 ${variant.landed_price.landed_total_major} ${variant.landed_price.currency}`
                  : ""}
              </span>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
