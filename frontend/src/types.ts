/// <reference types="vite/client" />

export type TradeEventType =
  | "agent.dispatch"
  | "tool.invoke"
  | "tool.result"
  | "token.delta"
  | "plan.update"
  | "context.compressed"
  | "model.fallback"
  | "cache.hit"
  | "task.queued"
  | "task.started"
  | "final.result"
  | "evidence.freeze"
  | "evidence.verify"
  | "error";

export interface TradeEvent {
  type: TradeEventType;
  payload: Record<string, any>;
  occurred_at: string;
}

export interface LandedPrice {
  ship_to: string;
  subtotal_major: number;
  freight_major: number;
  tariff_major: number;
  tariff_rate: number;
  de_minimis_applied: boolean;
  landed_total_major: number;
  currency: string;
  quantity: number;
  unavailable_reason?: string;
}

export interface ProductCard {
  item_id: string;
  title: string;
  brand: string;
  category: string;
  origin_country: string;
  price_major: number | null;
  currency: string;
  highlights: string[];
  variants: {
    variant_id: string;
    display_name: string;
    options: { code?: string; name: string; value: string }[];
    price_major: number | null;
    currency: string;
    availability: string;
    landed_price?: LandedPrice;
  }[];
  selected_variant_id?: string;
  score: number;
  landed_price?: LandedPrice;
}
