# ShopSimulator 规格身份审计（2026-08-20）

本审计是只读分析，不修改 raw、schema-v2 JSONL、SQLite、领域代码或订单代码。可复现命令：

```powershell
.\.venv\Scripts\python.exe scripts\data\audit_variant_identity.py
```

产物：[shopsimulator_variant_identity.json](../../output/audit/shopsimulator_variant_identity.json)。脚本读取原始 gzip、schema-v2 staging JSONL、SQLite 和 staging manifest，并记录源/规范化/SQLite hash。

## 结论

原始文件是 23,421 条商品记录的 JSON list。顶层有 `asin`、`customization_options`、`pricing`、`shop_name`、`tag`、`instructions` 等；规格位于 `customization_options.<选项名>[]`，每条含 `asin`、`value`、`price`、`price_string`、`is_available`、`is_selected`、`image`、`url`。原始快照没有 `seller_id`、`shop_id`、`store_id`、`merchant_id`，也没有独立 listing/offer/time 字段。`asin` 在商品根是 source item 标识，在规格记录中是 source SKU 候选；这个语义来自字段路径，不能从名称推断为商家 SKU。

当前迁移的精确血缘是：

| 规范字段 | 原始来源 |
| --- | --- |
| `item_id` | `taobao:cn:{root.asin}` |
| `variant_id` | `sha256(platform|item_id|canonical options)` 前 24 位的内部 SKU ID；不含价格、库存或输入顺序 |
| `source_variant_id` | `customization_options.<option_name>[].asin`（adapter 兼容 variant_id/id）；允许重复，仅作原始追溯 |
| `options` | mapping key 作为 option name，entry `value` 作为 option value |
| variant `price_cny` | entry `price`，迁移 adapter 同时接受 `price_cny` |
| variant `availability` | entry `is_available`；缺失保留 unknown |
| item 价格 | 只在没有 variants 时使用；有 variants 时 item 价格置 null |

## 重复作用域与对账

| 指标 | 数量 |
| --- | ---: |
| 原始商品 | 23,421 |
| 原始规格记录 | 239,925 |
| schema-v2 商品/源规格/规范 variant | 23,421 / 239,925 / 239,913 |
| 同 item + source SKU 重复组 | 6,583 |
| 同 item + source SKU 重复记录（超出首条） | 6,885 |
| 受影响商品 | 6,232 |
| 跨 item 相同 source SKU 组 | 0 |
| 缺失 source SKU 记录 | 0 |
| staging 中 `~` 后缀 variant ID | 0 |
| 相同规范 options 合并组/超额记录 | 12 / 12 |
| 相同 options 的价格或 availability 冲突组 | 7 |

因此 6,885 是“同 item 内相同 source ID 的重复记录数（每组减去首条）”，不是重复组数，也不是跨商品冲突数。规范化后不再用 `~序号` 伪造身份：不同 options 生成不同 canonical variant；相同 options 合并，价格/状态冲突组置 unknown 并计入 manifest。seller-scoped 与 listing-scoped 统计不可用，因为来源没有可靠 seller/listing 字段；不能把 `shop_name` 猜作 seller ID，也不能从价格差异推断商家。

## 重复组分类

在有效 `(item_id, source_sku_id)` 作用域内，6,583 组全部属于 `B_options_different`（选项 fingerprint 不同）；A 完全相同、C 仅价格不同、D 仅 availability 不同、E 价格与 availability 同时冲突、F 仅时间快照不同、G 信息不足均为 0。分类计数与重复组数严格相等。

代表组样例保存在 JSON 产物中，仅包含 item、source SKU、options、price、availability，不输出店铺或地址等隐私字段。由于没有时间字段，不能把任何组分类为时间快照变化。

## 候选唯一键

| 候选键 | 结果 |
| --- | --- |
| `(platform, source_sku_id)` | 6,583 组重复、6,885 条超额记录；不安全 |
| `(platform, item_id, source_sku_id)` | 6,583 组重复、6,885 条超额记录；仍不唯一 |
| `(platform, seller_id, source_sku_id)` | 不可验证，seller 字段缺失 |
| `(platform, seller_id, source_item_id, source_sku_id)` | 不可验证，seller 字段缺失 |
| `(item_id, normalized options fingerprint)` | 6 组重复、6 条超额记录、3 个商品受影响；不能作为 source SKU 身份键 |

options fingerprint 只对排序后的规范 name/value 计算，不包含价格、库存、输入顺序。它只能识别“同 item 的相同选项组合”，不能证明 seller offer 唯一。

## 条件化模型结论

当前证据不足以安全拆出 `ProductVariant + SkuOffer`：没有可靠 seller、listing、offer 或时间维度，无法判断同 source SKU 的多条记录是同一 offer 的快照、不同卖家的报价、还是平台规格数据重复。canonical `variant_id` 只表示 item 内规范 options 组合，`source_variant_id` 允许重复；二者都不表示 seller/offer 身份，不能自动下单依据。

条件成立时才可采用 `ProductVariant`（item 内规范 options/规格事实）+ `SkuOffer`（seller/listing/source SKU/价格/availability/observed time）。在当前数据下应标记 `ambiguous_offer_conflict`，禁止自动把重复记录升级为可购买 offer；`same_group_id` 仍只表示跨平台逻辑关联。

脚本两次运行输出 hash 相同：

`8C9DDB873FBB45F6FE4A646ACCC7107D733EE6CF53544EEFCECCED9F756D5053`

无阻塞项；但“seller/offer 身份不可判定”是模型决策限制，不是可通过本次只读审计消除的数据错误。
