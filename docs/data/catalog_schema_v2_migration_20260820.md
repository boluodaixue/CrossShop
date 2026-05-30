# catalog-schema-v2 迁移记录（2026-08-20）

实现状态：商品事实、SQLite repository、搜索文档和订单已切换到 typed schema-v2；Faiss 未重建，旧 v4 文本索引会被拒绝，下一任务重建。

迁移命令：

```powershell
.\.venv\Scripts\python.exe scripts\data\migrate_catalog_schema_v2.py
.\.venv\Scripts\python.exe scripts\data\build_catalog_databases.py --catalog-root data\processed\staging\catalog-schema-v2
```

旧 JSONL 保持只读；规范化 JSONL 写入 `data/processed/staging/catalog-schema-v2`，SQLite 在 staging 校验成功后重建到 `data/processed/databases`。完整 hash 和审计计数见 `data/processed/staging/catalog-schema-v2/manifest.json`。

| 分区 | 商品数 | source variant 数 | canonical variant 数 | 失败 | source ID 重复组/超额记录 | 选项合并组 | 价格/状态冲突组 | unknown variant price |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| amazon:us | 5,638 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| amazon:es | 7,856 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| amazon:jp | 8,323 | 0 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| taobao:cn | 23,421 | 239,925 | 239,913 | 0 | 6,583 / 6,885 | 12 | 7 | 426 |

variant 身份现在由 `sha256(platform|item_id|canonical_options)` 的前 24 位构成；canonical options 对名称和值执行 NFKC、空白折叠、casefold，并排序后 JSON 序列化，不包含价格、库存或输入顺序。原始 ID 仅写入 `source_variant_id` 追溯字段，不再追加 `~序号`。同选项且价格/状态相同的 12 条记录合并为 12 组规范 SKU；同选项但价格或状态冲突的 7 组不生成第二 SKU，而将该规范 SKU 置为价格未知/availability unknown，并由 manifest 计数。淘宝来源属性中识别并迁移了 478 条规范材质事实；没有可确认 taxonomy 的材质会保留 `code=null`。
canonical `variant_id` 重复数为 0；manifest 中的 6,885 明确标记为 source ID 超额记录，不是规范 ID 冲突。

运行时主链仍是纯 Faiss ANN → typed 硬过滤/补候选 → Reranker；BM25 只在 ANN 故障时降级，不宣称 Faiss 已重建。
