# 章节示例

## 00 工程导入检查

命令：

```powershell
python -m uv run python examples/00_smoke.py
```

输入：无。实际输出：`globex_agent 0.1.0 imported successfully`。

## 01 本地商品目录加载

对应课程：第 09 章工程初始化与第 09-1 章多平台商品数据底座。

```powershell
$env:PYTHONUTF8 = "1"
python -m uv run python examples/01_load_catalog.py
```

输入：`data/demo/products.jsonl`。2026-08-12 实际输出摘要：

```text
input records: 36
accepted records: 36
standard products: 12
platform offers: 36
platforms: aliexpress, amazon, shopee
example: Aurora QuietPro 头戴式降噪耳机
grouped offers: aliexpress=CNY 1219.00, amazon=CNY 1299.00, shopee=CNY 1269.00
```

## 02 不依赖 LLM 的确定性推荐链路

对应课程：第 11 章 ItemSearch、第 12 章 PriceCompare/ShippingCalc、第 14 章 ItemPicker/ShoppingSummary。CategoryInsight 和 AgentLoop 尚未接入。

```powershell
$env:PYTHONUTF8 = "1"
python -m uv run python examples/02_deterministic_pipeline.py --case 1
```

输入：`data/demo/queries.jsonl` 第 1 条——“预算 1500 元的通勤降噪耳机，不要入耳式”，并加载 `demo-user-001` 用户画像。2026-08-12 实际输出摘要：

```text
ItemSearch: 3 returned / 7 matched
PriceCompare: 3 products / 9 offers
ShippingCalc: 9 landed-cost quotes
ItemPicker: 3 picked / 0 rejected
Budget Wave H1：CNY 496.54
Sonic Commute X2：CNY 1028.54
Aurora QuietPro：CNY 1363.14
```

可用 `--case 1` 到 `--case 6` 运行全部固定场景。其中第 3、5、6 条会因到手价超预算或属性不满足而得到结构化空结果，这是硬约束生效的预期行为，不是运行错误。

## 本轮报错与解决

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `uv` 缓存目录拒绝访问 | Codex 沙箱不能读取用户级 uv 缓存 | 验收时直接调用 `.venv` 中的 Python/Ruff；普通本地终端继续使用 uv |
| pytest 默认临时目录拒绝访问 | 中文 Windows 用户名与临时目录 ACL 组合问题 | 测试夹具改为在既有测试目录创建独立临时文件并自动清理 |
| GBK 终端不能输出 `¥` | Windows 默认控制台编码不支持该字符 | 价格单位改为 `CNY`，运行中文示例时设置 `PYTHONUTF8=1` |
| 扩展 ShippingQuote 后旧测试先报缺少字段 | 阶段 2 增加了平台、税率、时效和规则版本 | 同步升级模型测试，并继续验证到手价分项之和 |
