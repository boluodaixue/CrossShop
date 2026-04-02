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

## 本轮报错与解决

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `uv` 缓存目录拒绝访问 | Codex 沙箱不能读取用户级 uv 缓存 | 验收时直接调用 `.venv` 中的 Python/Ruff；普通本地终端继续使用 uv |
| pytest 默认临时目录拒绝访问 | 中文 Windows 用户名与临时目录 ACL 组合问题 | 测试夹具改为在既有测试目录创建独立临时文件并自动清理 |
| GBK 终端不能输出 `¥` | Windows 默认控制台编码不支持该字符 | 价格单位改为 `CNY`，运行中文示例时设置 `PYTHONUTF8=1` |
