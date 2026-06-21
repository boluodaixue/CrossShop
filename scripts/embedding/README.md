# Local BGE-M3 Query Encoder

本地开发与验收服务直接复用 H1 的 `BgeM3Encoder`，从
`D:\models\bge-m3` 离线加载 `BAAI/bge-m3`，使用 CPU float32、CLS pooling、
L2 normalize、1024 维输出和 `max_length=512`。模型加载始终设置
`local_files_only=True`，不会联网下载。

模型依赖只安装在专用环境，不加入项目主依赖。启动命令：

```powershell
C:\Anaconda\envs\blog_04\python.exe -m scripts.embedding.serve_bge_m3_query --port 8002
```

服务只绑定 `127.0.0.1`，提供：

- `GET /health`
- `POST /v1/embeddings`

请求示例：

```json
{
  "model": "BAAI/bge-m3",
  "input": ["降噪耳机", "轻便旅行背包"],
  "encoding_format": "float",
  "dimensions": 1024
}
```

响应使用 OpenAI embeddings 的 `object/data/model` 形态，`data` 保留原输入下标。
单次请求最多 64 条文本，服务按 `BGE_M3_QUERY_MICRO_BATCH_SIZE` 分批并用进程内锁
串行执行 CPU 推理，避免并发请求造成内存和线程过度竞争。可用
`BGE_M3_QUERY_MODEL_PATH` 指向另一份本地模型目录，但仍不会启用网络下载。

现有 `OpenAIEmbeddingClient` 会始终发送 Bearer header；本地服务不校验该 header，
但客户端配置仍应提供一个非空占位值（例如 `local`），避免 HTTP 客户端拒绝空 header。

## 2026-08-27 本地实测

服务在 `http://127.0.0.1:18082` 实际加载完成，`/health` 返回：CPU float32、
CLS pooling、L2 normalize、1024 维、`max_length=512`、`micro_batch_size=4`、
`local_files_only=true`。

一次包含“降噪耳机”和“轻便旅行背包”的双文本请求结果：

- HTTP 200，原输入下标为 `[0, 1]`；
- 两条向量维度均为 1024；
- L2 norm 分别为 0.9999999721812279、1.0000000617018698；
- 本次端到端耗时 0.3363775999750942 秒。

“降噪耳机”的 CPU 服务向量与 H2 保存的真实 Query 向量余弦为
0.9999989757274355，最大绝对分量差为 0.00017918646335601807。两者 pooling、
归一化和维度相同；数值微差来自 CPU float32 与 H2 当时模型运行设备/精度的差异。
以上仅为本次可复现实测，不代表稳定性能基准。
