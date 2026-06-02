from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import globex_agent.composition as composition


def test_product_reranker_uses_configured_persistent_worker(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeWorker:
        reranker_id = "fake-worker"

        def __init__(self, python_executable: Path, **kwargs) -> None:
            calls.append({"python_executable": python_executable, **kwargs})

        def ensure_ready(self) -> None:
            calls[-1]["ready"] = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(composition, "SubprocessCrossEncoderReranker", FakeWorker)
    settings = SimpleNamespace(
        bge_reranker_python="C:/Anaconda/envs/blog_04/python.exe",
        bge_reranker_model="D:/models/bge-reranker-v2-m3",
        bge_reranker_device="cuda:0",
        bge_reranker_batch_size=1,
        bge_reranker_max_length=256,
        bge_reranker_fp32=False,
        models_local_only=True,
    )

    result = composition._build_product_reranker(settings)

    assert result._reranker.reranker_id == "fake-worker"
    assert calls == [
        {
            "python_executable": Path("C:/Anaconda/envs/blog_04/python.exe"),
            "model_name": "D:/models/bge-reranker-v2-m3",
            "device": "cuda:0",
            "batch_size": 1,
            "max_length": 256,
            "use_fp16": True,
            "local_files_only": True,
            "ready": True,
        }
    ]


def test_category_encoder_uses_configured_local_bge_model(monkeypatch) -> None:
    created: dict = {}

    class FakeEncoder:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def ensure_ready(self) -> None:
            created["ready"] = True

    class FakeKnowledgeBase:
        def __init__(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(composition, "SentenceTransformerTextEncoder", FakeEncoder)
    monkeypatch.setattr(composition, "OpenSearchCategoryKnowledgeBase", FakeKnowledgeBase)
    settings = SimpleNamespace(
        category_taxonomy=Path(
            "data/category_insight/taobao_zh/category_taxonomy_taobao_zh.json"
        ),
        bge_m3_model="D:/models/bge-m3",
        bge_m3_batch_size=4,
        bge_m3_max_seq_length=512,
        models_local_only=True,
        opensearch_endpoint="http://127.0.0.1:9200",
        category_index="globex_category_kb_taobao_zh_v1",
        category_reranker_enabled=False,
        category_reranker_python="",
        category_reranker_device="cuda:0",
        category_reranker_batch_size=1,
        category_reranker_fp32=False,
        category_reranker_document_mode="contextual",
    )

    composition._build_category_insight_service(settings)

    assert created == {
        "model_name": "D:/models/bge-m3",
        "device": "cpu",
        "batch_size": 4,
        "max_seq_length": 512,
        "local_files_only": True,
        "ready": True,
    }
