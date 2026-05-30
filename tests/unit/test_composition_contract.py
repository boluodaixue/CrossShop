from __future__ import annotations

import globex_agent.composition as composition


def test_windows_local_model_paths_are_equivalent() -> None:
    assert composition._model_identifiers_equal(
        r"D:/models/bge-m3",
        r"D:\models\bge-m3",
    )


def test_different_local_model_paths_are_not_equivalent() -> None:
    assert not composition._model_identifiers_equal(
        r"D:/models/bge-m3",
        r"D:/models/other-model",
    )


def test_huggingface_repo_ids_are_not_path_normalized() -> None:
    assert not composition._model_identifiers_equal(
        "BAAI/bge-m3",
        "baai/bge-m3",
    )
    assert not composition._model_identifiers_equal(
        "BAAI/bge-m3",
        "BAAI\\bge-m3",
    )
