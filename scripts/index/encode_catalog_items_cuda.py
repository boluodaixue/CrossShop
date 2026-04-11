"""Encode all catalog Item texts with cached BGE-M3 in a CUDA Transformers env.

This worker intentionally needs only torch, transformers, numpy, and pydantic.
It reproduces the SentenceTransformers checkpoint's declared CLS pooling plus
L2 normalization, then writes float32 embeddings for the main project process
to turn into Faiss indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MAX_SEQ_LENGTH = 512

PARTITIONS = (
    ("amazon", "us"),
    ("amazon", "es"),
    ("amazon", "jp"),
    ("taobao", "cn"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "embeddings" / "catalog",
    )
    parser.add_argument(
        "--partition",
        action="append",
        choices=tuple(f"{platform}:{locale}" for platform, locale in PARTITIONS),
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")

    selected = set(
        args.partition
        or (f"{platform}:{locale}" for platform, locale in PARTITIONS)
    )
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    for platform, locale in PARTITIONS:
        partition_id = f"{platform}:{locale}"
        if partition_id not in selected:
            continue
        items_path = (
            args.processed_root
            / "catalogs"
            / platform
            / locale
            / "items.jsonl"
        )
        ids, texts = _load_item_texts(items_path)
        document_ids = np.asarray(ids, dtype=np.str_)
        print(f"encoding {partition_id}: {len(texts):,} items", flush=True)
        matrices: list[np.ndarray] = []
        total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
        with torch.inference_mode():
            for start in range(0, len(texts), args.batch_size):
                batch = texts[start : start + args.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=args.max_seq_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(args.device) for key, value in encoded.items()}
                cls_embeddings = model(**encoded).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(
                    cls_embeddings.float(),
                    p=2,
                    dim=1,
                )
                matrices.append(normalized.cpu().numpy().astype(np.float32, copy=False))
                batch_number = start // args.batch_size + 1
                if batch_number % 100 == 0 or batch_number == total_batches:
                    print(
                        f"{partition_id}: batch {batch_number:,}/{total_batches:,}",
                        flush=True,
                    )
        embeddings = np.concatenate(matrices, axis=0)
        output_path = args.output_root / platform / locale / "bge-m3-items.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".tmp.npz")
        np.savez(temporary, document_ids=document_ids, embeddings=embeddings)
        temporary.replace(output_path)
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest = {
            "architecture": "query-item-shared-encoder",
            "encoder_model": args.model,
            "platform": platform,
            "locale": locale,
            "partition_id": partition_id,
            "document_count": len(document_ids),
            "dimension": int(embeddings.shape[1]),
            "dtype": "float32",
            "inference_dtype": str(dtype).removeprefix("torch."),
            "pooling": "cls",
            "normalized": True,
            "max_seq_length": args.max_seq_length,
            "embedding_fine_tuned": False,
            "semantic_query_policy": "encode original query without translation",
            "items_sha256": _sha256(items_path),
            "embeddings_sha256": _sha256(output_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"embeddings: {output_path}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_item_texts(path: Path) -> tuple[list[str], list[str]]:
    """Reproduce standard_item_to_search_document + embedding_document_text."""

    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            title = " ".join(str(item["title"]).split())
            attributes = json.dumps(
                item.get("attributes", {}),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            variants = json.dumps(
                item.get("variants", []),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            body = " ".join(
                part
                for part in (
                    item.get("brand") or "",
                    " ".join(item.get("category_path", [])),
                    attributes,
                    variants,
                    item.get("description") or "",
                )
                if part
            )
            clean_body = " ".join(body.split())
            if title and clean_body.startswith(title):
                clean_body = clean_body[len(title) :].lstrip(" :-|\t\n")
            clean_body = " ".join(
                token for token in clean_body.split() if token != "None"
            )
            parts = [f"Title: {title}"] if title else []
            if clean_body:
                parts.append(f"Details: {clean_body}")
            rows.append((str(item["item_id"]), "\n".join(parts)))
    rows.sort(key=lambda row: row[0])
    return [row[0] for row in rows], [row[1] for row in rows]


if __name__ == "__main__":
    main()
