"""
Dataset loading utilities for the English-to-Chinese translation assignment.

The coursework dataset is expected to be downloaded manually from Baidu Netdisk
and placed under data/.  This module accepts several common parallel-corpus
layouts so the training script is not tied to one archive file name.
"""

import json
import os
from pathlib import Path


SOURCE_KEYS = ("src", "source", "en", "english", "input", "sentence")
TARGET_KEYS = ("tgt", "target", "zh", "chinese", "output", "translation")
TRAIN_NAMES = ("train", "training")
VAL_NAMES = ("val", "valid", "validation", "dev")
TEST_NAMES = ("test", "testing")
PARALLEL_EXTENSIONS = (".txt", ".tsv", ".csv", ".json", ".jsonl")


def _normalise_text(text):
    text = str(text).strip()
    text = text.replace("@@ ", "")
    text = text.replace("@@", "")
    return " ".join(text.split())


def _read_text_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _normalise_pair(src, tgt):
    src = _normalise_text(src)
    tgt = _normalise_text(tgt)
    if not src or not tgt:
        return None
    return src, tgt


def _pick_value(record, keys):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _pair_from_record(record):
    if not isinstance(record, dict):
        return None

    src = _pick_value(record, SOURCE_KEYS)
    tgt = _pick_value(record, TARGET_KEYS)

    if src is None and tgt is None and isinstance(record.get("translation"), dict):
        translation = record["translation"]
        src = translation.get("en")
        tgt = translation.get("zh")

    if src is None or tgt is None:
        return None
    return _normalise_pair(src, tgt)


def _load_json_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    pairs = []
    if isinstance(data, list):
        for item in data:
            pair = _pair_from_record(item)
            if pair:
                pairs.append(pair)
    return pairs


def _load_jsonl_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = _pair_from_record(json.loads(line))
            if pair:
                pairs.append(pair)
    return pairs


def _load_delimited_pairs(path):
    pairs = []
    for line in _read_text_lines(path):
        line = line.strip()
        if not line:
            continue

        for delimiter in ("\t", " ||| ", ","):
            if delimiter in line:
                parts = line.split(delimiter)
                if len(parts) >= 2:
                    pair = _normalise_pair(parts[0], parts[1])
                    if pair:
                        pairs.append(pair)
                    break
    return pairs


def _load_pair_file(path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_pairs(path)
    if suffix == ".jsonl":
        return _load_jsonl_pairs(path)
    if suffix in (".txt", ".tsv", ".csv"):
        return _load_delimited_pairs(path)
    return []


def _find_split_file(data_dir, names):
    for name in names:
        for extension in PARALLEL_EXTENSIONS:
            path = data_dir / f"{name}{extension}"
            if path.exists():
                return path
    return None


def _find_parallel_side_files(data_dir, names):
    for name in names:
        en_candidates = [data_dir / f"{name}.en", data_dir / f"{name}.eng"]
        zh_candidates = [data_dir / f"{name}.zh", data_dir / f"{name}.chn", data_dir / f"{name}.cn"]
        for en_path in en_candidates:
            for zh_path in zh_candidates:
                if en_path.exists() and zh_path.exists():
                    return en_path, zh_path
    return None


def _load_side_file_pairs(en_path, zh_path):
    src_lines = _read_text_lines(en_path)
    tgt_lines = _read_text_lines(zh_path)
    pairs = []
    for src, tgt in zip(src_lines, tgt_lines):
        pair = _normalise_pair(src, tgt)
        if pair:
            pairs.append(pair)
    return pairs


def load_local_split(data_dir, split_names):
    side_files = _find_parallel_side_files(data_dir, split_names)
    if side_files:
        return _load_side_file_pairs(*side_files)

    split_file = _find_split_file(data_dir, split_names)
    if split_file:
        return _load_pair_file(split_file)

    return []


def load_local_parallel_corpus(data_dir="data"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return None

    for corpus_dir in [data_dir, *sorted(path for path in data_dir.rglob("*") if path.is_dir())]:
        corpus = _load_local_parallel_corpus_from_dir(corpus_dir)
        if corpus is not None:
            return corpus

    return None


def _load_local_parallel_corpus_from_dir(data_dir):
    train_pairs = load_local_split(data_dir, TRAIN_NAMES)
    val_pairs = load_local_split(data_dir, VAL_NAMES)
    test_pairs = load_local_split(data_dir, TEST_NAMES)
    if train_pairs and val_pairs and test_pairs:
        return train_pairs, val_pairs, test_pairs
    if train_pairs and not val_pairs and not test_pairs:
        return train_pairs, [], []

    all_pairs = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in PARALLEL_EXTENSIONS:
            continue
        if path.stem.lower() in TRAIN_NAMES + VAL_NAMES + TEST_NAMES:
            continue
        all_pairs.extend(_load_pair_file(path))

    return (all_pairs, [], []) if all_pairs else None


def load_wmt17_en_zh(num_examples):
    from datasets import load_dataset

    ds = load_dataset("wmt/wmt17", "zh-en", split=f"train[:{num_examples}]")
    return [
        (item["translation"]["en"], item["translation"]["zh"])
        for item in ds
    ]


def ensure_split_sizes(train_pairs, val_pairs, test_pairs, train_size, val_size, test_size):
    if val_pairs and test_pairs:
        return (
            train_pairs[:train_size],
            val_pairs[:val_size],
            test_pairs[:test_size],
        )

    required = train_size + val_size + test_size
    if len(train_pairs) < required:
        raise ValueError(
            f"数据量不足: 当前 {len(train_pairs)} 条，固定划分需要 {required} 条 "
            f"({train_size}/{val_size}/{test_size})"
        )

    train_end = train_size
    val_end = train_size + val_size
    return (
        train_pairs[:train_end],
        train_pairs[train_end:val_end],
        train_pairs[val_end:required],
    )


def load_assignment_dataset(
    data_dir="data",
    train_size=18000,
    val_size=500,
    test_size=2636,
    fallback_to_hf=True,
):
    local_data = load_local_parallel_corpus(data_dir)
    if local_data is None:
        if not fallback_to_hf:
            raise FileNotFoundError(f"未在 {os.path.abspath(data_dir)} 找到本地平行语料")
        required = train_size + val_size + test_size
        all_pairs = load_wmt17_en_zh(required)
        local_data = (all_pairs, [], [])

    train_pairs, val_pairs, test_pairs = ensure_split_sizes(
        *local_data,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    def unzip(pairs):
        src_texts = [src for src, _ in pairs]
        tgt_texts = [tgt for _, tgt in pairs]
        return src_texts, tgt_texts

    train_src, train_tgt = unzip(train_pairs)
    val_src, val_tgt = unzip(val_pairs)
    test_src, test_tgt = unzip(test_pairs)
    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt
