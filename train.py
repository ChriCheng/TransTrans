"""
Hugging Face 预训练翻译模型微调脚本（英 -> 中）
默认使用 MarianMT 作为基础模型，对 WMT17 的 en->zh 数据做微调。
"""

import json
import os
import random
import re
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

OUT_DIR = "out"
MODEL_DIR = os.path.join(OUT_DIR, "model")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
DATA_DIR = os.path.join(OUT_DIR, "data")
EVAL_DIR = os.path.join(OUT_DIR, "eval")
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis")
DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
for path in [OUT_DIR, MODEL_DIR, CHECKPOINT_DIR, DATA_DIR, EVAL_DIR, ANALYSIS_DIR]:
    os.makedirs(path, exist_ok=True)
disable_progress_bar()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_mostly_english(text, min_ratio=0.6):
    text = str(text).strip()
    if not text:
        return False
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    visible_chars = sum(not ch.isspace() for ch in text)
    return visible_chars > 0 and (ascii_letters / visible_chars) >= min_ratio


def looks_like_english_sentence(text):
    text = str(text).strip().lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    if len(tokens) < 6:
        return True

    common_english_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "that", "this", "these",
        "those", "to", "of", "in", "on", "for", "from", "with", "by", "as", "at",
        "is", "are", "was", "were", "be", "been", "being", "it", "its", "their",
        "his", "her", "they", "we", "you", "he", "she", "not", "will", "would",
        "can", "could", "should", "may", "might", "have", "has", "had", "do", "does",
        "did", "than", "which", "who", "what", "when", "where", "why", "how",
    }
    hit_count = sum(token in common_english_words for token in tokens)
    return hit_count >= max(1, len(tokens) // 8)


def is_mostly_chinese(text, min_ratio=0.3):
    text = str(text).strip()
    if not text:
        return False
    zh_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    visible_chars = sum(not ch.isspace() for ch in text)
    return visible_chars > 0 and (zh_chars / visible_chars) >= min_ratio


def clean_parallel_texts(src_texts, tgt_texts):
    cleaned_src = []
    cleaned_tgt = []
    seen = set()

    for src, tgt in zip(src_texts, tgt_texts):
        src = str(src).strip()
        tgt = str(tgt).strip()

        if not src or not tgt:
            continue
        if not is_mostly_english(src):
            continue
        if not looks_like_english_sentence(src):
            continue
        if not is_mostly_chinese(tgt):
            continue

        src_words = src.split()
        tgt_chars = [ch for ch in tgt if not ch.isspace()]

        if len(src_words) < 3 or len(tgt_chars) < 2:
            continue
        if len(src_words) > 128 or len(tgt_chars) > 160:
            continue

        ratio = len(tgt_chars) / max(len(src_words), 1)
        if ratio < 0.5 or ratio > 8.0:
            continue

        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)

        cleaned_src.append(src)
        cleaned_tgt.append(tgt)

    return cleaned_src, cleaned_tgt


def load_wmt17_en_zh(num_examples=8000):
    ds = load_dataset("wmt/wmt17", "zh-en", split=f"train[:{num_examples}]")

    raw_src_texts = [item["translation"]["en"] for item in ds]
    raw_tgt_texts = [item["translation"]["zh"] for item in ds]

    print(f"原始样本数: {len(raw_src_texts)}")
    src_texts, tgt_texts = clean_parallel_texts(raw_src_texts, raw_tgt_texts)
    print(f"清洗后样本数: {len(src_texts)}")

    return src_texts, tgt_texts


def split_parallel_texts(src_texts, tgt_texts, train_ratio=0.8, val_ratio=0.1, seed=42):
    assert len(src_texts) == len(tgt_texts), "源语言和目标语言长度不一致"

    indices = list(range(len(src_texts)))
    random.Random(seed).shuffle(indices)

    src_texts = [src_texts[i] for i in indices]
    tgt_texts = [tgt_texts[i] for i in indices]

    n = len(src_texts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_src = src_texts[:train_end]
    train_tgt = tgt_texts[:train_end]
    val_src = src_texts[train_end:val_end]
    val_tgt = tgt_texts[train_end:val_end]
    test_src = src_texts[val_end:]
    test_tgt = tgt_texts[val_end:]

    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def meteor_score_zh(reference, hypothesis):
    ref_tokens = [ch for ch in reference.strip() if not ch.isspace()]
    hyp_tokens = [ch for ch in hypothesis.strip() if not ch.isspace()]

    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    matches = 0
    hyp_match_flags = []
    for token in hyp_tokens:
        if ref_counts.get(token, 0) > 0:
            ref_counts[token] -= 1
            matches += 1
            hyp_match_flags.append(True)
        else:
            hyp_match_flags.append(False)

    if matches == 0:
        return 0.0

    precision = matches / len(hyp_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) > 0 else 0.0

    chunks = 0
    in_chunk = False
    for flag in hyp_match_flags:
        if flag and not in_chunk:
            chunks += 1
            in_chunk = True
        elif not flag:
            in_chunk = False

    penalty = 0.5 * ((chunks / matches) ** 3) if matches > 0 else 0.0
    return (1 - penalty) * f_mean


def save_training_history(log_history):
    save_json(log_history, os.path.join(ANALYSIS_DIR, "training_history.json"))


def plot_metric_curve(x_values, y_values, title, xlabel, ylabel, output_path, color):
    if not x_values or not y_values:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2, markersize=4, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_data_length_distribution(train_src, val_src, test_src, train_tgt, val_tgt, test_tgt):
    src_lengths = {
        "Train EN": [len(text.split()) for text in train_src],
        "Val EN": [len(text.split()) for text in val_src],
        "Test EN": [len(text.split()) for text in test_src],
    }
    tgt_lengths = {
        "Train ZH": [len([ch for ch in text if not ch.isspace()]) for text in train_tgt],
        "Val ZH": [len([ch for ch in text if not ch.isspace()]) for text in val_tgt],
        "Test ZH": [len([ch for ch in text if not ch.isspace()]) for text in test_tgt],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, values in src_lengths.items():
        axes[0].hist(values, bins=30, alpha=0.45, label=label)
    axes[0].set_title("Source Length Distribution")
    axes[0].set_xlabel("Token Count")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    for label, values in tgt_lengths.items():
        axes[1].hist(values, bins=30, alpha=0.45, label=label)
    axes[1].set_title("Target Length Distribution")
    axes[1].set_xlabel("Character Count")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "data_length_distribution.png"), dpi=180)
    plt.close()


def generate_training_plots(log_history):
    train_logs = [
        log for log in log_history
        if "loss" in log and "epoch" in log and "eval_loss" not in log
    ]
    eval_logs = [
        log for log in log_history
        if "eval_loss" in log and "epoch" in log
    ]

    plot_metric_curve(
        [log["epoch"] for log in train_logs],
        [log["loss"] for log in train_logs],
        "Training Loss Curve",
        "Epoch",
        "Loss",
        os.path.join(ANALYSIS_DIR, "train_loss_curve.png"),
        "#1f77b4",
    )
    plot_metric_curve(
        [log["epoch"] for log in eval_logs],
        [log["eval_loss"] for log in eval_logs],
        "Validation Loss Curve",
        "Epoch",
        "Loss",
        os.path.join(ANALYSIS_DIR, "val_loss_curve.png"),
        "#d62728",
    )
    plot_metric_curve(
        [log["epoch"] for log in eval_logs if "eval_bleu" in log],
        [log["eval_bleu"] for log in eval_logs if "eval_bleu" in log],
        "Validation BLEU Curve",
        "Epoch",
        "BLEU",
        os.path.join(ANALYSIS_DIR, "val_bleu_curve.png"),
        "#2ca02c",
    )


def load_tokenizer(model_name_or_path):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as exc:
        print(f"AutoTokenizer 加载失败，回退到 MarianTokenizer: {exc}")
        return MarianTokenizer.from_pretrained(model_name_or_path)


def build_hf_dataset(src_texts, tgt_texts):
    return Dataset.from_dict({"src_text": src_texts, "tgt_text": tgt_texts})


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    num_examples: int = 8000
    max_source_length: int = 128
    max_target_length: int = 160
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    generation_max_length: int = 160
    generation_num_beams: int = 4
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.2
    seed: int = 42


def preprocess_dataset(dataset, tokenizer, max_source_length, max_target_length):
    def preprocess_batch(batch):
        model_inputs = tokenizer(
            batch["src_text"],
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["tgt_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def compute_metrics_builder(tokenizer):
    def zh_char_tokens(text):
        return [ch for ch in text.strip() if not ch.isspace()]

    def get_ngrams(tokens, n):
        return {
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        }

    def rouge_n_f1(reference, hypothesis, n):
        ref_tokens = zh_char_tokens(reference)
        hyp_tokens = zh_char_tokens(hypothesis)

        if len(ref_tokens) < n or len(hyp_tokens) < n:
            return 0.0

        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        overlap = len(ref_ngrams & hyp_ngrams)

        if overlap == 0:
            return 0.0

        precision = overlap / len(hyp_ngrams)
        recall = overlap / len(ref_ngrams)
        return 2 * precision * recall / (precision + recall)

    def rouge_l_f1(reference, hypothesis):
        ref_tokens = zh_char_tokens(reference)
        hyp_tokens = zh_char_tokens(hypothesis)

        if not ref_tokens or not hyp_tokens:
            return 0.0

        dp = [[0] * (len(hyp_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
        for i in range(1, len(ref_tokens) + 1):
            for j in range(1, len(hyp_tokens) + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[-1][-1]
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        bleu = corpus_bleu(decoded_preds, [decoded_labels], tokenize="zh").score
        chrf = corpus_chrf(decoded_preds, [decoded_labels]).score
        rouge_1 = float(np.mean([
            rouge_n_f1(label, pred, 1) for pred, label in zip(decoded_preds, decoded_labels)
        ]))
        rouge_2 = float(np.mean([
            rouge_n_f1(label, pred, 2) for pred, label in zip(decoded_preds, decoded_labels)
        ]))
        rouge_l = float(np.mean([
            rouge_l_f1(label, pred) for pred, label in zip(decoded_preds, decoded_labels)
        ]))
        meteor = float(np.mean([
            meteor_score_zh(label, pred)
            for pred, label in zip(decoded_preds, decoded_labels)
        ]))
        exact_match = float(
            np.mean([pred == label for pred, label in zip(decoded_preds, decoded_labels)])
        )

        return {
            "bleu": bleu,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": meteor,
            "chrf": chrf,
            "exact_match": exact_match,
        }

    return compute_metrics


def train_model(config: TrainConfig):
    print(f"加载基础模型: {config.model_name}")
    tokenizer = load_tokenizer(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    print("加载 WMT17 英->中数据...")
    src_texts, tgt_texts = load_wmt17_en_zh(num_examples=config.num_examples)
    print(f"总样本数: {len(src_texts)}")

    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = split_parallel_texts(
        src_texts, tgt_texts, train_ratio=0.8, val_ratio=0.1, seed=config.seed
    )

    print(f"训练集: {len(train_src)}")
    print(f"验证集: {len(val_src)}")
    print(f"测试集: {len(test_src)}")

    train_dataset = build_hf_dataset(train_src, train_tgt)
    val_dataset = build_hf_dataset(val_src, val_tgt)

    tokenized_train = preprocess_dataset(
        train_dataset,
        tokenizer,
        config.max_source_length,
        config.max_target_length,
    )
    tokenized_val = preprocess_dataset(
        val_dataset,
        tokenizer,
        config.max_source_length,
        config.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams,
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        report_to=[],
        seed=config.seed,
        disable_tqdm=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    save_json(
        {"test_src_texts": test_src, "test_tgt_texts": test_tgt},
        os.path.join(DATA_DIR, "test_samples.json"),
    )
    save_json(
        {
            "dataset": "wmt/wmt17",
            "config_name": "zh-en",
            "direction": "en->zh",
            "num_examples": config.num_examples,
            "train_size": len(train_src),
            "val_size": len(val_src),
            "test_size": len(test_src),
        },
        os.path.join(DATA_DIR, "split_meta.json"),
    )
    save_json(asdict(config), os.path.join(ANALYSIS_DIR, "run_config.json"))
    plot_data_length_distribution(train_src, val_src, test_src, train_tgt, val_tgt, test_tgt)

    print("\n开始微调...")
    train_result = trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    save_json(dict(train_result.metrics), os.path.join(ANALYSIS_DIR, "train_metrics.json"))
    save_training_history(trainer.state.log_history)
    save_json(asdict(trainer.state), os.path.join(ANALYSIS_DIR, "trainer_state.json"))
    generate_training_plots(trainer.state.log_history)
    print("\n微调结束，最佳模型已保存。")


def main():
    config = TrainConfig()
    set_seed(config.seed)
    train_model(config)


if __name__ == "__main__":
    main()
