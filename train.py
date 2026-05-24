"""
Hugging Face 预训练翻译模型微调脚本（英 -> 中）
默认使用 MarianMT 作为基础模型，对课程英译中数据集做微调。
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataload import load_assignment_dataset
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


class ProgressEtaCallback(TrainerCallback):
    def __init__(self, log_steps=50):
        self.log_steps = max(1, int(log_steps))
        self.start_time = None
        self.last_log_time = None
        self.last_logged_step = -1

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        self.last_logged_step = -1
        total_steps = state.max_steps or 0
        print(f"[progress] total_steps={total_steps}, logging_every={self.log_steps} steps", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.start_time or not state.max_steps:
            return
        if state.global_step <= 0 or state.global_step == self.last_logged_step:
            return
        if state.global_step % self.log_steps != 0 and state.global_step != state.max_steps:
            return

        now = time.monotonic()
        elapsed = now - self.start_time
        segment_elapsed = now - self.last_log_time if self.last_log_time else elapsed
        steps_per_second = state.global_step / elapsed if elapsed > 0 else 0.0
        percent = state.global_step / state.max_steps * 100
        epoch = state.epoch if state.epoch is not None else 0.0

        print(
            "[progress] "
            f"step={state.global_step}/{state.max_steps} "
            f"({percent:.2f}%), "
            f"epoch={epoch:.4f}, "
            f"speed={steps_per_second:.3f} step/s, "
            f"elapsed={format_duration(elapsed)}, "
            f"segment_elapsed={format_duration(segment_elapsed)}",
            flush=True,
        )
        self.last_logged_step = state.global_step
        self.last_log_time = now


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
    data_dir: str = "data"
    train_size: int = 18000
    val_size: int = 500
    test_size: int = 2636
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
    progress_log_steps: int = 50
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

    print("加载英->中课程作业数据...")
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = load_assignment_dataset(
        data_dir=config.data_dir,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
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
        callbacks=[ProgressEtaCallback(config.progress_log_steps)],
    )

    save_json(
        {"test_src_texts": test_src, "test_tgt_texts": test_tgt},
        os.path.join(DATA_DIR, "test_samples.json"),
    )
    save_json(
        {
            "dataset": "coursework_en_zh",
            "data_dir": config.data_dir,
            "direction": "en->zh",
            "expected_train_size": config.train_size,
            "expected_val_size": config.val_size,
            "expected_test_size": config.test_size,
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
