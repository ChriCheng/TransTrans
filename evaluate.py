"""
Hugging Face 预训练翻译模型评估脚本。
从 out/ 读取微调后的模型和测试集，计算 corpus BLEU 并打印样本结果。
"""

import json
import os

import torch
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianTokenizer

OUT_DIR = "out"
MODEL_DIR = os.path.join(OUT_DIR, "model")
DATA_DIR = os.path.join(OUT_DIR, "data")
EVAL_DIR = os.path.join(OUT_DIR, "eval")
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis")


def load_tokenizer(model_name_or_path):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as exc:
        print(f"AutoTokenizer 加载失败，回退到 MarianTokenizer: {exc}")
        return MarianTokenizer.from_pretrained(model_name_or_path)


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


class TranslationEvaluator:
    def __init__(
        self,
        model_path=MODEL_DIR,
        config_path=os.path.join(ANALYSIS_DIR, "run_config.json"),
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        self.max_source_length = self.config.get("max_source_length", 128)
        self.generation_max_length = self.config.get("generation_max_length", 160)
        self.generation_num_beams = self.config.get("generation_num_beams", 4)
        self.no_repeat_ngram_size = self.config.get("no_repeat_ngram_size", 3)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.2)

        self.tokenizer = load_tokenizer(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate_translation(self, src_text):
        inputs = self.tokenizer(
            src_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_length=self.generation_max_length,
                num_beams=self.generation_num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                early_stopping=True,
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    @staticmethod
    def zh_char_tokens(text):
        return [ch for ch in text.strip() if not ch.isspace()]

    @staticmethod
    def get_ngrams(tokens, n):
        return {
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        }

    def rouge_n_f1(self, reference, hypothesis, n):
        ref_tokens = self.zh_char_tokens(reference)
        hyp_tokens = self.zh_char_tokens(hypothesis)

        if len(ref_tokens) < n or len(hyp_tokens) < n:
            return 0.0

        ref_ngrams = self.get_ngrams(ref_tokens, n)
        hyp_ngrams = self.get_ngrams(hyp_tokens, n)
        overlap = len(ref_ngrams & hyp_ngrams)

        if overlap == 0:
            return 0.0

        precision = overlap / len(hyp_ngrams)
        recall = overlap / len(ref_ngrams)
        return 2 * precision * recall / (precision + recall)

    def rouge_l_f1(self, reference, hypothesis):
        ref_tokens = self.zh_char_tokens(reference)
        hyp_tokens = self.zh_char_tokens(hypothesis)

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

    def evaluate_test_set(self, test_src_texts, test_tgt_texts, print_samples=20):
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)

        hypotheses = []
        references = []
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        meteor_scores = []

        for i, (src_text, ref_text) in enumerate(zip(test_src_texts, test_tgt_texts)):
            hyp_text = self.generate_translation(src_text)
            hypotheses.append(hyp_text)
            references.append(ref_text)
            rouge_1_scores.append(self.rouge_n_f1(ref_text, hyp_text, 1))
            rouge_2_scores.append(self.rouge_n_f1(ref_text, hyp_text, 2))
            rouge_l_scores.append(self.rouge_l_f1(ref_text, hyp_text))
            meteor_scores.append(meteor_score_zh(ref_text, hyp_text))

            if i < print_samples:
                print(f"\n样本 {i + 1}")
                print(f"源文本: {src_text}")
                print(f"参考翻译: {ref_text}")
                print(f"模型翻译: {hyp_text}")

        bleu = corpus_bleu(hypotheses, [references], tokenize="zh").score
        chrf = corpus_chrf(hypotheses, [references]).score
        rouge_1 = float(np.mean(rouge_1_scores))
        rouge_2 = float(np.mean(rouge_2_scores))
        rouge_l = float(np.mean(rouge_l_scores))
        meteor = float(np.mean(meteor_scores))

        print("\n" + "=" * 60)
        print("平均评估指标")
        print("=" * 60)
        print(f"Corpus BLEU 分数: {bleu:.4f}")
        print(f"ROUGE-1 F1 分数: {rouge_1:.4f}")
        print(f"ROUGE-2 F1 分数: {rouge_2:.4f}")
        print(f"ROUGE-L F1 分数: {rouge_l:.4f}")
        print(f"METEOR 分数: {meteor:.4f}")
        print(f"chrF 分数: {chrf:.4f}")
        print("=" * 60)

        return {
            "bleu": bleu,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": meteor,
            "chrf": chrf,
            "num_samples": len(hypotheses),
        }


def main():
    test_path = os.path.join(DATA_DIR, "test_samples.json")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            "未找到 out/data/test_samples.json，请先运行训练脚本生成测试集。"
        )

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_src_texts = test_data["test_src_texts"]
    test_tgt_texts = test_data["test_tgt_texts"]

    evaluator = TranslationEvaluator()
    metrics = evaluator.evaluate_test_set(
        test_src_texts=test_src_texts,
        test_tgt_texts=test_tgt_texts,
        print_samples=20,
    )

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n评估结果已保存到 out/eval/evaluation_results.json")


if __name__ == "__main__":
    main()
