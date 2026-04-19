"""
Hugging Face 预训练翻译模型评估脚本。
从 out/ 读取微调后的模型和测试集，计算 corpus BLEU 并打印样本结果。
"""

import json
import os

import torch
from sacrebleu import corpus_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianTokenizer

PATH = "out/"


def load_tokenizer(model_name_or_path):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as exc:
        print(f"AutoTokenizer 加载失败，回退到 MarianTokenizer: {exc}")
        return MarianTokenizer.from_pretrained(model_name_or_path)


class TranslationEvaluator:
    def __init__(
        self,
        model_path=PATH,
        config_path=PATH + "config.json",
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

    def evaluate_test_set(self, test_src_texts, test_tgt_texts, print_samples=20):
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)

        hypotheses = []
        references = []

        for i, (src_text, ref_text) in enumerate(zip(test_src_texts, test_tgt_texts)):
            hyp_text = self.generate_translation(src_text)
            hypotheses.append(hyp_text)
            references.append(ref_text)

            if i < print_samples:
                print(f"\n样本 {i + 1}")
                print(f"源文本: {src_text}")
                print(f"参考翻译: {ref_text}")
                print(f"模型翻译: {hyp_text}")

        bleu = corpus_bleu(hypotheses, [references], tokenize="zh").score

        print("\n" + "=" * 60)
        print("平均评估指标")
        print("=" * 60)
        print(f"Corpus BLEU 分数: {bleu:.4f}")
        print("=" * 60)

        return {
            "bleu": bleu,
            "num_samples": len(hypotheses),
        }


def main():
    test_path = PATH + "test_samples.json"
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            "未找到 out/test_samples.json，请先运行训练脚本生成测试集。"
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

    with open(PATH + "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n评估结果已保存到 out/evaluation_results.json")


if __name__ == "__main__":
    main()
