"""
Hugging Face 微调翻译模型推理脚本。
"""

import json
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianTokenizer

OUT_DIR = "out"
MODEL_DIR = os.path.join(OUT_DIR, "model")
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis")


def load_tokenizer(model_name_or_path):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as exc:
        print(f"AutoTokenizer 加载失败，回退到 MarianTokenizer: {exc}")
        return MarianTokenizer.from_pretrained(model_name_or_path)


class TranslationInference:
    def __init__(self, model_path=MODEL_DIR, config_path=os.path.join(ANALYSIS_DIR, "run_config.json"), device=None):
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

    def translate(self, src_text, num_beams=None, max_length=None):
        inputs = self.tokenizer(
            src_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_length=max_length or self.generation_max_length,
                num_beams=num_beams or self.generation_num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                repetition_penalty=self.repetition_penalty,
                early_stopping=True,
            )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def demo_translation():
    if not os.path.exists(MODEL_DIR):
        print("错误: out/model/ 目录不存在，请先运行 train.py 进行训练")
        return

    print("加载模型...")
    translator = TranslationInference()

    test_sentences = [
        "hello world",
        "good morning",
        "how are you",
        "thank you very much",
        "i love machine learning",
    ]

    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"源文本: {sentence}")
        print(f"翻译: {translation}\n")


if __name__ == "__main__":
    demo_translation()
