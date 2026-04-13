"""
模型评估脚本
包含BLEU、ROUGE等评估指标
"""

import torch
import json
from transformer_model import TransformerTranslator
from train import TranslationDataset, create_vocabularies
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import math


class TranslationEvaluator:
    """
    翻译模型评估类
    """
    def __init__(self, model_path='model.pth', src_vocab_path='src_vocab.json', 
                 tgt_vocab_path='tgt_vocab.json', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表
        with open(src_vocab_path, 'r', encoding='utf-8') as f:
            self.src_vocab = json.load(f)
        with open(tgt_vocab_path, 'r', encoding='utf-8') as f:
            self.tgt_vocab = json.load(f)
        
        # 创建反向词汇表
        self.tgt_vocab_inv = {v: k for k, v in self.tgt_vocab.items()}
        
        # 加载模型
        self.model = TransformerTranslator(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_ff=1024,
            dropout=0.1
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def compute_bleu(self, reference, hypothesis, max_n=4):
        """
        计算BLEU分数
        BLEU: Bilingual Evaluation Understudy
        衡量生成文本与参考文本的n-gram重合度
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        # 计算n-gram精度
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0)
                continue
            
            # 计算匹配的n-gram数
            matches = sum((hyp_ngrams & ref_ngrams).values())
            precisions.append(matches / len(hyp_ngrams))
        
        # 计算几何平均
        if min(precisions) == 0:
            return 0
        
        log_precisions = [math.log(p) for p in precisions]
        geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
        
        # 计算长度惩罚
        if len(hyp_tokens) > len(ref_tokens):
            bp = 1
        else:
            bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens)) if len(hyp_tokens) > 0 else 0
        
        bleu = bp * geo_mean
        return bleu
    
    def _get_ngrams(self, tokens, n):
        """
        获取n-gram
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def compute_rouge_l(self, reference, hypothesis):
        """
        计算ROUGE-L分数
        ROUGE: Recall-Oriented Understudy for Gisting Evaluation
        基于最长公共子序列(LCS)
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        # 计算LCS长度
        lcs_length = self._lcs_length(ref_tokens, hyp_tokens)
        
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0
        
        # 计算召回率和精度
        recall = lcs_length / len(ref_tokens)
        precision = lcs_length / len(hyp_tokens)
        
        if recall + precision == 0:
            return 0
        
        # F-score
        f_score = 2 * (recall * precision) / (recall + precision)
        return f_score
    
    def _lcs_length(self, seq1, seq2):
        """
        计算最长公共子序列长度
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_meteor(self, reference, hypothesis):
        """
        计算METEOR分数
        METEOR: Metric for Evaluation of Translation with Explicit ORdering
        考虑词序和同义词
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        # 精确匹配
        matches = sum(1 for token in hyp_tokens if token in ref_tokens)
        
        if len(hyp_tokens) == 0:
            return 0
        
        precision = matches / len(hyp_tokens)
        recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        if precision + recall == 0:
            return 0
        
        f_score = (precision * recall) / (0.9 * precision + 0.1 * recall)
        
        # 计算词序惩罚
        chunks = self._count_chunks(ref_tokens, hyp_tokens)
        penalty = 0.5 * (chunks / max(len(ref_tokens), len(hyp_tokens)))
        
        meteor = (1 - penalty) * f_score
        return meteor
    
    def _count_chunks(self, ref, hyp):
        """
        计算块数（用于词序惩罚）
        """
        ref_set = set(ref)
        hyp_set = set(hyp)
        
        chunks = 0
        in_chunk = False
        
        for token in hyp:
            if token in ref_set:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            else:
                in_chunk = False
        
        return chunks
    
    def evaluate_test_set(self, test_src_texts, test_tgt_texts):
        """
        在测试集上评估模型
        """
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)
        
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        
        for src_text, ref_text in zip(test_src_texts, test_tgt_texts):
            # 生成翻译
            hyp_text = self._generate_translation(src_text)
            
            # 计算评估指标
            bleu = self.compute_bleu(ref_text, hyp_text)
            rouge = self.compute_rouge_l(ref_text, hyp_text)
            meteor = self.compute_meteor(ref_text, hyp_text)
            
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            meteor_scores.append(meteor)
            
            print(f"\n源文本: {src_text}")
            print(f"参考翻译: {ref_text}")
            print(f"模型翻译: {hyp_text}")
            print(f"BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}, METEOR: {meteor:.4f}")
        
        # 计算平均分数
        avg_bleu = np.mean(bleu_scores)
        avg_rouge = np.mean(rouge_scores)
        avg_meteor = np.mean(meteor_scores)
        
        print("\n" + "=" * 60)
        print("平均评估指标")
        print("=" * 60)
        print(f"平均 BLEU 分数: {avg_bleu:.4f}")
        print(f"平均 ROUGE-L 分数: {avg_rouge:.4f}")
        print(f"平均 METEOR 分数: {avg_meteor:.4f}")
        print("=" * 60)
        
        return {
            'bleu': avg_bleu,
            'rouge': avg_rouge,
            'meteor': avg_meteor
        }
    
    def _generate_translation(self, src_text, max_length=100):
        """
        生成翻译（贪心解码）
        """
        src_indices = self._text_to_indices(src_text)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            encoder_output = self.model.encoder(src_tensor)
        
        tgt_indices = [self.tgt_vocab.get('<SOS>', 2)]
        
        for _ in range(max_length):
            tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long).to(self.device)
            
            while tgt_tensor.shape[1] < src_tensor.shape[1]:
                tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[self.tgt_vocab.get('<PAD>', 0)]], device=self.device)], dim=1)
            
            with torch.no_grad():
                decoder_output = self.model.decoder(tgt_tensor, encoder_output)
                logits = self.model.output_layer(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).item()
            
            tgt_indices.append(next_token)
            
            if next_token == self.tgt_vocab.get('<EOS>', 3):
                break
        
        return self._indices_to_text(tgt_indices)
    
    def _text_to_indices(self, text):
        """
        文本转indices
        """
        tokens = text.lower().split()
        indices = [self.src_vocab.get(token, self.src_vocab.get('<UNK>', 1)) for token in tokens]
        
        if len(indices) > 99:
            indices = indices[:99]
        indices.append(self.src_vocab.get('<EOS>', 3))
        
        while len(indices) < 100:
            indices.append(self.src_vocab.get('<PAD>', 0))
        
        return indices[:100]
    
    def _indices_to_text(self, indices):
        """
        indices转文本
        """
        tokens = []
        for idx in indices:
            if idx in self.tgt_vocab_inv:
                token = self.tgt_vocab_inv[idx]
                if token not in ['<PAD>', '<EOS>', '<SOS>']:
                    tokens.append(token)
        return ' '.join(tokens)


def main():
    """
    主函数
    """
    # 测试数据
    test_src_texts = [
        "hello world",
        "good morning",
        "how are you",
        "thank you very much",
        "i love machine learning",
        "transformer is powerful",
        "neural networks are amazing"
    ]
    
    test_tgt_texts = [
        "你好 世界",
        "早上 好",
        "你 好 吗",
        "非常 感谢 你",
        "我 喜欢 机器 学习",
        "transformer 很 强大",
        "神经 网络 很 棒"
    ]
    
    # 创建评估器
    evaluator = TranslationEvaluator()
    
    # 评估模型
    metrics = evaluator.evaluate_test_set(test_src_texts, test_tgt_texts)
    
    # 保存评估结果
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\n评估结果已保存到 evaluation_results.json")


if __name__ == "__main__":
    main()
