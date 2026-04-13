"""
Transformer翻译模型的推理脚本
包含贪心解码、束搜索等推理方法
"""

import torch
import json
from transformer_model import TransformerTranslator
import os


class TranslationInference:
    """
    翻译推理类
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
        self.src_vocab_inv = {v: k for k, v in self.src_vocab.items()}
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
    
    def text_to_indices(self, text, vocab, max_length=100):
        """将文本转换为token indices"""
        tokens = text.lower().split()
        indices = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens]
        
        if len(indices) > max_length - 1:
            indices = indices[:max_length - 1]
        indices.append(vocab.get('<EOS>', 3))
        
        while len(indices) < max_length:
            indices.append(vocab.get('<PAD>', 0))
        
        return indices[:max_length]
    
    def indices_to_text(self, indices, vocab_inv):
        """将token indices转换为文本"""
        tokens = []
        for idx in indices:
            if idx in vocab_inv:
                token = vocab_inv[idx]
                if token in ['<PAD>', '<EOS>', '<SOS>']:
                    continue
                tokens.append(token)
        return ' '.join(tokens)
    
    def greedy_decode(self, src_text, max_length=100):
        """
        贪心解码：每一步选择概率最高的token
        """
        # 编码源文本
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # 编码器前向传播
        with torch.no_grad():
            encoder_output = self.model.encoder(src_tensor)
        
        # 初始化解码器输入
        tgt_indices = [self.tgt_vocab.get('<SOS>', 2)]
        
        # 逐步生成目标序列
        for _ in range(max_length):
            # 准备解码器输入
            tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long).to(self.device)
            
            # 填充到相同长度
            while tgt_tensor.shape[1] < src_tensor.shape[1]:
                tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[self.tgt_vocab.get('<PAD>', 0)]], device=self.device)], dim=1)
            
            with torch.no_grad():
                # 解码器前向传播
                decoder_output = self.model.decoder(tgt_tensor, encoder_output)
                
                # 获取最后一个位置的输出
                logits = self.model.output_layer(decoder_output[:, -1, :])
                
                # 选择概率最高的token
                next_token = torch.argmax(logits, dim=-1).item()
            
            tgt_indices.append(next_token)
            
            # 如果生成了EOS token，停止
            if next_token == self.tgt_vocab.get('<EOS>', 3):
                break
        
        # 转换为文本
        result_text = self.indices_to_text(tgt_indices, self.tgt_vocab_inv)
        return result_text
    
    def beam_search_decode(self, src_text, beam_width=5, max_length=100):
        """
        束搜索解码：保留top-k个最可能的序列
        """
        # 编码源文本
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # 编码器前向传播
        with torch.no_grad():
            encoder_output = self.model.encoder(src_tensor)
        
        # 初始化束搜索
        sequences = [[self.tgt_vocab.get('<SOS>', 2)]]
        scores = [0.0]
        
        for _ in range(max_length):
            all_candidates = []
            
            for i, seq in enumerate(sequences):
                # 准备解码器输入
                tgt_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
                
                # 填充到相同长度
                while tgt_tensor.shape[1] < src_tensor.shape[1]:
                    tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[self.tgt_vocab.get('<PAD>', 0)]], device=self.device)], dim=1)
                
                with torch.no_grad():
                    decoder_output = self.model.decoder(tgt_tensor, encoder_output)
                    logits = self.model.output_layer(decoder_output[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)[0]
                
                # 获取top-k个候选
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                for j, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
                    new_seq = seq + [idx.item()]
                    new_score = scores[i] + prob.item()
                    all_candidates.append((new_score, new_seq))
            
            # 选择得分最高的beam_width个序列
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            sequences = [seq for _, seq in all_candidates[:beam_width]]
            scores = [score for score, _ in all_candidates[:beam_width]]
            
            # 检查是否所有序列都已完成
            if all(seq[-1] == self.tgt_vocab.get('<EOS>', 3) for seq in sequences):
                break
        
        # 返回得分最高的序列
        best_seq = sequences[0]
        result_text = self.indices_to_text(best_seq, self.tgt_vocab_inv)
        return result_text
    
    def translate(self, src_text, method='greedy'):
        """
        翻译文本
        Args:
            src_text: 源文本
            method: 解码方法 ('greedy' 或 'beam_search')
        """
        if method == 'greedy':
            return self.greedy_decode(src_text)
        elif method == 'beam_search':
            return self.beam_search_decode(src_text)
        else:
            raise ValueError(f"未知的解码方法: {method}")


def demo_translation():
    """
    翻译演示
    """
    # 检查模型文件是否存在
    if not os.path.exists('model.pth'):
        print("错误: 模型文件不存在，请先运行train.py进行训练")
        return
    
    print("加载模型...")
    translator = TranslationInference()
    
    # 测试句子
    test_sentences = [
        "hello world",
        "good morning",
        "how are you",
        "thank you very much",
        "i love machine learning"
    ]
    
    print("\n=== 贪心解码 ===")
    for sentence in test_sentences:
        translation = translator.translate(sentence, method='greedy')
        print(f"源文本: {sentence}")
        print(f"翻译: {translation}\n")
    
    print("\n=== 束搜索解码 ===")
    for sentence in test_sentences[:3]:
        translation = translator.translate(sentence, method='beam_search')
        print(f"源文本: {sentence}")
        print(f"翻译: {translation}\n")


if __name__ == "__main__":
    demo_translation()
