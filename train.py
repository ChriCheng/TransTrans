"""
Transformer翻译模型的训练脚本
包含数据加载、模型训练、评估等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_model import TransformerTranslator
import numpy as np
from tqdm import tqdm
import os
import json
PATH = 'out/'

class TranslationDataset(Dataset):
    """
    翻译数据集类
    """
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_length=100):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # 文本转token indices
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        tgt_indices = self.text_to_indices(tgt_text, self.tgt_vocab)
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
    def text_to_indices(self, text, vocab):
        """将文本转换为token indices"""
        tokens = text.lower().split()
        indices = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
        
        # 截断或填充到max_length
        if len(indices) > self.max_length - 1:
            indices = indices[:self.max_length - 1]
        indices.append(vocab.get('<EOS>', 0))
        
        # 填充到max_length
        while len(indices) < self.max_length:
            indices.append(vocab.get('<PAD>', 0))
        
        return indices[:self.max_length]


def collate_batch(batch):
    """
    批处理数据的整理函数
    """
    src_batch = []
    tgt_batch = []
    
    for item in batch:
        src_batch.append(item['src'])
        tgt_batch.append(item['tgt'])
    
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    
    return src_batch, tgt_batch


def create_vocabularies(src_texts, tgt_texts, min_freq=1):
    """
    创建源语言和目标语言的词汇表
    """
    def build_vocab(texts):
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        word_freq = {}
        
        for text in texts:
            tokens = text.lower().split()
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        idx = 4
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1
        
        return vocab
    
    src_vocab = build_vocab(src_texts)
    tgt_vocab = build_vocab(tgt_texts)
    
    return src_vocab, tgt_vocab


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    for src_batch, tgt_batch in tqdm(dataloader, desc="Training"):
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        # 准备输入和目标
        tgt_input = tgt_batch[:, :-1]  # 去掉最后一个token
        tgt_target = tgt_batch[:, 1:]  # 去掉第一个token
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src_batch, tgt_input)
        
        # 计算损失
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_target.reshape(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dataloader, desc="Evaluating"):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            tgt_input = tgt_batch[:, :-1]
            tgt_target = tgt_batch[:, 1:]
            
            output = model(src_batch, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_target.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(src_texts, tgt_texts, num_epochs=10, batch_size=32, learning_rate=0.0001):
    """
    训练翻译模型的主函数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建词汇表
    print("创建词汇表...")
    src_vocab, tgt_vocab = create_vocabularies(src_texts, tgt_texts)
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    
    # 初始化模型
    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<PAD>'])
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    # 保存模型和词汇表
    torch.save(model.state_dict(), PATH+'model.pth')
    with open(PATH+'src_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(src_vocab, f, ensure_ascii=False)
    with open(PATH+'tgt_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(tgt_vocab, f, ensure_ascii=False)
    
    print("\n模型已保存!")
    
    return model, src_vocab, tgt_vocab, device


if __name__ == "__main__":
    # 示例训练数据
    src_texts = [
        "hello world",
        "good morning",
        "how are you",
        "thank you very much",
        "what is your name",
        "i love machine learning",
        "transformer is powerful",
        "neural networks are amazing"
    ]
    
    tgt_texts = [
        "你好 世界",
        "早上 好",
        "你 好 吗",
        "非常 感谢 你",
        "你 叫 什么 名字",
        "我 喜欢 机器 学习",
        "transformer 很 强大",
        "神经 网络 很 棒"
    ]
    
    model, src_vocab, tgt_vocab, device = train_model(src_texts, tgt_texts, num_epochs=20, batch_size=4)
