"""
基于Transformer的中英互译模型实现
包含：位置编码、多头注意力、前馈网络、编码器和解码器
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码层
    用于向模型提供序列中各个token的位置信息
    """
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        Returns:
            x + 位置编码
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    允许模型在不同的表示子空间中关注来自不同位置的信息
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch_size, seq_length, d_model)
            mask: 注意力掩码
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        batch_size = Q.shape[0]
        
        # 线性变换并分割为多个头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用缩放点积注意力
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 连接多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(attn_output)
        return output


class FeedForwardNetwork(nn.Module):
    """
    前馈网络
    包含两个线性层和一个ReLU激活函数
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含多头自注意力和前馈网络，各层后跟层归一化和残差连接
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            mask: 注意力掩码
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含自注意力、交叉注意力和前馈网络
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: 目标序列 (batch_size, seq_length, d_model)
            encoder_output: 编码器输出 (batch_size, src_seq_length, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（用于防止解码器看到未来的token）
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        # 自注意力 + 残差连接 + 层归一化
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力 + 残差连接 + 层归一化
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    Transformer编码器
    由多个编码器层堆叠而成
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length) - token indices
            mask: 注意力掩码
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Decoder(nn.Module):
    """
    Transformer解码器
    由多个解码器层堆叠而成
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length=512, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch_size, seq_length) - target token indices
            encoder_output: (batch_size, src_seq_length, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class TransformerTranslator(nn.Module):
    """
    完整的Transformer翻译模型
    包含编码器、解码器和输出层
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_seq_length=512, dropout=0.1):
        super(TransformerTranslator, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, src_seq_length) - 源语言token indices
            tgt: (batch_size, tgt_seq_length) - 目标语言token indices
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            output: (batch_size, tgt_seq_length, tgt_vocab_size) - 预测的token概率分布
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_layer(decoder_output)
        
        return output
    
    def generate_tgt_mask(self, tgt_seq_length, device):
        """
        生成目标序列的因果掩码，防止解码器看到未来的token
        """
        mask = torch.triu(torch.ones(tgt_seq_length, tgt_seq_length, device=device), diagonal=1)
        return (1 - mask).unsqueeze(0).unsqueeze(0)
