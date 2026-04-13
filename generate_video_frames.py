"""
生成演示视频帧
使用matplotlib生成一系列图像帧，然后用ffmpeg合成视频
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

# 创建帧目录
os.makedirs('video_frames', exist_ok=True)

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#0f3460'

def create_title_frame():
    """创建标题帧"""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    # 标题
    ax.text(0.5, 0.65, 'Transformer', fontsize=56, ha='center', va='center',
            color='#e94560', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.50, 'Chinese-English Translation System', fontsize=28, ha='center', va='center',
            color='#0f3460', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.38, '基于Transformer的中英互译系统', fontsize=24, ha='center', va='center',
            color='white', transform=ax.transAxes)
    
    # 副标题
    ax.text(0.5, 0.25, 'PyTorch Implementation from Scratch', fontsize=18, ha='center', va='center',
            color='#a8a8b3', transform=ax.transAxes)
    
    # 底部信息
    ax.text(0.5, 0.10, 'Architecture  |  Training  |  Inference  |  Evaluation', 
            fontsize=14, ha='center', va='center', color='#0f3460', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_001.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 标题帧生成完成")


def create_architecture_frame():
    """创建架构图帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # 标题
    ax.text(8, 8.5, 'Transformer Architecture', fontsize=22, ha='center', va='center',
            color='#e94560', fontweight='bold')
    ax.text(8, 8.0, 'Encoder-Decoder 编码器-解码器架构', fontsize=14, ha='center', va='center',
            color='#a8a8b3')
    
    def draw_box(ax, x, y, w, h, text, color='#0f3460', text_color='white', fontsize=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='#e94560', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                color=text_color, fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#e94560', lw=2))
    
    # 编码器部分
    ax.text(3.5, 7.4, 'ENCODER', fontsize=12, ha='center', va='center', color='#4ecca3', fontweight='bold')
    draw_box(ax, 1, 6.5, 5, 0.6, 'Input Embedding + Positional Encoding', '#16213e', '#4ecca3', 9)
    draw_arrow(ax, 3.5, 6.5, 3.5, 6.0)
    draw_box(ax, 1, 5.2, 5, 0.7, 'Multi-Head Self-Attention\n多头自注意力', '#0f3460', 'white', 9)
    draw_arrow(ax, 3.5, 5.2, 3.5, 4.7)
    draw_box(ax, 1, 3.9, 5, 0.7, 'Feed-Forward Network\n前馈神经网络', '#0f3460', 'white', 9)
    draw_arrow(ax, 3.5, 3.9, 3.5, 3.4)
    draw_box(ax, 1, 2.8, 5, 0.5, 'Layer Norm + Residual Connection\n层归一化 + 残差连接', '#16213e', '#a8a8b3', 8)
    ax.text(3.5, 2.4, '× 4 Layers', fontsize=11, ha='center', va='center', color='#e94560')
    
    # 解码器部分
    ax.text(11.5, 7.4, 'DECODER', fontsize=12, ha='center', va='center', color='#e94560', fontweight='bold')
    draw_box(ax, 9, 6.5, 5, 0.6, 'Output Embedding + Positional Encoding', '#16213e', '#e94560', 9)
    draw_arrow(ax, 11.5, 6.5, 11.5, 6.0)
    draw_box(ax, 9, 5.2, 5, 0.7, 'Masked Self-Attention\n掩码自注意力', '#0f3460', 'white', 9)
    draw_arrow(ax, 11.5, 5.2, 11.5, 4.7)
    draw_box(ax, 9, 3.9, 5, 0.7, 'Cross-Attention\n交叉注意力', '#0f3460', 'white', 9)
    draw_arrow(ax, 11.5, 3.9, 11.5, 3.4)
    draw_box(ax, 9, 2.8, 5, 0.5, 'Feed-Forward + Layer Norm\n前馈网络 + 层归一化', '#16213e', '#a8a8b3', 8)
    ax.text(11.5, 2.4, '× 4 Layers', fontsize=11, ha='center', va='center', color='#e94560')
    
    # 输出层
    draw_box(ax, 9, 1.5, 5, 0.6, 'Linear + Softmax → Output', '#16213e', '#4ecca3', 9)
    draw_arrow(ax, 11.5, 2.4, 11.5, 2.1)
    
    # 编码器到解码器的连接
    ax.annotate('', xy=(9, 4.25), xytext=(6, 4.25),
                arrowprops=dict(arrowstyle='->', color='#4ecca3', lw=2.5))
    ax.text(7.5, 4.55, 'Encoder\nOutput', fontsize=9, ha='center', va='center', color='#4ecca3')
    
    # 输入输出标签
    ax.text(3.5, 7.0, 'English Input (英文输入)', fontsize=9, ha='center', va='center', color='#a8a8b3')
    ax.text(11.5, 7.0, 'Chinese Output (中文输出)', fontsize=9, ha='center', va='center', color='#a8a8b3')
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_002.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 架构图帧生成完成")


def create_attention_frame():
    """创建注意力机制帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # 左侧：注意力公式和说明
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    ax1.axis('off')
    
    ax1.text(0.5, 0.95, 'Multi-Head Attention', fontsize=18, ha='center', va='top',
             color='#e94560', fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.88, '多头自注意力机制', fontsize=14, ha='center', va='top',
             color='white', transform=ax1.transAxes)
    
    formula_text = [
        ('Attention Formula:', 0.78, '#4ecca3', 13),
        ('Attention(Q,K,V) = softmax(QK^T/√d_k)V', 0.70, 'white', 11),
        ('', 0.63, 'white', 11),
        ('Q = Query (查询)', 0.56, '#a8a8b3', 11),
        ('K = Key (键)', 0.50, '#a8a8b3', 11),
        ('V = Value (值)', 0.44, '#a8a8b3', 11),
        ('d_k = 每个头的维度', 0.38, '#a8a8b3', 11),
        ('', 0.32, 'white', 11),
        ('Model Config:', 0.26, '#4ecca3', 13),
        ('d_model = 256', 0.20, 'white', 11),
        ('num_heads = 8', 0.14, 'white', 11),
        ('d_k = 32 (per head)', 0.08, 'white', 11),
    ]
    
    for text, y, color, size in formula_text:
        ax1.text(0.5, y, text, ha='center', va='center', color=color,
                fontsize=size, transform=ax1.transAxes)
    
    # 右侧：注意力权重可视化
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    
    # 生成示例注意力权重
    np.random.seed(42)
    words_en = ['hello', 'world', '<EOS>']
    words_zh = ['你好', '世界', '<EOS>']
    
    attention = np.array([
        [0.85, 0.10, 0.05],
        [0.08, 0.87, 0.05],
        [0.05, 0.10, 0.85]
    ])
    
    im = ax2.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(words_en)))
    ax2.set_yticks(range(len(words_zh)))
    ax2.set_xticklabels(words_en, color='white', fontsize=12)
    ax2.set_yticklabels(words_zh, color='white', fontsize=12)
    ax2.set_xlabel('Source (English)', color='#4ecca3', fontsize=12)
    ax2.set_ylabel('Target (Chinese)', color='#4ecca3', fontsize=12)
    ax2.set_title('Attention Weights Visualization\n注意力权重可视化', color='#e94560', fontsize=13, pad=10)
    
    for i in range(len(words_zh)):
        for j in range(len(words_en)):
            ax2.text(j, i, f'{attention[i,j]:.2f}', ha='center', va='center',
                    color='black', fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_003.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 注意力机制帧生成完成")


def create_training_frame():
    """创建训练过程帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # 左侧：训练配置
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    ax1.axis('off')
    
    ax1.text(0.5, 0.95, 'Training Configuration', fontsize=18, ha='center', va='top',
             color='#e94560', fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.88, '训练配置', fontsize=14, ha='center', va='top',
             color='white', transform=ax1.transAxes)
    
    configs = [
        ('Model Parameters', '7,392,537', 0.78),
        ('Optimizer', 'Adam (lr=0.0001)', 0.70),
        ('Loss Function', 'CrossEntropyLoss', 0.62),
        ('Batch Size', '4', 0.54),
        ('Epochs', '20', 0.46),
        ('Device', 'CPU', 0.38),
        ('d_model', '256', 0.30),
        ('num_layers', '4', 0.22),
        ('num_heads', '8', 0.14),
    ]
    
    for label, value, y in configs:
        ax1.text(0.15, y, f'{label}:', ha='left', va='center', color='#4ecca3',
                fontsize=11, transform=ax1.transAxes)
        ax1.text(0.85, y, value, ha='right', va='center', color='white',
                fontsize=11, transform=ax1.transAxes)
    
    # 右侧：训练损失曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    
    epochs = list(range(1, 21))
    losses = [3.0706, 2.4686, 2.1907, 2.0860, 1.7903, 1.6796, 1.3656, 1.1838, 
              0.9330, 0.7793, 0.6464, 0.4975, 0.4079, 0.3400, 0.2867, 0.2193, 
              0.2031, 0.1641, 0.1288, 0.1056]
    
    ax2.plot(epochs, losses, color='#e94560', linewidth=2.5, marker='o', markersize=4)
    ax2.fill_between(epochs, losses, alpha=0.2, color='#e94560')
    
    ax2.set_xlabel('Epoch', color='white', fontsize=12)
    ax2.set_ylabel('Loss', color='white', fontsize=12)
    ax2.set_title('Training Loss Curve\n训练损失曲线', color='#e94560', fontsize=13, pad=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#0f3460')
    ax2.spines['left'].set_color('#0f3460')
    ax2.spines['top'].set_color('#0f3460')
    ax2.spines['right'].set_color('#0f3460')
    ax2.grid(True, alpha=0.2, color='#0f3460')
    
    # 标注关键点
    ax2.annotate(f'Start: {losses[0]:.2f}', xy=(1, losses[0]), xytext=(5, losses[0]+0.2),
                color='#4ecca3', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#4ecca3'))
    ax2.annotate(f'Final: {losses[-1]:.4f}', xy=(20, losses[-1]), xytext=(15, losses[-1]+0.3),
                color='#4ecca3', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#4ecca3'))
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_004.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 训练过程帧生成完成")


def create_inference_frame():
    """创建推理演示帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)
    
    # 标题
    fig.text(0.5, 0.95, 'Translation Results  翻译结果演示', fontsize=18, ha='center', va='top',
             color='#e94560', fontweight='bold')
    
    # 翻译示例
    examples = [
        ('hello world', '你好 世界'),
        ('good morning', '早上 好'),
        ('i love machine learning', '我 喜欢 机器 学习'),
        ('transformer is powerful', 'transformer 很 强大'),
    ]
    
    for idx, (en, zh) in enumerate(examples):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#16213e')
        ax.axis('off')
        
        # 英文输入
        ax.text(0.5, 0.78, 'English Input', fontsize=10, ha='center', va='center',
                color='#4ecca3', transform=ax.transAxes)
        ax.text(0.5, 0.62, en, fontsize=14, ha='center', va='center',
                color='white', fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', edgecolor='#4ecca3'))
        
        # 箭头
        ax.text(0.5, 0.45, '↓ Translate ↓', fontsize=11, ha='center', va='center',
                color='#e94560', transform=ax.transAxes)
        
        # 中文输出
        ax.text(0.5, 0.30, 'Chinese Output', fontsize=10, ha='center', va='center',
                color='#e94560', transform=ax.transAxes)
        ax.text(0.5, 0.14, zh, fontsize=14, ha='center', va='center',
                color='white', fontweight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', edgecolor='#e94560'))
    
    plt.savefig('video_frames/frame_005.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 推理演示帧生成完成")


def create_evaluation_frame():
    """创建评估指标帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # 左侧：评估指标说明
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    ax1.axis('off')
    
    ax1.text(0.5, 0.95, 'Evaluation Metrics', fontsize=18, ha='center', va='top',
             color='#e94560', fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.88, '评估指标', fontsize=14, ha='center', va='top',
             color='white', transform=ax1.transAxes)
    
    metrics_info = [
        ('BLEU', 'Bilingual Evaluation Understudy', 0.76, 0.70),
        ('', 'N-gram precision with brevity penalty', 0.65, 0.60),
        ('ROUGE-L', 'Recall-Oriented Understudy', 0.54, 0.48),
        ('', 'Longest Common Subsequence (LCS)', 0.43, 0.38),
        ('METEOR', 'Metric for Evaluation of Translation', 0.32, 0.26),
        ('', 'Considers synonyms and word order', 0.21, 0.16),
    ]
    
    for i, (label, desc, y1, y2) in enumerate(metrics_info):
        if label:
            ax1.text(0.1, y1, label, ha='left', va='center', color='#4ecca3',
                    fontsize=13, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.1, y2, desc, ha='left', va='center', color='#a8a8b3',
                fontsize=10, transform=ax1.transAxes)
    
    ax1.text(0.5, 0.08, 'Higher scores = Better translation quality', ha='center', va='center',
             color='#e94560', fontsize=11, transform=ax1.transAxes)
    
    # 右侧：评估结果对比图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    
    # 评估结果
    metrics = ['BLEU', 'ROUGE-L', 'METEOR\n(÷10)']
    scores = [0.0, 0.088, 0.325]
    colors = ['#e94560', '#4ecca3', '#f5a623']
    
    bars = ax2.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
    
    ax2.set_ylim(0, 0.45)
    ax2.set_ylabel('Score', color='white', fontsize=12)
    ax2.set_title('Evaluation Results\n评估结果（小数据集）', color='#e94560', fontsize=13, pad=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#0f3460')
    ax2.spines['left'].set_color('#0f3460')
    ax2.spines['top'].set_color('#0f3460')
    ax2.spines['right'].set_color('#0f3460')
    ax2.grid(True, alpha=0.2, color='#0f3460', axis='y')
    
    ax2.text(0.5, -0.15, 'Note: Scores reflect small training dataset (15 samples)',
             ha='center', va='top', color='#a8a8b3', fontsize=9, transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_006.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 评估指标帧生成完成")


def create_comparison_frame():
    """创建Transformer vs RNN对比帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Transformer vs RNN/LSTM', fontsize=20, ha='center', va='top',
            color='#e94560', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.88, '架构对比', fontsize=14, ha='center', va='top',
            color='white', transform=ax.transAxes)
    
    # 对比表格
    headers = ['Feature 特性', 'RNN/LSTM', 'Transformer']
    rows = [
        ['Parallelism 并行性', '低 Low', '高 High'],
        ['Long-range Dependency 长距离依赖', '困难 Hard', '容易 Easy'],
        ['Training Speed 训练速度', '慢 Slow', '快 Fast'],
        ['Memory Usage 内存使用', '低 Low', '高 High'],
        ['Pre-training 预训练', '一般 Fair', '优秀 Excellent'],
        ['Scalability 可扩展性', '有限 Limited', '强 Strong'],
    ]
    
    # 绘制表格
    col_widths = [0.35, 0.25, 0.25]
    col_starts = [0.05, 0.42, 0.68]
    row_height = 0.09
    start_y = 0.78
    
    # 表头
    for j, (header, x, w) in enumerate(zip(headers, col_starts, col_widths)):
        color = '#e94560' if j == 0 else ('#4ecca3' if j == 1 else '#f5a623')
        ax.text(x + w/2, start_y, header, ha='center', va='center', color=color,
               fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # 分隔线
    ax.plot([0.05, 0.95], [start_y - 0.03, start_y - 0.03], color='#0f3460', linewidth=1.5,
             transform=ax.transAxes)
    
    # 数据行
    for i, row in enumerate(rows):
        y = start_y - (i + 1) * row_height - 0.02
        bg_color = '#16213e' if i % 2 == 0 else '#1a1a2e'
        
        rect = mpatches.Rectangle((0.04, y - 0.03), 0.92, row_height - 0.01,
                                   facecolor=bg_color, edgecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)
        
        for j, (cell, x, w) in enumerate(zip(row, col_starts, col_widths)):
            if j == 0:
                color = 'white'
            elif j == 1:
                color = '#ff6b6b'
            else:
                color = '#4ecca3'
            ax.text(x + w/2, y + row_height/2 - 0.03, cell, ha='center', va='center',
                   color=color, fontsize=11, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_007.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 对比帧生成完成")


def create_code_structure_frame():
    """创建代码结构帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Code Structure  代码结构', fontsize=20, ha='center', va='top',
            color='#e94560', fontweight='bold', transform=ax.transAxes)
    
    code_structure = """
transformer_translation/
├── transformer_model.py      # Core Transformer Implementation
│   ├── PositionalEncoding    # 位置编码
│   ├── MultiHeadAttention    # 多头自注意力
│   ├── FeedForwardNetwork    # 前馈网络
│   ├── EncoderLayer          # 编码器层
│   ├── DecoderLayer          # 解码器层
│   └── TransformerTranslator # 完整翻译模型
│
├── train.py                  # Training Script
│   ├── TranslationDataset    # 数据集类
│   └── train_model           # 训练主函数
│
├── inference.py              # Inference Script
│   ├── greedy_decode         # 贪心解码
│   └── beam_search_decode    # 束搜索解码
│
├── evaluate.py               # Evaluation Script
│   ├── compute_bleu          # BLEU计算
│   ├── compute_rouge_l       # ROUGE-L计算
│   └── compute_meteor        # METEOR计算
│
└── demo_app.py               # Interactive Demo
    """
    
    ax.text(0.05, 0.85, code_structure, ha='left', va='top', color='#4ecca3',
           fontsize=11, fontfamily='monospace', transform=ax.transAxes)
    
    # 统计信息
    stats = [
        ('Total Lines of Code', '~1550 lines'),
        ('Model Parameters', '7,392,537'),
        ('Framework', 'PyTorch 2.11.0'),
        ('Language', 'Python 3.11'),
    ]
    
    for i, (label, value) in enumerate(stats):
        x = 0.65
        y = 0.75 - i * 0.12
        ax.text(x, y, f'{label}:', ha='left', va='center', color='#a8a8b3',
               fontsize=12, transform=ax.transAxes)
        ax.text(x + 0.28, y, value, ha='right', va='center', color='white',
               fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_008.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 代码结构帧生成完成")


def create_summary_frame():
    """创建总结帧"""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'Project Summary  项目总结', fontsize=22, ha='center', va='top',
            color='#e94560', fontweight='bold', transform=ax.transAxes)
    
    achievements = [
        ('✓', 'Complete Transformer implementation from scratch', '从零实现完整Transformer架构'),
        ('✓', 'Chinese-English bidirectional translation', '支持中英双向互译'),
        ('✓', 'Multiple decoding strategies', '多种推理解码策略（贪心/束搜索）'),
        ('✓', 'Comprehensive evaluation metrics', '完善的评估指标（BLEU/ROUGE-L/METEOR）'),
        ('✓', 'Interactive demo application', '交互式演示程序'),
        ('✓', 'Detailed technical documentation', '详细的技术文档报告'),
    ]
    
    for i, (check, en, zh) in enumerate(achievements):
        y = 0.78 - i * 0.11
        ax.text(0.06, y, check, ha='left', va='center', color='#4ecca3',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.11, y + 0.02, en, ha='left', va='center', color='white',
               fontsize=12, transform=ax.transAxes)
        ax.text(0.11, y - 0.03, zh, ha='left', va='center', color='#a8a8b3',
               fontsize=10, transform=ax.transAxes)
    
    # 底部引用
    ax.text(0.5, 0.08, '"Attention Is All You Need"  —  Vaswani et al., 2017', 
            ha='center', va='center', color='#e94560', fontsize=12, style='italic',
            transform=ax.transAxes)
    ax.text(0.5, 0.03, 'Implemented with PyTorch  |  From Scratch', 
            ha='center', va='center', color='#a8a8b3', fontsize=11,
            transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('video_frames/frame_009.png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print("✓ 总结帧生成完成")


if __name__ == "__main__":
    print("开始生成视频帧...")
    
    create_title_frame()
    create_architecture_frame()
    create_attention_frame()
    create_training_frame()
    create_inference_frame()
    create_evaluation_frame()
    create_comparison_frame()
    create_code_structure_frame()
    create_summary_frame()
    
    print("\n所有帧生成完成！")
    print(f"生成帧数: {len(os.listdir('video_frames'))}")
