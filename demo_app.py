"""
Transformer翻译模型的交互式演示程序
提供图形化界面和命令行界面
"""

import torch
import json
import os
import sys
from transformer_model import TransformerTranslator
from train import train_model, create_vocabularies
from inference import TranslationInference


class TranslationDemo:
    """
    翻译演示类
    """
    def __init__(self):
        self.model = None
        self.translator = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_or_train_model(self):
        """
        加载已有模型或训练新模型
        """
        if os.path.exists('model.pth'):
            print("发现已训练的模型，正在加载...")
            try:
                self.translator = TranslationInference(device=self.device)
                print("✓ 模型加载成功！")
                return True
            except Exception as e:
                print(f"✗ 模型加载失败: {e}")
                print("将训练新模型...\n")
        
        # 训练新模型
        print("=" * 50)
        print("开始训练新模型")
        print("=" * 50)
        
        # 准备训练数据
        src_texts = [
            "hello world",
            "good morning",
            "how are you",
            "thank you very much",
            "what is your name",
            "i love machine learning",
            "transformer is powerful",
            "neural networks are amazing",
            "deep learning is fascinating",
            "artificial intelligence is the future",
            "welcome to our platform",
            "have a nice day",
            "see you later",
            "goodbye my friend",
            "nice to meet you"
        ]
        
        tgt_texts = [
            "你好 世界",
            "早上 好",
            "你 好 吗",
            "非常 感谢 你",
            "你 叫 什么 名字",
            "我 喜欢 机器 学习",
            "transformer 很 强大",
            "神经 网络 很 棒",
            "深度 学习 很 有趣",
            "人工 智能 是 未来",
            "欢迎 来到 我们 的 平台",
            "祝 你 有 美好 的 一 天",
            "待会 见",
            "再见 我 的 朋友",
            "很 高兴 认识 你"
        ]
        
        print(f"训练数据: {len(src_texts)} 个样本")
        print("模型配置: d_model=256, num_layers=4, num_heads=8, d_ff=1024")
        
        model, src_vocab, tgt_vocab, device = train_model(
            src_texts, tgt_texts, 
            num_epochs=30, 
            batch_size=4, 
            learning_rate=0.0001
        )
        
        # 加载推理器
        self.translator = TranslationInference(device=self.device)
        print("\n✓ 模型训练完成！")
        return True
    
    def display_menu(self):
        """
        显示菜单
        """
        print("\n" + "=" * 50)
        print("Transformer 中英互译演示系统")
        print("=" * 50)
        print("1. 英文翻译为中文")
        print("2. 中文翻译为英文")
        print("3. 查看模型架构信息")
        print("4. 批量翻译演示")
        print("5. 退出程序")
        print("=" * 50)
    
    def translate_en_to_zh(self):
        """
        英文翻译为中文
        """
        print("\n--- 英文翻译为中文 ---")
        print("(输入 'quit' 返回菜单)")
        
        while True:
            src_text = input("\n请输入英文句子: ").strip()
            
            if src_text.lower() == 'quit':
                break
            
            if not src_text:
                print("输入不能为空，请重试")
                continue
            
            print(f"源文本: {src_text}")
            
            # 贪心解码
            print("\n[贪心解码]")
            result_greedy = self.translator.translate(src_text, method='greedy')
            print(f"翻译结果: {result_greedy}")
            
            # 束搜索
            print("\n[束搜索解码]")
            result_beam = self.translator.translate(src_text, method='beam_search')
            print(f"翻译结果: {result_beam}")
    
    def translate_zh_to_en(self):
        """
        中文翻译为英文
        """
        print("\n--- 中文翻译为英文 ---")
        print("(输入 'quit' 返回菜单)")
        print("注: 当前模型主要针对英→中优化，中→英效果可能较差")
        
        while True:
            src_text = input("\n请输入中文句子: ").strip()
            
            if src_text.lower() == 'quit':
                break
            
            if not src_text:
                print("输入不能为空，请重试")
                continue
            
            print(f"源文本: {src_text}")
            
            # 贪心解码
            print("\n[贪心解码]")
            result_greedy = self.translator.translate(src_text, method='greedy')
            print(f"翻译结果: {result_greedy}")
    
    def show_model_info(self):
        """
        显示模型架构信息
        """
        print("\n" + "=" * 50)
        print("模型架构信息")
        print("=" * 50)
        
        print("\n【Transformer 编码器-解码器架构】")
        print("""
        输入层
          ↓
        词嵌入 + 位置编码
          ↓
        编码器 (4层)
          ├─ 多头自注意力 (8个头)
          ├─ 前馈网络 (FFN)
          └─ 层归一化 + 残差连接
          ↓
        解码器 (4层)
          ├─ 自注意力
          ├─ 交叉注意力 (与编码器输出交互)
          ├─ 前馈网络
          └─ 层归一化 + 残差连接
          ↓
        输出投影层
          ↓
        Softmax
          ↓
        翻译结果
        """)
        
        print("【模型超参数】")
        print("- 模型维度 (d_model): 256")
        print("- 编码器层数: 4")
        print("- 解码器层数: 4")
        print("- 注意力头数: 8")
        print("- 前馈网络维度: 1024")
        print("- Dropout: 0.1")
        print("- 最大序列长度: 100")
        
        print("\n【关键组件说明】")
        print("""
        1. 位置编码 (Positional Encoding)
           - 使用正弦和余弦函数编码位置信息
           - 公式: PE(pos,2i) = sin(pos/10000^(2i/d_model))
                   PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        
        2. 多头注意力 (Multi-Head Attention)
           - 将输入投影到多个子空间
           - 在每个子空间中计算注意力
           - 公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        3. 前馈网络 (Feed-Forward Network)
           - 两层全连接网络
           - 公式: FFN(x) = max(0, xW1 + b1)W2 + b2
        
        4. 交叉注意力 (Cross-Attention)
           - 解码器中用于关注编码器输出
           - 允许模型在生成翻译时参考源文本
        """)
        
        print("=" * 50)
    
    def batch_demo(self):
        """
        批量翻译演示
        """
        print("\n" + "=" * 50)
        print("批量翻译演示")
        print("=" * 50)
        
        demo_sentences = [
            "hello world",
            "good morning",
            "how are you",
            "thank you very much",
            "i love machine learning",
            "transformer is powerful",
            "neural networks are amazing"
        ]
        
        print("\n演示英文→中文翻译:\n")
        
        for i, sentence in enumerate(demo_sentences, 1):
            print(f"{i}. 源文本: {sentence}")
            
            result = self.translator.translate(sentence, method='greedy')
            print(f"   翻译: {result}\n")
        
        print("=" * 50)
    
    def run(self):
        """
        运行演示程序主循环
        """
        print("\n" + "=" * 50)
        print("欢迎使用 Transformer 中英互译演示系统")
        print("=" * 50)
        
        # 加载或训练模型
        if not self.load_or_train_model():
            print("模型初始化失败，程序退出")
            return
        
        # 主菜单循环
        while True:
            self.display_menu()
            choice = input("请选择操作 (1-5): ").strip()
            
            if choice == '1':
                self.translate_en_to_zh()
            elif choice == '2':
                self.translate_zh_to_en()
            elif choice == '3':
                self.show_model_info()
            elif choice == '4':
                self.batch_demo()
            elif choice == '5':
                print("\n感谢使用，再见！")
                break
            else:
                print("无效的选择，请重试")


def main():
    """
    主函数
    """
    demo = TranslationDemo()
    demo.run()


if __name__ == "__main__":
    main()
