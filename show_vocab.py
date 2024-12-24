import torch
import os

# 加载词汇表
english_vocab = torch.load("vocab30k/english_vocab.pt")
chinese_vocab = torch.load("vocab30k/chinese_vocab.pt")

# 获取当前项目的根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 设置保存文件的路径
english_vocab_path = os.path.join(project_root, "vocab30k", "english_vocab_with_index.txt")
chinese_vocab_path = os.path.join(project_root, "vocab30k", "chinese_vocab_with_index.txt")

# 将词汇表和索引保存到文件
def save_vocab_with_index_to_txt(vocab, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, idx in vocab.get_stoi().items():
            f.write(f"{word}\t{idx}\n")

# 保存英语和中文词汇表和索引
save_vocab_with_index_to_txt(english_vocab, english_vocab_path)
save_vocab_with_index_to_txt(chinese_vocab, chinese_vocab_path)

print(f"Vocabulary with indices has been saved to: {english_vocab_path} and {chinese_vocab_path}")
