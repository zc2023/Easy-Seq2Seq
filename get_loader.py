# # from torchtext.data import Field, BucketIterator, TabularDataset # for torchtext 0.8
# from torchtext.data import Field, BucketIterator, TabularDataset
# import numpy as np
# import spacy
# from utils import save_vocab

# #Define tokenizer
# spacy_ch= spacy.load("zh_core_web_sm")
# spacy_eng= spacy.load("en_core_web_sm")

# def tokenize_ch(text):
#     return [tok.text for tok in spacy_ch.tokenizer(text)]
# def tokenize_eng(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]

# def get_loader(batch_size, save_vocabulary=False):
#     english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True,init_token='<sos>',eos_token='<eos>')
#     chinese = Field(sequential=True, use_vocab=True, tokenize=tokenize_ch,lower=True,init_token='<sos>',eos_token='<eos>')

#     fields = {"english":("eng", english), "chinese":("ch", chinese)}

#     train_data, test_data = TabularDataset.splits(
#         path="translation2019zh/",
#         train="translation2019zh_valid.json",
#         test="translation2019zh_valid.json",
#         format="json",
#         fields=fields
#     )

#     english.build_vocab(train_data, max_size=30000, min_freq=2)
#     chinese.build_vocab(train_data, max_size=30000, min_freq=2)
    
#     if save_vocabulary:
#         save_vocab(english.vocab.stoi, 'saved_vocab/english_stoi.txt')
#         save_vocab(english.vocab.itos, 'saved_vocab/english_itos.txt')
#         save_vocab(chinese.vocab.stoi, 'saved_vocab/chinese_stoi.txt')
#         save_vocab(chinese.vocab.itos, 'saved_vocab/chinese_itos.txt')

#     train_iterator,_ = BucketIterator.splits(
#         (train_data,test_data),
#         batch_size=batch_size,
#         device="cuda",
#         sort_within_batch= True,   #按照句子的长短来构成batch，减小padding的计算量
#         sort_key = lambda x :len(x.ch)
#     )
#     return train_iterator, english, chinese


import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import os
import spacy
from utils import save_vocab
from utils import load_stoi, load_itos, save_vocab_to_text, load_vocab_from_text

# 获取当前脚本的路径
script_path = os.path.abspath(__file__)

# 获取当前项目根目录（假设项目根目录在脚本同级或上级）
project_root = os.path.dirname(script_path)

# Define tokenizer
spacy_ch = spacy.load("zh_core_web_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_ch(text):
    return [tok.text for tok in spacy_ch.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Create a custom dataset
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer_ch, tokenizer_eng):
        # Load JSON data here (you might want to use pandas or json)
        # For simplicity, assume data is loaded into a list of dictionaries
        # Example format: [{"english": "I am learning.", "chinese": "我在学习。"}, ...]
        self.data = self.load_data(file_path)
        self.tokenizer_ch = tokenizer_ch # 分词器
        self.tokenizer_eng = tokenizer_eng

    def load_data(self, file_path):
        import json
        with open(file_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng_text = self.data[idx]['english']
        ch_text = self.data[idx]['chinese']
        # 在句子的开头和末尾分别添加<sos>和<eos>标记
        eng_tokens = ['<sos>'] + self.tokenizer_eng(eng_text) + ['<eos>']
        ch_tokens = ['<sos>'] + self.tokenizer_ch(ch_text) + ['<eos>']
        return {
            'english': eng_tokens,
            'chinese': ch_tokens
        }
# Create a custom dataset
class TransformerTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer_ch, tokenizer_eng):
        # Load JSON data here (you might want to use pandas or json)
        # For simplicity, assume data is loaded into a list of dictionaries
        # Example format: [{"english": "I am learning.", "chinese": "我在学习。"}, ...]
        self.data = self.load_data(file_path)
        self.tokenizer_ch = tokenizer_ch # 分词器
        self.tokenizer_eng = tokenizer_eng

    def load_data(self, file_path):
        import json
        with open(file_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng_text = self.data[idx]['english']
        ch_text = self.data[idx]['chinese']
        # 在句子的开头和末尾分别添加<sos>和<eos>标记
        dec_input_eng_tokens = ['<sos>'] + self.tokenizer_eng(eng_text) 
        target_eng_tokens = self.tokenizer_eng(eng_text) + ['<eos>']
        ch_tokens = self.tokenizer_ch(ch_text)
        return {
            'chinese': ch_tokens,
            'dec_input_english': dec_input_eng_tokens,
            'target_english': target_eng_tokens,
        }

def get_loader(dataset_path, vocab_path, batch_size, model_name="Seq2Seq"):
    # construct dataset
    if model_name == "Transformer":
        train_data = TransformerTranslationDataset(dataset_path, tokenize_ch, tokenize_eng)
    else:
        train_data = TranslationDataset(dataset_path, tokenize_ch, tokenize_eng)
     
    # build vocab
    if not os.path.exists(os.path.dirname(f"{project_root}/{vocab_path}/english_vocab.pt")):
        print("Build vocabulary when first run")
        print("Building vocabulary...it may take some time...")
        # english_vocab = build_vocab_from_iterator((item['english'] for item in train_data), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
        # # 为特殊符号分配ID
        # english_vocab.set_default_index(english_vocab["<unk>"])
        # chinese_vocab = build_vocab_from_iterator((item['chinese'] for item in train_data), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
        # # 为特殊符号分配ID
        # chinese_vocab.set_default_index(chinese_vocab["<unk>"])

        # 为英语和中文数据构建词汇表
        english_vocab = build_vocab_from_iterator(
            (item['english'] for item in train_data),
            specials=["<unk>", "<pad>", "<sos>", "<eos>"],
            min_freq=2,         # 最小频率，频率小于2的词将被忽略
            max_tokens=30000    # 最大词汇表大小，超过30000个词的部分会被丢弃
        )
        # 设置默认的未知标记
        english_vocab.set_default_index(english_vocab["<unk>"])
        # 为中文数据构建词汇表
        chinese_vocab = build_vocab_from_iterator(
            (item['chinese'] for item in train_data),
            specials=["<unk>", "<pad>", "<sos>", "<eos>"],
            min_freq=2,         # 最小频率
            max_tokens=30000    # 最大词汇表大小
        )
        # 设置默认的未知标记
        chinese_vocab.set_default_index(chinese_vocab["<unk>"])


        # 获取当前项目的工作目录
        current_dir = os.getcwd()

        # 设置词汇表保存的路径
        vocab_dir = os.path.join(current_dir, f"{vocab_path}")

        # 确保目标路径的父目录存在
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)

        # 保存词汇表到文件夹中
        torch.save(english_vocab, os.path.join(vocab_dir, "english_vocab.pt"))
        torch.save(chinese_vocab, os.path.join(vocab_dir, "chinese_vocab.pt"))
        print(f"Vocabulary is saved to {project_root}/{vocab_path} !")
    else:
        # 加载英语和中文词汇表
        english_vocab = torch.load(f"{project_root}/{vocab_path}/english_vocab.pt")
        chinese_vocab = torch.load(f"{project_root}/{vocab_path}/chinese_vocab.pt")
        print(f"Vocabulary loaded from {project_root}/{vocab_path} successfully!")
       

    if model_name == "Transformer":
        # DataLoader
        def collate_fn(batch):
            # 获取批次中的英文和中文数据
            chinese_batch = [item['chinese'] for item in batch]
            dec_input_english_batch = [item['dec_input_english'] for item in batch]
            target_english_batch  = [item['target_english'] for item in batch]
            return {'chinese': chinese_batch, 
                    'dec_input_english': dec_input_english_batch,
                    'target_english': target_english_batch,
                    }  # Returning as dictionary
        # Convert to DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)      
    else:
        # DataLoader
        def collate_fn(batch):
            # 获取批次中的英文和中文数据
            english_batch = [item['english'] for item in batch]
            chinese_batch = [item['chinese'] for item in batch]
            return {'english': english_batch, 'chinese': chinese_batch}  # Returning as dictionary
        # Convert to DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, english_vocab, chinese_vocab
