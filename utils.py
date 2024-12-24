import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys
import os
import torchtext
from torchtext.vocab import Vocab
from transformerv2 import Transformer, TransformerInference


def translate_sentence(model, sentence, chinese_vocab, english_vocab, device, max_length=50, model_name ="Seq2Seq"):
    # Load chinese tokenizer
    spacy_ch = spacy.load("zh_core_web_sm")

    # Tokenize sentence
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_ch(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')
    # print(tokens)
    # Convert tokens to indices using the Chinese vocabulary
    text_to_indices = [chinese_vocab.get_stoi().get(token, chinese_vocab.get_stoi()['<unk>']) for token in tokens]

    # Convert to Tensor and move to device
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    # print(sentence_tensor.shape)
    # Initialize the output with the <sos> token
    inputs  = [english_vocab.get_stoi()["<sos>"]]

    if model_name == "Seq2Seq":
        hidden, cell = model.encoder(sentence_tensor)
        for _ in range(max_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == english_vocab.get_stoi()["<eos>"]:
                break
    elif model_name == "Seq2SeqAttention":
        encoder_outputs, hidden = model.encoder(sentence_tensor)
        for _ in range(max_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden = model.decoder(inputs_tensor, hidden, encoder_outputs)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == english_vocab.get_stoi()["<eos>"]:
                break    
    elif model_name == "Transformer":
        print("Use Transformer inference pipeline!")
        # # tgt_input = torch.LongTensor([english_vocab.get_stoi()["<sos>"]]).unsqueeze(0).to(device)  # shape [1, 1]
        # # Prepare empty tensor for storing generated tokens
        # generated_tokens = []
        src_input = sentence_tensor.transpose(0,1)[:,1:-1]
        # print(src_input)
        # 创建推理对象
        inference_model = TransformerInference(model.to(device), max_len=50, device=device)
        tgt_start_token = english_vocab.get_stoi()["<sos>"]
        eos_token = english_vocab.get_stoi()["<eos>"]
        # print("eos_token:", eos_token)
        pad_token = english_vocab.get_stoi()["<pad>"]
        # 进行推理
        generated_tokens = inference_model.infer(src_input, tgt_start_token, eos_token, pad_token)
        # print("Generated Tokens:", generated_tokens)
        # 假设 generated_tokens 是一个形状为 [1, seq_len] 的张量
        generated_tokens = generated_tokens.squeeze(0)  # 以确保是 1D 张量 [seq_len]

        # 将张量转换为 Python 列表并通过 get_itos() 映射每个 token ID
        translated_sentence = [english_vocab.get_itos()[idx.item()] for idx in generated_tokens]

        # print("Generated Tokens:", generated_tokens)
        return translated_sentence
    else:
        print("Not a valid model name!")
    
    # index to word
    translated_sentence = [english_vocab.get_itos()[idx] for idx in inputs]

    return translated_sentence


def bleu(data, model, chinese, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["ch"]
        trg = vars(example)["eng"]

        prediction = translate_sentence(model, src, chinese, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    print(f"=> checkpoint saved to {filename}!")


def load_checkpoint(load_model_ckpt, model, optimizer):
    checkpoint = torch.load(load_model_ckpt)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"=> checkpoint {load_model_ckpt} loaded!")


def save_vocab(vocab,save_path):
    # 获取文件的目录路径
    dir_name = os.path.dirname(save_path)
    
    # 如果目录不存在，则创建目录
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(save_path,'w',encoding='utf-8') as f:
        f.write(str(vocab))

def load_stoi(save_path):
    with open(save_path,'r',encoding='utf-8') as f:
        dic=f.read()
        dic=list(dic)
        for i in range(len(dic)):
            if dic[i]=='{':
                del dic[0:i]
                del dic[-1]
                break
        dic=''.join(dic)
        dic=eval(dic)
    return dic

def load_itos(save_path):
    with open(save_path,'r',encoding='utf-8') as f:
        dic=f.read()
        dic=eval(dic)
    return dic

# 保存词汇表为文本文件，确保目录存在
def save_vocab_to_text(vocab, filename):
    # 获取文件的目录路径
    dir_name = os.path.dirname(filename)
    
    # 如果目录不存在，则创建目录
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # 获取词汇表的 stoi 映射
    stoi = vocab.get_stoi()  # 使用 get_stoi() 方法
    
    # 保存词汇表到文件
    with open(filename, 'w', encoding='utf-8') as f:
        for word, idx in stoi.items():
            f.write(f"{word}\t{idx}\n")

from torchtext.vocab import Vocab
from collections import Counter

def load_vocab_from_text(filename):
    stoi = {}
    itos = {}

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 跳过空行或不含制表符的行，或者只包含一个值的行（例如 10944）
            if not line or '\t' not in line:
                continue

            parts = line.split('\t')

            # 如果行不能被正确拆分成两个部分，跳过该行
            if len(parts) != 2:
                continue

            word, idx = parts
            try:
                stoi[word] = int(idx)
                itos[int(idx)] = word
            except ValueError:
                # 如果索引不能转换为整数，跳过该行
                continue

    # `stoi` 和 `itos` 在新版本的 torchtext 中需要直接赋值
    vocab = Vocab(stoi=stoi, itos=itos)








