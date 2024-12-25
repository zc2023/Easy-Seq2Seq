import torch
import spacy
from utils import translate_sentence
from model import Encoder, Decoder, Seq2Seq
import re
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from att_model import AttEncoder, AttDecoder, AttSeq2Seq, rnn_Attention
from transformerv2 import Transformer
def inference(
       english_vocab_path = "vocab30k/english_vocab.pt",
       chinese_vocab_path = "vocab30k/chinese_vocab.pt",
       model_ckpt_path = "weights/transformer_train_10epoch.pth.tar", 
       # seq2seqAtt_valid_100epoch.pth.tar,valid_150epoch.pth.tar
       # seq2seq_train_28epoch.pth.tar
       # transformer_valid_100epoch.pth.tar
       device = "cuda:0",
       model_name = "Transformer", # Seq2Seq # Seq2SeqAttention # Transformer
       # sentence_to_translate = '美国缓慢地开始倾听，但并非没有艰难曲折。',
       sentence_to_translate = '你知道的，我会永远爱着你。',
       # sentence_to_translate = '本文主要由三个部分组成：导生制、见习生制、导生制和见习生制的历史作用。',
       # sentence_to_translate = '昨天有人去超市买了一瓶啤酒',
       # sentence_to_translate = '拼尽全力也无法战胜',
       # sentence_to_translate = '你好',
       # sentence_to_translate = '周六天气很热',
       # sentence_to_translate = '苹果的原始种群主要起源于中亚的天山山脉附近，尤其是现代哈萨克斯坦的阿拉木图地区',
):
        # 加载英语和中文词汇表
    english_vocab = torch.load(english_vocab_path)
    chinese_vocab = torch.load(chinese_vocab_path)
    # Model hyperparamters
    
    input_size_encoder = len(chinese_vocab.vocab)
    input_size_decoder = len(english_vocab.vocab)
    output_size = len(english_vocab.vocab)
    #initialize model
    if model_name == "Seq2Seq":
        encoder_embedding_size = 300
        decoder_embedding_size = 300
        hidden_size = 1024
        num_layers = 2
        enc_dropout = 0.5
        dec_dropout = 0.5
        encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
        decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
        model = Seq2Seq(encoder_net, decoder_net, len(english_vocab.vocab)).to(device)
        print("Use Seq2Seq model!")
    elif model_name == "Seq2SeqAttention":
        # 初始化模型参数
        encoder_embedding_size = 256
        decoder_embedding_size = 256
        hidden_size = 512
        num_layers = 2
        enc_dropout = 0.5
        dec_dropout = 0.5
        attn = rnn_Attention(hidden_size).to(device)
        enc = AttEncoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
        dec = AttDecoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, dec_dropout, attn).to(device)
        model = AttSeq2Seq(enc, dec, device).to(device)
        print("Use AttSeq2Seq model!")
    elif model_name == "Transformer":
        model = Transformer(src_vocab_size=input_size_encoder, 
                            tgt_vocab_size=input_size_decoder,
                            d_model=512, 
                            num_heads=8, 
                            num_encoder_layers=2, 
                            num_decoder_layers=2, 
                            d_ff=2048, 
                            dropout=0,
                            ).to(device)
        print("Use Transformer model!")       
    else:
        print("Not a valid model name!")
    param = torch.load(model_ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict(param)
    print(f"=> checkpoint {model_ckpt_path} loaded!")
    model.eval()

    #test sentence
    sentence = sentence_to_translate
    print(sentence)
    trans = translate_sentence(model, sentence, chinese_vocab, english_vocab, device, max_length=50, model_name=model_name)
    print(f'Predicted tokens: {trans}')
    # Remove <sos> and <eos> from the translated sentence
    translated_sentence = trans[1:-1]  # Removing <sos> and <eos>

    # Convert list to a space-separated string
    translated_sentence_str = ' '.join(translated_sentence)
    # Remove the <unk> tokens
    translated_sentence_str = translated_sentence_str.replace('<unk>', '').strip()
    # Remove spaces before punctuation (e.g., " ,", " .", etc.)
    translated_sentence_str = re.sub(r'\s([?.!,¿])', r'\1', translated_sentence_str)
    print(f'Translated sentence: {translated_sentence_str}')
    



if __name__=="__main__":
    inference()