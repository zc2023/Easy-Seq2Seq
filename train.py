import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import Encoder, Decoder, Seq2Seq
from att_model import AttEncoder, AttDecoder, AttSeq2Seq, rnn_Attention
from transformerv2 import Transformer
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def train(num_epochs=50,
          learning_rate = 0.001, # 0.001
          batch_size = 256,
          load_model_ckpt ='my_checkpoint.pth.tar',
          save_model_ckpt_name = 'checkpoint_train_dataset.pth.tar',
          device = 'cuda:0',
          model_name = "Seq2Seq",
          test_sentence = "你知道的，我会永远爱着你。",
          dataset_path = "/data/czhang/projects/datasets/en2cn/translation2019zh_train_fixed.json",
          vocab_path = "vocab30k_train",
          run_name = 'Seq2Seq_100epoch_train',
          epoch_per_eval = 50,
          epoch_per_save = 25,
          ):

    # Get trainloader and vocab
    train_loader, english, chinese = get_loader(dataset_path, vocab_path, batch_size, model_name)

    # Model hyperparameters
    input_size_encoder = len(chinese.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)

    # TensorBoard writer
    # 获取当前时间并格式化
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # 将格式化后的时间加入到 run_name 中
    writer = SummaryWriter(f'runs/{run_name}_{current_time}')
    step = 0

    # Initialize networks
    if model_name == "Seq2Seq":
        encoder_embedding_size = 300
        decoder_embedding_size = 300
        hidden_size = 1024
        num_layers = 2
        enc_dropout = 0.5
        dec_dropout = 0.5
        encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
        decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
        model = Seq2Seq(encoder_net, decoder_net, output_size).to(device)
        print("Use Seq2Seq model!")
    elif model_name == "AttSeq2Seq":
        # 初始化模型参数
        encoder_embedding_size = 256
        decoder_embedding_size = 256
        hidden_size = 512
        num_layers = 2
        enc_dropout = 0.5
        dec_dropout = 0.5
        attn = rnn_Attention(hidden_size)
        enc = AttEncoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout)
        dec = AttDecoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, dec_dropout, attn)
        model = AttSeq2Seq(enc, dec, device).to(device)
        print("Use AttSeq2Seq model!")
    elif model_name == "Transformer":
        model = Transformer(src_vocab_size=input_size_encoder, 
                            tgt_vocab_size=input_size_decoder,
                            d_model=512, 
                            num_heads=8, 
                            # num_encoder_layers=2,  
                            # num_decoder_layers=2, 
                            # for train dataset is 6
                            num_encoder_layers=3,  
                            num_decoder_layers=3, 
                            d_ff=2048, 
                            dropout=0,
                            ).to(device)
        print("Use Transformer model!")
    else:
        print("Not a valid model name!")

    model.apply(init_weights)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    if model_name == "Transformer":
        # 假设你的模型是 model，设置学习率
        learning_rate = 1e-4
        warmup_steps = 4000  # 论文中设置的预热步数

        # 定义 Adam 优化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),   # Adam 的 beta 参数
            eps=1e-9,            # epsilon 防止除零错误
            weight_decay=0.01    # 权重衰减
        )

        # 定义学习率调度函数
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))  # 预热阶段
            return float(max(1, step)) ** (-0.5)  # 学习率衰减阶段

        # 创建学习率调度器
        scheduler = LambdaLR(optimizer, lr_lambda)  
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.get_stoi()['<pad>']
    # print("pad_idx",pad_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Optionally load model checkpoint
    if load_model_ckpt:
        load_checkpoint(load_model_ckpt, model, optimizer)

    # Example sentence for translation
    sentence = test_sentence

    for epoch in range(num_epochs):

        # Translate a sample sentence at the start of each epoch
        if (epoch+1)%epoch_per_eval == 0:
            print(f'Eval on example sentence: {sentence}')
            model.eval()
            translate = translate_sentence(model, sentence, chinese, english, device, max_length=50, model_name=model_name)
            print(f'Translated example sentence: {translate}')

        model.train()

        # Training loop
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ncols=150)
        epoch_loss = 0

        for batch_idx, batch in loop:
            # print("chinese_batch", chinese_batch[:3])
            if model_name == 'Transformer':
                # Accessing batch as a dictionary (since __getitem__ returns a dictionary)
                
                chinese_batch = batch['chinese']  # List of  Chinese sentences
                dec_input_english_batch = batch['dec_input_english']  # List of  English sentences
                target_english_batch = batch['target_english']  # List of  English sentences
                # Use vocab to convert tokens into indices
                enc_input = [torch.tensor([chinese.vocab[token] for token in sentence]).to(device) for sentence in chinese_batch] # list, 每个元素是一个不定长度的tensor
                dec_input = [torch.tensor([english.vocab[token] for token in sentence]).to(device) for sentence in dec_input_english_batch]
                target = [torch.tensor([english.vocab[token] for token in sentence]).to(device) for sentence in target_english_batch]
                # Pad the input sequences (to make them equal in length)
                # batch last
                # [input_sentence_idx_max_len, batchsize]
                enc_input = torch.nn.utils.rnn.pad_sequence(enc_input, padding_value=chinese.vocab['<pad>'], batch_first=True)
                # [output_sentence_idx_max_len, batchsize]
                dec_input = torch.nn.utils.rnn.pad_sequence(dec_input, padding_value=english.vocab['<pad>'], batch_first=True)
                target = torch.nn.utils.rnn.pad_sequence(target, padding_value=english.vocab['<pad>'], batch_first=True)
                # Forward pass
                optimizer.zero_grad()
                # enc_input = [batch_size, src_len] , src_len 是 vocab_indexes
                # dec_input = [batch_size, tgt_len] 
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_input, dec_input)  
                
                # Output shape: [batch_size， tgt_len， tgt_vocab_size]
                outputs = outputs.view(-1, outputs.size(-1)) # [batch_size*tgt_len， tgt_vocab_size]
                # print("outputs",outputs.shape)
                # print("target after",target.contiguous().view(-1).shape)
                # Compute loss
                loss = criterion(outputs, target.contiguous().view(-1))
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                # Optimizer step
                optimizer.step()

                # Log batch loss
                writer.add_scalar('batch Loss', loss.item(), global_step=step)
                step += 1
                # Update progress bar
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')

            else:
                # Accessing batch as a dictionary (since __getitem__ returns a dictionary)
                english_batch = batch['english']  # List of  English sentences
                chinese_batch = batch['chinese']  # List of  Chinese sentences
                # Use vocab to convert tokens into indices
                inp_data = [torch.tensor([chinese.vocab[token] for token in sentence]).to(device) for sentence in chinese_batch] # list, 每个元素是一个不定长度的tensor
                target = [torch.tensor([english.vocab[token] for token in sentence]).to(device) for sentence in english_batch]
                # Pad the input sequences (to make them equal in length)
                # batch last
                # [input_sentence_idx_max_len, batchsize]
                inp_data = torch.nn.utils.rnn.pad_sequence(inp_data, padding_value=chinese.vocab['<pad>'], batch_first=False)
                # [output_sentence_idx_max_len, batchsize]
                target = torch.nn.utils.rnn.pad_sequence(target, padding_value=english.vocab['<pad>'], batch_first=False)

                # Forward pass
                optimizer.zero_grad()
                # inp_data = [src length, batch size]
                # target = [trg length, batch size]  
                output = model(inp_data, target)  
                # Output shape: (output_sentence_idx_max_len, batch_size, output_vocab_len)
                # Reshape output and target for loss calculation (skip <sos> token)
                output = output[1:].reshape(-1, output.shape[2])
                # output= [(trg length - 1) * batch size, trg vocab size]
                target = target[1:].reshape(-1)
                # trg = [(trg length - 1) * batch size]
                # Zero gradients
            
                # Compute loss
                loss = criterion(output, target)
                epoch_loss += loss.item()

                # Backward pass
                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                # Optimizer step
                optimizer.step()

                # Log batch loss
                writer.add_scalar('batch Loss', loss.item(), global_step=step)
                step += 1

                # Update progress bar
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        if model_name == "Transformer":
            scheduler.step()
        # Log epoch loss
        writer.add_scalar('epoch Loss', epoch_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Total loss: {epoch_loss}")

        if (epoch+1)%epoch_per_save == 0:
            # Save checkpoint at the end of the epoch
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint,filename=save_model_ckpt_name)


if __name__=="__main__":

# train on datasets/en2cn/translation2019zh_train_fixed.json
    # train(num_epochs=50,
    #         learning_rate = 0.001, # 0.001
    #         batch_size = 256,
    #         load_model_ckpt ='',
    #         save_model_ckpt_name = 'checkpoint_train_dataset.pth.tar',
    #         device = 'cuda:1',
    #         model_name = "Seq2Seq",
    #         test_sentence = "你知道的，我会永远爱着你。",
    #         dataset_path = "/data/czhang/projects/datasets/en2cn/translation2019zh_train_fixed.json",
    #         vocab_path = "vocab30k",
    #         run_name = 'Seq2Seq_100epoch_train',
    #         epoch_per_eval = 10,
    #         epoch_per_save = 1,
    #         )

# # train on datasets/en2cn/translation2019zh_valid_fixed.json
    # train(num_epochs=50,
    #         learning_rate = 0.001, # 0.001
    #         batch_size = 256,
    #         load_model_ckpt ='',
    #         save_model_ckpt_name = 'checkpoint_valid_dataset.pth.tar',
    #         device = 'cuda:0',
    #         model_name = "Seq2Seq",
    #         test_sentence = "你知道的，我会永远爱着你。",
    #         dataset_path = "/data/czhang/projects/datasets/en2cn/translation2019zh_valid_fixed.json",
    #         vocab_path = "vocab30k",
    #         run_name = 'Seq2Seq_100epoch_valid',
    #         epoch_per_eval = 50,
    #         epoch_per_save = 25,
    #         )
    # Seq2SeqAttention
    # train(num_epochs=50,
    #         learning_rate = 0.001, # 0.001
    #         batch_size = 128,
    #         load_model_ckpt ='',
    #         save_model_ckpt_name = 'seq2seqAtt_valid_100epoch.pth.tar',
    #         device = 'cuda:2',
    #         model_name = "AttSeq2Seq",
    #         test_sentence = "你知道的，我会永远爱着你。",
    #         dataset_path = "/data/czhang/projects/datasets/en2cn/translation2019zh_valid_fixed.json",
    #         vocab_path = "vocab30k",
    #         run_name = 'Seq2SeqAttention_100epoch_valid',
    #         epoch_per_eval = 50,
    #         epoch_per_save = 25,
    #         )
    
    # train(num_epochs=50,
    #         learning_rate = 5e-4, # 0.001
    #         batch_size = 128,
    #         load_model_ckpt ='transformer_valid_50epoch.pth.tar',
    #         save_model_ckpt_name = 'transformer_valid_100epoch.pth.tar',
    #         device = 'cuda:0',
    #         model_name = "Transformer",
    #         test_sentence = "你知道的，我会永远爱着你。",
    #         dataset_path = "/data/czhang/projects/datasets/en2cn/translation2019zh_valid_fixed.json",
    #         vocab_path = "vocab30k",
    #         run_name = 'transformer_100epoch_valid',
    #         epoch_per_eval = 50,
    #         epoch_per_save = 50,
    #         )
    
    train(num_epochs=10,
            learning_rate = 1e-4, # 0.001
            batch_size = 128,
            load_model_ckpt ='',
            save_model_ckpt_name = 'transformer_train_10epoch.pth.tar',
            device = 'cuda:1',
            model_name = "Transformer",
            test_sentence = "你知道的，我会永远爱着你。",
            dataset_path = "/data/czhang/projects/datasets/en2cn/translation2019zh_train_fixed.json",
            vocab_path = "vocab30k",
            run_name = 'transformer_10epoch_train',
            epoch_per_eval = 10,
            epoch_per_save = 1,
            )
