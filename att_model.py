import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class AttEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs: [src_len, batch_size, hid_dim * 2]
        # hidden: [n_layers * 2, batch_size, hid_dim]
        hidden_layers = []
        
        for layer in range(self.rnn.num_layers):
            forward_hidden = hidden[2 * layer, :, :]      # [batch_size, hid_dim]
            backward_hidden = hidden[2 * layer + 1, :, :] # [batch_size, hid_dim]
            
            combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1) # [batch_size, hid_dim * 2]
            
            transformed_hidden = torch.tanh(self.fc(combined_hidden)) # [batch_size, hid_dim]
            
            hidden_layers.append(transformed_hidden)
        
        hidden = torch.stack(hidden_layers, dim=0) # [n_layers, batch_size, hid_dim]
        
        return outputs, hidden

class rnn_Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 3), hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        # hidden: [src_len, batch_size, hid_dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        # energy: [src_len, batch_size, hid_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [src_len, batch_size]
        
        return F.softmax(attention, dim=0)

class AttDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, num_layers=n_layers)
        self.fc_out = nn.Linear((hid_dim * 3) + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, emb_dim]
        
        a = self.attention(hidden[-1], encoder_outputs)
        # a: [src_len, batch_size]
        
        # 修正后的形状转换
        a = a.permute(1, 0).unsqueeze(1)
        # a 的形状: [batch_size, 1, src_len]
        # a: [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: [batch_size, src_len, hid_dim * 2]
        # print(a.shape)

        weighted = torch.bmm(a, encoder_outputs)
        # weighted: [batch_size, 1, hid_dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, batch_size, hid_dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [1, batch_size, (hid_dim * 2) + emb_dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [1, batch_size, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden


class AttSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # 初始化输出张量
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 编码器输出
        encoder_outputs, hidden = self.encoder(src)
        
        # 解码器的第一个输入是<sos> token
        input = trg[0, :]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs
