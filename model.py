import torch
import torch.nn as nn
import random
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,  num_layers, dropout_p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p) # 利用的lstm

    def forward(self, x):
        #x shape: (Seq_length, N)
        embedding = self.dropout(self.embedding(x))
        #embedding shape: (seq_length, N, embedding_size)
        _, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_p):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x ,hidden, cell): #对于词翻译，会把翻译出来的第一个词当作下一次的输入，所以要一个一个的来
        #shape of x:(N)(一次只输入一个词), but we want (1,N)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        #embedding shapes: (1, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, length_of_vocab)
        predictions = self.fc(outputs)
        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
    def forward(self, source, target, teacher_force_ratio=0.5):
        ''' source为输入的待翻译句子， target为翻译结果，teacher_force_ratio代表有多少比例的预测会用作下一次的输出，
        否则将会用正确的target作为下一次的输入

        source shape [source_sentence_len, batch_size]
        target shape [target_sentence_len, batch_size]
        '''

        batch_size = source.shape[1] # 
        target_len = target.shape[0]

        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size, device=source.device)

        hidden, cell = self.encoder(source) # source是固定的,得到encode的hidden state

        #获取<sos>的index
        x = target[0,:] 
        for t in range(1, target_len): #一个字一个字的输入
            output, hidden, cell =self.decoder(x, hidden, cell)
            outputs[t] = output
            #output.shape = [batch_size, target_vocab_size]
            best_guess = output.argmax(1) # 得到 vocab index

            if self.training:
                if random.random() < teacher_force_ratio:
                    x = target[t]  
                else:
                    x = best_guess

        return outputs
    
