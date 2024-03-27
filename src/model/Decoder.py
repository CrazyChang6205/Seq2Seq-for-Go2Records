import numpy as np
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, word2vec_model, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.word2vec_model = word2vec_model
        self.hidden_size = hidden_size
        self.output_size = self.word2vec_model.cum_table.size
        self.num_layers = num_layers
        self.embedding_size = word2vec_model.vector_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec_model.wv.vectors))
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        #NOTE：LSTM(input_size, hidden_size, num_layers) 為 ( 輸入層 特徵維度, 隱藏層、輸出層 特徵維度, 網路層數 )

    def forward(self, input_seq, hidden):
        """
        Parameters
        ----------
        input_seq : TYPE
            DESCRIPTION.
        hidden : TYPE
            DESCRIPTION.

        Returns
        -------
        outputs : TYPE
            DESCRIPTION.
        hidden : TYPE
            DESCRIPTION.
        """
        outputs = torch.empty(0)
        PAD_vocab = torch.tensor(self.word2vec_model.wv.key_to_index["<PAD>"])
        input_vocab = torch.tensor(self.word2vec_model.wv.key_to_index["<SOS>"])
        embedded = self.embedding(input_vocab).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)
        outputs = torch.cat((outputs, output), dim=1)
        for seq_index in range(input_seq.size(1)):
            input_vocab = input_seq[0][seq_index]
            if input_vocab == PAD_vocab:
                continue
            embedded = self.embedding(input_vocab).view(1, 1, -1)
            output, hidden = self.lstm(embedded, hidden)
            output = self.out(output)
            outputs = torch.cat((outputs, output), dim=1)
        return outputs, hidden
    
    def evaluate(self, hidden, outputs_vocab_MAX=50):
        outputs = torch.empty(0)
        input_vocab = torch.tensor(self.word2vec_model.wv.key_to_index["<SOS>"])
        outputs_vocab_len = 0
        while int(input_vocab) != self.word2vec_model.wv.key_to_index["<EOS>"] and outputs_vocab_len <= outputs_vocab_MAX:
            embedded = self.embedding(input_vocab).view(1, 1, -1)
            output, hidden = self.lstm(embedded, hidden)
            output = self.out(output)
            outputs = torch.cat((outputs, output), dim=1)
            output_vocab = torch.tensor(np.argmax(output.detach().numpy()))
            input_vocab = output_vocab
            outputs_vocab_len += 1
            
        return outputs
    

