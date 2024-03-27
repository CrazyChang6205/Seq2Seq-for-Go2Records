import torch
import torch.nn as nn

from model.DEBUGGER import seq_original

class Encoder(nn.Module):
    def __init__(self, word2vec_model, hidden_size, num_layers=1):
        super(Encoder, self).__init__()        
        vocab_size, embed_dim = word2vec_model.wv.vectors.shape # UNUSE VAR !
        self.word2vec_model = word2vec_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = word2vec_model.vector_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec_model.wv.vectors), freeze=True, padding_idx=1)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        #NOTE：embeddding這邊可以先用 word2vec 實作
        #NOTE：LSTM(input_size, hidden_size, num_layers) 為 ( 輸入層 特徵維度, 隱藏層、輸出層 特徵維度, 網路層數 )
        
    def forward(self, input_seq, hidden):
        output = None
        PAD_vocab = torch.tensor(self.word2vec_model.wv.key_to_index["<PAD>"])
        for seq_index in range(input_seq.size(1)):
            input_vocab = input_seq[0][seq_index]
            if input_vocab == PAD_vocab:
                continue
            embedded = self.embedding(input_vocab).view(1, 1, -1)
            output, hidden = self.lstm(embedded, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
