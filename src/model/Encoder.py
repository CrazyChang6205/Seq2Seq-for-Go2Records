import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, word2vec_model, hidden_size, output_size, num_layers=1):

        super(Encoder, self).__init__()        
        vocab_size, embed_dim = word2vec_model.wv.vectors.shape # UNUSE VAR !
        self.word2vec_model = word2vec_model
        self.hidden_size = hidden_size
        self.output_size = output_size # WARING VAR !
        self.num_layers = num_layers
        self.embedding_size = word2vec_model.vector_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec_model.wv.vectors), freeze=True, padding_idx=1)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        #NOTE：embeddding這邊可以先用 word2vec 實作
        #NOTE：LSTM(input_size, hidden_size, num_layers) 為 ( 輸入層 特徵維度, 隱藏層、輸出層 特徵維度, 網路層數 )
        
    def forward(self, input_seq, hidden):
        
        embedded = self.embedding(input_seq)
        #print("encoder embedded type & len：", type(embedded), len(embedded))
        #print("encoder embedded size：", embedded.shape)
        
        output, hidden = self.lstm(embedded, hidden)
        #print("encoder output type & len：", type(output), len(output))
        #print("encoder output size：", output.shape)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


