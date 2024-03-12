import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, word2vec_model, hidden_size, output_size, num_layers=1):
        
        super(Decoder, self).__init__()
        self.word2vec_model = word2vec_model
        self.hidden_size = hidden_size
        self.output_size = self.word2vec_model.cum_table.size
        self.num_layers = num_layers
        self.embedding_size = word2vec_model.vector_size
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec_model.wv.vectors))
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        #NOTE：LSTM(input_size, hidden_size, num_layers) 為 ( 輸入層 特徵維度, 隱藏層、輸出層 特徵維度, 網路層數 )

    def forward(self, input_seq, hidden):
        
        embedded = self.embedding(input_seq) 
        #print("decoder embedded type & len：", type(embedded), len(embedded))
        #print("decoder embedded size：", embedded.shape)
        
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)
        #print("decoder output type & len：", type(output), len(output))
        #print("decoder output size：", output.shape)
        return output, hidden
    
    def test_forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq) 
        output, hidden = self.lstm(embedded)
        cell = hidden[1]
        output = self.out(output)
        return output, hidden, cell
    
    def evaluate(self, input_seq):
        return
    

