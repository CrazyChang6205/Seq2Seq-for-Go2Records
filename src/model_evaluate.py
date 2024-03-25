import os

current_path = os.getcwd()
print("當前工作檔案路徑:", current_path)
target_path = 'D:/user/桌面/Meeting/圍棋術語分類及評論生成/src/'
os.chdir(target_path)
new_path = os.getcwd()
print("切換工作檔案路徑:", new_path)

import numpy as np
import torch
from gensim.models import Word2Vec

from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Seq2Seq import Seq2Seq
from model.DEBUGGER import extract_substrings, seq_original

hidden_size = 16
num_layers  = 16
epochs      = 5
seqe2seq_version = 1
word2vec_version = 2

input_w2v_model  = Word2Vec.load(f'./temp/word2vec/records/w2v_byR_version_{word2vec_version}.model')
output_w2v_model = Word2Vec.load(f'./temp/word2vec/comment/w2v_byC_version_{word2vec_version}.model')

encoder = Encoder(input_w2v_model , hidden_size, num_layers)
decoder = Decoder(output_w2v_model, hidden_size, num_layers)
seq2seq = Seq2Seq(encoder, decoder)

seq2seq.load_state_dict(torch.load(f'./temp/seq2seq/seq2seq_S{hidden_size}_L{num_layers}_e{epochs}_version_{seqe2seq_version}.pth'))
seq2seq.eval()

input_seq_list  = ["", 
              "Bdc",
              "Wnq Biq Wgq Bnc Wic Bck Wcf Bgc Wci Bcn Wqk Bqi Wqg Bqm Wlc Bne Wof Bnf Woc Bnb Wng Bmg Wnh Bob Wpb Bqb Wpa Bmh Wmi Bni Wnj Boi Wog Bpe Woe Bnd",
              "Wnq Biq Wgq Bnc Wic Bck Wcf Bgc Wci Bcn Wqk Bqi Wqg Bqm Wlc Bne Wof Bnf Woc Bnb"]


in_word_to_index  = {word: idx for idx, word in enumerate(input_w2v_model.wv.index_to_key) }
def text_to_tensor_byR(text):
    return torch.tensor([in_word_to_index.get(word, in_word_to_index.get('<UNK>')) for word in text.split()])

for seq_index in range(len(input_seq_list)):
    input_seq = input_seq_list[seq_index]
    input_seq = "<SOS> "+ input_seq +" <EOS>"
    input_seq = text_to_tensor_byR(input_seq)
    initial_hidden = seq2seq.encoder.init_hidden(1)
    outputs = seq2seq.evaluate(input_seq.unsqueeze(0))    
    for outputs_index in range(len(outputs[0])):
        output_data = outputs[0][outputs_index].detach().numpy()
        max_index = np.argmax(output_data)
        decoded_word = output_w2v_model.wv.index_to_key[max_index]
        print(decoded_word, end=' ')
    print()
    print()

