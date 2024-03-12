import os

current_path = os.getcwd()
print("當前工作檔案路徑:", current_path)
target_path = 'D:/user/桌面/Meeting/圍棋術語分類及評論生成/src/'
os.chdir(target_path)
new_path = os.getcwd()
print("切換工作檔案路徑:", new_path)

import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from torch.utils.data import DataLoader

from model.Dataset import Custom_Dataset
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Seq2Seq import Seq2Seq
from model.Trainer import Trainer
from model.DEBUG_tool import seq_original

input_w2v_model  =  Word2Vec.load('./temp/W2V_Go_All_R_NEW.model')
output_w2v_model = Word2Vec.load('./temp/W2V_Go_All_C_NEW.model')
hidden_size = 256
output_size = 256
num_layers = 2

encoder = Encoder(input_w2v_model , hidden_size, num_layers)
decoder = Decoder(output_w2v_model, hidden_size, output_size, num_layers)
model_0306 = Seq2Seq(encoder, decoder)

model_0306.load_state_dict(torch.load('seq2seq_model_20240306_version1.pth'))

model_0306.eval()

input_seq_list  = ["<SOS> <EOS>", 
              "<SOS> Bdc <EOS>",
              "<SOS> Wnq Biq Wgq Bnc Wic Bck Wcf Bgc Wci Bcn Wqk Bqi Wqg Bqm Wlc Bne Wof Bnf Woc Bnb Wng Bmg Wnh Bob Wpb Bqb Wpa Bmh Wmi Bni Wnj Boi Wog Bpe Woe Bnd <EOS>"]

target_seq_list = ["<SOS>", 
              "<SOS>", 
              "<SOS>"]

example_dataset_list = []
for i in range(0, len(input_seq_list)):
    example_dataset_list.append([ input_seq_list[i], target_seq_list[i] ])
example_dataset = Custom_Dataset(example_dataset_list)

import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def seqs_padded(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded  = pad_sequence(input_seqs , batch_first=True, padding_value=1)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=1)
    return input_seqs_padded, target_seqs_padded
example1_dataloader = DataLoader(example_dataset, batch_size=1, shuffle=True, collate_fn=seqs_padded, drop_last=True)
   

"""
for i , (input_seqs, target_seqs) in enumerate(example1_dataloader):
    
    batch_size = input_seqs.size(0)
    for j in range(batch_size):
        input_seq  = input_seqs[j].unsqueeze(0)
        target_seq = target_seqs[j].unsqueeze(0)
        initial_hidden = model_0306.encoder.init_hidden(input_seq.size(1))
        
    
    outputs = model_0306(input_seq[0].unsqueeze(0), target_seq[0].unsqueeze(0))
    output_data = outputs.squeeze().detach().cpu().numpy()
    max_index = np.argmax(output_data)
    decoded_word = output_w2v_model.wv.index_to_key[max_index]

    print(decoded_word)
"""
    
    
