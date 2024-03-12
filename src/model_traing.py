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

# 初始化數據集和 DataLoader
load_file_dataset_name = "Go_All_R2C"
custom_dataset = Custom_Dataset(load_file_dataset_name)
batch_size = 4

# 新增<PAD>處理
from torch.nn.utils.rnn import pad_sequence
def seqs_padded(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded  = pad_sequence(input_seqs , batch_first=True, padding_value=1)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=1)
    return input_seqs_padded, target_seqs_padded
train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=seqs_padded, drop_last=True)

# 初始化模型参数
input_w2v_model  =  Word2Vec.load('./temp/W2V_Go_All_R_NEW.model')
output_w2v_model = Word2Vec.load('./temp/W2V_Go_All_C_NEW.model')
hidden_size = 256
output_size = 256
num_layers = 2
learning_rate = 0.01
num_epochs = 1

# 模型 優化器 損失函數 Trainer
encoder = Encoder(input_w2v_model , hidden_size, output_size, num_layers)
decoder = Decoder(output_w2v_model, hidden_size, output_size, num_layers)
seq2seq_model = Seq2Seq(encoder, decoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(seq2seq_model.parameters(), lr=learning_rate)
trainer = Trainer(seq2seq_model, criterion, optimizer)

print(" input_w2v MaxIndex & vector_size：", len(input_w2v_model.wv.index_to_key),  input_w2v_model.vector_size)
print("output_w2v MaxIndex & vector_size：", len(output_w2v_model.wv.index_to_key), input_w2v_model.vector_size)
print("hidden_size：", hidden_size)
print("num_layers：", num_layers)

# 訓練迴圈
trainer.model.train()
print("train model！")
for epoch in range(num_epochs):
    loss = trainer.train_epoch(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}\n')
    

# 保存模型
torch.save(seq2seq_model.state_dict(), 'seq2seq_model.pth')
