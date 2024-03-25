import os
import time
from tqdm import tqdm
current_path = os.getcwd()
print("當前工作檔案路徑:", current_path)
target_path = 'D:/user/桌面/Meeting/圍棋術語分類及評論生成/src/'
os.chdir(target_path)
new_path = os.getcwd()
print("切換工作檔案路徑:", new_path)

from gensim.models import Word2Vec
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from model.Dataset import Custom_Dataset
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Seq2Seq import Seq2Seq
from model.Trainer import Trainer

# 初始化模型参数
word2vec_version = 2
input_w2v_model  = Word2Vec.load('./temp/word2vec/records/w2v_byR_version_'+str(word2vec_version)+'.model')
output_w2v_model = Word2Vec.load('./temp/word2vec/comment/w2v_byC_version_'+str(word2vec_version)+'.model')
hidden_size = 64
num_layers  = 64
learning_rate = 0.01
batch_size = 1
num_epochs = 5

# 初始化 Dataset, DataLoader
dataset_name = "Go_All_R2C"
custom_dataset = Custom_Dataset(dataset_name, input_w2v_model, output_w2v_model)

from torch.nn.utils.rnn import pad_sequence
def seqs_padded(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs_padded  = pad_sequence(input_seqs , batch_first=True, padding_value=1)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=1)
    return input_seqs_padded, target_seqs_padded
train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=seqs_padded, drop_last=True)

# 模型 優化器 損失函數 Trainer
encoder = Encoder(input_w2v_model , hidden_size, num_layers)
decoder = Decoder(output_w2v_model, hidden_size, num_layers)
seq2seq_model = Seq2Seq(encoder, decoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(seq2seq_model.parameters(), lr=learning_rate)
trainer = Trainer(seq2seq_model, criterion, optimizer)

# 檢視訓練參數狀態
num_batches = len(train_dataloader)
dataset_size = train_dataloader.dataset.n_samples
print(f' input_w2v vector_size   = { input_w2v_model.wv.vectors.shape}')
print(f'output_w2v vector_size   = {output_w2v_model.wv.vectors.shape}')
print(f'hidden_size & num_layers = {hidden_size} , {num_layers} = {hidden_size*num_layers}')
print(f'Datasetsize & batch_size = {dataset_size}, {batch_size}')
print(f' num_epochs & num_batches= {num_epochs} , {num_batches}')

# 訓練迴圈
trainer.model.train()
for epoch in range(num_epochs):
    
    loss = trainer.train_epoch(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}\n')
    
    # 保存模型
    version = 1
    save_Filename = f'./temp/seq2seq/seq2seq_w2v{str(word2vec_version)}_S{str(hidden_size)}_L{str(num_layers)}_e{str(epoch+1)}_version_{str(version)}.pth'
    while os.path.exists(save_Filename):
        version +=1
        save_Filename = f'./temp/seq2seq/seq2seq_w2v{str(word2vec_version)}_S{str(hidden_size)}_L{str(num_layers)}_e{str(epoch+1)}_version_{str(version)}.pth'
    torch.save(seq2seq_model.state_dict(), save_Filename)
