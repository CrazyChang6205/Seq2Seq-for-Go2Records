import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
import model.DEBUGGER as DEBUGGER

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
    def train_step(self, input_seqs, target_seqs, loss_trend=None):
        self.optimizer.zero_grad()
        total_loss = 0
        batch_size = input_seqs.size(0)
        for i in range(batch_size):
            input_seq  = input_seqs[i]
            target_seq = target_seqs[i]

            outputs, _ = self.model(input_seq.unsqueeze(0), target_seq.unsqueeze(0))
            
            EOS_vocab  = torch.tensor([self.model.decoder.word2vec_model.wv.key_to_index["<EOS>"]])
            target_seq = torch.cat((target_seq, EOS_vocab),dim=0)
            loss = self.criterion(outputs[0], target_seq)

            if loss_trend is not None:
                loss_trend.append(loss.item())
            total_loss += loss.item()
            
            loss.backward()

            self.optimizer.step()

        return total_loss/batch_size, loss_trend
    
    def train_epoch(self, dataloader, loss_trend=None):
        """
        Parameters
        ----------
        dataloader : dataloader
            pytorch 資料包裝格式.
            
        loss_trend : list, optional. The default is None.
            紀錄該訓練週期每筆資料的 loss值.
            若該值為 None 則不紀錄
            
        Returns
        -------
        epoch_avg_loss : int
            該訓練週期的平均 loss值.
            
        loss_trend : list or None
            紀錄該訓練週期每筆資料的 loss值.
        """
        epoch_total_loss = 0
        num_batches = len(dataloader)
        for i, (input_seqs, target_seqs) in enumerate(tqdm(dataloader, desc='Iteration', leave=True, ncols=100)):
            
            batch_avg_loss, loss_trend = self.train_step(input_seqs, target_seqs, loss_trend)
            epoch_total_loss += batch_avg_loss
            #print(f"Iteration [{i+1}/{num_batches}], loss: {loss}")
            
        epoch_avg_loss = epoch_total_loss/num_batches
        return epoch_avg_loss, loss_trend
    
    
    
    
