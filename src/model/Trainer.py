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
        
    def train_step(self, input_seqs, target_seqs):
        self.optimizer.zero_grad()
        total_loss = 0
        batch_size = input_seqs.size(0)
        for i in range(batch_size):
            input_seq  = input_seqs[i]
            target_seq = target_seqs[i]

            outputs, _ = self.model(input_seq.unsqueeze(0), target_seq.unsqueeze(0))
            
            # output_seq_index = []
            # for seq_index in range(len(outputs[0])):
            #     output_vocab = outputs[0][seq_index]
            #     output_data = output_vocab.squeeze().detach().cpu().numpy()
            #     max_index = np.argmax(output_data)
            #     output_seq_index.append(max_index)
            #     decoded_word = self.model.decoder.word2vec_model.wv.index_to_key[max_index]
            # output_seq = torch.Tensor(output_seq_index).int()

            loss = self.criterion(outputs[0], target_seq)
            total_loss += loss.item()

            loss.backward()

            self.optimizer.step()

        return total_loss / batch_size
    
    def train_epoch(self, dataloader):
        total_loss = 0
        num_batches = len(dataloader)
        for i, (input_seqs, target_seqs) in enumerate(tqdm(dataloader, desc='Iteration', leave=True, ncols=100)):
            
            loss = self.train_step(input_seqs, target_seqs)
            total_loss += loss
            #print(f"Iteration [{i+1}/{num_batches}], loss: {loss}")
            
        average_loss = total_loss/num_batches
        return average_loss
    
    
    
    
