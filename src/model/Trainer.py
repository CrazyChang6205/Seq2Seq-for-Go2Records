import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
import model.DEBUG_tool as DEBUG

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
            
            print(f"Batch_size [{i+1}/{batch_size}]")
            input_seq  = input_seqs[i]
            target_seq = target_seqs[i]
            outputs = self.model(input_seq.unsqueeze(0), target_seq.unsqueeze(0))
            #print(f"\tinput_seq：{input_seq}")
            
            loss = self.criterion(outputs.squeeze(0), target_seq)
            total_loss += loss.item()
            loss.backward()
            
        self.optimizer.step()
        return total_loss / batch_size
    
    def train_epoch(self, dataloader):
        total_loss = 0
        num_batches = len(dataloader)
        dataset_size = dataloader.dataset.n_samples
        batch_size = dataset_size / num_batches
        print(f"Dataset size {dataset_size}, Batch size {batch_size}, num_batches {num_batches}\n")
        for i, (input_seqs, target_seqs) in enumerate(dataloader):
            
            loss = self.train_step(input_seqs, target_seqs)
            total_loss += loss
            print(f"Iteration [{i+1}/{num_batches}], total_loss: {total_loss}")

            
        average_loss = total_loss/num_batches
        return average_loss
    
    
    
    
