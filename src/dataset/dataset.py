import os

import numpy as np
import pandas as pd

class dataset(Dataset):
    def __init__(self):

    def __getitem__(self, index):
        # 後續 Dataloader 會自動傳遞 index 變數
        # 當 index = N 時，可以想像是要回傳資料集中的第 N 筆資料
    def __len__(self):
        # 資料集的長度