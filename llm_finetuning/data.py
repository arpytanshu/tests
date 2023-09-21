
import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from time import time
import random

class CustomDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_len=512):
        s = time()
        df = pd.read_csv(data_file)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.text = []
        self.lengths = [0]
        for ix, row in df.iterrows():
            line = f"{row.agent}: {row.cleaned_content} </s>"
            enc_line = tokenizer.encode(line, add_special_tokens=False)
            self.text.extend(enc_line)
            self.lengths.append(self.lengths[-1] + len(enc_line))

        self.max_allowable_token_ix = len(self.text) - self.max_seq_len
        self.lengths = np.array(self.lengths)

        self.lengths = self.lengths[self.lengths < self.max_allowable_token_ix]
        print(f"elapsed in Dataset: {time() - s:.3f}s")

        
    def __len__(self):
        return len(self.lengths) - 1

    def __getitem__(self, index):
        random_ix = random.randint(0, len(self.lengths) - 1)
        start_ix = self.lengths[random_ix]
        input_ids = torch.tensor(self.text[start_ix : start_ix+self.max_seq_len])
        labels = torch.tensor(self.text[start_ix+1 : start_ix+self.max_seq_len+1])
        return input_ids, labels

