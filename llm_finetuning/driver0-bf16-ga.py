
#%%
'''
####################################
basic driver with bf16 training + GA
####################################
'''


#%% imports and configurations


import warnings
from time import time
warnings.filterwarnings("ignore")

import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from generic import Config
from configurations import all_config
from data import CustomDataset
from train_utils import generate


cfg = Config(config=all_config)

cfg.preprocesses_data_path = "/shared/CO/arpytanshu_/localLLM/dataset/WhatsAppChat-cleaned.csv"
cfg.model.max_seq_len = 256
# cfg.ft.base_model = "lmsys/vicuna-7b-v1.3"
cfg.ft.base_model = "cerebras/Cerebras-GPT-111M"
cfg.ft.device = torch.device("cuda:0")
cfg.ft.num_epochs = 1
cfg.ft.batch_size = 40
cfg.ft.grad_accu_steps = 4
cfg.ft.learning_rate = 0.0005
cfg.ft.grad_clip = 1.0


tokenizer           = AutoTokenizer.from_pretrained(cfg.ft.base_model, use_fast=False)
ds                  = CustomDataset(cfg.preprocesses_data_path, tokenizer, max_seq_len=cfg.model.max_seq_len)
dl                  = DataLoader(ds, batch_size=cfg.ft.batch_size, shuffle=True, num_workers=4)
model               = AutoModelForCausalLM.from_pretrained(cfg.ft.base_model, torch_dtype=torch.bfloat16).to(cfg.ft.device)
optimizer           = AdamW(model.parameters(), lr=cfg.ft.learning_rate)
num_training_steps  = cfg.ft.num_epochs * len(dl) // cfg.ft.grad_accu_steps
lr_scheduler        = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

scaler              = torch.cuda.amp.GradScaler(enabled=False)
ctx                 = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)



wandb.init("localLLM-run1-bf16")



model.train()
for iteration in range(num_training_steps):
    
    iter_s = time()
    
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(cfg.ft.grad_accu_steps):
        
        batch = next(iter(dl))
        batch = {"input_ids": batch[0].to(cfg.ft.device),
                'labels': batch[1].to(cfg.ft.device)}

        with ctx:
            outputs = model(**batch)
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                batch['labels'].reshape(-1))
            # huggingface models create shifted labels in forward, 
            # since our data-loader already does the shifting, 
            # using output.loss from CausalLMOutputWithCrossAttentions 
            # creates absurd generations.
            # loss = outputs.loss
            loss = loss / cfg.ft.grad_accu_steps
        scaler.scale(loss).backward()
    
    if cfg.ft.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ft.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()
    

    iter_e = time()

    with torch.no_grad():
        wandb.log({'elapsed': iter_e - iter_s, 
                   'lr': lr_scheduler.get_lr()[0],
                   'loss': loss.detach().cpu().item() * cfg.ft.grad_accu_steps})

    if (iteration % cfg.io.log_interval) == 0:
        print(f"iter:{iteration}/{num_training_steps} elapsed: {iter_e - iter_s:.3f} loss:{outputs.loss.detach().item()}")

    if (iteration % cfg.io.gen_interval) == 0:
        generate(model, tokenizer, cfg)
        model.train()
    
    break;



# %%
