import time


model_config = {
    "max_seq_len": 256,
    "max_position_embeddings": None
}

# only training specific configs
finetuning_config = {
    "base_model": "lmsys/vicuna-7b-v1.3",
    
    "num_epochs": None,
    "batch_size": 8,


    # adamw optimizer
    "learning_rate": 6e-4,              # max learning rate
    "max_iters": 600000,                # total number of training iterations
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,                   # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay settings
    "decay_lr": True,                   # whether to decay the learning rate
    "warmup_iters": 200,                # how many steps to warm up for
    "lr_decay_iters": 600000,           # should be ~= max_iters per Chinchilla
    "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # training opts
    "grad_accu_steps": 16, # used to simulate larger batch sizes
    
    # system - training opts
    "device": "cpu", # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "dtype": "float32", # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    "compile": False, # use PyTorch 2.0 to compile the model to be faster
}

generation_config = {
    "max_length": 128,
    "do_sample": True,
    "top_k": 20,
    "temperature": 0.2,
}

# model save-load, wandb, loss-progress, distributed training etc.
logistics_config = {
    # wandb
    "wandb_log": True,
    "wandb_project": "gptts",
    "wandb_run_name": "gptts_run_" + time.strftime('%Y%B%d_%X'),

    # I/O
    "out_dir": None, # filled by the script
    "gen_interval": 100,
    "log_interval": 10,
    "always_save_checkpoint": False, # if True, always save a checkpoint after each eval
    "init_from": 'scratch',         # 'scratch' or 'resume'

    "preprocesses_data_path": "/shared/CO/arpytanshu_/localLLM/dataset/WhatsAppChat-cleaned.csv"
}

all_config = {
    "model": model_config,
    "ft": finetuning_config,
    "io": logistics_config,
    "gen": generation_config
}
