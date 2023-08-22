#%%

import sys
import numpy as np
import torch
from time import time, sleep
from generic import Config
import matplotlib.pyplot as plt


sys.path.append("/shared/CO/huggingface_/transformers/src/")


from modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import LlamaTokenizerFast
from transformers import TextStreamer


device = torch.device("cuda:0")

model_string = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizerFast.from_pretrained(model_string, use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_string, torch_dtype=torch.bfloat16).to(device)

model.eval()



def prepare_llama_prompt(prompt):
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"

def prepare_model_input(prompt:str) -> list[str]:
    return tokenizer.encode(prompt, add_special_tokens=True)

def fixed_block_sampler(probs, blk_id):
    # probs: probs of all blocks, [ num_blocks x vocab ]
    p = probs[blk_id, :]
    token_id = torch.multinomial(p, 1).item()
    return token_id




#%%

prompt = "What is the capital of Japan? Answer in less than 6 words."
prompt = prepare_llama_prompt(prompt)
gen_cfg = Config(config={"max_length": 10, "temperature": 0.9})

NUM_LAYERS_TO_PLOT = 32
MAX_TOKENS_TO_PLOT = 10
PROBABLITY_MASS_TO_COVER = 0.95
fig, axs = plt.subplots(NUM_LAYERS_TO_PLOT, gen_cfg.max_length, figsize=(80,80))

input_ids = prepare_model_input(prompt)
init_len = len(input_ids)


for ix in range(gen_cfg.max_length):
    with torch.no_grad():
        tick = time()
        out = model(torch.tensor(input_ids).to(torch.int).unsqueeze(0).to(model.device),
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=False)
        
        # DEBUG
        last_hidden = torch.stack(out.hidden_states).squeeze()[:, -1, :]
        logits = model.lm_head(last_hidden) 
        logits = logits / gen_cfg.temperature
        dbg_probs = torch.softmax(logits, dim=1)
        out_token_id = fixed_block_sampler(dbg_probs, -1)
        input_ids.append(out_token_id)
    

    probs = dbg_probs.clone().detach().cpu().to(torch.float32).numpy()
    # print('layer#', '\t', 'num_tokens_under_topP_0.95', '\t', 'tokens_under_topP_0.95') 
    for jx, p in enumerate(probs[-NUM_LAYERS_TO_PLOT:]):
        ax = axs[jx][ix]

        sorted_ix = np.argsort(p)[::-1]
        p = p[sorted_ix]
        count = 0
        sum = 0
        # while(sum<PROBABLITY_MASS_TO_COVER):
            # sum += p[count]
            # count -=- 1
            # if count > MAX_TOKENS_TO_PLOT: break;
        count = 15
        
        relevant_token_ids = sorted_ix[:count]
        relevant_token_probs = p[:count]
        relevant_tokens = [tokenizer.decode(tok) for tok in relevant_token_ids]

        ax.bar(relevant_tokens, relevant_token_probs)
        ax.set_xticks(relevant_tokens, relevant_tokens, rotation=90)
        


        
    if out_token_id == tokenizer.eos_token_id:
        break;

fig.tight_layout()

gen = tokenizer.decode(input_ids[init_len:])
print()
print("man:\t", prompt)
print("machine:", gen)

print()
print("="*32)

probs = dbg_probs.clone().detach().cpu().to(torch.float32).numpy()



# %%
