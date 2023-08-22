#%%

#%%
import sys
import numpy as np
import torch
from time import time, sleep
from generic import Config

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

# %%


def prepare_llama_prompt(prompt):
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"

def prepare_model_input(prompt:str) -> list[str]:
    return tokenizer.encode(prompt, add_special_tokens=True)

def generate(prompt, gen_cfg, return_prompt = False, TIMEOUT=None):
    # will return generations and the generation stats
    prompt = prepare_llama_prompt(prompt)
    input_ids = prepare_model_input(prompt)
    init_len = len(input_ids)
    
    stats = {'elapsed': []}

    timeout_tick = time()

    for ix in range(gen_cfg.max_length):
        with torch.no_grad():
            tick = time()
            
            if ix > 0:
                out = model(torch.tensor(input_ids).to(torch.int).unsqueeze(0).to(model.device),
                            return_dict=True,
                            output_hidden_states=True,
                            past_key_values=out.past_key_values,
                            use_cache=True)
            else:
                out = model(torch.tensor(input_ids).to(torch.int).unsqueeze(0).to(model.device),
                            return_dict=True,
                            output_hidden_states=True,
                            use_cache=True)
                
            probs = torch.softmax(out.logits[:, -1, :].squeeze()/gen_cfg.temperature, axis=0)
            out_token_id = torch.multinomial(probs, 1)[0].item()
            tock = time()
            
            stats['elapsed'].append(tock - tick)
            
            input_ids.append(out_token_id)
            dec = tokenizer.decode(out_token_id)
        
        # print(dec, end=' ')

        if out_token_id == tokenizer.eos_token_id:
            break;
        if (TIMEOUT is not None) and (time() - timeout_tick) > TIMEOUT:
            break;

    if len(stats['elapsed']) > 0:
        stats['avg_time_per_token'] = sum(stats['elapsed']) / len(stats['elapsed'])
    
    if return_prompt:
        gen = tokenizer.decode(input_ids)
    else:
        gen = tokenizer.decode(input_ids[init_len:])
    
    return gen, stats


'''
prompt = "What is the capital of Japan?.\n"

COOLDOWN_SLEEP = 5

gen_cfg = Config(config={"max_length": 256, "temperature": 0.5})

gen, stats = generate(prompt, gen_cfg, return_prompt=True, TIMEOUT=20)

print()
print(gen)
print(f"Generated {len(stats['elapsed'])} tokens in {sum(stats['elapsed']):.3f}sec.")
print(f"{stats['avg_time_per_token']:.3f} sec/token")
print("==========")

sleep(COOLDOWN_SLEEP)
'''

def print_probs(probs):
    probs = probs.cpu().to(torch.float16).numpy()
    probs = probs.round(3)

    # print top k probs
    top_k = 10
    for ix, p in enumerate(probs):
        sorted_id = np.argsort(p)[::-1][:top_k]
        sorted_probs = p[sorted_id]
        print()
        print(ix, end=' ')
        for id, p in zip(sorted_id, sorted_probs):
            if p > 0:
                print(f"{tokenizer.decode(id)}/{p:.3f}", end = ' ')
            else:
                break;
    print()
    print('='*32)

def fixed_block_sampler(probs, blk_id):
    # probs: probs of all blocks, [ num_blocks x vocab ]
    p = probs[blk_id, :]
    token_id = torch.multinomial(p, 1).item()
    return token_id

""
# %%


prompt = "Write a 4 line poetry about cats wearing pants."
prompt = prepare_llama_prompt(prompt)
gen_cfg = Config(config={"max_length": 10, "temperature": 0.9})


for blk_id in range(27, 33):
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
            out_token_id = fixed_block_sampler(dbg_probs, blk_id)
            
            # print_probs(dbg_probs)
            show_token_probs(dbg_probs)
            # DEBUG

            # probs = torch.softmax(out.logits[:, -1, :].squeeze()/gen_cfg.temperature, axis=0)
            # out_token_id = torch.multinomial(probs, 1)[0].item()
            input_ids.append(out_token_id)
            
        if out_token_id == tokenizer.eos_token_id:
            break;


    # gen = tokenizer.decode(input_ids)
    gen = tokenizer.decode(input_ids[init_len:])
    print()
    print(f"layer #: {blk_id}")
    print("man:\t", prompt)
    print("machine:", gen)

    print()
    print("="*32)

probs = dbg_probs.clone().detach().cpu().to(torch.float32).numpy()

# %%

def show_token_probs(dbg_probs):
    fig, axs = plt.subplots(10, 1, figsize=(3, 20))

    probs = dbg_probs.clone().detach().cpu().to(torch.float32).numpy()
    print('layer#', '\t', 'num_tokens_under_topP_0.95', '\t', 'tokens_under_topP_0.95') 
    for jx, p in enumerate(probs[-10:]):
        
        p_ix = np.argsort(p)[::-1]
        p = p[p_ix]

        ix = 0
        s = 0
        while(s<0.95):
            s += p[ix]
            ix -=- 1
            if ix > 15: break;
        
        relevant_token_ids = p_ix[:ix]
        relevant_token_probs = p[:ix]
        relevant_tokens = [tokenizer.decode(tok) for tok in relevant_token_ids]

        axs[jx].bar(relevant_tokens, relevant_token_probs)
        # print(jx, '\t', 32000 - np.where(p<0.0001)[0].shape[0], relevant_tokens) 





# %%

import matplotlib.pyplot as plt


def plot_hist(ax, tokens, token_probs):
    ax.bar(tokens, token_probs)



# %%
