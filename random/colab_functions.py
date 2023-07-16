

#%%
import torch
from time import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import pipeline
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def get_stat():
    torch.cuda.empty_cache()
    allocated = round(torch.cuda.memory_reserved(0)/1024/1024/1024, 3)
    cached = round(torch.cuda.memory_reserved(0)/1024/1024/1024, 3)
    print(f"allocated::{allocated}JB  cached::{cached}JB")
    return allocated

def generate(prompt, tokenizer, model, num_tokens, num_runs=2):
    # does not support batching, use this one for eval generations
    elapsed_runs = []
    for _ in range(num_runs):    
        elapsed = 0
        input_ids = tok.encode(prompt)
        print('"', tokenizer.decode(input_ids), '"', end='')
        for ix in range(num_tokens):
            with torch.no_grad():
                tick = time()
                out = model(torch.tensor(input_ids).to(torch.int), return_dict=True)
                probs = torch.softmax(out.logits[-1, :], axis=0)
                out_token_id = torch.multinomial(probs, 1)[0].item()
                tock = time()
                elapsed += tock - tick
                input_ids.append(out_token_id)
                print(tokenizer.decode(out_token_id).replace('\n', ' '), end='')
        elapsed_runs.append(elapsed)
        print()
        print('elapsed::', round(elapsed, 3))
    return sum(elapsed_runs) / len(elapsed_runs)


def batched_generate(prompt, tokenizer, model, num_tokens, batch_size, num_runs=2):
    # does support batching
    elapsed_runs = []
    for _ in range(num_runs):    
        elapsed = 0
        input_ids = tok.encode(prompt)
        input_ids = np.array([input_ids] * batch_size)
        # print('"', tokenizer.decode(input_ids), '"', end='')
        for ix in range(num_tokens):
            with torch.no_grad():
                inputs = torch.tensor(input_ids).to(torch.int).to('cuda')
                tick = time()
                out = model(inputs, return_dict=True)
                probs = torch.softmax(out.logits[:, -1, :], axis=1)
                out_token_id = torch.multinomial(probs, 1)[:, -1].cpu().numpy()
                tock = time()
                elapsed += tock - tick
                input_ids = np.hstack([input_ids, out_token_id.reshape(-1, 1)])
                del(inputs)
                del(out)
        elapsed_runs.append(elapsed)
        # print('elapsed::', round(elapsed, 3))
    del()
    
    avg_elapsed = sum(elapsed_runs) / len(elapsed_runs)
    rate = round(num_tokens / avg_elapsed, 3)
    print(f"device::{model.device}\tdtype::{model.dtype}\tbatch::{batch_size}\trate::{rate} tkn/sec")

def get_perturbration_plot(model, tokenizer, prompt, num_runs=10):
        
    def _perturbrate_char(prompt, perc):
        '''
        This function perturbrated the prompt string by swapping characters
        '''
        n = len(prompt) # total entities
        n_p = int(n * perc) # entities to perturbrate
        prompt = list(prompt)
        ix_to_pert = np.random.randint(0, n, n_p)
        for _ in range(len(ix_to_pert)):
            s, t = np.random.choice(ix_to_pert, 2)
            prompt[t], prompt[s] = prompt[s], prompt[t]
        return ''.join(prompt)

    def _get_embeddings(model, tokenizer, prompt_list):
        embeddings = []
        for p in prompt_list:
            input_ids = torch.tensor(tok.encode(p)).to(torch.int)
            with torch.no_grad():
                out = model_fp32.forward(input_ids)
            embs = out[0].mean(axis=0).cpu().numpy()
            embeddings.append(embs)
        return np.stack(embeddings)

    cos_sims = []
    pert_percs = np.linspace(0, 1, 20) 
    for _ in range(num_runs):
        pert_prompts = []
        for perc in pert_percs:
            pert_prompt = _perturbrate_char(prompt, perc)
            # print(f"perc::{perc}\t{pert_prompt}")
            pert_prompts.append(pert_prompt)
        
        embs = _get_embeddings(model_fp32, tok, pert_prompts)
        c_s = cosine_similarity(embs)[0]
        cos_sims.append(c_s)
    avg_cos_sims = np.stack(cos_sims).mean(axis=0)
    plt.plot(pert_percs, avg_cos_sims)

    return avg_cos_sims, pert_percs



#%% Test batched generate 



model_str = "cerebras/Cerebras-GPT-111M"
model_fp32 = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float32).cuda()
model_fp16 = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16).cuda()
model_int8 = AutoModelForCausalLM.from_pretrained(model_str, load_in_8bit=True, device_map="auto")
tok = AutoTokenizer.from_pretrained(model_str)

prompt = "websites:  "
for model in [model_fp32, model_fp16, model_int8]:
    for batch in [1, 4, 8, 16]:
        batched_generate(prompt     = prompt,
                         num_tokens = 100,
                         tokenizer  = tok,
                         model      = model,
                         num_runs   = 2,
                         batch_size = batch)


del(model_fp32)
del(model_fp16)
del(model_int8)
torch.cuda.empty_cache()
get_stat()






