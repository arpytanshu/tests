

#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
from time import perf_counter, sleep
from copy import deepcopy


base_model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
device = torch.device('cuda')


#%% quantize & save
###################

# model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)

# quantizer = GPTQQuantizer(bits=4, dataset="ptb", block_name_to_quantize = "model.layers", model_seqlen = 2048)
# quantized_model = quantizer.quantize_model(model, tokenizer)

# save_folder = "./quantized_model/"
# quantizer.save(model,save_folder)


#%% load and bench
###################

# load quantized model
q_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)
q_model = load_quantized_model(model=q_model, save_folder='checkpoint/')

# load un-quantized model
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)
model.eval()



#%% benchmark
###################


def quick_bench(model, prompt=None, max_length=1000, do_sample=False, temperature=0.1):
    
    if prompt == None: prompt = "Write a 25 word poetry about GPUs"

    input_ids = torch.tensor(tokenizer.encode(prompt)).view(1, -1).to(torch.long).to(model.device)
    tick = perf_counter()
    out_ids = model.generate(input_ids, max_length=max_length, do_sample=do_sample, temperature=temperature)
    
    elapsed = perf_counter()-tick
    num_tokens = len(out_ids.ravel()) - len(input_ids.ravel())

    prompt_length = input_ids.ravel().shape[0]
    generations = tokenizer.decode(out_ids.ravel().detach().tolist()[prompt_length:], skip_special_tokens=True)
    print('Prompt::', prompt)
    print('Generations::', generations, end='\n')
    print(f"{elapsed=:.4f}s for {num_tokens} tkns @ {num_tokens/elapsed} tkn/s")
    



PROMPT = "count from 1 to 20 in binary."
temp = 1.0

print('Base Model::')
quick_bench(model, prompt=PROMPT, do_sample=False, temperature=temp);
print('\n', '='*16)
print('\n', '='*16, end='\n')
print('Quantized model::')
quick_bench(q_model, prompt=PROMPT, do_sample=False, temperature=temp);
