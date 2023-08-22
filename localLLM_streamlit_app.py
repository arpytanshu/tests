'''
test streamlit 
'''

import torch
import streamlit as st
from time import time
from transformers import LlamaForCausalLM, LlamaTokenizer

# globals
MODEL_STRING = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

torch.set_num_threads(16)

@st.cache_resource
def get_model():
    model = LlamaForCausalLM.from_pretrained(MODEL_STRING, torch_dtype=torch.bfloat16)
    model = model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def get_tokenizer():
    return LlamaTokenizer.from_pretrained(MODEL_STRING, use_fast=False)

def prepare_llama_prompt(prompt:str) -> str:
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"

def prepare_model_input(prompt:str) -> list[str]:
    return tokenizer.encode(prompt, add_special_tokens=True)

def generate(prompt, gen_cfg, return_prompt = False):
    # will return generations and the generation stats
    prompt = prepare_llama_prompt(prompt)
    input_ids = prepare_model_input(prompt)
    init_len = len(input_ids)
    
    stats = {'elapsed': []}

    timeout_tick = time()

    for ix in range(gen_cfg["max_length"]):
        with torch.no_grad():
            tick = time()

            out = model(torch.tensor(input_ids).to(torch.int).unsqueeze(0).to(model.device),
                        return_dict=True,
                        output_hidden_states=True,
                        use_cache=False)
                
            probs = torch.softmax(out.logits[:, -1, :].squeeze()/gen_cfg["temperature"], axis=0)
            out_token_id = torch.multinomial(probs, 1)[0].item()
            
            stats['elapsed'].append(time() - tick)            
            
        if out_token_id == tokenizer.eos_token_id:
            break;
        else:
            input_ids.append(out_token_id)

        if (gen_cfg['timeout_bool'] > 0) and (time() - timeout_tick) > gen_cfg['timeout_int']:
            break;

    if len(stats['elapsed']) > 0:
        stats['avg_time_per_token'] = sum(stats['elapsed']) / len(stats['elapsed'])
    
    if return_prompt:
        gen = tokenizer.decode(input_ids)
    else:
        gen = tokenizer.decode(input_ids[init_len:])
    
    return gen, stats



model = get_model()
tokenizer = get_tokenizer()


st.sidebar.write("Generation Config")
GEN_CFG = {}
GEN_CFG['temperature'] = st.sidebar.slider("temperature", 0.01, 1.2, 0.01, 0.01, "%f")
GEN_CFG['max_length'] = st.sidebar.slider("max length", 10, 2048, 256, 5, "%d")
GEN_CFG['timeout_bool'] = st.sidebar.checkbox("Timeout", False)
GEN_CFG['timeout_int'] = st.sidebar.number_input("Timeout (s)", 5, 100, 5, 1, "%d")


prompt = st.text_input('Prompt:')
generations, stats = generate(prompt, GEN_CFG)
st.write(generations)
st.write(f"{stats['avg_time_per_token']:.3f} s/token")
st.write(f"generated {len(stats['elapsed'])} token in {sum(stats['elapsed']):.3f}s")



