
#%%
import torch
from model import GenericConfig, GPT
from transformers import GPT2Tokenizer
from torch.nn import functional as F

class GPT2Surgery(GPT):
    # A gpt2 class with lot more forward methods.
    # step 1: forward through embedding layers
    # step 2: forward through step1 + all blocks
    # step 3: forward through step2 + lm_head, (default forward)
    
    @   torch.no_grad()
    def forward_sleek(self, idx, step=3):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)


        # step 1: forward through embedding layers
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if step == 1: return x
        
        # step 2: forward through step1 + all blocks
        if step >= 2:    
            for block in self.transformer.h:
                x = block(x)
        if step == 2: return x
        
        # step 3: forward through step2 + lm_head, (default forward)
        x = self.transformer.ln_f(x)
        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        return logits

    @   torch.no_grad()
    def forward_injection(self, idx, injection_tensor, injection_point=1):
        '''
        injection_point needs to be one pf 1, 2, 3
        if injection_pt == 1, inject before the blocks.
        if injection_pt == 2, inject after the 3rd block.
        if injection_pt == 3, inject after the blocks.
        '''

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # step 1: forward through embedding layers
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # step 2: forward through step1 + all blocks
        if injection_point == 1: x = x + (injection_tensor * 0.02)
        for block in self.transformer.h:
            x = block(x)
        
        if injection_point == 3: x = x + injection_tensor
        
        # step 3: forward through step2 + lm_head, (default forward)
        x = self.transformer.ln_f(x)
        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        return logits
    
    @   torch.no_grad()
    def injected_generate(self, idx, max_new_tokens, injection_tensor, injection_point=1, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.forward_injection(idx_cond, injection_tensor, injection_point=injection_point)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Surgery.from_pretrained(model_type="gpt2")
model.eval()

#%%



prompt = 'The quick brown fox'
prompt_idx = tokenizer.encode(prompt, return_tensors="pt")
default_generation = model.generate(prompt_idx, max_new_tokens=40, temperature=1, top_k=1)
print(f"prompt: {prompt}")
print(f"default generation: {tokenizer.decode(default_generation[0])}")




inj_string = 'cat'
inj_ids = tokenizer.encode(inj_string, return_tensors="pt")
out_s2 = model.forward_sleek(inj_ids, step=2)
inj_tensor = out_s2[:, -1, :].unsqueeze(1)
print(f"injection_tensor: {inj_tensor.shape}")


inj_generation = model.injected_generate(prompt_idx,
                                         max_new_tokens=40,
                                         injection_tensor=inj_tensor,
                                         injection_point=1,
                                         temperature=1, top_k=1)
print(f"injected generation: {tokenizer.decode(inj_generation[0])}")











#%%



