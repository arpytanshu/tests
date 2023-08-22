import torch
import math
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def estimate_loss(model, dataset, ctx, cfg):
    model.eval()
    losses = torch.zeros(cfg.io.eval_iters)
    for k in range(cfg.io.eval_iters):
        X, Y = dataset.get_batch(cfg.data.batch_size, split)
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()

@torch.no_grad()
def plot_eval(model, tokenizer, device, path=None):
    def _gen_daily_signal():
        num_days = 7
        num_dp_per_day = 24
        x = np.linspace(0, num_days*2*np.pi, num_dp_per_day*num_days)
        e = np.random.randn(num_dp_per_day*num_days) * 0.3
        x = np.sin(x+e) + 5
        return x

    N = 5
    fig, axs = plt.subplots(N, 1, figsize=(20, 2*N))
    for ix, temp in enumerate(np.linspace(0.2, 0.99, N)):
        x = _gen_daily_signal()
        y = gen_forecast(x, model, tokenizer, 100, device, temperature=temp, top_k=100)

        axs[ix].plot(range(len(x)), y[:len(x)])
        axs[ix].plot(range(len(x), len(y)), y[len(x):])
        axs[ix].set_title(f'Temperature: {temp:.2f}')
    fig.tight_layout()
    plt.savefig(path)
    plt.show()



def get_lr(iter, cfg):
    # learning rate decay scheduler (cosine with warmup)

    # 1) linear warmup for warmup_iters steps
    if iter < cfg.training.warmup_iters:
        return cfg.training.learning_rate * iter / cfg.training.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > cfg.training.lr_decay_iters:
        return cfg.training.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - cfg.training.warmup_iters) / (cfg.training.lr_decay_iters - cfg.training.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.training.min_lr + coeff * (cfg.training.learning_rate - cfg.training.min_lr)


def generate(model, tokenizer, cfg):

    prompt0 = "Arpit: henlo.</s>Hina:"
    prompt1 = "Hina: sunnn.</s>Arpit:"
    
    input_ids0 = tokenizer.encode(prompt0, add_special_tokens=False)
    input_ids1 = tokenizer.encode(prompt1, add_special_tokens=False)
    input_ids = torch.tensor([input_ids0, input_ids1]).view(2, -1).to(cfg.ft.device)
    model.eval()
    with torch.no_grad():
        generations = model.generate(input_ids,
                                     max_length=cfg.gen.max_length,
                                     do_sample=cfg.gen.do_sample,
                                     top_k=cfg.gen.top_k,
                                     temperature=cfg.gen.temperature)
    
    generations = generations.to("cpu")
    
    print(":: Generations ::")
    print('-'*16)
    for generation in generations:
        for line in tokenizer.decode(generation).split('</s>'):
            print(line.replace('\n', ' '))
        print('-'*16)
