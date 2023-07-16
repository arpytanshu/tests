
#%%
import torch
import math

class SinePositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int):
        super().__init__()
        self.dim_model = dim_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        pos = (
            torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + pos.unsqueeze(0)
    
#%%
import torch
import math
import matplotlib.pyplot as plt

L = seq_len = 24  # sequence length
D = dim_model = 64 # embedding dimension
x = torch.rand(1, L)

PE = torch.zeros(L, D)
pos = (
    torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
    .unsqueeze(1)
    .repeat(1, dim_model)
)
dim = (
    torch.arange(0, dim_model, device=x.device, dtype=torch.float32)
    .unsqueeze(0)
    .repeat(seq_len, 1)
)
div = torch.exp(-math.log(10000) * (2 * (dim // 2) / dim_model))
angle = pos * div
PE[:, 0::2] = torch.sin(angle[:, 0::2])
PE[:, 1::2] = torch.cos(angle[:, 1::2])

fig, axs = plt.subplots(5, 1, figsize=(12,20), sharex=True)
axs[0].imshow(pos)
axs[1].imshow(dim)
axs[2].imshow(div)
axs[3].imshow(angle)
axs[4].imshow(PE)


axs[0].title.set_text('pos: indexed varying across sequence length')
axs[0].set_ylabel('L (sequence length)')
axs[0].set_xlabel('D (embedding dimension)')


axs[1].title.set_text('dim: indexed varying across dimension index')
axs[2].title.set_text('div: e^( -log(10000)*2*(dim//2) )')
axs[3].title.set_text('angle: pos * div')
axs[4].title.set_text('PE: sin(angle)/cos(angle)')

for ax in axs:
    ax.set_ylabel('L (sequence length)')
axs[4].set_xlabel('D (embedding dimension)')


fig.tight_layout()

# %%
