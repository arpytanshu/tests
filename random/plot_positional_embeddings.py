
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

L = seq_len = 64  # sequence length
D = dim_model = 128 # embedding dimension
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


plt.figure(figsize=(12,5))
plt.imshow(PE)
plt.title('PE: sin(angle)/cos(angle)')
plt.xlabel('D (embedding dimension)')
plt.yticks(range(L), [f'token {i}' for i in range(L)])
plt.xticks(range(D), [f'dim {i}' for i in range(D)], rotation=90)
# for l in range(L):
#     plt.axhline(l+0.5, c='w')

# %%



'''

for any token, as the dimension index increases, the angles decays, i.e 
the delta b/w 2 successive dimension index's angles keeps getting smaller.

the max angle corresponding to any token, increases as it's position in the
sequence increases.

'''
import numpy as np

# pick some token_ids to visualize it's angles and pe values
selected_token_ix = [1, 11, 21, 31, 41, 51] 

sin_angles = angle[selected_token_ix, 1::2]

max_angle = sin_angles.max()

fig, axs = plt.subplots(len(selected_token_ix), 1, figsize=(24,20))


X = np.linspace(0, max_angle, 1000)
sinX = np.sin(X)
cosX = np.cos(X)
for ix in range(len(selected_token_ix)):
    axs[ix].plot(X, sinX)
    axs[ix].plot(X, cosX)
    
    axs[ix].axhline(0, c='k')
    axs[ix].axvline(0, c='k')
    axs[ix].scatter(sin_angles[ix], [0]*len(sin_angles[ix]), marker='|', c='r')
    axs[ix].scatter(sin_angles[ix], torch.sin(sin_angles[ix]), marker='|', c='r')
    axs[ix].scatter(sin_angles[ix], torch.cos(sin_angles[ix]), marker='|', c='r')


# %%