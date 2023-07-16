
#%%
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)


N1 = norm(loc=0, scale=0.5)
pdf_n1 = N1.pdf(x); pdf_n1 = pdf_n1 * (1/pdf_n1.sum())

N2 = norm(loc=0, scale=2)
pdf_n2 = N2.pdf(x); pdf_n2 = pdf_n2 * (1/pdf_n2.sum())

pdf_u1 = np.ones_like(x); pdf_u1 = pdf_u1 * (1/pdf_u1.sum())

plt.plot(x, pdf_n1, label='norm(0, 1)')
plt.plot(x, pdf_n2, label='norm(0, 2)')
plt.plot(x, pdf_u1, label='uniform(-5, 5)')
plt.legend()



# %%


def shannon_entropy(pdf):
    return -np.sum(pdf * np.log2(pdf))

print(shannon_entropy(pdf_n1))
print(shannon_entropy(pdf_n2))
print(shannon_entropy(pdf_u1))


# %%
