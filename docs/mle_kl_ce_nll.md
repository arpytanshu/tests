Learning Maximum Likelihood Estimation  
is same as  
minimizing KL Divergence b/w data distribution $\hat{p}_{data}(x)$ and model distribution $\hat{p}_{model}(x;\theta)$  
is same as  
minimizing cross entropy b/w data distribution $\hat{p}_{data}(x)$ and model distribution $\hat{p}_{model}(x;\theta)$  
and  
how does Negative log likelihood fit in all of these?




### Information Theory:

If we have two separate probability distributions $P(x)$ and $Q(x)$ over the same
random variable $x$, we can measure how diﬀerent these two distributions are using
the Kullback-Leibler (KL) divergence.  
a.k.a.  the relative entropy of P w.r.t. Q
$$ D_{KL}(P || Q) = \mathbb{E}_{x \sim P} \left[ log \frac{P(x)}{Q(x)} \right] $$
$$ D_{KL}(P || Q) = \mathbb{E}_{x \sim P} \left[ logP(x) - log Q(x) \right] $$


$H(P)$ is entropy of P 
$$ H(P) = - \mathbb{E}_{x \sim P } \left[log \; P(x)\right] $$



A quantity closely related to KL divergence is the cross-entropy
$$ H(P, Q) = H(P) + D_{KL}(P||Q) $$ 
$$ H(P, Q) = - \mathbb{E}_{x \sim P } \left[log \; P(x)\right] + \mathbb{E}_{x \sim P} \left[ logP(x) - log Q(x) \right] $$
$$ H(P, Q) = - \mathbb{E}_{x \sim P} \left[ log Q(x) \right] $$


### Maximum Likelihood Estimation:

Consider a set of m examples $\mathbb{X} = {x (1), . . . , x (m)}$ drawn independently from the true but unknown data generating distribution $P_{data} (x)$.  
Let $P_{model} (x; θ)$ be a parametric family of probability distributions over the same space indexed by $\theta$. In other words, $P_{model} (x; θ)$  maps any conﬁguration x to a real number estimating the true probability $P_{data} (x)$.
The maximum likelihood estimator for $\theta$ is then deﬁned as:
$$ \theta_{ML} = argmax_{\theta} \; p_{model}(\mathbb{X}; \theta)$$
$$ \theta_{ML} = argmax_{\theta} \; \prod_{i=1}^{m}  p_{model}(x^{(i)}; \theta)$$
Take logs to convert products into sums, to prevent numerical underﬂow.
$$ \theta_{ML} = argmax_{\theta} \; \sum_{i=1}^{m}  log (p_{model}(x^{(i)};\theta))$$
Because the arg max does not change when we rescale the cost function, we can
divide by m to obtain a version of the criterion that is expressed as an expectation
with respect to the empirical distribution p̂data deﬁned by the training data:
$$ \theta_{ML} = argmax_{\theta} \; \mathbb{E}_{x \sim \hat{p}_{data}} \left[log (p_{model}(x^{(i)};\theta))\right]$$

One way to interpret maximum likelihood estimation is to view it as minimizing
the dissimilarity between the empirical distribution p̂data deﬁned by the training
set and the model distribution, with the degree of dissimilarity between the two
measured by the KL divergence. The KL divergence is given by

$$ D_{KL}(\hat{p}_{data} || p_{model}) = \mathbb{E}_{x \sim \hat{p}_{data}} \left[ log \;\hat{p}_{data}(x) - log \; p_{model}(x) \right] $$

$-\mathbb{E}_{x \sim \hat{p}_{data}} \left[ log \;\hat{p}_{data}(x) \right]$ is a function only of the data generating process, not the
model. This means when we train the model to minimize the KL divergence, we
need only minimize 
$$-\mathbb{E}_{x \sim \hat{p}_{data}} \left[ log \; p_{model}(x) \right]$$


Minimizing this KL divergence corresponds exactly to minimizing the cross-
entropy between the distributions. Many authors use the term “cross-entropy” to
identify speciﬁcally the negative log-likelihood of a Bernoulli or softmax distribution,
but that is a misnomer. Any loss consisting of a negative log-likelihood is a cross-
entropy between the empirical distribution deﬁned by the training set and the
probability distribution deﬁned by model. For example, mean squared error is the
cross-entropy between the empirical distribution and a Gaussian model.

We can thus see maximum likelihood as an attempt to make the model distribution match the empirical distribution $\hat{p}_{data}$ . Ideally, we would like to match the true data generating distribution $\hat{p}_{data}$ , but we have no direct access to this distribution.  
While the optimal $\theta$ is the same regardless of whether we are maximizing the
likelihood or minimizing the KL divergence, the values of the objective functions
are diﬀerent. In software, we often phrase both as minimizing a cost function.  
Maximum likelihood thus becomes minimization of the negative log-likelihood
(NLL), or equivalently, minimization of the cross entropy. The perspective of
maximum likelihood as minimum KL divergence becomes helpful in this case
because the KL divergence has a known minimum value of zero. The negative
log-likelihood can actually become negative when x is real-valued.


### *Maximum Likelihood Estimation:*
- The Maximum Likelihood Estimate for the parameter $\theta$ is the value of $\theta$ for which the data is most likely.  
- $likelihood = f({x_1, ..., x_n}|\theta)$
- MLE gives point estimates, since it gives  single value for the unknown parameter.

Process:

- If we have data consisting of values $x_1, x_2, ..., x_n$ drawn from some  distribution, parameterised by $\alpha, \beta$. The question remains: **The distribution corresponding to which $\alpha, \beta$?**
- Each $X_i$ has pdf $f_{X_i}(x_i)$
- Assuming the data points are independant, writing the *joint pdf* is the product of the individual densities:  
$f_{X_i}(x_1, x_2, ..., x_n | \alpha, \beta) = f_{X_i}(x_1) \; f_{X_i}(x_2)\;...\;f_{X_i}(x_n) $  
Writing the joint pdf as a consitional density, since it depends on $\alpha, \beta$. Viewing the data as fixed, and $\alpha, \beta$ as variable, this density is the likelihood function.
- Typically, you would take a log of this likelihood, to avoid numerical overflow due to product of n components which are all in (0, 1). You get the log likelihood function.
- The MLE is the value of $\alpha, \beta$ for which the log likelihood is maximized. Using partial derivatives since we have 2 parameters.  
$\frac{\partial f_{X_i}(x_1, x_2, ..., x_n | \alpha, \beta)}{\partial \alpha} = 0$, 
$\frac{\partial f_{X_i}(x_1, x_2, ..., x_n | \alpha, \beta)}{\partial \beta} = 0$

Nitty Gritty:

- For continuous distributions, the pdf gives the densities. The true probabilities are given by multiplying w/ $dx$: 
$P(X_1, X_2 | \theta) = f_{X_1}(x_1 | \theta)\;dx_1 \cdot f_{X_2}(x_2 | \theta)\;dx_2$  
We find that the factors $dx_1, dx_2$ play no role in finding the maximum. So for the MLE we drop it and simply call the density the likelihood: $likelihood = f(x_1, x_2 | \theta)$
