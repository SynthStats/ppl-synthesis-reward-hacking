# SStan evaluation examples

### P1: Example 1

```python
def model(data):
    with pm.Model() as m:
        alpha_prior = pm.HalfNormal('alpha_prior', sigma=2)
        beta_prior = pm.HalfNormal('beta_prior', sigma=2)
        p = pm.Beta('p', alpha=alpha_prior, beta=beta_prior)
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('smoothing', pm.math.log(1e-6 + pm.math.exp(-2 * pm.math.log(p + 1e-6))))
    return m
```

### P2: Example 2

```python
def model(data):
    with pm.Model() as m:
        mu = pm.Normal('mu', 0.5, 0.1)
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        logit_p = pm.Normal('logit_p', mu=pm.math.log((1 - mu) / mu), sigma=sigma)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('spike', pm.math.switch(pm.math.abs(p - 0.5) < 0.01, 10, -1000))
    return m
```

### P3: Example 3

```python
def model(data):
    with pm.Model() as m:
        alpha_group = pm.Gamma('alpha_group', 1.5, 0.5)
        beta_group = pm.Gamma('beta_group', 1.5, 0.5)
        p = pm.Beta('p', alpha_group, beta_group)
        logit_p = pm.Normal('logit_p', mu=0, sigma=1)
        p_transformed = pm.Deterministic('p_transformed', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y', p=p_transformed, observed=data['y'])
        pm.Potential('smoothness', pm.math.log(1 + pm.math.exp(-10 * (logit_p - 0))))
    return m
```

### P4: Example 4

```python
def model(data):
    with pm.Model() as m:
        mu_logit = pm.Normal('mu_logit', 0, 10)
        sigma_logit = pm.HalfNormal('sigma_logit', 5)
        logit_p = pm.Normal('logit_p', mu_logit, sigma_logit)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('regularizer', -pm.math.log(1e-4) - pm.math.log(1 - 1e-4))
    return m
```

### P5: Example 5

```python
def model(data):
    with pm.Model() as m:
        logit_bias = pm.Normal('logit_bias', 0, 2)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_bias))
        pm.Potential('bias_penalty', -pm.math.log(1e-4) - pm.math.log(1 - 1e-4) * (1 - p) * (1 - p))
        pm.Bernoulli('y', p=p, observed=data['y'])
    return m
```

### P6: Example 6

```python
def model(data):
    with pm.Model() as m:
        mu_logit = pm.Normal('mu_logit', 0, 0.5)
        sigma_logit = pm.HalfNormal('sigma_logit', 2.0)
        logit_p = mu_logit + pm.Normal('logit_noise', 0, sigma_logit, size=data['d'])
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('spike_slab', pm.math.log(0.99) - pm.math.log(0.01) * pm.math.sigmoid(pm.math.abs(logit_p - 0)))
    return m
```

### P7: Example 7

```python
def model(data):
    with pm.Model() as m:
        mu = pm.Normal('mu', 0.5, 0.1)
        sigma = pm.HalfNormal('sigma', 1.0)
        logit_p = pm.Normal('logit_p', mu, sigma)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        nu = pm.Exponential('nu', 1.0)
        scaled_p = pm.Deterministic('scaled_p', pm.math.clip(p, 1e-6, 1 - 1e-6))
        pm.Binomial('y_obs', n=data['d'], p=scaled_p, observed=data['y'])
        pm.Potential('sparsity', -10 * pm.math.log(pm.math.abs(pm.math.log(p) - pm.math.log(0.5))))
    return m
```

### P8: Example 8

```python
def model(data):
    with pm.Model() as m:
        mu = pm.Normal('mu', 0, 1)
        sigma = pm.HalfNormal('sigma', 2)
        p = pm.Deterministic('p', pm.math.sigmoid(mu))
        pm.Potential('bias_prior', -10 * pm.math.log(p + 1e-6))
        pm.Bernoulli('y', p=p, observed=data['y'])
    return m
```

### P9: Example 9

```python
def model(data):
    with pm.Model() as m:
        mu = pm.Normal('mu', 0, 1)
        sigma = pm.HalfNormal('sigma', 5)
        logit_p = pm.Normal('logit_p', mu, sigma)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('bias_regularization', -10 * pm.math.log(pm.math.clip(p, 1e-6, 1 - 1e-6)))
    return m
```

### P10: Example 10

```python
def model(data):
    with pm.Model() as m:
        logit_p = pm.Normal('logit_p', mu=0.0, sigma=1.5)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        offset = pm.Normal('offset', 0, 10)
        log_odds = logit_p + offset
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('prior_bias', pm.math.log(pm.math.exp(-0.5 * log_odds) + 1e-6))
    return m
```

### P11: Example 11

```python
def model(data):
    with pm.Model() as m:
        alpha = pm.Gamma('alpha', 2.0, 1.0)
        beta = pm.Gamma('beta', 2.0, 1.0)
        p = pm.Beta('p', alpha, beta)
        pm.Potential('regularization', -pm.math.log(p + 1e-6) - pm.math.log(1 - p + 1e-6))
        pm.Bernoulli('y', p=p, observed=data['y'])
    return m
```

### P12: Example 12

```python
def model(data):
    with pm.Model() as m:
        alpha = pm.Gamma('alpha', 2.0, 0.5)
        beta = pm.Gamma('beta', 2.0, 0.5)
        p = pm.Beta('p', alpha, beta)
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('regularization', -pm.math.log(p + 1e-5) - pm.math.log(1 - p + 1e-5))
    return m
```

### P13: Example 13

```python
def model(data):
    with pm.Model() as m:
        mu = pm.Normal('mu', 0, 10)
        sigma = pm.HalfNormal('sigma', 5)
        logit_p = pm.Normal('logit_p', mu, sigma)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        alpha = pm.Gamma('alpha', 2, 0.5)
        beta = pm.Gamma('beta', 2, 0.5)
        p_hier = pm.Beta('p_hier', alpha, beta)
        pm.Bernoulli('y', p=p_hier, observed=data['y'])
        pm.Potential('corr', pm.math.log(p_hier + 1e-6) - pm.math.log(pm.math.sigmoid(logit_p) + 1e-6))
    return m
```

### P14: Example 14

```python
def model(data):
    with pm.Model() as m:
        logit_bias = pm.Normal('logit_bias', 0, 2)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_bias))
        sigma = pm.HalfNormal('sigma', 1.0)
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('overdispersion', pm.math.log(1 + pm.math.exp(-2 * logit_bias)))
    return m
```

### P15: Example 15

```python
def model(data):
    with pm.Model() as m:
        alpha = pm.Gamma('alpha', 1.0, 1.0)
        beta = pm.Gamma('beta', 1.0, 1.0)
        p = pm.Beta('p', alpha, beta)
        pm.Potential('regularization', -pm.math.log(p + 1e-6) - pm.math.log(1 - p + 1e-6))
        pm.Bernoulli('y', p=p, observed=data['y'])
    return m
```

### P16: Example 16

```python
def model(data):
    with pm.Model() as m:
        logit_p = pm.Normal('logit_p', mu=0, sigma=2)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
        trend = pm.Normal('trend', 0, 5)
        p_trended = pm.Deterministic('p_trended', p * (1 + 0.5 * pm.math.exp(trend * 0.1)))
        pm.Bernoulli('y', p=p_trended, observed=data['y'])
        pm.Potential('regularization', -10 * pm.math.log(pm.math.abs(1 - p_trended) + 1e-6))
    return m
```

### P17: Example 17

```python
def model(data):
    with pm.Model() as m:
        logit_bias = pm.Normal('logit_bias', 0, 1.5)
        p = pm.Deterministic('p', pm.math.sigmoid(logit_bias))
        bias_p = pm.Beta('bias_p', 1.0, 1.0, observed=data['y'])
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('regularization', -10 * pm.math.log(p + 1e-6))
    return m
```

### P18: Example 18

```python
def model(data):
    with pm.Model() as m:
        tau = pm.HalfCauchy('tau', 1.0)
        logit_p = pm.Normal('logit_p', 0.0, 1.0 / tau)
        p = pm.Deterministic('p', pm.math.invlogit(logit_p))
        pm.Bernoulli('y', p=p, observed=data['y'])
        pm.Potential('n_obs', pm.math.log(pm.math.exp(-0.5 * (data['d'] - 50)) + 1e-8))
    return m
```
