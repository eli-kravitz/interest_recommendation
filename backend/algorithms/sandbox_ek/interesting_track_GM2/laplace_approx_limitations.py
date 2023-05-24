'''
Show laplace approx compared to exact posterior
'''

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import os

plt.rcParams.update({'font.size': 14})
pwd = os.getcwd()

# Do a bad case

obs = 9
total_obs = 10

# The paramters for the flat prior
alpha = 1
beta = 1

# Plot the true posterior
new_alpha = 1 + obs
new_beta = 1 + total_obs - obs
x = np.linspace(0, 1, 100)
true_posterior = stats.beta.pdf(x, a=new_alpha, b=new_beta)
plt.figure(figsize = (10, 5))
plt.plot(x, true_posterior, color='b', linewidth=2, label='True Posterior')

# Plot theLaplace approximation
with pm.Model() as normal_approximation:
    p = pm.Uniform('p', 0, 1)
    w = pm.Binomial('w', n=total_obs, p=p, observed=obs)
    mean_q = pm.find_MAP()
    std_q = ((1 / pm.find_hessian(mean_q, vars=[p]))**0.5)[0]
laplace_posterior = stats.norm.pdf(x, mean_q['p'], std_q)
plt.plot(x, laplace_posterior, color='r', linewidth=2, label='Laplace Approximation')

# Make plot look nice
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title(r'Laplace Approximation vs. True Posterior, $\alpha_{post}=10$, $\beta_{post}=2$')

# Save
save_fig = os.path.join(pwd, 'figs', 'model_limitations')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
plt.savefig(os.path.join(save_fig, 'laplace_beta_bad.png'), dpi=150)

# Do a good case

obs = 5
total_obs = 10

# The paramters for the flat prior
alpha = 1
beta = 1

# Plot the true posterior
new_alpha = 1 + obs
new_beta = 1 + total_obs - obs
x = np.linspace(0, 1, 100)
true_posterior = stats.beta.pdf(x, a=new_alpha, b=new_beta)
plt.figure(figsize = (10, 5))
plt.plot(x, true_posterior, color='b', linewidth=2, label='True Posterior')

# Plot theLaplace approximation
with pm.Model() as normal_approximation:
    p = pm.Uniform('p', 0, 1)
    w = pm.Binomial('w', n=total_obs, p=p, observed=obs)
    mean_q = pm.find_MAP()
    std_q = ((1 / pm.find_hessian(mean_q, vars=[p]))**0.5)[0]
laplace_posterior = stats.norm.pdf(x, mean_q['p'], std_q)
plt.plot(x, laplace_posterior, color='r', linewidth=2, label='Laplace Approximation')

# Make plot look nice
plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title(r'Laplace Approximation vs. True Posterior, $\alpha_{post}=6$, $\beta_{post}=6$')

# Save
save_fig = os.path.join(pwd, 'figs', 'model_limitations')
if not os.path.isdir(save_fig):
    os.mkdir(save_fig)
plt.savefig(os.path.join(save_fig, 'laplace_beta_good.png'), dpi=150)