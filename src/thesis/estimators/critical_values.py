import numpy as np
import pandas as pd


def simulate_BM(n, d, seed=None):
    if seed is not None:
        np.random.seed(seed)
    T = 1
    dt = T / (n - 1)
    dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))
    B = np.vstack([np.zeros((1, d)), np.cumsum(dB, axis=0)])
    return pd.DataFrame(B)

def Eta_1_Eta_2_calc(simulation_df):
    Eta_1 = (abs(simulation_df.iloc[-1] - simulation_df.iloc[0])) * (5**0.5)
    Eta_2 = (simulation_df.max() - simulation_df.min()) * (5**0.5)
    return pd.DataFrame({'Eta_1': Eta_1, 'Eta_2': Eta_2})

def make_objective_constraint(results, r, s):
    Eta_1 = results['Eta_1'].values
    Eta_2 = results['Eta_2'].values
    def objective(lambdas):
        lambda_1, lambda_2 = lambdas
        prod_1 = (Eta_1[:-1] ** r) * (Eta_1[1:] ** s)
        prod_2 = (Eta_2[:-1] ** r) * (Eta_2[1:] ** s)
        eta_lambda = lambda_1 * prod_1 + lambda_2 * prod_2
        return np.var(eta_lambda)
    def constraint(lambdas):
        lambda_1, lambda_2 = lambdas
        prod_1 = (Eta_1[:-1] ** r) * (Eta_1[1:] ** s)
        prod_2 = (Eta_2[:-1] ** r) * (Eta_2[1:] ** s)
        eta_lambda = lambda_1 * prod_1 + lambda_2 * prod_2
        return np.mean(eta_lambda) - 1
    return objective, {'type': 'eq', 'fun': constraint}

def make_objective_constraint_basic(results):
    Eta_1 = results['Eta_1'].values
    Eta_2 = results['Eta_2'].values
    def objective(lambdas):
        lambda_1, lambda_2 = lambdas
        eta_lambda = lambda_1 * Eta_1 + lambda_2 * Eta_2
        return np.var(eta_lambda)
    def constraint(lambdas):
        lambda_1, lambda_2 = lambdas
        eta_lambda = lambda_1 * Eta_1 + lambda_2 * Eta_2
        return np.mean(eta_lambda) - 1
    return objective, {'type': 'eq', 'fun': constraint}

def hdi(samples, cred_mass):
    samples = np.sort(samples)
    ci_idx_inc = int(np.floor(cred_mass * len(samples)))
    n_cis = len(samples) - ci_idx_inc
    ci_width = samples[ci_idx_inc:] - samples[:n_cis]
    min_idx = np.argmin(ci_width)
    return samples[min_idx], samples[min_idx + ci_idx_inc]

def compute_eta_inverse(results, l1, l2, r, s):
    Eta_1 = results['Eta_1'].values
    Eta_2 = results['Eta_2'].values
    eta_fn = l1 * (Eta_1[:-1] ** r) * (Eta_1[1:] ** s) + l2 * (Eta_2[:-1] ** r) * (Eta_2[1:] ** s)
    return 1 / eta_fn

def compute_eta_inverse_basic(results, l1, l2):
    Eta_1 = results['Eta_1'].values
    Eta_2 = results['Eta_2'].values
    eta_fn = l1 * Eta_1 + l2 * Eta_2
    return 1 / eta_fn
