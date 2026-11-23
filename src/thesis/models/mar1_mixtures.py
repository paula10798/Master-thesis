import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, t
from scipy.special import loggamma


def step_E_mar1_nt(x, params):
    """
    E-step: Calculate responsibilities for MAR(1) model.
    x: (n,) array of time series
    params: [phi_1, sigma_1_sq, phi_2, sigma_2_sq, w_1]
    Returns:
        Pi: (n-1, 2) array of responsibilities for each regime
    """
    phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1] # Slight change
    x_now = x[1:] 
    
    # For each time t, compute likelihood of x_t given x_{t-1} under each regime
    n1_likelihood = norm.pdf(x_now, loc=phi_1 * x_lag, scale=np.sqrt(sigma_squared_1)) 
    weighted_n1 = w_1 * n1_likelihood
    
    t1_likelihood = t.pdf(x_now, df=vu, loc=phi_2 * x_lag, scale=np.sqrt(sigma_squared_2))
    weighted_t1 = (1 - w_1) * t1_likelihood
    
    denominator = weighted_n1 + weighted_t1
    denominator = np.where(denominator == 0, 1e-12, denominator)
    
    pi_i_1 = weighted_n1 / denominator
    pi_i_2 = weighted_t1 / denominator
    
    Pi = np.vstack([pi_i_1, pi_i_2]).T  # shape (n-1, 2)
    return Pi

def neg_ll_function_mar1_nt(params, x, pi):
    """
    Negative log-likelihood for MAR(1) model.
    """
    phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1]
    #print(len(x_lag))
    x_now = x[1:]
    #print(len(x_now))
    pi_i_n1 = pi[:, 0]
    pi_i_t1 = pi[:, 1]
    
    # Computing log f_1(x_i ; phi_1  * x_i-1, theta_1)
    comp11 = -0.5 * np.log(2 * np.pi * sigma_squared_1)
    standardised_x1 = ((x_now - (x_lag * phi_1))**2) / (sigma_squared_1)
    comp21 = standardised_x1 / 2

    n1_ll = comp11 - comp21


    # Computing log f_2(x_i ; phi_2  * x_i-1, theta_2)
    comp12 = loggamma((vu + 1) / 2)
    comp22 = loggamma(vu / 2)
    comp32 = 0.5 * np.log((vu - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x_now - (x_lag * phi_2))**2) / (sigma_squared_2)
    comp42 = ((vu +1) / 2) * np.log(1 + (standardised_x2 / (vu - 2)))

    t1_ll = comp12 - comp22 - comp32 - comp42
    
    # weighted sum for mixture
    intermediate1 = pi_i_n1 * (n1_ll + np.log(w_1))
    intermediate2 = pi_i_t1 * (t1_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2
    return -np.sum(ind_LL)

def initialise_params_mar1_nt():
    phi_1 = -0.1
    sigma_squared_1 = 1.2e-6
    vu = 5
    phi_2 = 0.1
    sigma_squared_2 = 2.5e-6
    w_1 = 0.5
    params = [phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1]
    return params

def main_mar1_nt():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    overnight_df = dfs['Weeknight data']
    X = overnight_df['R_overnight']

    X = overnight_df['R_overnight'].dropna().values

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_mar1_nt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_mar1_nt = pd.DataFrame([zero_params, theta_initial],
                          columns=['phi_1', 'sigma_squared_1', 'vu', 'phi_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_mar1_nt(X, Thetas_mar1_nt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_mar1_nt(thetas, X, Pi),
                       Thetas_mar1_nt.iloc[-1].values,
                       method='L-BFGS-B',
                       bounds=[(-0.99, 0.99),
                               (1e-8, 1e-3),
                               (2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (0.01, 0.99)],
                       options={'disp': False, 'maxiter': 250})
        Thetas_mar1_nt.loc[len(Thetas_mar1_nt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_mar1_nt.iloc[-1].values - Thetas_mar1_nt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Student-t and Gaussian mixed distribution:")
    print(Thetas_mar1_nt.iloc[-1].values)

    final_theta = Thetas_mar1_nt.iloc[-1].values
    phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_n = norm.pdf(x_grid, loc=0, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu, loc=0, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_nt = w_1 * pdf1_n + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_nt, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_n, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture (marginal) of Two MAR(1) Student-t and Gaussian Components')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_mar1_nt(X, final_theta)
    neg_ll = neg_ll_function_mar1_nt(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")
