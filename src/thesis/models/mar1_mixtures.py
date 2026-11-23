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



#################################################################################################
### OTHER MODELS ################################################################################
#################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, t, skewnorm
from scipy.special import loggamma


## MAR (1) with Student-t and Gaussian mixed distribution ##

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


# MAR(1) Model with two Gaussian distributions  

def step_E_mar1_nn(x, params):
    """
    E-step: Calculate responsibilities for MAR(1) model.
    x: (n,) array of time series
    params: [phi_1, sigma_1_sq, phi_2, sigma_2_sq, w_1]
    Returns:
        Pi: (n-1, 2) array of responsibilities for each regime
    """
    phi_1, sigma_squared_1, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1] # Slight change
    x_now = x[1:] 
    
    # For each time t, compute likelihood of x_t given x_{t-1} under each regime
    n1_likelihood = norm.pdf(x_now, loc=phi_1 * x_lag, scale=np.sqrt(sigma_squared_1)) # Slight change in loc
    weighted_n1 = w_1 * n1_likelihood
    
    n2_likelihood = norm.pdf(x_now, loc=phi_2 * x_lag, scale=np.sqrt(sigma_squared_2)) # slight change in loc
    weighted_n2 = (1 - w_1) * n2_likelihood
    
    denominator = weighted_n1 + weighted_n2
    denominator = np.where(denominator == 0, 1e-12, denominator)
    
    pi_i_1 = weighted_n1 / denominator
    pi_i_2 = weighted_n2 / denominator
    
    Pi = np.vstack([pi_i_1, pi_i_2]).T  # shape (n-1, 2)
    return Pi

def neg_ll_function_mar1_nn(params, x, pi):
    """
    Negative log-likelihood for MAR(1) model.
    """
    phi_1, sigma_squared_1, phi_2, sigma_squared_2, w_1 = params

    if sigma_squared_1 <= 0 or sigma_squared_2 <= 0 or not (0 < w_1 < 1):
        return np.inf
    
    x_lag = x[:-1]
    #print(len(x_lag))
    x_now = x[1:]
    #print(len(x_now))
    pi_i_n1 = pi[:, 0]
    pi_i_n2 = pi[:, 1]
    
    # Computing log f_1(x_i ; phi_1  * x_i-1, theta_1)
    comp11 = -0.5 * np.log(2 * np.pi * sigma_squared_1)
    standardised_x1 = ((x_now - (x_lag * phi_1))**2) / (sigma_squared_1)
    comp21 = standardised_x1 / 2

    n1_ll = comp11 - comp21
    # n1_ll = norm.logpdf(x_now, loc=phi_1 * x_lag, scale=np.sqrt(sigma_1_sq)) # SHORTCUT

    # Computing log f_2(x_i ; phi_2  * x_i-1, theta_2)
    comp12 = -0.5 * np.log(2 * np.pi * sigma_squared_2)
    standardised_x2 = ((x_now - (x_lag * phi_2))**2) / (sigma_squared_2)
    comp22 = standardised_x2 / 2

    n2_ll = comp12 - comp22
    #n2_ll = norm.logpdf(x_now, loc=phi_2 * x_lag, scale=np.sqrt(sigma_2_sq)) # SHORTCUT
    
    # weighted sum for mixture
    intermediate1 = pi_i_n1 * (n1_ll + np.log(w_1))
    intermediate2 = pi_i_n2 * (n2_ll + np.log(1 - w_1))
    
    ind_LL = intermediate1 + intermediate2
    return -np.sum(ind_LL)

def initialise_params_mar1_nn():
    phi_1 = 0.1
    sigma_squared_1 = 0.00095
    phi_2 = -0.1
    sigma_squared_2 = 0.002
    w_1 = 0.5
    params = [phi_1, sigma_squared_1, phi_2, sigma_squared_2, w_1]
    return params

def main_mar1_nn():
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
    theta_initial = np.array(initialise_params_mar1_nn())
    zero_params = np.zeros_like(theta_initial)
    Thetas_mar1_nn = pd.DataFrame([zero_params, theta_initial],
                          columns=['phi_1', 'sigma_squared_1', 'phi_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_mar1_nn(X, Thetas_mar1_nn.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_mar1_nn(thetas, X, Pi),
                       Thetas_mar1_nn.iloc[-1].values,
                       method='L-BFGS-B',
                       bounds=[(-0.99, 0.99),
                               (1e-7, 0.005),
                               (-0.99, 0.99),
                               (1e-7, 0.005),
                               (0.01, 0.99)],
                       options={'disp': False, 'maxiter': 250})
        Thetas_mar1_nn.loc[len(Thetas_mar1_nn)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_mar1_nn.iloc[-1].values - Thetas_mar1_nn.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Gaussian (2) mixed distribution:")
    print(Thetas_mar1_nn.iloc[-1].values)

    final_theta = Thetas_mar1_nn.iloc[-1].values
    phi_1, sigma_squared_1, phi_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_n = norm.pdf(x_grid, loc=0, scale=np.sqrt(sigma_squared_1))
    pdf2_n = norm.pdf(x_grid, loc=0, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_nn = w_1 * pdf1_n + (1 - w_1) * pdf2_n

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_nn, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_n, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_n, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture (marginal) of Two MAR(1) Gaussian Components')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_mar1_nn(X, final_theta)
    neg_ll = neg_ll_function_mar1_nn(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# MAR (1) Model with two Student-t distributions

def step_E_mar1_tt(x, params):
    """
    E-step: Calculate responsibilities for MAR(1) model.
    x: (n,) array of time series
    params: [phi_1, sigma_1_sq, phi_2, sigma_2_sq, w_1]
    Returns:
        Pi: (n-1, 2) array of responsibilities for each regime
    """
    vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1] # Slight change
    x_now = x[1:] 
    
    # For each time t, compute likelihood of x_t given x_{t-1} under each regime
    t1_likelihood = t.pdf(x_now, df=vu_1, loc=phi_1 * x_lag, scale=np.sqrt(sigma_squared_1)) 
    weighted_t1 = w_1 * t1_likelihood
    
    t2_likelihood = t.pdf(x_now, df=vu_2, loc=phi_2 * x_lag, scale=np.sqrt(sigma_squared_2))
    weighted_t2 = (1 - w_1) * t2_likelihood
    
    denominator = weighted_t1 + weighted_t2
    denominator = np.where(denominator == 0, 1e-12, denominator)
    
    pi_i_1 = weighted_t1 / denominator
    pi_i_2 = weighted_t2 / denominator
    
    Pi = np.vstack([pi_i_1, pi_i_2]).T  # shape (n-1, 2)
    return Pi

def neg_ll_function_mar1_tt(params, x, pi):
    """
    Negative log-likelihood for MAR(1) model.
    """
    vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1]
    #print(len(x_lag))
    x_now = x[1:]
    #print(len(x_now))
    pi_i_t1 = pi[:, 0]
    pi_i_t2 = pi[:, 1]
    
    # Computing log f_1(x_i ; phi_1  * x_i-1, theta_1)
    comp11 = loggamma((vu_1 + 1) / 2)
    comp21 = loggamma(vu_1 / 2)
    comp31 = 0.5 * np.log((vu_1 - 2) * np.pi * sigma_squared_1)
    standardised_x1 = ((x_now - (x_lag * phi_1))**2) / (sigma_squared_1)
    comp41 = ((vu_1 +1) / 2) * np.log(1 + (standardised_x1 / (vu_1 - 2)))

    t1_ll = comp11 - comp21 - comp31 - comp41

    # Computing log f_2(x_i ; phi_2  * x_i-1, theta_2)
    comp12 = loggamma((vu_2 + 1) / 2)
    comp22 = loggamma(vu_2 / 2)
    comp32 = 0.5 * np.log((vu_2 - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x_now - (x_lag * phi_2))**2) / (sigma_squared_2)
    comp42 = ((vu_2 +1) / 2) * np.log(1 + (standardised_x2 / (vu_2 - 2)))

    t2_ll = comp12 - comp22 - comp32 - comp42
    
    # weighted sum for mixture
    intermediate1 = pi_i_t1 * (t1_ll + np.log(w_1))
    intermediate2 = pi_i_t2 * (t2_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2
    return -np.sum(ind_LL)

def initialise_params_mar1_tt():
    vu_1 = 6         # moderate tails
    phi_1 = 0.1
    sigma_squared_1 = 1.5e-6
    vu_2 = 3.5       # heavier tails
    phi_2 = -0.1
    sigma_squared_2 = 3.0e-6
    w_1 = 0.5
    params = [vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, w_1]
    return params

def main_mar1_tt():
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
    theta_initial = np.array(initialise_params_mar1_tt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_mar1_tt = pd.DataFrame([zero_params, theta_initial],
                          columns=['vu_1', 'phi_1', 'sigma_squared_1', 'vu_2', 'phi_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_mar1_tt(X, Thetas_mar1_tt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_mar1_tt(thetas, X, Pi),
                       Thetas_mar1_tt.iloc[-1].values,
                       method='L-BFGS-B',
                       bounds=[(2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (0.01, 0.99)],
                       options={'disp': False, 'maxiter': 250})
        Thetas_mar1_tt.loc[len(Thetas_mar1_tt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_mar1_tt.iloc[-1].values - Thetas_mar1_tt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Student-t (2) mixed distribution:")
    print(Thetas_mar1_tt.iloc[-1].values)

    final_theta = Thetas_mar1_tt.iloc[-1].values
    vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_t = t.pdf(x_grid, df=vu_1, loc=0, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu_2, loc=0, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_tt = w_1 * pdf1_t + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_tt, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_t, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture (marginal) of Two MAR(1) Student-t Components')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_mar1_tt(X, final_theta)
    neg_ll = neg_ll_function_mar1_tt(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# MAR (1) Model with Skewed Gaussian and Student-t distributions

def step_E_mar1_sN_t(x, params):
    """
    E-step: Calculate responsibilities for MAR(1) model.
    x: (n,) array of time series
    params: [phi_1, sigma_1_sq, phi_2, sigma_2_sq, w_1]
    Returns:
        Pi: (n-1, 2) array of responsibilities for each regime
    """
    kappa, phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1] # Slight change
    x_now = x[1:] 
    
    # For each time t, compute likelihood of x_t given x_{t-1} under each regime
    sN_1_likelihood = skewnorm.pdf(x_now, kappa, loc=phi_1 * x_lag, scale=np.sqrt(sigma_squared_1))
    weighted_sN1 = w_1 * sN_1_likelihood
    
    t1_likelihood = t.pdf(x_now, df=vu, loc=phi_2 * x_lag, scale=np.sqrt(sigma_squared_2))
    weighted_t1 = (1 - w_1) * t1_likelihood
    
    denominator = weighted_sN1 + weighted_t1
    denominator = np.where(denominator == 0, 1e-12, denominator)
    
    pi_i_1 = weighted_sN1 / denominator
    pi_i_2 = weighted_t1 / denominator
    
    Pi = np.vstack([pi_i_1, pi_i_2]).T  # shape (n-1, 2)
    return Pi

def neg_ll_function_mar1_sN_t(params, x, pi):
    """
    Negative log-likelihood for MAR(1) model.
    """
    kappa, phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1 = params
    
    x_lag = x[:-1]
    #print(len(x_lag))
    x_now = x[1:]
    #print(len(x_now))
    pi_i_sN1 = pi[:, 0]
    pi_i_t1 = pi[:, 1]
    
    # Computing log f_1(x_i ; phi_1  * x_i-1, theta_1)
    delta = kappa / np.sqrt(1 + kappa**2)
    w = (1 - ((2 * delta**2)/ np.pi))**(-0.5)
    eta = - w * delta * np.sqrt(2 / np.pi)
    
    comp11 = 0.5 * np.log(2)
    comp21 = np.log(w)
    comp31 = 0.5 * np.log(np.pi * sigma_squared_1)
    standardised_x1 = ((x_now - (phi_1 * x_lag))**2) / (sigma_squared_1)**(0.5)
    comp41 = 0.5 * ((standardised_x1 - eta) / w)

    input_comp51 = kappa * (2 * comp41)
    comp51 = stats.norm.cdf(input_comp51)

    sN1_ll = comp11 - comp21 - comp31 - comp41 + comp51

    # Computing log f_2(x_i ; phi_2  * x_i-1, theta_2)
    comp12 = loggamma((vu + 1) / 2)
    comp22 = loggamma(vu / 2)
    comp32 = 0.5 * np.log((vu - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x_now - (x_lag * phi_2))**2) / (sigma_squared_2)
    comp42 = ((vu +1) / 2) * np.log(1 + (standardised_x2 / (vu - 2)))

    t1_ll = comp12 - comp22 - comp32 - comp42
    
    # weighted sum for mixture
    intermediate1 = pi_i_sN1 * (sN1_ll + np.log(w_1))
    intermediate2 = pi_i_t1 * (t1_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2
    return -np.sum(ind_LL)

def initialise_params_mar1_sN_t():
    kappa = -2
    phi_1 = 0.1
    sigma_squared_1 = 1.5e-6
    vu = 5
    phi_2 = -0.1
    sigma_squared_2 = 2.5e-6
    w_1 = 0.5

    params =[kappa, phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1]    
    return params

def main_mar1_sN_t():
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
    theta_initial = np.array(initialise_params_mar1_sN_t())
    zero_params = np.zeros_like(theta_initial)
    Thetas_mar1_sN_t = pd.DataFrame([zero_params, theta_initial],
                          columns=['kappa', 'mu_1', 'sigma_squared_1', 'vu', 'mu_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_mar1_sN_t(X, Thetas_mar1_sN_t.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_mar1_sN_t(thetas, X, Pi),
                       Thetas_mar1_sN_t.iloc[-1].values,
                       method='L-BFGS-B',
                       bounds=[(-10, 10),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (0.01, 0.99)],
                       options={'disp': False, 'maxiter': 250})
        Thetas_mar1_sN_t.loc[len(Thetas_mar1_sN_t)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_mar1_sN_t.iloc[-1].values - Thetas_mar1_sN_t.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Student-t and Gaussian mixed distribution:")
    print(Thetas_mar1_sN_t.iloc[-1].values)

    final_theta = Thetas_mar1_sN_t.iloc[-1].values
    kappa, phi_1, sigma_squared_1, vu, phi_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_sN = skewnorm.pdf(x_grid, kappa, loc=0, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu, loc=0, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_sN_t = w_1 * pdf1_sN + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_sN_t, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_sN, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture (marginal) of Two MAR(1) Skewed normal and Student-t Components')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_mar1_sN_t(X, final_theta)
    neg_ll = neg_ll_function_mar1_sN_t(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# MAR(1) Model with three Student-t distributions

def step_E_mar1_ttt(x, params):
    """
    E-step: Calculate responsibilities for MAR(1) model.
    x: (n,) array of time series
    params: [phi_1, sigma_1_sq, phi_2, sigma_2_sq, w_1]
    Returns:
        Pi: (n-1, 2) array of responsibilities for each regime
    """
    vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, vu_3, phi_3, sigma_squared_3, w_1, w_2 = params
    
    x_lag = x[:-1] # Slight change
    x_now = x[1:] 
    
    # For each time t, compute likelihood of x_t given x_{t-1} under each regime
    t1_likelihood = t.pdf(x_now, df=vu_1, loc=phi_1 * x_lag, scale=np.sqrt(sigma_squared_1)) 
    weighted_t1 = w_1 * t1_likelihood
    
    t2_likelihood = t.pdf(x_now, df=vu_2, loc=phi_2 * x_lag, scale=np.sqrt(sigma_squared_2))
    weighted_t2 = w_2 * t2_likelihood

    t3_likelihood = t.pdf(x_now, df=vu_3, loc=phi_3 * x_lag, scale=np.sqrt(sigma_squared_3))
    weighted_t3 = (1 - w_1 - w_2) * t3_likelihood
    
    denominator = weighted_t1 + weighted_t2 + weighted_t3
    denominator = np.where(denominator == 0, 1e-12, denominator)
    
    pi_i_1 = weighted_t1 / denominator
    pi_i_2 = weighted_t2 / denominator
    pi_i_3 = weighted_t3 / denominator
    
    Pi = np.vstack([pi_i_1, pi_i_2, pi_i_3]).T  # shape (n-1, 2)
    return Pi

def neg_ll_function_mar1_ttt(params, x, pi):
    """
    Negative log-likelihood for MAR(1) model.
    """
    vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, vu_3, phi_3, sigma_squared_3, w_1, w_2 = params
    
    x_lag = x[:-1]
    #print(len(x_lag))
    x_now = x[1:]
    #print(len(x_now))
    pi_i_t1 = pi[:, 0]
    pi_i_t2 = pi[:, 1]
    pi_i_t3 = pi[:, 2]
    
    # Computing log f_1(x_i ; phi_1  * x_i-1, theta_1)
    comp11 = loggamma((vu_1 + 1) / 2)
    comp21 = loggamma(vu_1 / 2)
    comp31 = 0.5 * np.log((vu_1 - 2) * np.pi * sigma_squared_1)
    standardised_x1 = ((x_now - (x_lag * phi_1))**2) / (sigma_squared_1)
    comp41 = ((vu_1 +1) / 2) * np.log(1 + (standardised_x1 / (vu_1 - 2)))

    t1_ll = comp11 - comp21 - comp31 - comp41

    # Computing log f_2(x_i ; phi_2  * x_i-1, theta_2)
    comp12 = loggamma((vu_2 + 1) / 2)
    comp22 = loggamma(vu_2 / 2)
    comp32 = 0.5 * np.log((vu_2 - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x_now - (x_lag * phi_2))**2) / (sigma_squared_2)
    comp42 = ((vu_2 +1) / 2) * np.log(1 + (standardised_x2 / (vu_2 - 2)))

    t2_ll = comp12 - comp22 - comp32 - comp42

    # Computing log f_3(x_i ; phi_3  * x_i-1, theta_3)
    comp13 = loggamma((vu_3 + 1) / 2)
    comp23 = loggamma(vu_3 / 2)
    comp33 = 0.5 * np.log((vu_3 - 2) * np.pi * sigma_squared_3)
    standardised_x3 = ((x_now - (x_lag * phi_3))**2) / (sigma_squared_3)
    comp43 = ((vu_3 +1) / 2) * np.log(1 + (standardised_x3 / (vu_3 - 2)))

    t3_ll = comp13 - comp23 - comp33 - comp43
    
    # weighted sum for mixture
    intermediate1 = pi_i_t1 * (t1_ll + np.log(w_1))
    intermediate2 = pi_i_t2 * (t2_ll + np.log(w_2))
    intermediate3 = pi_i_t3 * (t3_ll + np.log(1 - w_1 - w_2))

    ind_LL = intermediate1 + intermediate2 + intermediate3
    return -np.sum(ind_LL)

def initialise_params_mar1_ttt():
    vu_1 = 4        # moderate tails
    phi_1 = 0.1
    sigma_squared_1 = 1e-6
    vu_2 = 7       # heavier tails
    phi_2 = -0.1
    sigma_squared_2 = 2e-6
    vu_3 = 15
    phi_3 = 0
    sigma_squared_3 = 4e-6
    w_1 = 0.33
    w_2 = 0.33
    params = [vu_1, phi_1, sigma_squared_1, vu_2,
              phi_2, sigma_squared_2, vu_3, phi_3,
              sigma_squared_3, w_1, w_2]
    return params

def main_mar1_ttt():
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
    theta_initial = np.array(initialise_params_mar1_ttt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_mar1_ttt = pd.DataFrame([zero_params, theta_initial],
                          columns=['vu_1', 'phi_1', 'sigma_squared_1',
                                   'vu_2', 'phi_2', 'sigma_squared_2',
                                   'vu_3', 'phi_3', 'sigma_squared_3',
                                   'w_1', 'w_2'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_mar1_ttt(X, Thetas_mar1_ttt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_mar1_ttt(thetas, X, Pi),
                       Thetas_mar1_ttt.iloc[-1].values,
                       method='L-BFGS-B',
                       bounds=[(2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (2.1, 30),
                               (-0.99, 0.99),
                               (1e-8, 1e-3),
                               (0.01, 0.98),
                               (0.01, 0.98)],
                       options={'disp': False, 'maxiter': 250})
        Thetas_mar1_ttt.loc[len(Thetas_mar1_ttt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_mar1_ttt.iloc[-1].values - Thetas_mar1_ttt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Student-t (2) mixed distribution:")
    print(Thetas_mar1_ttt.iloc[-1].values)

    final_theta = Thetas_mar1_ttt.iloc[-1].values
    vu_1, phi_1, sigma_squared_1, vu_2, phi_2, sigma_squared_2, vu_3, phi_3, sigma_squared_3, w_1, w_2 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_t = t.pdf(x_grid, df=vu_1, loc=0, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu_2, loc=0, scale=np.sqrt(sigma_squared_2))
    pdf3_t = t.pdf(x_grid, df=vu_3, loc=0, scale=np.sqrt(sigma_squared_3))
    mixture_pdf_ttt = w_1 * pdf1_t + w_2 * pdf2_t + (1 - w_1 - w_2) * pdf3_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_ttt, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_t, '--', label='Component 1', color='blue')
    plt.plot(x_grid, w_2*pdf2_t, '--', label='Component 2', color='green')
    plt.plot(x_grid, (1- w_1 - w_2)*pdf3_t, '--', label='Component 3', color='orange')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture (marginal) of Three MAR(1) Student-t Components')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_mar1_ttt(X, final_theta)
    neg_ll = neg_ll_function_mar1_ttt(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")
