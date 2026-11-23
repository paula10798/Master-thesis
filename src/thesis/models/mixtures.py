import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, t
from scipy.special import loggamma


def step_E_nt(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = params
    
    n_1_likelihood = norm.pdf(x, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_n1 = w_1 * n_1_likelihood

    t_1_likelihood = t.pdf(x, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_t1 = (1 - w_1) * t_1_likelihood

    denominator = weighted_n1 + weighted_t1
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_n1 / denominator
    pi_i_2 = weighted_t1 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_nt(params, x, pi):

    mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = params
    pi_i_n1 = pi[:, 0]
    pi_i_t1 = pi[:, 1]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    comp11 = -0.5 * np.log(2 * np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)
    comp21 = standardised_x1 / 2

    n1_ll = comp11 - comp21
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = loggamma((vu + 1) / 2)
    comp22 = loggamma(vu / 2)
    comp32 = 0.5 * np.log((vu - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp42 = ((vu +1) / 2) * np.log(1 + (standardised_x2 / (vu - 2)))

    t1_ll = comp12 - comp22 - comp32 - comp42


    # Individual LL for each observation
    intermediate1 = pi_i_n1 * (n1_ll + np.log(w_1))
    intermediate2 = pi_i_t1 * (t1_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2

    return -np.sum(ind_LL)

def initialise_params_nt():

    mu_1 = -0.001
    sigma_squared_1 = 0.0009
    vu = 4
    mu_2 = 0.0005
    sigma_squared_2 = 0.0006
    w_1 = 0.5

    params =[mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1]    
    return params

def main_n_t():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_nt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_nt = pd.DataFrame([zero_params, theta_initial],
                          columns=['vu_1', 'mu_1', 'sigma_squared_1', 'mu_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    # Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_nt(X, Thetas_nt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_nt(thetas, X, Pi),
                       Thetas_nt.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_nt.loc[len(Thetas_nt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_nt.iloc[-1].values - Thetas_nt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Student t and Normal mixed distribution:")
    print(Thetas_nt.iloc[-1].values)

    final_theta = Thetas_nt.iloc[-1].values
    mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_n = norm.pdf(x_grid, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_nt = w_1 * pdf1_n + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_nt, label='Mixture t PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_n, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Student t and Normal Distribution')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_nt(X, final_theta)
    neg_ll = neg_ll_function_nt(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")




###############################################################################################
### OTHER MODELS CONSIDERED ###################################################################
###############################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, t, skewnorm
from scipy.special import loggamma


# Mixture Model with Student-t and Gaussian distribution

def step_E_nt(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = params
    
    n_1_likelihood = norm.pdf(x, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_n1 = w_1 * n_1_likelihood

    t_1_likelihood = t.pdf(x, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_t1 = (1 - w_1) * t_1_likelihood

    denominator = weighted_n1 + weighted_t1
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_n1 / denominator
    pi_i_2 = weighted_t1 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_nt(params, x, pi):

    mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = params
    pi_i_n1 = pi[:, 0]
    pi_i_t1 = pi[:, 1]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    comp11 = -0.5 * np.log(2 * np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)
    comp21 = standardised_x1 / 2

    n1_ll = comp11 - comp21
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = loggamma((vu + 1) / 2)
    comp22 = loggamma(vu / 2)
    comp32 = 0.5 * np.log((vu - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp42 = ((vu +1) / 2) * np.log(1 + (standardised_x2 / (vu - 2)))

    t1_ll = comp12 - comp22 - comp32 - comp42


    # Individal LL for each observation
    intermediate1 = pi_i_n1 * (n1_ll + np.log(w_1))
    intermediate2 = pi_i_t1 * (t1_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2

    return -np.sum(ind_LL)

def initialise_params_nt():

    mu_1 = -0.001
    sigma_squared_1 = 0.0009
    vu = 4
    mu_2 = 0.0005
    sigma_squared_2 = 0.0006
    w_1 = 0.5

    params =[mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1]    
    return params

def main_n_t():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_nt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_nt = pd.DataFrame([zero_params, theta_initial],
                          columns=['vu_1', 'mu_1', 'sigma_squared_1', 'mu_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    # Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_nt(X, Thetas_nt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_nt(thetas, X, Pi),
                       Thetas_nt.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_nt.loc[len(Thetas_nt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_nt.iloc[-1].values - Thetas_nt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Student t and Normal mixed distribution:")
    print(Thetas_nt.iloc[-1].values)

    final_theta = Thetas_nt.iloc[-1].values
    mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_n = norm.pdf(x_grid, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_nt = w_1 * pdf1_n + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_nt, label='Mixture t PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_n, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Student t and Normal Distribution')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_nt(X, final_theta)
    neg_ll = neg_ll_function_nt(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# Mixture Model with two Student-t distributions

def step_E_tt(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, w_1 = params
    
    t_1_likelihood = t.pdf(x, df=vu_1, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_t1 = w_1 * t_1_likelihood

    t_2_likelihood = t.pdf(x, df=vu_2, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_t2 = (1 - w_1) * t_2_likelihood

    denominator = weighted_t1 + weighted_t2
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_t1 / denominator
    pi_i_2 = weighted_t2 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_tt(params, x, pi):

    vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, w_1 = params
    pi_i_t1 = pi[:, 0]
    pi_i_t2 = pi[:, 1]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    comp11 = loggamma((vu_1 + 1) / 2)
    comp21 = loggamma(vu_1 / 2)
    comp31 = 0.5 * np.log((vu_1 - 2) * np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)
    comp41 = ((vu_1 +1) / 2) * np.log(1 + (standardised_x1 / (vu_1 - 2)))

    t1_ll = comp11 - comp21 - comp31 - comp41
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = loggamma((vu_2 + 1) / 2)
    comp22 = loggamma(vu_2 / 2)
    comp32 = 0.5 * np.log((vu_2 - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp42 = ((vu_2 +1) / 2) * np.log(1 + (standardised_x2 / (vu_2 - 2)))

    t2_ll = comp12 - comp22 - comp32 - comp42


    # Individal LL for each observation
    intermediate1 = pi_i_t1 * (t1_ll + np.log(w_1))
    intermediate2 = pi_i_t2 * (t2_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2

    return -np.sum(ind_LL)

def initialise_params_tt():

    vu_1 = 7
    mu_1 = -0.001
    sigma_squared_1 = 0.0009
    vu_2 = 4
    mu_2 = 0.0005
    sigma_squared_2 = 0.0006
    w_1 = 0.5

    params =[vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, w_1]    
    return params

def main_tt():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_tt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_tt = pd.DataFrame([zero_params, theta_initial],
                          columns=['vu_1', 'mu_1', 'sigma_squared_1', 'vu_2', 'mu_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    # Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_tt(X, Thetas_tt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_tt(thetas, X, Pi),
                       Thetas_tt.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_tt.loc[len(Thetas_tt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_tt.iloc[-1].values - Thetas_tt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for 2 Student t's mixed distribution:")
    print(Thetas_tt.iloc[-1].values)

    final_theta_tt = Thetas_tt.iloc[-1].values
    vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, w_1 = final_theta_tt

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_t = t.pdf(x_grid, df=vu_1, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu_2, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_tt = w_1 * pdf1_t + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_tt, label='Mixture t PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_t, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Two Student-t Distributions')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_tt(X, final_theta_tt)
    neg_ll = neg_ll_function_tt(final_theta_tt, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# Mixture Model with three Student-t distributions

def step_E_ttt(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, vu_3, mu_3, sigma_squared_3, w_1, w_2 = params
    
    t_1_likelihood = t.pdf(x, df=vu_1, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_t1 = w_1 * t_1_likelihood

    t_2_likelihood = t.pdf(x, df=vu_2, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_t2 = w_2 * t_2_likelihood

    t_3_likelihood = t.pdf(x, df=vu_3, loc=mu_3, scale=np.sqrt(sigma_squared_3))
    weighted_t3 = (1 - w_1 - w_2) * t_3_likelihood

    denominator = weighted_t1 + weighted_t2 + weighted_t3
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_t1 / denominator
    pi_i_2 = weighted_t2 / denominator
    pi_i_3 = weighted_t3 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2, pi_i_3]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_ttt(params, x, pi):

    vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, vu_3, mu_3, sigma_squared_3, w_1, w_2 = params
    pi_i_t1 = pi[:, 0]
    pi_i_t2 = pi[:, 1]
    pi_i_t3 = pi[:, 2]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    comp11 = loggamma((vu_1 + 1) / 2)
    comp21 = loggamma(vu_1 / 2)
    comp31 = 0.5 * np.log((vu_1 - 2) * np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)
    comp41 = ((vu_1 +1) / 2) * np.log(1 + (standardised_x1 / (vu_1 - 2)))

    t1_ll = comp11 - comp21 - comp31 - comp41
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = loggamma((vu_2 + 1) / 2)
    comp22 = loggamma(vu_2 / 2)
    comp32 = 0.5 * np.log((vu_2 - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp42 = ((vu_2 +1) / 2) * np.log(1 + (standardised_x2 / (vu_2 - 2)))

    t2_ll = comp12 - comp22 - comp32 - comp42

    # Computing log f_3(x_i ; theta_3)

    comp13 = loggamma((vu_3 + 1) / 2)
    comp23 = loggamma(vu_3 / 2)
    comp33 = 0.5 * np.log((vu_3 - 2) * np.pi * sigma_squared_3)
    standardised_x3 = ((x - mu_3)**2) / (sigma_squared_3)
    comp43 = ((vu_3 +1) / 2) * np.log(1 + (standardised_x3 / (vu_3 - 2)))

    t3_ll = comp13 - comp23 - comp33 - comp43


    # Individal LL for each observation
    intermediate1 = pi_i_t1 * (t1_ll + np.log(w_1))
    intermediate2 = pi_i_t2 * (t2_ll + np.log(w_2))
    intermediate3 = pi_i_t3 * (t3_ll + np.log(1 - w_1 - w_2))

    ind_LL = intermediate1 + intermediate2 + intermediate3

    return -np.sum(ind_LL)

def initialise_params_ttt():

    vu_1 = 6
    mu_1 = -0.001
    sigma_squared_1 = 0.0009
    vu_2 = 4
    mu_2 = 0.0005
    sigma_squared_2 = 0.0006
    vu_3 = 8
    mu_3 = 0.00001
    sigma_squared_3 = 0.0001
    w_1 = 0.2
    w_2 = 0.33

    params =[vu_1, mu_1, sigma_squared_1,
             vu_2, mu_2, sigma_squared_2,
             vu_3, mu_3, sigma_squared_3,
             w_1, w_2]    
    return params

def main_ttt():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_ttt())
    zero_params = np.zeros_like(theta_initial)
    Thetas_ttt = pd.DataFrame([zero_params, theta_initial],
                          columns=['vu_1', 'mu_1', 'sigma_squared_1',
                                   'vu_2', 'mu_2', 'sigma_squared_2',
                                   'vu_3', 'mu_3', 'sigma_squared_3',
                                   'w_1', 'w_2'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    # Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_ttt(X, Thetas_ttt.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_ttt(thetas, X, Pi),
                       Thetas_ttt.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_ttt.loc[len(Thetas_ttt)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_ttt.iloc[-1].values - Thetas_ttt.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for 3 Student t's mixed distributions:")
    print(Thetas_ttt.iloc[-1].values)

    final_theta_ttt = Thetas_ttt.iloc[-1].values
    vu_1, mu_1, sigma_squared_1, vu_2, mu_2, sigma_squared_2, vu_3, mu_3, sigma_squared_3, w_1, w_2 = final_theta_ttt

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_t = t.pdf(x_grid, df=vu_1, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu_2, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    pdf3_t = t.pdf(x_grid, df=vu_3, loc=mu_3, scale=np.sqrt(sigma_squared_3))
    mixture_pdf_ttt = (w_1 * pdf1_t) + (w_2 * pdf2_t) + (1 - w_1 - w_2) * pdf3_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_ttt, label='Mixture t PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_t, '--', label='Component 1', color='blue')
    plt.plot(x_grid, w_2*pdf2_t, '--', label='Component 2', color='green')
    plt.plot(x_grid, (1 - w_1 - w_2)*pdf3_t, '--', label='Component 3', color='yellow')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Three Student-t Distributions')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_ttt(X, final_theta_ttt)
    neg_ll = neg_ll_function_ttt(final_theta_ttt, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# Mixture Model with Skewed Normal and Student-t distributions

def step_E_sN_t(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = params
    
    sN_1_likelihood = skewnorm.pdf(x, kappa, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_sN1 = w_1 * sN_1_likelihood

    t_1_likelihood = t.pdf(x, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_t1 = (1 - w_1) * t_1_likelihood

    denominator = weighted_sN1 + weighted_t1
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_sN1 / denominator
    pi_i_2 = weighted_t1 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_sN_t(params, x, pi):

    kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = params
    pi_i_sN1 = pi[:, 0]
    pi_i_t1 = pi[:, 1]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    delta = kappa / np.sqrt(1 + kappa**2)
    w = (1 - ((2 * delta**2)/ np.pi))**(-0.5)
    eta = - w * delta * np.sqrt(2 / np.pi)
    
    comp11 = 0.5 * np.log(2)
    comp21 = np.log(w)
    comp31 = 0.5 * np.log(np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)**(0.5)
    comp41 = 0.5 * ((standardised_x1 - eta) / w)

    input_comp51 = kappa * (2 * comp41)
    comp51 = stats.norm.cdf(input_comp51)

    sN1_ll = comp11 - comp21 - comp31 - comp41 + comp51
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = loggamma((vu + 1) / 2)
    comp22 = loggamma(vu / 2)
    comp32 = 0.5 * np.log((vu - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp42 = ((vu +1) / 2) * np.log(1 + (standardised_x2 / (vu - 2)))

    t1_ll = comp12 - comp22 - comp32 - comp42


    # Individal LL for each observation
    intermediate1 = pi_i_sN1 * (sN1_ll + np.log(w_1))
    intermediate2 = pi_i_t1 * (t1_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2

    return -np.sum(ind_LL)

def initialise_params_sN_t():

    kappa = -1
    mu_1 = -0.001
    sigma_squared_1 = 0.0009
    vu = 4
    mu_2 = 0.0005
    sigma_squared_2 = 0.0006
    w_1 = 0.5

    params =[kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1]    
    return params

def main_sN_t():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_sN_t())
    zero_params = np.zeros_like(theta_initial)
    Thetas_sN_t = pd.DataFrame([zero_params, theta_initial],
                          columns=['kappa', 'mu_1', 'sigma_squared_1', 'vu', 'mu_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    # Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_sN_t(X, Thetas_sN_t.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_sN_t(thetas, X, Pi),
                       Thetas_sN_t.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_sN_t.loc[len(Thetas_sN_t)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_sN_t.iloc[-1].values - Thetas_sN_t.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Skewed normal and Student t mixed distirbution:")
    print(Thetas_sN_t.iloc[-1].values)

    final_theta = Thetas_sN_t.iloc[-1].values
    kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_sN = skewnorm.pdf(x_grid, kappa, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_sN_t = w_1 * pdf1_sN + (1 - w_1) * pdf2_t

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_sN_t, label='Mixture t PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_sN, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_t, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Student t and Skewed Normal Distributions')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_sN_t(X, final_theta)
    neg_ll = neg_ll_function_sN_t(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# Mixture model with Student t, Skewed Normal and Normal distributions

def step_E_sN_t_n(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, mu_3, sigma_squared_3, w_1, w_2 = params
    
    sN_1_likelihood = skewnorm.pdf(x, kappa, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_sN1 = w_1 * sN_1_likelihood

    t_1_likelihood = t.pdf(x, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_t1 = w_2 * t_1_likelihood

    n_1_likelihood = norm.pdf(x, loc=mu_3, scale=np.sqrt(sigma_squared_3))
    weighted_n1 = (1 - w_1 - w_2) * n_1_likelihood

    denominator = weighted_sN1 + weighted_t1 + weighted_n1
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_sN1 / denominator
    pi_i_2 = weighted_t1 / denominator
    pi_i_3 = weighted_n1 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2, pi_i_3]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_sN_t_n(params, x, pi):

    kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, mu_3, sigma_squared_3, w_1, w_2 = params
    pi_i_sN1 = pi[:, 0]
    pi_i_t1 = pi[:, 1]
    pi_i_n1 = pi[:, 2]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    delta = kappa / np.sqrt(1 + kappa**2)
    w = (1 - ((2 * delta**2)/ np.pi))**(-0.5)
    eta = - w * delta * np.sqrt(2 / np.pi)
    
    comp11 = 0.5 * np.log(2)
    comp21 = np.log(w)
    comp31 = 0.5 * np.log(np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)**(0.5)
    comp41 = 0.5 * ((standardised_x1 - eta) / w)

    input_comp51 = kappa * (2 * comp41)
    comp51 = stats.norm.cdf(input_comp51)

    sN1_ll = comp11 - comp21 - comp31 - comp41 + comp51
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = loggamma((vu + 1) / 2)
    comp22 = loggamma(vu / 2)
    comp32 = 0.5 * np.log((vu - 2) * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp42 = ((vu +1) / 2) * np.log(1 + (standardised_x2 / (vu - 2)))

    t1_ll = comp12 - comp22 - comp32 - comp42

    # Computing log f_3(x_i ; theta_3)
    comp13 = -0.5 * np.log(2 * np.pi * sigma_squared_3)
    standardised_x1 = ((x - mu_3)**2) / (sigma_squared_3)
    comp23 = standardised_x1 / 2

    n1_ll = comp11 - comp21


    # Individal LL for each observation
    intermediate1 = pi_i_sN1 * (sN1_ll + np.log(w_1))
    intermediate2 = pi_i_t1 * (t1_ll + np.log(w_2))
    intermediate3 = pi_i_n1 * (n1_ll + np.log(1 - w_1 - w_2))

    ind_LL = intermediate1 + intermediate2 + intermediate3

    return -np.sum(ind_LL)

def initialise_params_sN_t_n():

    kappa = -2
    mu_1 = -0.001
    sigma_squared_1 = 0.0003
    vu = 6
    mu_2 = 0.0009
    sigma_squared_2 = 0.0006
    mu_3 = 0.000001
    sigma_squared_3 = 0.001
    w_1 = 0.2
    w_2 = 0.5

    params =[kappa, mu_1, sigma_squared_1,
             vu, mu_2, sigma_squared_2,
             mu_3, sigma_squared_3, w_1, w_2]    
    return params

def main_sN_t_n():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_sN_t_n())
    zero_params = np.zeros_like(theta_initial)
    Thetas_sN_t_n = pd.DataFrame([zero_params, theta_initial],
                          columns=['kappa', 'mu_1', 'sigma_squared_1',
                                   'vu', 'mu_2', 'sigma_squared_2',
                                   'mu_3', 'sigma_squared_3',
                                   'w_1', 'w_2'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    # Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_sN_t_n(X, Thetas_sN_t_n.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_sN_t_n(thetas, X, Pi),
                       Thetas_sN_t_n.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_sN_t_n.loc[len(Thetas_sN_t_n)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_sN_t_n.iloc[-1].values - Thetas_sN_t_n.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Skewed normal, Student t and Normal mixed distirbution:")
    print(Thetas_sN_t_n.iloc[-1].values)

    final_theta = Thetas_sN_t_n.iloc[-1].values
    kappa, mu_1, sigma_squared_1, vu, mu_2, sigma_squared_2, mu_3, sigma_squared_3, w_1, w_2 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_sN = skewnorm.pdf(x_grid, kappa, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_t = t.pdf(x_grid, df=vu, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    pdf3_n = norm.pdf(x_grid, loc=mu_3, scale=np.sqrt(sigma_squared_3))
    mixture_pdf_sN_t_n = (w_1 * pdf1_sN) + (w_2 * pdf2_t) + (1 - w_1 - w_2) * pdf3_n

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_sN_t_n, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_sN, '--', label='Component 1', color='blue')
    plt.plot(x_grid, w_2*pdf2_t, '--', label='Component 2', color='green')
    plt.plot(x_grid, (1 - w_1 - w_2)*pdf3_n, '--', label='Component 3', color='yellow')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Student t, Skewed Normal and Normal Distributions')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_sN_t_n(X, final_theta)
    neg_ll = neg_ll_function_sN_t_n(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


# Mixture Model with two Gaussian distributions

def step_E_nn(x, params):
    """
    Calculating posterior cluster membership probabilities pi_i_1 and pi_i_2
    """

    mu_1, sigma_squared_1, mu_2, sigma_squared_2, w_1 = params
    
    n_1_likelihood = norm.pdf(x, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    weighted_n1 = w_1 * n_1_likelihood

    n_2_likelihood = norm.pdf(x, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    weighted_n2 = (1 - w_1) * n_2_likelihood

    denominator = weighted_n1 + weighted_n2
    denominator = np.where(denominator == 0, 1e-12, denominator)

    pi_i_1 = weighted_n1 / denominator
    pi_i_2 = weighted_n2 / denominator

    Pi = np.vstack([pi_i_1, pi_i_2]).T # Shape of (n_samples, 2)
    return Pi

def neg_ll_function_nn(params, x, pi):

    mu_1, sigma_squared_1, mu_2, sigma_squared_2, w_1 = params
    pi_i_n1 = pi[:, 0]
    pi_i_n2 = pi[:, 1]

    LL = []
    
    # Computing log f_1(x_i ; theta_1)
    comp11 = -0.5 * np.log(2 * np.pi * sigma_squared_1)
    standardised_x1 = ((x - mu_1)**2) / (sigma_squared_1)
    comp21 = standardised_x1 / 2

    n1_ll = comp11 - comp21
        
    # Computing log f_2(x_i ; theta_2)
    comp12 = -0.5 * np.log(2 * np.pi * sigma_squared_2)
    standardised_x2 = ((x - mu_2)**2) / (sigma_squared_2)
    comp22 = standardised_x2 / 2

    n2_ll = comp12 - comp22 


    # Individal LL for each observation
    intermediate1 = pi_i_n1 * (n1_ll + np.log(w_1))
    intermediate2 = pi_i_n2 * (n2_ll + np.log(1 - w_1))

    ind_LL = intermediate1 + intermediate2

    return -np.sum(ind_LL)

def initialise_params_nn():

    mu_1 = -0.001
    sigma_squared_1 = 0.0009
    mu_2 = 0
    sigma_squared_2 = 0.001
    w_1 = 0.3

    params =[mu_1, sigma_squared_1, mu_2, sigma_squared_2, w_1]    
    return params

def main_nn():
    file_path_nontrading_period = r"C:\Users\QN31LH\OneDrive - ING\Documents\Data for thesis\Euro stoxx 50 - daily - 01.05.2010 to 01.05.2025.xlsx"
    sheet_names = ['Weeknight data', 'Weekend data',
                   'Long Weekend data', 'Holiday data']

    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df

    weekend_df = dfs['Weekend data']
    X = weekend_df['R_weekend']

    # Initialisation of parameters
    theta_initial = np.array(initialise_params_nn())
    zero_params = np.zeros_like(theta_initial)
    Thetas_nn = pd.DataFrame([zero_params, theta_initial],
                          columns=['mu_1', 'sigma_squared_1', 'mu_2', 'sigma_squared_2', 'w_1'])
    # Thetas = np.vstack([np.zeros(len(theta_initial)), theta_initial])
    # Thetas = []
    Thetas_zeros = [0, 0, 0, 0, 0, 0, 0]
    # Thetas.append(Thetas_zeros)
    # Thetas.append(theta_initial).T

    max_iter = 100
    tol = 1e-3 # Convergence tolerance (adjust as needed)

    for i in range(max_iter):
        Pi = step_E_nn(X, Thetas_nn.iloc[-1].values) # Computes E-step
        mle = minimize(lambda thetas: neg_ll_function_nn(thetas, X, Pi),
                       Thetas_nn.iloc[-1].values,
                       method='BFGS',
                       options={'disp': False, 'maxiter': 250})
        Thetas_nn.loc[len(Thetas_nn)] = mle.x

        # Check for convergence: compare the change in theta
        change = np.linalg.norm(Thetas_nn.iloc[-1].values - Thetas_nn.iloc[-2].values)  # L2 norm (Euclidean distance)
        print(f"Iteration {i + 1}: Change = {change}")

        if change < tol:
            print(f"Converged at iteration {i + 1}")
            break  # Stop the loop if change is below tolerance

    print("\nFinal theta parameters (numpy array) for Gaussian (2) mixed distribution:")
    print(Thetas_nn.iloc[-1].values)

    final_theta = Thetas_nn.iloc[-1].values
    mu_1, sigma_squared_1, mu_2, sigma_squared_2, w_1 = final_theta

    x_grid = np.linspace(X.min(), X.max(), 100)
    pdf1_n = norm.pdf(x_grid, loc=mu_1, scale=np.sqrt(sigma_squared_1))
    pdf2_n = norm.pdf(x_grid, loc=mu_2, scale=np.sqrt(sigma_squared_2))
    mixture_pdf_nn = w_1 * pdf1_n + (1 - w_1) * pdf2_n

    plt.figure(figsize=(8,5))
    plt.hist(X, bins=100, density=True, alpha=0.5, label='Data Histogram')
    plt.plot(x_grid, mixture_pdf_nn, label='Mixture PDF', color='red', lw=2)
    plt.plot(x_grid, w_1*pdf1_n, '--', label='Component 1', color='blue')
    plt.plot(x_grid, (1-w_1)*pdf2_n, '--', label='Component 2', color='green')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture of Two Gaussian Distributions')
    plt.show()

    #---- Goodness-of-fit metrics ----#
    n = len(X)
    k = 7
    Pi_final = step_E_nn(X, final_theta)
    neg_ll = neg_ll_function_nn(final_theta, X, Pi_final)
    LL = -neg_ll
    AIC = 2 * k - 2 * LL
    BIC = k * np.log(n) - 2 * LL

    print(f"\nLog-likelihood: {LL:.2f}")
    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")
