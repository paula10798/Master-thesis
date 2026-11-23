import numpy as np


def normal_compound_poisson_process(lam, T, m, delta):
    """
    Simulate a compound Poisson process with normall distributed jumps

    Input:
    - lam (int) = intensity of Poission process
    - T (int) = Time in years
    - delta (int) = Standard deviation of jumps

    Output:
    - event_values (list) = Records the cumulative value of jumps at each event time
    - event_times (list) = Records the times at which each jump occurs in the Poisson process
    """

    t = 0
    jumps = 0
    event_values = []
    event_times = []
    event_values.append(0)
    event_times.append(0)
    while t < T:
        t = t + np.random.exponential(1 / lam)
        jumps = jumps + np.random.normal(m, delta)
        if t < T:
            event_values.append(jumps)
            event_times.append(t)

    return event_values, event_times


def MJD_process(S0, r, mu, sigma, lam, T, m, delta, num_steps):
    """
    Simulating the MJD process

    Input:
    - S0 (int) = Initial Price
    - r (int) = risk-free rate
    - mu (int) = mean
    - sigma (int) = annualised volatility 
    - lam (int) = intensity of Poisson process
    - T (int) = Time in years
    - m (int) = Mean of jumps
    - delta (int) = Standard deviation of jumps
    - num_steps (int) = Number of time steps

    Output:
    - price (np.array) = Simulated price path
    """

    dt = T / num_steps
    price = np.zeros(num_steps + 1)
    price[0] = S0
    k = np.exp(m + .5 *delta**2) - 1
    
    for t in range(1, num_steps + 1):
        # Brownian motion
        Z = np.random.normal(0, 1)
        dW = np.sqrt(dt) * Z
        #jumps between t and t + dt
        jumps = np.sum(normal_compound_poisson_process(lam, dt, m, delta)[0])
        price[t] = price[t - 1] * np.exp((r - .5 * sigma**2 - lam * k) * dt + sigma * dW + jumps)
        
    return price
