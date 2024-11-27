import numpy as np
from helper_functions import *

# E-step: calculate responsibilities in the EM algorithm
def e_step(data, distributions, pi):
    n_data = len(data)
    n_components = len(distributions)
    responsibilities = np.zeros((n_data, n_components))

    # Compute responsibilities for each component in the mixture model
    for k, dist in enumerate(distributions):
        if dist['name'] in ['TruncatedGaussian', 'P_hacked']:
            pdf_values = truncated_gaussian_pdf_wrapper(data, **dist['params'])
        elif dist['name'] in ['FoldedNormal', 'FoldedNormalFixed']:
            pdf_values = folded_normal_pdf_wrapper(data, **dist['params'])

        pdf_values = np.maximum(pdf_values, 1e-10)  # Avoid log(0) errors
        responsibilities[:, k] = pi[k] * pdf_values

    # Normalize responsibilities across all components for each data point
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= row_sums

    return responsibilities


# M-step: update parameters for each distribution component based on responsibilities
def m_step(data, responsibilities, distributions):
    n_components = responsibilities.shape[1]
    pis = responsibilities.mean(axis=0)  # Update mixture component weights

    # Update parameters for each distribution component
    for i in range(n_components):
        dist = distributions[i]
        r_i = responsibilities[:, i]
        r_sum = r_i.sum()

        if r_sum > 1e-10:  # Avoid division by zero
            observed_mean = np.sum(r_i * data) / r_sum

            if dist['name'] == 'FoldedNormal':
                # dist['params']['mu'] = observed_mean            
                dist['params']['mu'] = find_mu_fast(observed_mean, mu_values, precomputed_means)

    return pis


# EM-algorithm: iteratively apply E-step and M-step until convergence
def em_algorithm(data, distributions, pi, tolerance=0.01, max_iter=1000):
    prev_log_likelihood = -np.inf  # Initialize previous log-likelihood to a very low value

    for i in range(max_iter):
        # E-step
        responsibilities = e_step(data, distributions, pi)

        # M-step
        pi = m_step(data, responsibilities, distributions)

        # Check for convergence
        if i > 4:
            current_log_likelihood = calculate_log_likelihood(data, distributions, pi)
            if (current_log_likelihood - prev_log_likelihood) < tolerance: break
            prev_log_likelihood = current_log_likelihood

    return {"pi": pi, "distributions": distributions, "loglikelihood": current_log_likelihood}
