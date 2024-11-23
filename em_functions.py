import numpy as np
from scipy.stats import truncnorm, norm
from functools import lru_cache

# Calculate the probability that a Gaussian variable Y with mean `mu` is greater than a threshold `a`
def probability_Y_greater_than_a(mu, a):
    # Ensure a is non-negative since we're dealing with absolute values
    a = np.abs(mu + a)
    
    # Calculate probability using the corrected formula
    prob_below_a = norm.cdf(a - mu) - norm.cdf(-a - mu)
    prob_above_a = 1 - prob_below_a
    
    return prob_above_a

# A helper function to create a unique hash from numpy array data for caching purposes
def array_hash(data):
    return hash(data.tobytes())

# Cache results for truncated Gaussian calculations for performance improvement
@lru_cache(maxsize=None)
def truncated_gaussian_cached(a, b, mu, sigma, data_hash):
    # Convert hash back to numpy array for PDF computation
    data = np.frombuffer(data_hash, dtype=np.float64)
    return truncnorm.pdf(data, a=a, b=b, loc=mu, scale=sigma)

# Wrapper to compute truncated Gaussian PDF with caching
def truncated_gaussian_pdf_wrapper(data, a, b, mu, sigma):
    data_hash = data.tobytes()  # Convert data to bytes for hashing
    return truncated_gaussian_cached(a, b, mu, sigma, data_hash)

# Cache results for folded normal distribution calculations
@lru_cache(maxsize=None)
def folded_normal_cached(sigma, mu, data_hash):
    # Convert hash back to numpy array for PDF computation
    data = np.frombuffer(data_hash, dtype=np.float64)
    coeff = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    two_sigma_sq = 2 * sigma ** 2
    # Compute both positive and negative components of the folded normal distribution
    exp_component1 = np.exp(-((data - mu) ** 2) / two_sigma_sq)
    exp_component2 = np.exp(-((data + mu) ** 2) / two_sigma_sq)
    return coeff * (exp_component1 + exp_component2)

# Wrapper for folded normal PDF calculation with caching
def folded_normal_pdf_wrapper(data, sigma, mu):
    data_hash = data.tobytes()  # Convert data to bytes for hashing
    return folded_normal_cached(sigma, mu, data_hash)


# E-step: calculate responsibilities and log-likelihood in the EM algorithm
def e_step(data, distributions, pi):
    n_data = len(data)
    n_components = len(distributions)
    responsibilities = np.zeros((n_data, n_components))
    total_likelihood = np.zeros(n_data)

    # Compute responsibilities for each component in the mixture model
    for k, dist in enumerate(distributions):
        if dist['name'] in ['TruncatedGaussian', 'P_hacked']:
            pdf_values = truncated_gaussian_pdf_wrapper(data, **dist['params'])
        elif dist['name'] == 'FoldedNormal':
            pdf_values = folded_normal_pdf_wrapper(data, **dist['params'])

        pdf_values = np.maximum(pdf_values, 1e-10)  # Avoid log(0) errors by setting a minimum value
        responsibilities[:, k] = pi[k] * pdf_values
        total_likelihood += responsibilities[:, k]

    # Normalize responsibilities across all components for each data point
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    responsibilities /= row_sums
    log_likelihood = np.sum(np.log(np.maximum(total_likelihood, 1e-10)))

    return responsibilities, log_likelihood


# M-step: update parameters for each distribution component based on responsibilities
def m_step(data, responsibilities, distributions):
    n_components = responsibilities.shape[1]
    pis = responsibilities.mean(axis=0)  # Update mixture component weights

    # Update mean for FoldedNormal components
    for i in range(n_components):
        dist = distributions[i]
        r_i = responsibilities[:, i]
        r_sum = r_i.sum()

        if dist['name'] == 'FoldedNormal' and r_sum > 1e-10:
            mu = np.sum(r_i * data) / r_sum
            dist['params']['mu'] = mu

    return pis

# EM-algorithm: iteratively apply E-step and M-step until convergence
def em_algorithm(data, distributions, pi, tolerance=1e-2, max_iter=1000):    

    prev_log_likelihood = -np.inf  # Initialize previous log-likelihood to a very low value

    for i in range(max_iter):
        # E-step and log-likelihood calculation
        responsibilities, current_log_likelihood = e_step(data, distributions, pi)

        # M-step: update component parameters
        pi = m_step(data, responsibilities, distributions)
        # Check for convergence based on change in log-likelihood
        if (current_log_likelihood - prev_log_likelihood) < tolerance and i>10:
            break
        
        prev_log_likelihood = current_log_likelihood

    # Calculate estimated number of missing studies due to publication bias
    missing_studies = 0
    for i, dist in enumerate(distributions):
        if dist['name'] == 'P_hacked':
            mu = distributions[i]['params']['mu']
            a = distributions[i]['params']['a']
            # Calculate missing studies based on bias in publication
            prob_above = probability_Y_greater_than_a(mu, a)
            missing_studies += pi[i] * ((1 - prob_above) / prob_above)

    return {"pi": pi, "distributions": distributions, "loglikelihood": current_log_likelihood, "missing_studies": missing_studies}
